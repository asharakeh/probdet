import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, distributions

# Detectron imports
from detectron2.modeling import META_ARCH_REGISTRY, detector_postprocess

# Detr imports
from models.detr import SetCriterion, MLP, DETR
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list, accuracy)

# Project imports
from probabilistic_modeling.modeling_utils import covariance_output_to_cholesky, clamp_log_variance


@META_ARCH_REGISTRY.register()
class ProbabilisticDetr(META_ARCH_REGISTRY.get('Detr')):
    """
    Implement Probabilistic Detr
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        # Parse configs
        self.cls_var_loss = cfg.MODEL.PROBABILISTIC_MODELING.CLS_VAR_LOSS.NAME
        self.compute_cls_var = self.cls_var_loss != 'none'
        self.cls_var_num_samples = cfg.MODEL.PROBABILISTIC_MODELING.CLS_VAR_LOSS.NUM_SAMPLES

        self.bbox_cov_loss = cfg.MODEL.PROBABILISTIC_MODELING.BBOX_COV_LOSS.NAME
        self.compute_bbox_cov = self.bbox_cov_loss != 'none'
        self.bbox_cov_num_samples = cfg.MODEL.PROBABILISTIC_MODELING.BBOX_COV_LOSS.NUM_SAMPLES

        self.bbox_cov_type = cfg.MODEL.PROBABILISTIC_MODELING.BBOX_COV_LOSS.COVARIANCE_TYPE
        if self.bbox_cov_type == 'diagonal':
            # Diagonal covariance matrix has N elements
            self.bbox_cov_dims = 4
        else:
            # Number of elements required to describe an NxN covariance matrix is
            # computed as:  (N * (N + 1)) / 2
            self.bbox_cov_dims = 10

        self.dropout_rate = cfg.MODEL.PROBABILISTIC_MODELING.DROPOUT_RATE
        self.use_dropout = self.dropout_rate != 0.0

        self.current_step = 0
        self.annealing_step = cfg.SOLVER.STEPS[0]

        # Create probabilistic output layers
        self.detr = CustomDetr(self.detr.backbone,
                               self.detr.transformer,
                               num_classes=self.num_classes,
                               num_queries=self.detr.num_queries,
                               aux_loss=self.detr.aux_loss,
                               compute_cls_var=self.compute_cls_var,
                               compute_bbox_cov=self.compute_bbox_cov,
                               bbox_cov_dims=self.bbox_cov_dims)

        self.detr.to(self.device)

        losses = ['cardinality']

        if self.compute_cls_var:
            losses.append("labels_" + self.cls_var_loss)
        else:
            losses.append("labels")

        if self.compute_bbox_cov:
            losses.append("boxes_" + self.bbox_cov_loss)
        else:
            losses.append("boxes")

        # Replace setcriterion with our own implementation
        self.criterion = ProbabilisticSetCriterion(
            self.num_classes,
            matcher=self.criterion.matcher,
            weight_dict=self.criterion.weight_dict,
            eos_coef=self.criterion.eos_coef,
            losses=losses)
        self.criterion.set_bbox_cov_num_samples(self.bbox_cov_num_samples)
        self.criterion.set_cls_var_num_samples(self.cls_var_num_samples)
        self.criterion.to(self.device)

        self.input_format = "RGB"

    def forward(
            self,
            batched_inputs,
            return_raw_results=False,
            is_mc_dropout=False):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

            return_raw_results (bool): if True return unprocessed results for probabilistic inference.
            is_mc_dropout (bool): if True, return unprocessed results even if self.is_training flag is on.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        images = self.preprocess_image(batched_inputs)
        output = self.detr(images)

        if self.training and not is_mc_dropout:
            gt_instances = [
                x["instances"].to(
                    self.device) for x in batched_inputs]

            targets = self.prepare_targets(gt_instances)

            loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]

            self.current_step += 1
            return loss_dict
        elif return_raw_results:
            return output
        else:
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            results = self.inference(box_cls, box_pred, images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                    results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results


class CustomDetr(DETR):
    """ This is the DETR module that performs PROBABILISTIC object detection """

    def __init__(self, backbone, transformer, num_classes,
                 num_queries,
                 aux_loss=False,
                 compute_cls_var=False,
                 compute_bbox_cov=False,
                 bbox_cov_dims=4):

        super().__init__(backbone, transformer, num_classes, num_queries, aux_loss)
        hidden_dim = self.transformer.d_model

        self.compute_cls_var = compute_cls_var
        if self.compute_cls_var:
            self.class_var_embed = nn.Linear(hidden_dim, num_classes + 1)
            nn.init.normal_(self.class_var_embed.weight, std=0.0001)
            nn.init.constant_(self.class_var_embed.bias, 2 * np.log(0.01))

        self.compute_bbox_cov = compute_bbox_cov
        if self.compute_bbox_cov:
            self.bbox_covar_embed = MLP(
                hidden_dim, hidden_dim, bbox_cov_dims, 3)

    def forward(self, samples: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask,
                              self.query_embed.weight, pos[-1])[0]

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()

        # Only change to detr code happens here. We need to expose the features from
        # the transformer to compute variance parameters.
        out = {'pred_logits': outputs_class[-1],
               'pred_boxes': outputs_coord[-1]}

        if self.compute_cls_var:
            cls_var_out = self.class_var_embed(hs[-1])
            out.update({'pred_logits_var': cls_var_out})
        if self.compute_bbox_cov:
            bbox_cov_out = self.bbox_covar_embed(hs[-1])
            out.update({'pred_boxes_cov': bbox_cov_out})

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                outputs_class, outputs_coord)
        return out


class ProbabilisticSetCriterion(SetCriterion):

    """
    This is custom set criterion to allow probabilistic estimates
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        super().__init__(num_classes, matcher, weight_dict, eos_coef, losses)
        self.probabilistic_loss_weight = 0.0
        self.bbox_cov_num_samples = 1000
        self.cls_var_num_samples = 1000

    def set_bbox_cov_num_samples(self, bbox_cov_num_samples):
        self.bbox_cov_num_samples = bbox_cov_num_samples

    def set_cls_var_num_samples(self, cls_var_num_samples):
        self.cls_var_num_samples = cls_var_num_samples

    def loss_labels_att(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL + Loss attenuation)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        outputs must contain the mean pred_logits and the variance pred_logits_var
        """
        if 'pred_logits_var' not in outputs:
            return self.loss_labels(outputs, targets, indices, num_boxes, log)

        assert 'pred_logits' in outputs

        src_logits = outputs['pred_logits']
        src_logits_var = outputs['pred_logits_var']

        src_logits_var = torch.sqrt(torch.exp(src_logits_var))

        univariate_normal_dists = distributions.normal.Normal(
            src_logits, scale=src_logits_var)
        pred_class_stochastic_logits = univariate_normal_dists.rsample(
            (self.cls_var_num_samples,))
        pred_class_stochastic_logits = pred_class_stochastic_logits.view(
            pred_class_stochastic_logits.shape[1],
            pred_class_stochastic_logits.shape[2] * pred_class_stochastic_logits.shape[0],
            -1)

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J]
                                      for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target_classes = torch.unsqueeze(target_classes, dim=0)
        target_classes = torch.repeat_interleave(
            target_classes, self.cls_var_num_samples, dim=0)
        target_classes = target_classes.view(
            target_classes.shape[1],
            target_classes.shape[2] *
            target_classes.shape[0])

        loss_ce = F.cross_entropy(
            pred_class_stochastic_logits.transpose(
                1, 2), target_classes, self.empty_weight)

        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this
            # one here
            losses['class_error'] = 100 - \
                accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_boxes_var_nll(
            self,
            outputs,
            targets,
            indices,
            num_boxes):
        """Compute the losses related to the bounding boxes, the nll probabilistic regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        if 'pred_boxes_cov' not in outputs:
            return self.loss_boxes(outputs, targets, indices, num_boxes)

        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        src_vars = clamp_log_variance(outputs['pred_boxes_cov'][idx])

        target_boxes = torch.cat([t['boxes'][i]
                                  for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        if src_vars.shape[1] == 4:
            loss_nll = 0.5 * torch.exp(-src_vars) * loss_bbox + 0.5 * src_vars
        else:
            forecaster_cholesky = covariance_output_to_cholesky(
                src_vars)
            if forecaster_cholesky.shape[0] != 0:
                multivariate_normal_dists = distributions.multivariate_normal.MultivariateNormal(
                    src_boxes, scale_tril=forecaster_cholesky)
                loss_nll = - \
                    multivariate_normal_dists.log_prob(target_boxes)
            else:
                loss_nll = loss_bbox

        loss_nll_final = loss_nll.sum() / num_boxes

        # Collect all losses
        losses = dict()
        losses['loss_bbox'] = loss_nll_final
        # Add iou loss
        losses = update_with_iou_loss(
            losses, src_boxes, target_boxes, num_boxes)

        return losses

    def loss_boxes_energy(
            self,
            outputs,
            targets,
            indices,
            num_boxes):
        """Compute the losses related to the bounding boxes, the energy distance loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        if 'pred_boxes_cov' not in outputs:
            return self.loss_boxes(outputs, targets, indices, num_boxes)

        assert 'pred_boxes' in outputs

        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i]
                                  for t, (_, i) in zip(targets, indices)], dim=0)

        # Begin probabilistic loss computation
        src_vars = clamp_log_variance(outputs['pred_boxes_cov'][idx])
        forecaster_cholesky = covariance_output_to_cholesky(
            src_vars)
        multivariate_normal_dists = distributions.multivariate_normal.MultivariateNormal(
            src_boxes, scale_tril=forecaster_cholesky)

        # Define Monte-Carlo Samples
        distributions_samples = multivariate_normal_dists.rsample(
            (self.bbox_cov_num_samples + 1,))

        distributions_samples_1 = distributions_samples[0:self.bbox_cov_num_samples, :, :]
        distributions_samples_2 = distributions_samples[1:
                                                        self.bbox_cov_num_samples + 1, :, :]

        # Compute energy score. Smooth L1 loss is preferred in this case to
        # maintain the proper scoring properties.
        loss_covariance_regularize = - F.l1_loss(
            distributions_samples_1,
            distributions_samples_2,
            reduction="sum") / self.bbox_cov_num_samples  # Second term

        gt_proposals_delta_samples = torch.repeat_interleave(
            target_boxes.unsqueeze(0), self.bbox_cov_num_samples, dim=0)

        loss_first_moment_match = 2 * F.l1_loss(
            distributions_samples_1,
            gt_proposals_delta_samples,
            reduction="sum") / self.bbox_cov_num_samples  # First term

        loss_energy = loss_first_moment_match + loss_covariance_regularize

        # Normalize and add losses
        loss_energy_final = loss_energy.sum() / num_boxes

        # Collect all losses
        losses = dict()
        losses['loss_bbox'] = loss_energy_final
        # Add iou loss
        losses = update_with_iou_loss(
            losses, src_boxes, target_boxes, num_boxes)

        return losses

    def loss_boxes_smm(
            self,
            outputs,
            targets,
            indices,
            num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss, SMM variance and Covariance loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        if 'pred_boxes_cov' not in outputs:
            return self.loss_boxes(outputs, targets, indices, num_boxes)

        assert 'pred_boxes' in outputs

        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i]
                                  for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        # Begin probabilistic loss computation
        src_vars = clamp_log_variance(outputs['pred_boxes_cov'][idx])

        errors = (src_boxes - target_boxes)
        if src_vars.shape[1] == 4:
            second_moment_matching_term = F.l1_loss(
                torch.exp(src_vars), errors ** 2, reduction='none')
        else:
            errors = torch.unsqueeze(errors, 2)
            gt_error_covar = torch.matmul(
                errors, torch.transpose(errors, 2, 1))

            # This is the cholesky decomposition of the covariance matrix.
            # We reconstruct it from 10 estimated parameters as a
            # lower triangular matrix.
            forecaster_cholesky = covariance_output_to_cholesky(
                src_vars)

            predicted_covar = torch.matmul(
                forecaster_cholesky, torch.transpose(
                    forecaster_cholesky, 2, 1))

            second_moment_matching_term = F.l1_loss(
                predicted_covar, gt_error_covar, reduction='none')

        loss_smm = second_moment_matching_term.sum() / num_boxes

        # Normalize and add losses
        loss_bbox_final = loss_bbox.sum() / num_boxes
        loss_smm_final = loss_smm + loss_bbox_final

        # Collect all losses
        losses = dict()
        losses['loss_bbox'] = loss_smm_final
        # Add iou loss
        losses = update_with_iou_loss(
            losses, src_boxes, target_boxes, num_boxes)

        return losses

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'labels_loss_attenuation': self.loss_labels_att,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'boxes_negative_log_likelihood': self.loss_boxes_var_nll,
            'boxes_energy_loss': self.loss_boxes_energy,
            'boxes_second_moment_matching': self.loss_boxes_smm,
            'masks': self.loss_masks,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)


def update_with_iou_loss(losses, src_boxes, target_boxes, num_boxes):
    loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
        box_ops.box_cxcywh_to_xyxy(src_boxes),
        box_ops.box_cxcywh_to_xyxy(target_boxes)))
    losses['loss_giou'] = loss_giou.sum() / num_boxes
    return losses
