import numpy as np
import torch
import torch.nn.functional as F


# DETR imports
from detr.util.box_ops import box_cxcywh_to_xyxy

# Detectron Imports
from detectron2.structures import Boxes

# Project Imports
from probabilistic_inference import inference_utils
from probabilistic_inference.inference_core import ProbabilisticPredictor
from probabilistic_modeling.modeling_utils import covariance_output_to_cholesky, clamp_log_variance


class DetrProbabilisticPredictor(ProbabilisticPredictor):
    def __init__(self, cfg):
        super().__init__(cfg)
        # These are mock variables to be compatible with probabilistic detectron library. No NMS is performed for DETR.
        # Only needed for ensemble methods
        self.test_nms_thresh = 0.5
        self.test_topk_per_image = self.model.detr.num_queries

    def detr_probabilistic_inference(self,
                                     input_im):

        outputs = self.model(input_im,
                             return_raw_results=True,
                             is_mc_dropout=self.mc_dropout_enabled)

        image_width = input_im[0]['image'].shape[2]
        image_height = input_im[0]['image'].shape[1]

        # Handle logits and classes
        predicted_logits = outputs['pred_logits'][0]
        if 'pred_logits_var' in outputs.keys():
            predicted_logits_var = outputs['pred_logits_var'][0]
            box_cls_dists = torch.distributions.normal.Normal(
                predicted_logits, scale=torch.sqrt(
                    torch.exp(predicted_logits_var)))
            predicted_logits = box_cls_dists.rsample(
                (self.model.cls_var_num_samples,))
            predicted_prob_vectors = F.softmax(predicted_logits, dim=-1)
            predicted_prob_vectors = predicted_prob_vectors.mean(0)
        else:
            predicted_prob_vectors = F.softmax(predicted_logits, dim=-1)

        predicted_prob, classes_idxs = predicted_prob_vectors[:, :-1].max(-1)
        # Handle boxes and covariance matrices
        predicted_boxes = outputs['pred_boxes'][0]

        # Rescale boxes to inference image size (not COCO original size)
        pred_boxes = Boxes(box_cxcywh_to_xyxy(predicted_boxes))
        pred_boxes.scale(scale_x=image_width, scale_y=image_height)
        predicted_boxes = pred_boxes.tensor

        # Rescale boxes to inference image size (not COCO original size)
        if 'pred_boxes_cov' in outputs.keys():
            predicted_boxes_covariance = covariance_output_to_cholesky(
                outputs['pred_boxes_cov'][0])
            predicted_boxes_covariance = torch.matmul(
                predicted_boxes_covariance, predicted_boxes_covariance.transpose(
                    1, 2))

            transform_mat = torch.tensor([[[1.0, 0.0, -0.5, 0.0],
                                           [0.0, 1.0, 0.0, -0.5],
                                           [1.0, 0.0, 0.5, 0.0],
                                           [0.0, 1.0, 0.0, 0.5]]]).to(self.model.device)
            predicted_boxes_covariance = torch.matmul(
                torch.matmul(
                    transform_mat,
                    predicted_boxes_covariance),
                transform_mat.transpose(
                    1,
                    2))

            scale_mat = torch.diag_embed(
                torch.as_tensor(
                    (image_width,
                     image_height,
                     image_width,
                     image_height),
                    dtype=torch.float32)).to(
                self.model.device).unsqueeze(0)
            predicted_boxes_covariance = torch.matmul(
                torch.matmul(
                    scale_mat,
                    predicted_boxes_covariance),
                torch.transpose(scale_mat, 2, 1))
        else:
            predicted_boxes_covariance = []

        return predicted_boxes, predicted_boxes_covariance, predicted_prob, classes_idxs, predicted_prob_vectors

    def post_processing_standard_nms(self, input_im):
        """
        This function produces results using standard non-maximum suppression. The function takes into
        account any probabilistic modeling method when computing the results.

        Args:
            input_im (list): an input im list generated from dataset handler.

        Returns:
            result (instances): object instances

        """
        outputs = self.detr_probabilistic_inference(input_im)

        return inference_utils.general_standard_nms_postprocessing(
            input_im, outputs)

    def post_processing_output_statistics(self, input_im):
        """
        Output statistics does not make much sense for DETR architecture. There is some redundancy due to forced 100
        detections per image, but cluster sizes would be too small for meaningful estimates. Might implement it later
        on.
        """
        raise NotImplementedError
        pass

    def post_processing_mc_dropout_ensembles(self, input_im):
        if self.cfg.PROBABILISTIC_INFERENCE.ENSEMBLES.BOX_MERGE_MODE == 'pre_nms':
            raise NotImplementedError
        else:
            # Merge results:
            results = [
                inference_utils.general_standard_nms_postprocessing(
                    input_im,
                    self.detr_probabilistic_inference(input_im),
                    self.test_nms_thresh,
                    self.test_topk_per_image) for _ in range(
                    self.num_mc_dropout_runs)]

            # Append per-ensemble outputs after NMS has been performed.
            ensemble_pred_box_list = [
                result.pred_boxes.tensor for result in results]
            ensemble_pred_prob_vectors_list = [
                result.pred_cls_probs for result in results]
            ensembles_class_idxs_list = [
                result.pred_classes for result in results]
            ensembles_pred_box_covariance_list = [
                result.pred_boxes_covariance for result in results]

            return inference_utils.general_black_box_ensembles_post_processing(
                input_im,
                ensemble_pred_box_list,
                ensembles_class_idxs_list,
                ensemble_pred_prob_vectors_list,
                ensembles_pred_box_covariance_list,
                self.test_nms_thresh,
                self.test_topk_per_image,
                self.cfg.PROBABILISTIC_INFERENCE.AFFINITY_THRESHOLD,
                is_generalized_rcnn=True,
                merging_method=self.cfg.PROBABILISTIC_INFERENCE.ENSEMBLES.BOX_FUSION_MODE)

    def post_processing_ensembles(self, input_im, model_dict):
        if self.cfg.PROBABILISTIC_INFERENCE.ENSEMBLES.BOX_MERGE_MODE == 'pre_nms':
            raise NotImplementedError
        else:
            outputs_list = []
            for model in model_dict:
                self.model = model
                outputs_list.append(
                    self.post_processing_standard_nms(input_im))

            # Merge results:
            ensemble_pred_box_list = []
            ensemble_pred_prob_vectors_list = []
            ensembles_class_idxs_list = []
            ensembles_pred_box_covariance_list = []
            for results in outputs_list:
                # Append per-ensemble outputs after NMS has been performed.
                ensemble_pred_box_list.append(results.pred_boxes.tensor)
                ensemble_pred_prob_vectors_list.append(results.pred_cls_probs)
                ensembles_class_idxs_list.append(results.pred_classes)
                ensembles_pred_box_covariance_list.append(
                    results.pred_boxes_covariance)

            return inference_utils.general_black_box_ensembles_post_processing(
                input_im,
                ensemble_pred_box_list,
                ensembles_class_idxs_list,
                ensemble_pred_prob_vectors_list,
                ensembles_pred_box_covariance_list,
                self.test_nms_thresh,
                self.test_topk_per_image,
                self.cfg.PROBABILISTIC_INFERENCE.AFFINITY_THRESHOLD,
                is_generalized_rcnn=True,
                merging_method=self.cfg.PROBABILISTIC_INFERENCE.ENSEMBLES.BOX_FUSION_MODE)

    def post_processing_bayes_od(self, input_im):
        """
        Since there is no NMS step in DETR, bayesod is not implemented. Although possible to add NMS
        and implement it later on.
        """
        raise NotImplementedError
        pass
