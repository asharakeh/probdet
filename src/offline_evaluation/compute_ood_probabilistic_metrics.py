import itertools
import os
import torch
import ujson as json
import pickle

from prettytable import PrettyTable

# Detectron imports
from detectron2.engine import launch

# Project imports
from core.evaluation_tools import scoring_rules
from core.evaluation_tools.evaluation_utils import eval_predictions_preprocess
from core.setup import setup_config, setup_arg_parser
from probabilistic_inference.inference_utils import get_inference_output_dir

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(
        args,
        cfg=None,
        min_allowed_score=None):

    # Setup config
    if cfg is None:
        cfg = setup_config(args, random_seed=args.random_seed, is_testing=True)

    cfg.defrost()
    cfg.ACTUAL_TEST_DATASET = args.test_dataset

    # Setup torch device and num_threads
    torch.set_num_threads(cfg.DATALOADER.NUM_WORKERS)

    # Build path to gt instances and inference output
    inference_output_dir = get_inference_output_dir(
        cfg['OUTPUT_DIR'],
        args.test_dataset,
        args.inference_config,
        args.image_corruption_level)

    if min_allowed_score is None:
        # Check if F-1 Score has been previously computed ON THE ORIGINAL
        # DATASET, and not on VOC.
        try:
            train_set_inference_output_dir = get_inference_output_dir(
                cfg['OUTPUT_DIR'],
                cfg.DATASETS.TEST[0],
                args.inference_config,
                0)
            with open(os.path.join(train_set_inference_output_dir, "mAP_res.txt"), "r") as f:
                min_allowed_score = f.read().strip('][\n').split(', ')[-1]
                min_allowed_score = round(float(min_allowed_score), 4)
        except FileNotFoundError:
            # If not, process all detections. Not recommended as the results might be influenced by very low scoring
            # detections that would normally be removed in robotics/vision
            # applications.
            min_allowed_score = 0.0

    # Get matched results by either generating them or loading from file.
    with torch.no_grad():
        try:
            preprocessed_predicted_instances = torch.load(
                os.path.join(
                    inference_output_dir,
                    "preprocessed_predicted_instances_odd_{}.pth".format(min_allowed_score)),
                map_location=device)
        # Process predictions
        except FileNotFoundError:
            prediction_file_name = os.path.join(
                inference_output_dir,
                'coco_instances_results.json')
            predicted_instances = json.load(open(prediction_file_name, 'r'))
            preprocessed_predicted_instances = eval_predictions_preprocess(
                predicted_instances, min_allowed_score=min_allowed_score, is_odd=True)
            torch.save(
                preprocessed_predicted_instances,
                os.path.join(
                    inference_output_dir,
                    "preprocessed_predicted_instances_odd_{}.pth".format(min_allowed_score)))

        predicted_boxes = preprocessed_predicted_instances['predicted_boxes']
        predicted_cov_mats = preprocessed_predicted_instances['predicted_covar_mats']
        predicted_cls_probs = preprocessed_predicted_instances['predicted_cls_probs']

        predicted_boxes = list(itertools.chain.from_iterable(
            [predicted_boxes[key] for key in predicted_boxes.keys()]))
        predicted_cov_mats = list(itertools.chain.from_iterable(
            [predicted_cov_mats[key] for key in predicted_cov_mats.keys()]))
        predicted_cls_probs = list(itertools.chain.from_iterable(
            [predicted_cls_probs[key] for key in predicted_cls_probs.keys()]))

        num_false_positives = len(predicted_boxes)
        valid_idxs = torch.as_tensor(
            [i for i in range(num_false_positives)]).to(device)

        predicted_boxes = torch.stack(predicted_boxes, 1).transpose(0, 1)
        predicted_cov_mats = torch.stack(predicted_cov_mats, 1).transpose(0, 1)
        predicted_cls_probs = torch.stack(
            predicted_cls_probs,
            1).transpose(
            0,
            1)

        false_positives_dict = {
            'predicted_box_means': predicted_boxes,
            'predicted_box_covariances': predicted_cov_mats,
            'predicted_cls_probs': predicted_cls_probs}

        false_positives_reg_analysis = scoring_rules.compute_reg_scores_fn(
            false_positives_dict, valid_idxs)

        if cfg.MODEL.META_ARCHITECTURE == 'ProbabilisticRetinaNet':
            predicted_class_probs, predicted_class_idx = predicted_cls_probs.max(
                1)
            false_positives_dict['predicted_score_of_gt_category'] = 1.0 - \
                predicted_class_probs
            false_positives_cls_analysis = scoring_rules.sigmoid_compute_cls_scores(
                false_positives_dict, valid_idxs)

        else:
            false_positives_dict['predicted_score_of_gt_category'] = predicted_cls_probs[:, -1]
            _, predicted_class_idx = predicted_cls_probs[:, :-1].max(
                1)
            false_positives_cls_analysis = scoring_rules.softmax_compute_cls_scores(
                false_positives_dict, valid_idxs)

        # Summarize and print all
        table = PrettyTable()
        table.field_names = (['Output Type',
                              'Number of Instances',
                              'Cls Ignorance Score',
                              'Cls Brier/Probability Score',
                              'Reg Ignorance Score',
                              'Reg Energy Score'])
        table.add_row(
            [
                "False Positives:",
                num_false_positives,
                '{:.4f}'.format(
                    false_positives_cls_analysis['ignorance_score_mean'],),
                '{:.4f}'.format(
                    false_positives_cls_analysis['brier_score_mean']),
                '{:.4f}'.format(
                    false_positives_reg_analysis['total_entropy_mean']),
                '{:.4f}'.format(
                    false_positives_reg_analysis['fp_energy_score_mean'])])
        print(table)

        text_file_name = os.path.join(
            inference_output_dir,
            'probabilistic_scoring_res_odd_{}.txt'.format(min_allowed_score))

        with open(text_file_name, "w") as text_file:
            print(table, file=text_file)

        dictionary_file_name = os.path.join(
            inference_output_dir,
            'probabilistic_scoring_res_odd_{}.pkl'.format(min_allowed_score))
        false_positives_reg_analysis.update(false_positives_cls_analysis)
        with open(dictionary_file_name, "wb") as pickle_file:
            pickle.dump(false_positives_reg_analysis, pickle_file)


if __name__ == "__main__":
    # Create arg parser
    arg_parser = setup_arg_parser()

    args = arg_parser.parse_args()
    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
