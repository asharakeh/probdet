import cv2
import numpy as np
import os
import ujson as json

# Detectron imports
from detectron2.data import MetadataCatalog
from detectron2.engine import launch

# Project imports
from core.setup import setup_config, setup_arg_parser
from core.evaluation_tools import evaluation_utils
from core.visualization_tools.probabilistic_visualizer import ProbabilisticVisualizer as Visualizer
from probabilistic_inference.inference_utils import get_inference_output_dir


def main(
        args,
        cfg=None,
        iou_min=None,
        iou_correct=None,
        min_allowed_score=None):
    # Setup config
    if cfg is None:
        cfg = setup_config(args, random_seed=args.random_seed, is_testing=True)

    cfg.defrost()
    cfg.ACTUAL_TEST_DATASET = args.test_dataset

    # Build path to gt instances and inference output
    inference_output_dir = get_inference_output_dir(
        cfg['OUTPUT_DIR'],
        args.test_dataset,
        args.inference_config,
        args.image_corruption_level)

    # Get thresholds to perform evaluation on
    if iou_min is None:
        iou_min = args.iou_min
    if iou_correct is None:
        iou_correct = args.iou_correct
    if min_allowed_score is None:
        # Check if F-1 Score has been previously computed ON THE ORIGINAL
        # DATASET such as COCO even when evaluating on VOC.
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

    # get preprocessed instances
    preprocessed_predicted_instances, preprocessed_gt_instances = evaluation_utils.get_per_frame_preprocessed_instances(
        cfg, inference_output_dir, min_allowed_score)

    # get metacatalog and image infos
    meta_catalog = MetadataCatalog.get(args.test_dataset)
    images_info = json.load(open(meta_catalog.json_file, 'r'))['images']

    # Loop over all images and visualize errors
    for image_info in images_info:
        image_id = image_info['id']
        image = cv2.imread(
            os.path.join(
                meta_catalog.image_root,
                image_info['file_name']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        predicted_box_means = {
            image_id: preprocessed_predicted_instances['predicted_boxes'][image_id]}
        predicted_box_covariances = {
            image_id: preprocessed_predicted_instances['predicted_covar_mats'][image_id]}
        predicted_cls_probs = {
            image_id: preprocessed_predicted_instances['predicted_cls_probs'][image_id]}
        gt_box_means = {
            image_id: preprocessed_gt_instances['gt_boxes'][image_id]}
        gt_cat_idxs = {
            image_id: preprocessed_gt_instances['gt_cat_idxs'][image_id]}

        # Perform matching
        matched_results = evaluation_utils.match_predictions_to_groundtruth(
            predicted_box_means,
            predicted_cls_probs,
            predicted_box_covariances,
            gt_box_means,
            gt_cat_idxs,
            iou_min=iou_min,
            iou_correct=iou_correct)

        true_positives = matched_results['true_positives']
        duplicates = matched_results['duplicates']
        localization_errors = matched_results['localization_errors']
        false_positives = matched_results['false_positives']
        false_negatives = matched_results['false_negatives']

        # Plot True Positive Detections In Blue
        v = Visualizer(
            image,
            meta_catalog,
            scale=2.0)

        gt_boxes = true_positives['gt_box_means'].cpu().numpy()
        true_positive_boxes = true_positives['predicted_box_means'].cpu(
        ).numpy()
        false_positives_boxes = false_positives['predicted_box_means'].cpu(
        ).numpy()
        duplicates_boxes = duplicates['predicted_box_means'].cpu().numpy()
        localization_errors_boxes = localization_errors['predicted_box_means'].cpu(
        ).numpy()

        # Get category labels
        gt_cat_idxs = true_positives['gt_cat_idxs'].cpu().numpy()
        # Get category mapping dictionary:
        train_thing_dataset_id_to_contiguous_id = MetadataCatalog.get(
            cfg.DATASETS.TRAIN[0]).thing_dataset_id_to_contiguous_id
        test_thing_dataset_id_to_contiguous_id = MetadataCatalog.get(
            args.test_dataset).thing_dataset_id_to_contiguous_id

        thing_dataset_id_to_contiguous_id = evaluation_utils.get_test_thing_dataset_id_to_train_contiguous_id_dict(
            cfg, args, train_thing_dataset_id_to_contiguous_id, test_thing_dataset_id_to_contiguous_id)
        class_list = MetadataCatalog.get(
            cfg.DATASETS.TRAIN[0]).as_dict()['thing_classes']

        if gt_cat_idxs.shape[0] > 0:
            gt_labels = [class_list[thing_dataset_id_to_contiguous_id[gt_class]]
                         for gt_class in gt_cat_idxs[:, 0]]
        else:
            gt_labels = []

        if cfg.MODEL.META_ARCHITECTURE != "ProbabilisticRetinaNet":
            if len(true_positives['predicted_cls_probs'] > 0):
                _, true_positive_classes = true_positives['predicted_cls_probs'][:, :-1].max(
                    1)
            else:
                true_positive_classes = np.array([])

            if len(duplicates['predicted_cls_probs']) > 0:
                _, duplicates_classes = duplicates['predicted_cls_probs'][:, :-1].max(
                    1)
            else:
                duplicates_classes = np.array([])

            if len(localization_errors['predicted_cls_probs']) > 0:
                _, localization_errors_classes = localization_errors['predicted_cls_probs'][:, :-1].max(
                    1)
            else:
                localization_errors_classes = np.array([])

            if len(false_positives['predicted_cls_probs']) > 0:
                _, false_positives_classes = false_positives['predicted_cls_probs'][:, :-1].max(
                    1)
            else:
                false_positives_classes = np.array([])

        else:
            if len(true_positives['predicted_cls_probs'] > 0):
                _, true_positive_classes = true_positives['predicted_cls_probs'].max(
                    1)
            else:
                true_positive_classes = np.array([])

            if len(duplicates['predicted_cls_probs']) > 0:

                _, duplicates_classes = duplicates['predicted_cls_probs'].max(
                    1)
            else:
                duplicates_classes = np.array([])

            if len(localization_errors['predicted_cls_probs']) > 0:
                _, localization_errors_classes = localization_errors['predicted_cls_probs'].max(
                    1)
            else:
                localization_errors_classes = np.array([])

            if len(false_positives['predicted_cls_probs']) > 0:
                _, false_positives_classes = false_positives['predicted_cls_probs'].max(
                    1)
            else:
                false_positives_classes = np.array([])

        if len(true_positives['predicted_cls_probs'] > 0):
            true_positive_classes = true_positive_classes.cpu(
            ).numpy()
            true_positive_labels = [class_list[tp_class]
                                    for tp_class in true_positive_classes]
        else:
            true_positive_labels = []

        if len(duplicates['predicted_cls_probs']) > 0:
            duplicates_classes = duplicates_classes.cpu(
            ).numpy()
            duplicates_labels = [class_list[d_class]
                                 for d_class in duplicates_classes]
        else:
            duplicates_labels = []

        if len(localization_errors['predicted_cls_probs']) > 0:
            localization_errors_classes = localization_errors_classes.cpu(
            ).numpy()
            localization_errors_labels = [class_list[le_class]
                                          for le_class in localization_errors_classes]
        else:
            localization_errors_labels = []

        if len(false_positives['predicted_cls_probs']) > 0:
            false_positives_classes = false_positives_classes.cpu(
            ).numpy()
            false_positives_labels = [class_list[fp_class]
                                      for fp_class in false_positives_classes]
        else:
            false_positives_labels = []

        # Overlay true positives in blue
        _ = v.overlay_instances(
            boxes=gt_boxes,
            assigned_colors=[
                'lime' for _ in gt_boxes],
            labels=gt_labels,
            alpha=1.0)
        plotted_true_positive_boxes = v.overlay_instances(
            boxes=true_positive_boxes,
            assigned_colors=[
                'dodgerblue' for _ in true_positive_boxes],
            alpha=1.0,
            labels=true_positive_labels)
        cv2.imshow(
            'True positive detections with IOU greater than {}'.format(iou_correct),
            cv2.cvtColor(
                plotted_true_positive_boxes.get_image(),
                cv2.COLOR_RGB2BGR))

        # Plot False Positive Detections In Red
        v = Visualizer(
            image,
            meta_catalog,
            scale=2.0)

        _ = v.overlay_instances(
            boxes=gt_boxes,
            assigned_colors=[
                'lime' for _ in gt_boxes],
            labels=gt_labels,
            alpha=0.7)
        plotted_false_positive_boxes = v.overlay_instances(
            boxes=false_positives_boxes,
            assigned_colors=[
                'red' for _ in false_positives_boxes],
            alpha=1.0,
            labels=false_positives_labels)
        cv2.imshow(
            'False positive detections with IOU less than {}'.format(iou_min),
            cv2.cvtColor(
                plotted_false_positive_boxes.get_image(),
                cv2.COLOR_RGB2BGR))

        # Plot Duplicates
        v = Visualizer(
            image,
            meta_catalog,
            scale=2.0)

        _ = v.overlay_instances(
            boxes=gt_boxes,
            assigned_colors=[
                'lime' for _ in gt_boxes],
            labels=gt_labels,
            alpha=0.7)

        plotted_duplicates_boxes = v.overlay_instances(
            boxes=duplicates_boxes,
            assigned_colors=[
                'magenta' for _ in duplicates_boxes],
            alpha=1.0,
            labels=duplicates_labels)
        cv2.imshow(
            'Duplicate Detections',
            cv2.cvtColor(
                plotted_duplicates_boxes.get_image(),
                cv2.COLOR_RGB2BGR))

        # Plot localization errors
        v = Visualizer(
            image,
            meta_catalog,
            scale=2.0)

        _ = v.overlay_instances(
            boxes=gt_boxes,
            assigned_colors=[
                'lime' for _ in gt_boxes],
            labels=gt_labels,
            alpha=0.7)
        plotted_localization_errors_boxes = v.overlay_instances(
            boxes=localization_errors_boxes,
            assigned_colors=['aqua' for _ in localization_errors_boxes],
            alpha=1.0,
            labels=localization_errors_labels)
        cv2.imshow(
            'Detections with localization errors between minimum IOU = {} and maximum IOU = {}'.format(
                iou_min, iou_correct), cv2.cvtColor(
                plotted_localization_errors_boxes.get_image(), cv2.COLOR_RGB2BGR))

        # Plot False Negatives Detections In Brown
        if len(false_negatives['gt_box_means']) > 0:
            false_negatives_boxes = false_negatives['gt_box_means'].cpu(
            ).numpy()
            false_negatives_classes = false_negatives['gt_cat_idxs'].cpu(
            ).numpy()
            false_negatives_labels = [class_list[thing_dataset_id_to_contiguous_id[gt_class[0]]]
                                      for gt_class in false_negatives_classes.tolist()]
        else:
            false_negatives_boxes = np.array([])
            false_negatives_labels = []

        v = Visualizer(
            image,
            meta_catalog,
            scale=2.0)

        plotted_false_negative_boxes = v.overlay_instances(
            boxes=false_negatives_boxes,
            assigned_colors=[
                'coral' for _ in false_negatives_boxes],
            alpha=1.0,
            labels=false_negatives_labels)
        cv2.imshow(
            'False negative ground truth.',
            cv2.cvtColor(
                plotted_false_negative_boxes.get_image(),
                cv2.COLOR_RGB2BGR))

        cv2.waitKey(0)
        cv2.destroyAllWindows()


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
