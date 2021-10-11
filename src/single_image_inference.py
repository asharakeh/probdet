"""
Probabilistic Detectron Single Image Inference Script
"""
import core
import cv2
import json
import os
import sys
import torch
import tqdm

# This is very ugly. Essential for now but should be fixed.
sys.path.append(os.path.join(core.top_dir(), 'src', 'detr'))

# Detectron imports
from detectron2.engine import launch
from detectron2.data import MetadataCatalog
from detectron2.data.transforms import ResizeShortestEdge

# Project imports
from core.evaluation_tools.evaluation_utils import get_train_contiguous_id_to_test_thing_dataset_id_dict
from core.setup import setup_config, setup_arg_parser
from probabilistic_inference.inference_utils import instances_to_json, get_inference_output_dir, build_predictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    # Setup config
    cfg = setup_config(args,
                       random_seed=args.random_seed,
                       is_testing=True)
    # Make sure only 1 data point is processed at a time. This simulates
    # deployment.
    cfg.defrost()
    cfg.SOLVER.IMS_PER_BATCH = 1

    cfg.MODEL.DEVICE = device.type

    # Create inference output directory
    inference_output_dir = args.output_dir
    os.makedirs(inference_output_dir, exist_ok=True)

    # Get category mapping dictionary. Mapping here is from coco-->coco
    train_thing_dataset_id_to_contiguous_id = MetadataCatalog.get(
        cfg.DATASETS.TRAIN[0]).thing_dataset_id_to_contiguous_id
    test_thing_dataset_id_to_contiguous_id = MetadataCatalog.get(
        cfg.DATASETS.TRAIN[0]).thing_dataset_id_to_contiguous_id

    # If both dicts are equal or if we are performing out of distribution
    # detection, just flip the test dict.
    cat_mapping_dict = get_train_contiguous_id_to_test_thing_dataset_id_dict(
        cfg,
        args,
        train_thing_dataset_id_to_contiguous_id,
        test_thing_dataset_id_to_contiguous_id)

    # Build predictor
    cfg.MODEL.WEIGHTS = os.path.expanduser(args.model_ckpt)
    predictor = build_predictor(cfg)

    # List images in image folder
    image_folder = os.path.expanduser(args.image_dir)
    image_list = os.listdir(image_folder)

    # Construct image resizer
    resizer = ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TEST, max_size=cfg.INPUT.MAX_SIZE_TEST)

    final_output_list = []
    with torch.no_grad():
        with tqdm.tqdm(total=len(image_list)) as pbar:
            for idx, input_file_name in enumerate(image_list):
                # Read image, apply shortest edge resize, and change to channel first position
                cv2_image = cv2.imread(os.path.join(image_folder, input_file_name))
                shape = cv2_image.shape
                height = shape[0]
                width = shape[1]
                output_transform = resizer.get_transform(cv2_image)
                cv2_image = output_transform.apply_image(cv2_image)
                input_im_tensor = torch.tensor(cv2_image).to(
                ).permute(2, 0, 1)
                input_im = [dict({'filename': input_file_name,
                                  'image_id': input_file_name,
                                  'height': height,
                                  'width': width,
                                  'image': input_im_tensor})]

                # Perform inference
                outputs = predictor(input_im)

                # predictor.visualize_inference(input_im, outputs)

                final_output_list.extend(
                    instances_to_json(
                        outputs,
                        input_im[0]['image_id'],
                        cat_mapping_dict))
                pbar.update(1)

                with open(os.path.join(inference_output_dir, 'results.json'), 'w') as fp:
                    json.dump(final_output_list, fp, indent=4,
                              separators=(',', ': '))


if __name__ == "__main__":
    # Create arg parser
    arg_parser = setup_arg_parser()
    args = arg_parser.parse_args()
    # Support single gpu inference only.
    args.num_gpus = 1
    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
