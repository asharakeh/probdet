import os

# Detectron imports
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances

# Project imports
import core.datasets.metadata as metadata


def setup_all_datasets(dataset_dir, image_root_corruption_prefix=None):
    """
    Registers all datasets as instances from COCO

    Args:
        dataset_dir(str): path to dataset directory

    """
    setup_coco_dataset(
        dataset_dir,
        image_root_corruption_prefix=image_root_corruption_prefix)

    setup_openim_dataset(dataset_dir)
    setup_openim_odd_dataset(dataset_dir)


def setup_coco_dataset(dataset_dir, image_root_corruption_prefix=None):
    """
    sets up coco dataset following detectron2 coco instance format. Required to not have flexibility on where the dataset
    files can be.
    """
    train_image_dir = os.path.join(dataset_dir, 'train2017')

    if image_root_corruption_prefix is not None:
        test_image_dir = os.path.join(
            dataset_dir, 'val2017' + image_root_corruption_prefix)
    else:
        test_image_dir = os.path.join(dataset_dir, 'val2017')

    train_json_annotations = os.path.join(
        dataset_dir, 'annotations', 'instances_train2017.json')
    test_json_annotations = os.path.join(
        dataset_dir, 'annotations', 'instances_val2017.json')

    register_coco_instances(
        "coco_2017_custom_train",
        {},
        train_json_annotations,
        train_image_dir)
    MetadataCatalog.get(
        "coco_2017_custom_train").thing_classes = metadata.COCO_THING_CLASSES
    MetadataCatalog.get(
        "coco_2017_custom_train").thing_dataset_id_to_contiguous_id = metadata.COCO_THING_DATASET_ID_TO_CONTIGUOUS_ID

    register_coco_instances(
        "coco_2017_custom_val",
        {},
        test_json_annotations,
        test_image_dir)
    MetadataCatalog.get(
        "coco_2017_custom_val").thing_classes = metadata.COCO_THING_CLASSES
    MetadataCatalog.get(
        "coco_2017_custom_val").thing_dataset_id_to_contiguous_id = metadata.COCO_THING_DATASET_ID_TO_CONTIGUOUS_ID


def setup_openim_dataset(dataset_dir):
    """
    sets up openimages dataset following detectron2 coco instance format. Required to not have flexibility on where the dataset
    files can be.

    Only validation is supported.
    """
    test_image_dir = os.path.join(dataset_dir, 'images')

    test_json_annotations = os.path.join(
        dataset_dir, 'COCO-Format', 'val_coco_format.json')

    register_coco_instances(
        "openimages_val",
        {},
        test_json_annotations,
        test_image_dir)
    MetadataCatalog.get(
        "openimages_val").thing_classes = metadata.COCO_THING_CLASSES
    MetadataCatalog.get(
        "openimages_val").thing_dataset_id_to_contiguous_id = metadata.OPENIMAGES_THING_DATASET_ID_TO_CONTIGUOUS_ID


def setup_openim_odd_dataset(dataset_dir):
    """
    sets up openimages out-of-distribution dataset following detectron2 coco instance format. Required to not have flexibility on where the dataset
    files can be.

    Only validation is supported.
    """
    test_image_dir = os.path.join(dataset_dir, 'images')

    test_json_annotations = os.path.join(
        dataset_dir, 'COCO-Format', 'val_coco_format.json')

    register_coco_instances(
        "openimages_ood_val",
        {},
        test_json_annotations,
        test_image_dir)
    MetadataCatalog.get(
        "openimages_ood_val").thing_classes = metadata.COCO_THING_CLASSES
    MetadataCatalog.get(
        "openimages_ood_val").thing_dataset_id_to_contiguous_id = metadata.OPENIMAGES_THING_DATASET_ID_TO_CONTIGUOUS_ID