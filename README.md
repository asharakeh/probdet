# Probabilistic Detectron2
This repository contains the official implementation of [Estimating and Evaluating Regression Predictive Uncertainty in Deep Object Detectors](https://openreview.net/forum?id=YLewtnvKgR7&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2021%2FConference%2FAuthors%23your-submissions)). 

This code extends the [detectron2](https://github.com/facebookresearch/detectron2) framework to estimate bounding box covariance matrices, and
is meant to be a starter kit for entering the domain of probabilistic object detection.

## Disclaimer
This research code was produced by one person with a single set of eyes, it may contain bugs and errors that I did not notice by the time of release.

## ToDo
 1. Code cleanup
 2. Add configurations that enable full usage of all functions in repo.
 3. Make evaluator more modular. Im hoping to write an evaluator class when time allows.
 4. Update DETR to [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR) for faster convergence.
 5. Update repository with more uncertainty estimation mechanisms for both classification and regression tasks such as [Dirichlet prior networks](https://arxiv.org/abs/1802.10501).
 
## Requirements
#### Software Support:
Name | Supported Versions
--- | --- |
Ubuntu |20.04
Python |3.8
CUDA |11.0, 11.1
Cudnn |8.0.1
PyTorch |1.7.1

To install requirements choose between a python virtualenv or build a docker image using the provided Dockerfile.
```
# Clone repo
git clone https://github.com/asharakeh/probabilistic_detectron.git
cd probabilistic_detectron
git submodule update --init --recursive
```

1. Virtual Environment Creation:
```
# Create python virtual env
mkvirtualenv probdet

# Add library path to virtual env
add2virtualenv src

# Install requirements
cat requirements.txt | xargs -n 1 -L 1 pip install
```

2. Docker Image
```
# Clone repo
git clone https://github.com/asharakeh/probdet.git
cd probdet/Docker

# Build docker image
sh build.sh 
```

## Datasets

### COCO Dataset
Download the COCO Object Detection Dataset [here](https://cocodataset.org/#home). 
The COCO dataset folder should have the following structure:
<br>

     └── COCO_DATASET_ROOT
         |
         ├── annotations
         ├── train2017
         └── val2017

To create the corrupted datasets using Imagenet-C corruptions, run the following code:

``` python src/core/datasets/generate_coco_corrupted_dataset.py --dataset-dir=COCO_DATASET_ROOT ```

### OpenImages Datasets      
Download our OpenImages validation splits [here](https://drive.google.com/file/d/1IfdL6vW-EoK6UZob3Jh0hZ62qmDlQlrY/view?usp=sharing).
We created a tarball that contains both shifted and out-of-distribution data splits used in our paper to make our repo easier to use. Do not modify or rename the internal folders as those paths are
hard coded in the dataset reader. We will refer to the root folder extracted from the tarball as ```OPENIM_DATASET_ROOT```.

## Training
To train the models in the paper, use this command:

``` train
python src/train_net.py
--num-gpus xx
--dataset-dir COCO_DATASET_ROOT
--config-file COCO-Detection/architecture_name/config_name.yaml
--random-seed xx
--resume
```

For an explanation of all command line arguments, use ```python src/train_net.py -h```

## Evaluation
To run model inference after training, use this command:
```eval
python src/apply_net.py 
--dataset-dir TEST_DATASET_ROOT 
--test-dataset test_dataset_name 
--config-file path/to/config.yaml 
--inference-config /path/to/inference/config.yaml 
--random-seed xx
--image-corruption-level xx
```

For an explanation of all command line arguments, use ```python src/apply_net.py  -h```

```--image-corruption-level``` can vary between 0-5, with 0 being the original COCO dataset with no corruption. 
In addition, ```--image-corruption-level``` has no effect when used with OpenImages dataset splits.

`--test-dataset` can be one of `coco_2017_custom_val`, `openimages_val`, or `openimages_ood_val`. `--dataset-dir` corresponds to the root directory of the dataset used.
Evaluation code will run inference on the test dataset and then will generate mAP, Negative Log Likelihood, Brier Score, Energy Score, and Calibration Error results. If only evaluation of metrics is required,
add `--eval-only` to the above code snippet.

## Configurations in the paper
We provide a list of config combinations that generate the architectures used in our paper:

Method Name | Config File | Inference Config File | Model
--- | --- | --- |---
Deterministic RetinaNet | retinanet_R_50_FPN_3x.yaml| standard_nms.yaml | [retinanet_R_50_FPN_3x.pth](https://drive.google.com/file/d/1a7ibPAO44ixQb2boLdZa96fk26SUZrl8/view?usp=sharing)
RetinaNet NLL | retinanet_R_50_FPN_3x_reg_var_nll.yaml | standard_nms.yaml | [retinanet_R_50_FPN_3x_reg_var_nll.pth](https://drive.google.com/file/d/11SghCRPC6R9joJq2aT1qYrUGVb6Xr0RM/view?usp=sharing)
RetinaNet DMM | retinanet_R_50_FPN_3x_reg_var_dmm.yaml | standard_nms.yaml | [retinanet_R_50_FPN_3x_reg_var_dmm.pth](https://drive.google.com/file/d/1zvm-SvW_FgXj8Y0umsXqYFxk5o2f7sNf/view?usp=sharing)
RetinaNet ES | retinanet_R_50_FPN_3x_reg_var_es.yaml | standard_nms.yaml | [retinanet_R_50_FPN_3x_reg_var_es.pth](https://drive.google.com/file/d/1R0WFyeZIabtQ7V0YuUirqcqkWd5WHTB9/view?usp=sharing)
--- | --- | --- | ---
Deterministic FasterRCNN | faster_rcnn_R_50_FPN_3x.yaml| standard_nms.yaml |[faster_rcnn_R_50_FPN_3x.pth](https://drive.google.com/file/d/10nCvLldTjMiIdfL3YWqkQNsmbL9s84dx/view?usp=sharing)
FasterRCNN NLL | faster_rcnn_R_50_FPN_3x_reg_covar_nll.yaml | standard_nms.yaml |[faster_rcnn_R_50_FPN_3x_reg_covar_nll.pth](https://drive.google.com/file/d/1RPvvmcKfG8AZQFyyWJdDBP16il3YaTnd/view?usp=sharing)
FasterRCNN DMM | faster_rcnn_R_50_FPN_3x_reg_var_dmm.yaml | standard_nms.yaml |[faster_rcnn_R_50_FPN_3x_reg_var_dmm.pth](https://drive.google.com/file/d/181-CXHezq6aa5S1xvfEOEMKEdP5tAyWX/view?usp=sharing)
FasterRCNN ES | faster_rcnn_R_50_FPN_3x_reg_var_es.yaml | standard_nms.yaml |[faster_rcnn_R_50_FPN_3x_reg_var_es.pth](https://drive.google.com/file/d/1Vm_eBSjl8n1T5JFLaLgXSdAg1bawJ1ky/view?usp=sharing)
--- | --- | --- | ---
Deterministic DETR | detr_R_50.yaml| standard_nms.yaml | [detr_R_50.pth](https://drive.google.com/file/d/18o-e9ZPMjXMw1c1XQQIVIWQYC2eogclZ/view?usp=sharing)
DETR NLL | detr_R_50_reg_var_nll.yaml | standard_nms.yaml | [detr_R_50_reg_var_nll.pth](https://drive.google.com/file/d/1iuk5OIF8UO2jg7PdpCZA1qlxtgQzmv54/view?usp=sharing)
DETR DMM| detr_R_50_reg_var_dmm.yaml | standard_nms.yaml | [detr_R_50_reg_var_dmm.pth](https://drive.google.com/file/d/1qXV4szLZTSdIkiWPhDLSyttL3Pj_gu8z/view?usp=sharing)
DETR ES| detr_R_50_reg_var_es.yaml | standard_nms.yaml | [detr_R_50_reg_var_es.pth](https://drive.google.com/file/d/1Kgll1Ez0cLo_Wut07LJQef7eGP3xuG6_/view?usp=sharing)

Experiments in the paper were performed on 5 models trained and evaluated using random seeds [0, 1000, 2000, 3000, 4000]. 
The variance in performance between different seeds was seen to be negligible, and the results of the top performing seed were reported. 

## Additional Configurations
The repo supports many more variants including dropout and ensemble methods for estimating epistemic uncertainty.
We provide a list of config combinations that generate the architectures used in our paper:

Method Name | Config File | Inference Config File
--- | --- | ---
RetinaNet Classification [Loss Attenuation](https://arxiv.org/abs/1703.04977) | retinanet_R_50_FPN_3x_cls_la.yaml | standard_nms.yaml
RetinaNet [Dropout](https://arxiv.org/pdf/1506.02142.pdf) Post-NMS Uncertainty Computation| retinanet_R_50_FPN_3x_dropout.yaml | mc_dropout_ensembles_post_nms_mixture_of_gaussians.yaml
RetinaNet [Dropout](https://arxiv.org/pdf/1506.02142.pdf) Pre-NMS Uncertainty Computation| retinanet_R_50_FPN_3x_dropout.yaml | mc_dropout_ensembles_pre_nms.yaml
RetinaNet [BayesOD](https://arxiv.org/abs/1903.03838) with NLL loss| retinanet_R_50_FPN_3x_reg_var_nll.yaml | bayes_od.yaml
RetinaNet [BayesOD](https://arxiv.org/abs/1903.03838) with ES loss| retinanet_R_50_FPN_3x_reg_var_es.yaml | bayes_od.yaml
RetinaNet [BayesOD](https://arxiv.org/abs/1903.03838) with ES loss and [Dropout](https://arxiv.org/pdf/1506.02142.pdf)| retinanet_R_50_FPN_3x_reg_var_es_dropout.yaml | bayes_od_mc_dropout.yaml
RetinaNet [Ensembles](https://papers.nips.cc/paper/2017/file/9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf) Post-NMS Uncertainty Estimation with NLL loss| retinanet_R_50_FPN_3x_reg_var_nll.yaml (Need to train 5 Models with different random seeds) | ensembles_post_nms_mixture_of_gaussians.yaml
RetinaNet [Ensembles](https://papers.nips.cc/paper/2017/file/9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf) Pre-NMS Uncertainty Estimation with NLL loss| retinanet_R_50_FPN_3x_reg_var_nll.yaml (Need to train 5 Models with different random seeds) | ensembles_pre_nms.yaml
RetinaNet [Ensembles](https://papers.nips.cc/paper/2017/file/9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf) Post-NMS Uncertainty Estimation with ES loss| retinanet_R_50_FPN_3x_reg_var_es.yaml (Need to train 5 Models with different random seeds) | ensembles_post_nms_mixture_of_gaussians.yaml
RetinaNet [Ensembles](https://papers.nips.cc/paper/2017/file/9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf) Pre-NMS Uncertainty Estimation with ES loss| retinanet_R_50_FPN_3x_reg_var_es.yaml (Need to train 5 Models with different random seeds) | ensembles_pre_nms.yaml
--- | --- | ---
FasterRCNN Classification [Loss Attenuation](https://arxiv.org/abs/1703.04977) | faster_rcnn_R_50_FPN_3x_cls_la.yaml | standard_nms.yaml
FasterRCNN [Dropout](https://arxiv.org/pdf/1506.02142.pdf) Post-NMS Uncertainty Computation| faster_rcnn_R_50_FPN_3x_dropout.yaml | mc_dropout_ensembles_post_nms_mixture_of_gaussians.yaml
FasterRCNN [Dropout](https://arxiv.org/pdf/1506.02142.pdf) Pre-NMS Uncertainty Computation| faster_rcnn_R_50_FPN_3x_dropout.yaml | mc_dropout_ensembles_pre_nms.yaml
FasterRCNN [BayesOD](https://arxiv.org/abs/1903.03838) with NLL loss| faster_rcnn_R_50_FPN_3x_reg_var_nll.yaml | bayes_od.yaml
FasterRCNN [BayesOD](https://arxiv.org/abs/1903.03838) with ES loss| faster_rcnn_R_50_FPN_3x_reg_var_es.yaml | bayes_od.yaml
FasterRCNN [BayesOD](https://arxiv.org/abs/1903.03838) with ES loss and [Dropout](https://arxiv.org/pdf/1506.02142.pdf)| retinanet_R_50_FPN_3x_reg_var_es_dropout.yaml | bayes_od_mc_dropout.yaml
FasterRCNN [Ensembles](https://papers.nips.cc/paper/2017/file/9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf) Post-NMS Uncertainty Estimation with NLL loss| faster_rcnn_R_50_FPN_3x_reg_var_nll.yaml (Need to train 5 Models with different random seeds) | ensembles_post_nms_mixture_of_gaussians.yaml
FasterRCNN [Ensembles](https://papers.nips.cc/paper/2017/file/9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf) Pre-NMS Uncertainty Estimation with NLL loss| faster_rcnn_R_50_FPN_3x_reg_var_nll.yaml (Need to train 5 Models with different random seeds) | ensembles_pre_nms.yaml
FasterRCNN [Ensembles](https://papers.nips.cc/paper/2017/file/9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf) Post-NMS Uncertainty Estimation with ES loss| faster_rcnn_R_50_FPN_3x_reg_var_es.yaml (Need to train 5 Models with different random seeds) | ensembles_post_nms_mixture_of_gaussians.yaml
FasterRCNN [Ensembles](https://papers.nips.cc/paper/2017/file/9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf) Pre-NMS Uncertainty Estimation with ES loss| faster_rcnn_R_50_FPN_3x_reg_var_es.yaml (Need to train 5 Models with different random seeds) | ensembles_pre_nms.yaml
--- | --- | ---
DETR Classification [Loss Attenuation](https://arxiv.org/abs/1703.04977) | detr_R_50_cls_la.yaml | standard_nms.yaml
DETR [Dropout](https://arxiv.org/pdf/1506.02142.pdf)| detr_R_50.yaml (dropout is included in original implementation of DETR) | mc_dropout_ensembles_post_nms_mixture_of_gaussians.yaml
DETR [Ensembles](https://papers.nips.cc/paper/2017/file/9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf) with NLL loss| detr_R_50_reg_var_nll.yaml (Need to train 5 Models with different random seeds) | ensembles_post_nms_mixture_of_gaussians.yaml
DETR [Ensembles](https://papers.nips.cc/paper/2017/file/9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf) with ES loss| detr_R_50_reg_var_es.yaml (Need to train 5 Models with different random seeds) | ensembles_post_nms_mixture_of_gaussians.yaml

DETR has no NMS post-processing, and as such does not support BayesOD NMS replacement. The repo also supports many additional lower performing configurations. I will continue developing it and add additional configurations when time allows.

## Citation
If you use this code, please cite our paper:
```
@inproceedings{
harakeh2021estimating,
title={Estimating and Evaluating Regression Predictive Uncertainty in Deep Object Detectors},
author={Harakeh, Ali and Waslander, Steven L.},
booktitle={9th International Conference on Learning Representations (ICLR)},
year={2021},
url={https://openreview.net/forum?id=YLewtnvKgR7},
}
```

## License
This code is released under the [Apache 2.0 License](LICENSE.md).