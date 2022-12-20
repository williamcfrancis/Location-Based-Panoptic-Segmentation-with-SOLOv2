# SOLOv2-based Efficient Panoptic Segmentation 

## System Requirements
* Linux 
* Python 3.7
* PyTorch 1.7
* CUDA 10.2
* GCC 7 or 8

## Dependencies
Install the following frameworks

- [EfficientNet-Pytorch](https://github.com/lukemelas/EfficientNet-PyTorch) for the backbone
- [detectron2](https://github.com/facebookresearch/detectron2) for the instance head
- [In-Place Activated BatchNorm](https://github.com/mapillary/inplace_abn)
- [COCO 2018 Panoptic Segmentation Task API (Beta version)](https://github.com/cocodataset/panopticapi) to compute panoptic quality metric

## Install Dependencies
### For EfficientPS
- Install [Albumentation](https://albumentations.ai/)
```
pip install -U albumentations
```
- Install [Pytorch lighting](https://www.pytorchlightning.ai/)
```
pip install pytorch-lightning
```
- Install [Inplace batchnorm](https://github.com/mapillary/inplace_abn)
```
pip install inplace-abn
```
- Install [EfficientNet Pytorch](https://github.com/lukemelas/EfficientNet-PyTorch)
```
pip install efficientnet_pytorch
```
- Install [Detecron 2 dependencies](https://github.com/facebookresearch/detectron2)
```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```
- Install [Panoptic api](https://github.com/cocodataset/panopticapi)
```
pip install git+https://github.com/cocodataset/panopticapi.git
```
### For SOLOv2
Install the dependencies by running
```
pip install pycocotools
pip install numpy
pip install scipy
pip install torch==1.5.1 torchvision==0.6.1
pip install mmcv
```

## Dataset Preparation

1. Download the GtFine and leftimg8bit files of the Cityscapes dataset from https://www.cityscapes-dataset.com/ and unzip the `leftImg8bit_trainvaltest.zip` and `gtFine_trainvaltest.zip` into `data/cityscapes`
2. The dataset needs to be converted into coco format using the conversion tool in mmdetection:
* Clone the repository using `git clone https://github.com/open-mmlab/mmdetection.git`
* Enter the repository using `cd mmdetection`
* Install cityscapescripts using `pip install cityscapesscripts`
* Run the script as 
```
python tools/dataset_converters/cityscapes.py \
    data/cityscapes/ \
    --nproc 8 \
    --out-dir data/cityscapes/annotations
```
3. Create the panoptic images json file:
* Clone the repository using `git clone https://github.com/mcordts/cityscapesScripts.git`
* Install it using `pip install git+https://github.com/mcordts/cityscapesScripts.git`
* Run the script using `python cityscapesScripts/cityscapesscripts/preparation/createPanopticImgs.py`

Now the folder structure for the dataset should look as follows:
```
EfficientPS
└── data
    └── cityscapes
        ├── annotations
        ├── train
        ├── cityscapes_panoptic_val.json
        └── val
```
## How to train

### SOLOv2
- Go into the SOLOv2 folder using `cd SOLOv2`
- Modify `config.yaml` to change the paths
- Run `python setup.py develop`
- Run `train.py`

### EfficientPS
- Go into the SOLOv2 folder using `cd ..` and `cd EfficientPS`
- Run `train_net.py`

## How to run inference
1. Go into the SOLOv2 folder using `cd SOLOv2`
2. Run `python eval.py`. This will save the SOLOv2 masks in `EfficientPS/solo_outputs`
3. Now go into the EfficientPS folder using `cd ..` and `cd EfficientPS`
4. Run the combined evaluation using `python solo_fusion.py`

The results will be saved in `EfficientPS/Outputs`
