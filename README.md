# Improved-Efficient-Panoptic-Segmentation


Panoptic segmentation is a scene understanding problem that combines the prediction from both instance and semantic segmentation into a general unified output.
Our proposed project implements and modifies the state-of-the-art EfficientPS model by changing the instance segmentation head from Mask-RCNN to SOLOv2.

### Why SOLOv2?
![image](https://user-images.githubusercontent.com/38180831/203141810-3c0e51b8-7a79-46ff-b0de-532efb184231.png)

## EfficientPS Architecture

The original EfficientPS paper: [here](https://arxiv.org/abs/2004.02307)\
Code from the authors of EfficientPS: [here](https://github.com/DeepSceneSeg/EfficientPS)

![image](https://user-images.githubusercontent.com/38180831/203141883-79fd1093-eb06-4be8-8ddc-b9c8e63a911e.png)

## How to use

- Download Cityscape Dataset:
```
git clone https://github.com/mcordts/cityscapesScripts.git
# City scapes script
pip install git+https://github.com/mcordts/cityscapesScripts.git
# Panoptic
CITYSCAPES_DATASET=/path/to/abovementioned/cityscapes python cityscapesscripts/preparation/createPanopticImgs.py
```
- Install [pytorch](https://pytorch.org/)
```
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```
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
- Install [detecron 2 dependencies](https://github.com/facebookresearch/detectron2)
```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```
- Install [Panoptic api](https://github.com/cocodataset/panopticapi)
```
pip install git+https://github.com/cocodataset/panopticapi.git
```
- Modify `config.yaml`
- Run `train_net.py`

## Choice of implementation

**1 - Original Configuration of the authors**

```python
Training config
	Solver: SGD
		lr: 0.007
		momentum: 0.9
	Batch_size: 16
	Image_size: 1024 x 2048
	Norm: SyncInplaceBN
	Augmentation:
		- RandomCrop
		- RandomFlip
			- Normalize
	Warmup: 200 iterations 1/3 lr to lr
	Scheduler: StepLR
		- step [120, 144]
		- total epoch: 160
```

**2 - Adapted configuration to my resources**

The authors trained their models using 16 NVIDIA Titan X GPUs. Due to the fact that I only had one GPU to train the model, I could not use the same configuration. Here is a summary of the necessary implementation decisions:

- I first wanted on a batch size of one to keep the same image size. But the `1024 x 2048` image did not fit into memory. So I reduced the size of the images by 2 leading to `512 x 1024` images.
- Still I could not fit many images into memory, in order to increase the batch size and the speed of the training I decided to use **mixed precision training**. Mixed precision training is simply combining single precision (32 bit) tensor with half precision (16bit) tensor. Using 16bit tensor frees up a lot of memory and also speeds up the overall training, but it can also reduce the performance of the overall training. (More information in the [paper](https://arxiv.org/pdf/1802.00930.pdf))
- Using a smaller images size and 16 bit precision enabled me to have a `batch size` of 3 images.
- For the optimizer, I decided to use `Adam` which is more stable and so requires less optimisation to reach good performances, I reduced the learning rate base on the ratio of batch size between their implementation and mine, giving me a learning rate of `1.3e-3` . Base on my experiments changing the learning rate did not seem to make a big impact.
- Since I was not able to train for the number of epochs used during the training (160 epochs), I decided to use `ReduceLROnPlateau` as a scheduler in order to optimize my performance on a small number of epochs.
- For the augmentations:
    - I did not use `RandomCrop`, mainly because I did not have time to optimize the creation of the batch with different image sizes. In their case they have one image per batch, so the problem does not occur. I could have still used random scale on higher scale in order to perform random cropping but it was not my top priority.
    - `RandomFlip` and `Normalisation` (with the statistics of the dataset) are applied
- On the testing pipeline: I did not do multiscaling for the testing procedure.

To sum up, we have:

```python
Training config
	Solver: Adam
		lr: 1.3e-3
	Batch_size: 3
	Image_size: 512 x 1024
	Norm: InplaceBN
	Augmentation:
		- RandomFlip
			- Normalize
	Warmup: 500 iterations lr/500 to lr
	Scheduler: ReduceLROnPlateau
		- patience 3
		- min lr: 1e-4
```


### Why EfficientPS?

Early research explored various techniques for Instance segmentation and Semantic segmentation separately. Initial panoptic segmentation methods heuristically combine predictions from state-of-the-art instance segmentation network and semantic segmentation network in a post-processing step. However, they suffered from large computational overhead, redundancy in learning and discrepancy between the predictions of each network.\
Recent works implemented top-down manner with shared components or in a bottom-up manner sequentially. This again did not utilize component sharing and suffered from low computational efficiency, slow runtimes and subpar results.\
EfficientPS:
- Shared backbone: EfficientNet
- Feature aligning semantic head, modified Mask R-CNN
- Panoptic fusion module: dynamic fusion of logits based on mask confidences
- Jointly optimized end-to-end, Depth-wise separable conv, Leaky ReLU
- 2 way FPN : semantically rich multiscale features

### Novelty of this approach

We replace the Mask-RCNN architecture from the instance head with a SOLOv2 architecture in order to improve the instance segmentation of the EfficientPS model.\
The Mask-RCNN losses now will be replaced by SOLOv2’s Focal Loss for semantic category classification and DiceLoss for mask prediction.\
This approach of using a location-based instance segmentation for panoptic segmentation will improve upon the performance metrics.\



## Results

![image](https://user-images.githubusercontent.com/38180831/203145055-325e047d-db78-437c-b103-bc42593e2c6f.png)
![image](https://user-images.githubusercontent.com/38180831/203145086-789ef0b7-25c7-4269-b468-a5673fecf22f.png)



