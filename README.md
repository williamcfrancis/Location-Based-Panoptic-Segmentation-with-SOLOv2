# Improved-Efficient-Panoptic-Segmentation


Panoptic segmentation is a scene understanding problem that combines the prediction from both instance and semantic segmentation into a general unified output
Our proposed project implements and modifies the state-of-the-art EfficientPS model by changing the instance segmentation head from Mask-RCNN to SOLOv2
![image](https://user-images.githubusercontent.com/38180831/203141810-3c0e51b8-7a79-46ff-b0de-532efb184231.png)

## EfficientPS Architecture

The original EfficientPS paper: [here](https://arxiv.org/abs/2004.02307)\
Code from the authors of EfficientPS: [here](https://github.com/DeepSceneSeg/EfficientPS)

![image](https://user-images.githubusercontent.com/38180831/203141883-79fd1093-eb06-4be8-8ddc-b9c8e63a911e.png)

## Panoptic Fusion Module

![image](https://user-images.githubusercontent.com/38180831/203141954-b923bc2c-88ed-427a-87af-8549646bf600.png)

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

## Dataset 

![image](https://user-images.githubusercontent.com/38180831/203142596-a41e1ffe-8763-4370-8f61-695053aa1a77.png)

## Results

![image](https://user-images.githubusercontent.com/38180831/203145055-325e047d-db78-437c-b103-bc42593e2c6f.png)
![image](https://user-images.githubusercontent.com/38180831/203145086-789ef0b7-25c7-4269-b468-a5673fecf22f.png)



