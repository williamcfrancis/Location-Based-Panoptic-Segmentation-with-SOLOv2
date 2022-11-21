# Improved-Efficient-Panoptic-Segmentation

Panoptic segmentation is a scene understanding problem that combines the prediction from both instance and semantic segmentation into a general unified output
Our proposed project implements and modifies the state-of-the-art EfficientPS model by changing the instance segmentation head from Mask-RCNN to SOLOv2
![image](https://user-images.githubusercontent.com/38180831/203141810-3c0e51b8-7a79-46ff-b0de-532efb184231.png)

## EfficientPS Architecture

![image](https://user-images.githubusercontent.com/38180831/203141883-79fd1093-eb06-4be8-8ddc-b9c8e63a911e.png)

## Panoptic Fusion Module

![image](https://user-images.githubusercontent.com/38180831/203141954-b923bc2c-88ed-427a-87af-8549646bf600.png)

### Why EfficientPS?

![image](https://user-images.githubusercontent.com/38180831/203142358-5586e8e2-6d2c-4f97-ae28-04e47eba6247.png)

![image](https://user-images.githubusercontent.com/38180831/203142384-10802c2e-8056-46ef-9d6f-3178d5e6b23a.png = 200x)

### Novelty of this approach

We replace the Mask-RCNN architecture from the instance head with a SOLOv2 architecture in order to improve the instance segmentation of the EfficientPS model
The Mask-RCNN losses now will be replaced by SOLOv2’s Focal Loss for semantic category classification and DiceLoss for mask prediction
This approach of using a location-based instance segmentation for panoptic segmentation will improve upon the performance metrics

## Dataset 

![image](https://user-images.githubusercontent.com/38180831/203142596-a41e1ffe-8763-4370-8f61-695053aa1a77.png)

### Performance Metrics


Panoptic Quality:


Segmentation Quality:


Recognition Quality:![image](https://user-images.githubusercontent.com/38180831/203142711-e4276897-70be-4a99-a9ba-8b5e68ea5871.png)
