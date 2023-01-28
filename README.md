# Real Time Object Tracking with OpenCV and YOLOv4 model
Real-time object tracking is a technique used to detect and track objects in a video stream in real-time. First of all it must be clear that what is the difference between object detection and object tracking:
1. Object detection is the detection on every single frame and frame after frame. 
2. Object tracking does frame-by-frame tracking but keeps the history of where the object is at a time after time.

# YOLOv4 Object Detection Model
Object detection using YOLOv4 from scratch and have some basic concept over object detection model via the flow diagram.

![This is an image](/images/AO.png)

YOLOv4 is a convolutional neural network (CNN) based object detection model. It uses a single neural network to predict bounding boxes and class probabilities directly from full images in one pass. The architecture of YOLOv4 consists of several layers, including:

1. A backbone network, which is responsible for extracting feature maps from the input image. In YOLOv4, the backbone network is a variant of the CSPDarknet architecture, which is a combination of the Darknet and Cross Stage Partial (CSP) architectures.

2. A neck network, which is used to fuse feature maps from the backbone network and extract higher-level features. In YOLOv4, the neck network consists of several SPP (Spatial Pyramid Pooling) and PAN (Path Aggregation Network) blocks.

3. A head network, which is used to predict bounding boxes and class probabilities from the features extracted by the neck network. The head network in YOLOv4 consists of several YOLO (You Only Look Once) blocks, which are similar to the YOLOv3 blocks but with some modifications.

4. A auxiliary network, which is used to enhance the feature maps and improve the accuracy of the prediction, The auxiliary network in YOLOv4 consists of SPADE (Spatially Adaptive Normalization) blocks and PAN blocks.

Overall, YOLOv4 architecture is more efficient and accurate than YOLOv3.

[Download file yolov4 model](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights) and save it into YOLOv4_model folder.


# Libraries
    import cv2
    import numpy as np
    import math

# Discussion and Analysis
This project is about tracking an object frame by frame from scratch level to understand the basic concept of tracking where I have understand how the tracking process will be done in real time. Although, this method won't give us good result because I just use some basic mathematical logic to complete this project but I had lots of knoweledge and logic to explore.

We have other techniques to track an object much more efficiently and accurately. Some of the techniques or architectures to track an object such as SORT and DeepSORT. These architechtures used [Kalman Filter](https://www.youtube.com/playlist?list=PLX2gX-ftPVXU3oUFNATxGXY90AULiqnWT), IOU matching, Cosine Distance and [Hungarian Algorthim](https://www.youtube.com/watch?v=cQ5MsiGaDY8&t=225s&ab_channel=CompSci) to track an object effectively.

# Object Tracking Demo
[Object Tracking in shopping mall](/output/object_tracking_1.avi)

# References
[Object Tracking with Opencv and Python](https://pysource.com/2021/01/28/object-tracking-with-opencv-and-python/)

# License
The MIT License (MIT). Please see [License File](/LICENSE) for more information.
