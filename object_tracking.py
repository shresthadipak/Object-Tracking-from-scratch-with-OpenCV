import cv2
import numpy as np
from objectDetectionModule import objectDetector

detector = objectDetector()

img = cv2.imread('images/image1.jpg')

class_ids, boxes, confidences, indexes = detector.object_detect(img)


colors = np.random.uniform(0, 255, size=(len(boxes), 3))

for i, conf in zip(range(len(boxes)), confidences):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(detector.classes[class_ids[i]])
        color = colors[i]
        text = label+ ' ' +str(round(conf, 2))

        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

cv2.imshow('Object Tracking', img)   
cv2.waitKey(0)     