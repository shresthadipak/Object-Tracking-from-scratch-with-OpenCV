import cv2
import numpy as np
from objectDetectionModule import objectDetector

detector = objectDetector()

cap = cv2.VideoCapture('videos/los_angeles.mp4')

count = 0
center_points = []

while True:
    ret, img = cap.read()

    h, w, _ = img.shape

    # define screen resolution
    screen_res = 1280, 720
    scale_width = screen_res[0] / img.shape[1]
    scale_height = screen_res[1] / img.shape[0]
    scale = min(scale_width, scale_height)

    img = cv2.resize(img, (int(w * scale), int(h * scale)), cv2.INTER_AREA)

    count +=1
    if not ret:
        break

    class_ids, boxes, confidences, indexes = detector.object_detect(img)


    colors = np.random.uniform(0, 255, size=(len(boxes), 3))

    for i, conf in zip(range(len(boxes)), confidences):
        if i in indexes:
            x, y, w, h = boxes[i]
            cx = int((x+x+w)/2)
            cy = int((y+y+h)/2)
            center_points.append((cx, cy))
            label = str(detector.classes[class_ids[i]])
            color = colors[i]
            text = label+ ' ' +str(round(conf, 2))
            if label == 'car':
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                # cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    for pt in center_points:
        cv2.circle(img, pt, 3, (0,0,255), -1)

    cv2.imshow('Object Tracking', img) 

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()         