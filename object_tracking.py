import cv2
import numpy as np
from objectDetectionModule import objectDetector
import math

detector = objectDetector()

cap = cv2.VideoCapture('videos/raw_video.mp4')

# Video Output
width = int(cap.get(3))
height = int(cap.get(4))
fps = int(cap.get(5))/2
out = cv2.VideoWriter('output/object_tracking_1.avi', cv2.VideoWriter_fourcc(*'MPEG'), fps, (width, height))

# Initilize 
count = 0
center_pts_prev_frame = []

tracking_objects = {}
track_id = 1

while True:
    ret, img = cap.read()
    count +=1
    if not ret:
        break

    # Define the current center points
    center_pts_curr_frame = []

    # define the coordinates of an objects
    bbox = []


    # define screen resolution
    # h, w, _ = img.shape
    # screen_res = 1280, 720
    # scale_width = screen_res[0] / img.shape[1]
    # scale_height = screen_res[1] / img.shape[0]
    # scale = min(scale_width, scale_height)

    # img = cv2.resize(img, (int(w * scale), int(h * scale)), cv2.INTER_AREA)

    
    # Detect an object on frame    
    class_ids, boxes, confidences, indexes = detector.object_detect(img)
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))
    for i, conf in zip(range(len(boxes)), confidences):
        if i in indexes:
            x, y, w, h = boxes[i]

            cx = int((x+x+w)/2)
            cy = int((y+y+h)/2)
            center_pts_curr_frame.append((cx, cy))

            bbox.append((x, y, w, h))

            label = str(detector.classes[class_ids[i]])
            color = colors[i]
            # text = label+ ' ' +str(round(conf, 2))
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            # cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # check this condition in beginners frame to track objects
    if count <= 2:
        for pt1 in center_pts_curr_frame:
            for pt2 in center_pts_prev_frame:
                distance = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])

                if distance < 20:
                    tracking_objects[track_id] = pt1
                    track_id += 1
    else:
        tracking_objects_copy = tracking_objects.copy()
        center_pts_curr_frame_copy =  center_pts_curr_frame.copy()

        for object_id, pt2 in tracking_objects_copy.items():
            object_exists = False
            for pt1 in center_pts_curr_frame_copy:
                distance = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])

                # update Ids position
                if distance < 20:
                    tracking_objects[object_id] = pt1   
                    object_exists = True
                    if pt1 in center_pts_curr_frame:
                        center_pts_curr_frame.remove(pt1)
                    continue    

            # Remove the Ids lost
            if not object_exists:
                tracking_objects.pop(object_id) 


        # add a new ids for an object
        for pt in center_pts_curr_frame:
            tracking_objects[track_id] = pt
            track_id += 1               


    for object_id, box in zip(tracking_objects.items(), bbox):
        # cv2.circle(img, pt, 3, (0, 0, 255), -1)
        cv2.putText(img, f'ID: #{str(object_id[0])}', (box[0], box[1] - 5),0, 0.5, (0, 0, 255), 1)

    print('CURRENT FRAME')
    print(center_pts_curr_frame)

    print('PREVIOUS FRAME')
    print(center_pts_prev_frame)

    print(tracking_objects)

    out.write(img)
    # cv2.imshow('Object Tracking', img) 

    # Make a previous center points
    center_pts_prev_frame = center_pts_curr_frame.copy()

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()         