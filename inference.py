import datetime
from  ultralytics import YOLO
import numpy as np
import cv2
 
 
CONFIDENCE_THRESHOLD = 0.5
GREEN = (0,255,20)
 
frame_wid = 640
frame_hyt = 480
 
video_cap = "cars.mp4"
cap = cv2.VideoCapture(video_cap)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total Frames: {total_frames}")
frame_ht = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_wd = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

print(f"frame width: {frame_wd}, frame height: {frame_ht}")
# fourcc = cv2.VideoWriter_fourcc(*"XVID")  # not needed
codec = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("./droneshot.mp4", codec, 30, (frame_wd, frame_ht))

 
# loading the model
model = YOLO("yolov8m.pt")
 
print(" READING FRAMES..............")
 
while True:
    start = datetime.datetime.now()
    ret, frame = cap.read()
    print("helloo............")
 
    if not ret:
        print("NO FRAMES TO READ ...............")
        break
 
    # frame = cv2.resize(frame ,frame_wid,frame_hyt)
    detections = model.predict(source=[frame] ,conf =CONFIDENCE_THRESHOLD)
 
    DP = detections[0]
 
    if len(DP)!= 0:
        for i  in  range(len(detections[0])):
            print(i)
 
            boxes = detections[0].boxes
            box = boxes[i]  # returns one box
            clsID = box.cls.detach().cpu().numpy()[0]
            conf = box.conf.detach().cpu().numpy()[0]
            bb = box.xyxy.detach().cpu().numpy()[0]
 
            cv2.rectangle(
                frame,
                (int(bb[0]), int(bb[1])),
                (int(bb[2]), int(bb[3])),
                GREEN,
                2,
            )
 
 
        end = datetime.datetime.now()
        total = (end - start).total_seconds()
        print(f"Time to process 1 frame: {total * 1000:.0f} milliseconds")
 
        fps = f"FPS: {1 / total:.2f}"
        print("FPSf",fps)
        cv2.putText(frame, fps, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)
        
        out.write(frame)
    # show the frame to our screen
    # cv2.imshow("Frame", frame)
    # if cv2.waitKey(1) == ord("q"):
    #     break    
 
cap.release()
cv2.destroyAllWindows()