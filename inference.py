import datetime
from  ultralytics import YOLO
import numpy as np
import cv2
import time
 
 
CONFIDENCE_THRESHOLD = 0.25
BLUE = (255,20,20)
RED = (0,0,255)
LABEL_COLOR = (255,255,255)
 
frame_wid = 640
frame_hyt = 480
 
video_cap = "./person4.mp4"
cap = cv2.VideoCapture(video_cap)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total Frames: {total_frames}")
frame_ht = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_wd = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

print(f"frame width: {frame_wd}, frame height: {frame_ht}")
# fourcc = cv2.VideoWriter_fourcc(*"XVID")  # not needed
codec = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("outputperson2.mp4", codec, 30, (frame_wd, frame_ht))

 
# loading the model
model = YOLO("yolov8n-face.pt")

frame_count = 0
start_time = time.time()
 
while True:
    
    ret, frame = cap.read()
    print("helloo............")
    
    if not ret:
        print("NO FRAMES TO READ ...............")
        break
    frame_start = time.time()
    # frame = cv2.resize(frame ,frame_wid,frame_hyt)
    detections = model.predict(source=[frame] ,conf =CONFIDENCE_THRESHOLD)
 
    DP = detections[0]
 
    if len(DP)!= 0:
        
        for i  in  range(len(detections[0])):
            label = detections[0].names
            label = label[0]
            boxes = detections[0].boxes
            box = boxes[i]  # returns one box
            clsID = box.cls.detach().cpu().numpy()[0]
            conf = box.conf.detach().cpu().numpy()[0]
            bb = box.xyxy.detach().cpu().numpy()[0]
            print("LABEL" ,label)
            cv2.rectangle(
                frame,
                (int(bb[0]), int(bb[1])),
                (int(bb[2]), int(bb[3])),
                BLUE,
                8,)
            
            t_width, t_height = 340 ,70
            cv2.rectangle(frame, (int(bb[0]), int(bb[1]) - t_height - 10), (int(bb[0]) + t_width, int(bb[1])), RED , -1)
            cv2.putText(frame, str(label)+ ":" + str(round(conf ,2)), (int(bb[0]), int(bb[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 2, LABEL_COLOR, 5)
        
        frame_count +=1
        frame_end = time.time()
        total_elapsed_time = frame_end - start_time
        
        fps = frame_count/total_elapsed_time
 
        print("FPS",fps)
        cv2.putText(frame, f"FPS :{fps:.2f}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)
        
        
        out.write(frame)

    # show the frame to our screen
    frame = cv2.resize(frame , (640 ,480))
    cv2.imshow("Frame", frame)
    time.sleep(0.02)
    if cv2.waitKey(1) == ord("q"):
        break    
 
cap.release()
cv2.destroyAllWindows()
