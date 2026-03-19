"""
Comparison between YOLOv8n to my OpenCV implementation of Object Detection
"""

#pip install ultralytics
from ultralytics import YOLO
import os

"""
Seeing how YOLOv8n performs on test video
"""
#Pretrained YOLOv8 Model
model = YOLO("yolov8n.pt")

#Input Video Path
#input = 'test_video.mp4'
input = 'short_video.mp4'    #Use for faster run time

#Perform YOLO predictions on the frame
results = model(source=input, conf = 0.5, save = True, show = True)

"""
OpenCv Object Detection
"""

import numpy as np
import cv2
import random

#Gets the names of the classes
class_names = results[0].names

#Write class names to a txt file while removing the number associted with it
file_path = "names.txt"
with open(file_path, 'w') as file:
    for idx, name in class_names.items():
        file.write(f"{name}\n")

print(f"Class names written to {file_path}")

#Open name file in read mode
names = open("names.txt", 'r')
data = names.read()
names_list = data.split("\n")

#Gets rid of empty last index
names_list.pop()      
names.close()


#Capture video
cap = cv2.VideoCapture(input)

#Check if capture opened
if not cap.isOpened():
    print("Video file didn't open")
    exit()

#Get h/w of video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#Define Codec and VideoWriter Object
fourcc = cv2.VideoWriter_fourcc('F','M','P','4')
output_file = 'output_video.avi'
out = cv2.VideoWriter(output_file, fourcc, 20.0, (width, height))


#OpenCV model ---- mobileNet SSD Method
configPath ='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'      
weighsPath ='frozen_inference_graph.pb' 
cvmodel = cv2.dnn_DetectionModel(weighsPath, configPath)
cvmodel.setInputSize(width,height)                      #Sets limit of input size (pixels, pixels)
cvmodel.setInputScale(1.0/127.5)                    #Normalize pixels 
cvmodel.setInputMean((127.5,127,5,127.5))
cvmodel.setInputSwapRB(True)                        #BGR to RGB

#Dictionary Map to class indicies to unique colors
class_colors = {class_index: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for class_index in range(1, 81)}

#OpenCV needs to go frame by frame
while True:
    ret, frame = cap.read()

    if not ret:
        break

    #Pass each frame into model
    ClassIndex, confidence, bbox = cvmodel.detect(frame, confThreshold= 0.5)

    #Creates boxes and labels each object identified, the same class will get same color (similar to YOLOv8)
    if(len(ClassIndex != 0)):
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            if(ClassInd <= 80):

                #Use the unique color associated with class index
                color = class_colors[ClassInd]
                cv2.rectangle(frame, boxes ,color ,2)

                #Add class name and confidence to label
                label = f"{names_list[ClassInd - 1]}: {conf:.2f}"
                cv2.putText(frame, label, (boxes[0]+10, boxes[1]+40), fontFace = cv2.FONT_HERSHEY_PLAIN, fontScale = 1, color = color, thickness = 2)

    #Write Frame to output file 
    out.write(frame)
   
   
    #Display video frame with detected objects
    cv2.imshow('Object Detection OpenCV', frame)        #Continous loops allow for real-time object detection
    if cv2.waitKey(1) & 0xFF == ord(' '):               #Press Spacebar to end visualization early - will cause next visualization to end early - let this run for a bit
        break

cap.release()
out.release()
cv2.destroyAllWindows()

"""
Display both Videos Side By Side Using MatPlotLib
"""
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

cap1 = cv2.VideoCapture("output_video.avi")
cap2 = cv2.VideoCapture("runs/detect/predict/test_video.avi")     
#cap2 = cv2.VideoCapture("runs/detect/predict/short_video.avi")          #if using short_vid change to this line                
#Check if opened
if not cap1.isOpened():
    print("OpenCV Video File didn't open")
    exit()
if not cap2.isOpened():
    print("YOLOv8 Video File didn't open")
    exit()

#Set up Matplotlib figure and axes
fig, (ax1,ax2) = plt.subplots(1, 2, figsize = (15,10))
ax1.axis("off")
ax2.axis("off")

# Function to update the display in the animation
def update(frame):

    # Read frames from OpenCV model
    ret1, frame1 = cap1.read()

    if not ret1:
        anim.event_source.stop()  # Stop the animation when frames are finished
        plt.close(fig)  # Close the Matplotlib window

    # Read frames from YOLOv8 video
    ret2, frame2 = cap2.read()

    if not ret2:
        anim.event_source.stop()  # Stop the animation when frames are finished
        plt.close(fig)  # Close the Matplotlib window

    # Display frames side by side
    ax1.clear()
    ax2.clear()

    # OpenCV Object Detection
    frame1 = np.array(frame1)
    ax1.imshow(frame1[:, :, ::-1])
    ax1.set_title("OpenCV Object Detection")
    ax1.axis("off")

    # YOLOv8 Object Detection
    frame2 = np.array(frame2)
    ax2.imshow(frame2[:, :, ::-1])
    ax2.set_title("YOLOv8 Object Detection")
    ax2.axis("off")

# Create the animation
total_frames = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
anim = FuncAnimation(fig, update, frames=total_frames, interval=10, repeat=False)  # Adjust frames and interval as needed

plt.show()

# Release resources
cap1.release()
cap2.release()
out.release()