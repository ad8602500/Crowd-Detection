import cv2
import torch
from ultralytics import YOLO
import numpy as np
import pandas as pd
from scipy.spatial import distance

# Load YOLOv5 model 
model = YOLO("yolov5s.pt")  

# Open video file
video_path = "dataset_video.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))


out = cv2.VideoWriter("crowd_detected_output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# List to store crowd detection events
crowd_frames = []
frame_number = 0

# Define minimum distance threshold for crowd detection
MIN_DISTANCE = 50  
CROWD_THRESHOLD = 5  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_number += 1

    # Run YOLOv5 model on the frame
    results = model(frame)

    # Extract bounding boxes and class labels
    people_boxes = []
    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, conf, cls = box.tolist()
            if int(cls) == 0:  
                people_boxes.append(((x1 + x2) / 2, (y1 + y2) / 2))  

    # Detect crowded areas based on distance between people
    if len(people_boxes) > 1:
        distances = distance.cdist(people_boxes, people_boxes, "euclidean")
        close_people = sum(np.sum(distances < MIN_DISTANCE, axis=1) > 1)

        # If crowd is detected, store event
        if close_people >= CROWD_THRESHOLD:
            crowd_frames.append([frame_number, close_people])
            cv2.putText(frame, f"CROWD ALERT: {close_people} people", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

   
    for (cx, cy) in people_boxes:
        cv2.circle(frame, (int(cx), int(cy)), 5, (0, 255, 0), -1)

    # Write frame to output video
    out.write(frame)

    # Display the frame (Optional)
    cv2.imshow("Crowd Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save crowd events to CSV
if crowd_frames:
    df = pd.DataFrame(crowd_frames, columns=["Frame Number", "Person Count in Crowd"])
    df.to_csv("crowd_detection_results.csv", index=False)
    print(f"Crowd detection results saved to 'crowd_detection_results.csv' with {len(crowd_frames)} events.")
else:
    print("No crowd events detected.")

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print("Crowd detection video saved as 'crowd_detected_output.mp4'")
