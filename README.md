# Crowd-Detection

Crowd Detection Using YOLOv5
In this project, I implemented a crowd detection system using a pre-trained YOLOv5 model to identify and analyze crowds in a video. The system detects persons in each frame, calculates distances between them, and identifies a crowd when three or more people stand close together for 10 consecutive frames.

Key Features:
Person Detection: Uses YOLOv5 to detect people in a video.
Crowd Identification: Determines if a group of people is standing close together based on a distance threshold.
Event Logging: Records crowd events with frame numbers and person count in a CSV file.
Video Output: Highlights detected crowds in a processed video.
Outputs:
crowd_detected_output.mp4 → Video with detected crowds.
crowd_detection_results.csv → Log of crowd events.


Overview
This project detects crowds in a video using a pre-trained YOLOv5 model and logs detected crowd events. A crowd is defined as three or more persons standing close together for at least 10 consecutive frames. The output includes a video with marked detections and a CSV file logging crowd events.

Requirements
pip install ultralytics opencv-python numpy pandas scipy

Installation & Usage
Clone the Repository (if applicable)

git clone <link>
cd <repository_folder>
Install Required Packages

pip install -r requirements.txt
Run the Script
Outputs:

crowd_detected_output.mp4: Video with detected crowds highlighted.
crowd_detection_results.csv: Log of detected crowd events (Frame Number, Person Count).

Logic Behind Crowd Detection
Person Detection: Uses YOLOv5 to detect people in each frame.
Distance Calculation: Computes distances between detected persons using Euclidean distance.
Crowd Identification: If three or more people are standing close (within a distance threshold) for at least 10 consecutive frames, it logs the event.
Output Logging: Saves the detected crowd information in a CSV file and highlights it in the video.
