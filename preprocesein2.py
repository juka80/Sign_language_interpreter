import json
import cv2
import numpy as np
import os

# Define the path to the JSON files and the data folder
data_folder = os.getcwd()
nslt_json_path = os.path.join(data_folder, 'nslt_100.json')
wlasl_json_path = os.path.join(data_folder, 'WLASL_v0.3.json')


# Load the NSLT JSON data
with open(nslt_json_path, 'r') as file:
    nslt_data = json.load(file)

# Load the WLASL JSON data
with open(wlasl_json_path, 'r') as file:
    wlasl_data = json.load(file)

# Function to process videos
def process_video(video_path, bbox, frame_start, frame_end):
    cap = cv2.VideoCapture(video_path)
    current_frame = 0
    processed_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        current_frame += 1

        if current_frame < frame_start or (frame_end != -1 and current_frame > frame_end):
            continue

        # Przycinanie do ROI
        x, y, w, h = bbox
        roi = frame[y:y+h, x:x+w]

        # Zastosowanie filtrów
        roi_filtered = cv2.GaussianBlur(roi, (5, 5), 0)
        roi_filtered = cv2.medianBlur(roi_filtered, 5)
        roi_filtered = cv2.Canny(roi_filtered, 100, 200)
        # Można dodać CLAHE i inne filtry w tym miejscu
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        if len(frame.shape) == 3: 
            lab = cv2.cvtColor(roi_filtered, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_clahe = clahe.apply(l)
            lab_clahe = cv2.merge((l_clahe, a, b))
            clahe_filtered = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        else:
            clahe_filtered = clahe.apply(roi_filtered)

        yield clahe_filtered

        # Dodanie przetworzonej klatki do listy
        processed_frames.append(roi_filtered)

    cap.release()
    #return processed_frames

def match_data_and_process(nslt_data, wlasl_data):
    for entry in wlasl_data:
        gloss = entry['gloss']
        for instance in entry['instances']:
            video_id = instance['video_id']
            bbox = instance['bbox']
            frame_start = instance['frame_start']
            frame_end = instance['frame_end']
            video_path = os.path.join(data_folder, f"{video_id}.mp4")

            if os.path.exists(video_path):
                process_video(video_path, bbox, frame_start, frame_end)
                print("ZNALZALEM!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            #else:
                    
               # print(f"Video {video_id} for gloss {gloss} not found in the data folder.")


# Main execution
match_data_and_process(nslt_data, wlasl_data)
