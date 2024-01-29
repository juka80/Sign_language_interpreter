import json
import cv2
import numpy as np
import os

# Definicja funkcji do wczytywania danych JSON
def load_json_data(json_file_path):
    with open(json_file_path, 'r') as file:
        return json.load(file)

# Definicja funkcji do przetwarzania wideo
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

        # Dodanie przetworzonej klatki do listy
        processed_frames.append(roi_filtered)

    cap.release()
    return processed_frames

# Główny skrypt
def main():
    data_folder = 'videos'  # Tutaj wpisz właściwą ścieżkę do folderu z danymi
    json_file_path = os.path.join(data_folder, 'nslt_100.json')
    json_data = load_json_data(json_file_path)

    for entry in json_data:
        for instance in entry['instances']:
            video_id = instance['video_id']
            video_path = os.path.join(data_folder, f"{video_id}.mp4")
            bbox = instance['bbox']
            frame_start = instance['frame_start']
            frame_end = instance['frame_end']
            
            if os.path.exists(video_path):
                processed_frames = process_video(video_path, bbox, frame_start, frame_end)
                # Tutaj możesz zapisywać przetworzone klatki lub robić z nimi coś innego
            else:
                print(f"Video {video_id} not found in the data folder.")

if __name__ == '__main__':
    main()
