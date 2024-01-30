import json
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from skimage.feature import hog
from skimage import exposure
from skimage import exposure

# Define the path to the JSON files and the data folder
data_folder = os.path.join(r"C:\Studia\Modelowanie\Projekt_Informatyka\videos")
data_folder2 = os.getcwd()
nslt_json_path = os.path.join(data_folder2, 'nslt_100.json')
wlasl_json_path = os.path.join(data_folder2, 'WLASL_v0.3.json')

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

        # Crop to ROI (Region of Interest)
        x, y, w, h = bbox
        roi = frame[y:y+h, x:x+w]

        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_applied = clahe.apply(gray)

        # Apply Gaussian and Median Filters
        filtered = cv2.GaussianBlur(clahe_applied, (5, 5), 0)
        filtered = cv2.medianBlur(filtered, 5)

        # Apply Canny Edge Detection
        edged = cv2.Canny(filtered, 100, 200)

        # Add processed frame to list
        processed_frames.append(edged)

    cap.release()
    return processed_frames

def extract_features(processed_frames, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9):
    # Initialize the list to hold the features of the video
    video_features = []

    for frame in processed_frames:
        # Compute HOG features for the frame
        hog_features, hog_image = hog(frame,
                                      orientations=orientations,
                                      pixels_per_cell=pixels_per_cell,
                                      cells_per_block=cells_per_block,
                                      block_norm='L2-Hys',
                                      visualize=True)
        
        # Optionally, you can normalize the histogram to improve performance across different video conditions
        hog_features = exposure.rescale_intensity(hog_features, in_range=(0, 10))

        # Append the features of this frame to the video's feature list
        video_features.append(hog_features)

    # Flatten the list of features into a single one-dimensional list
    video_features = np.concatenate(video_features, axis=0)

    return video_features
                
features = []
labels = []

for entry in wlasl_data:
    gloss = entry['gloss']
    for instance in entry['instances']:
        video_id = instance['video_id']
        bbox = instance['bbox']
        frame_start = instance['frame_start']
        frame_end = instance['frame_end']
        video_path = os.path.join(data_folder, f"{video_id}.mp4")

        if os.path.exists(video_path):
            processed_frames = process_video(video_path, bbox, frame_start, frame_end)
            video_features = extract_features(processed_frames)
            features.append(video_features)
            labels.append(gloss)
        else:
            print(f"Video {video_id} for gloss {gloss} not found in the data folder.")

# Convert lists to arrays for machine learning
features = np.array(features)
labels = np.array(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Define the machine learning pipeline
pipeline = make_pipeline(StandardScaler(), PCA(n_components=0.95), SVC(kernel='rbf', class_weight='balanced'))

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model
print(f"Training Accuracy: {pipeline.score(X_train, y_train)}")
print(f"Testing Accuracy: {pipeline.score(X_test, y_test)}")
