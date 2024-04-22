# Face Recognition System

This repository contains three main components:
1. **Static Face Recognition** - Recognizes faces from a single captured photo.
2. **Live Face Recognition** - Recognizes faces in real-time from a webcam feed.
3. **Face Encodings** - Generates and stores encodings for known faces for later recognition.

## Setup

To run the scripts in this repository, you will need Python installed along with several dependencies:
- `opencv-python` for capturing images from a webcam.
- `dlib` and `face_recognition` for face detection and recognition capabilities.
- `numpy` and `pickle` for handling data.

Install these dependencies using pip:
```bash
pip install opencv-python dlib face_recognition numpy
```

## Generating Custom Face Encodings

To generate custom face encodings, create a folder named "faces" and inside that folder create a folder for each person. The folders name should be the persons name, inside each persons folder place individual face images. Each image should be in JPEG or PNG format.

Run the encoding script to generate and save face encodings:
```bash
python encode_faces.py
```

## Static Face Recognition

This script captures a single photo from your webcam, then processes it to detect and recognize faces based on the pre-generated encodings.

- **To run this script:**
```bash
python static_recognition.py
```
Press 'c' to capture the photo during live preview.

## Live Face Recognition

This script provides a live video feed from your webcam and performs face recognition every 2 seconds (120 frames), updating the recognition results in real-time.

- **To run this script:**
```bash
python live_recognition.py
```
Press 'q' to quit the live feed. The live feed shows recognized faces with labels.

## Repository Structure

- `encode_faces.py` - Script to generate and store face encodings.
- `static_recognition.py` - Script to perform face recognition on a single captured photo.
- `live_recognition.py` - Script for real-time face recognition using webcam.
- `encodings.pickle` - Serialized file containing face encodings.
- `names.pickle` - Serialized file containing corresponding names for the encodings.
- `dlib_face_recognition_resnet_model_v1.dat` - Dlib face recognition model file.
- `shape_predictor_68_face_landmarks.dat` - Dlib model for detecting facial landmarks.
