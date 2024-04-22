This repository contains three main components:

Static Face Recognition - Recognizes faces from a single captured photo (photo captured live from webcam on user input).

Live Face Recognition - Recognizes faces in real-time from a webcam feed.

Face Encodings - Generates and stores encodings for known faces for later recognition.

Setup
To run the scripts in this repository, you will need Python installed along with several dependencies:

opencv-python for capturing images from a webcam.
dlib and face_recognition for face detection and recognition capabilities.
numpy and pickle for handling data.
Install these dependencies using pip:

pip install opencv-python dlib face_recognition numpy

Generating Custom Face Encodings
To generate custom face encodings, create a folder named "faces" and inside that folder create a folder for each person. The folders name should be the persons name, inside each persons folder place individual face images. Each image should be in JPEG or PNG format.



