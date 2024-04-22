import cv2
import dlib
import numpy as np
import pickle
import os

# Load known face encodings and names
encodings_file = 'encodings.pickle'
names_file = 'names.pickle'
if os.path.exists(encodings_file) and os.path.exists(names_file):
    with open(encodings_file, 'rb') as ef:
        known_face_encodings = pickle.load(ef)
    with open(names_file, 'rb') as nf:
        known_face_names = pickle.load(nf)
else:
    print("Error: Encoding files not found.")
    exit(1)

# Initialize Dlib's face detector and the face recognition model
detector = dlib.get_frontal_face_detector()
predictor_path = 'shape_predictor_68_face_landmarks.dat'
face_rec_model_path = 'dlib_face_recognition_resnet_model_v1.dat'
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

def live_face_recognition():
    video_capture = cv2.VideoCapture(0)
    frame_skip = 120  # Process every 120 frames
    frame_count = 0
    labels = []  # To store the face labels

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Unable to capture image from webcam.")
            break

        # Clear previous drawings and labels every 600 frames
        if frame_count % frame_skip == 0:
            labels.clear()  # Clear old labels for new processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            dets = detector(rgb_frame, 1)
            print(f"Detected {len(dets)} faces.")

            for k, d in enumerate(dets):
                shape = sp(rgb_frame, d)
                face_descriptor = facerec.compute_face_descriptor(rgb_frame, shape)
                distances = np.linalg.norm(known_face_encodings - np.array(face_descriptor), axis=1)
                name = "Unknown"
                if len(distances) > 0 and np.min(distances) < 0.6:
                    name = known_face_names[np.argmin(distances)]

                labels.append((d, name))
                print(f"Face {k + 1}, Name: {name}")  # Debug output for face name recognition

        # Draw the results from the latest processed frame
        for d, name in labels:
            cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 0), 2)
            cv2.putText(frame, name, (d.left() + 6, d.bottom() - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('Live Face Recognition', frame)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

live_face_recognition()
