import cv2
import face_recognition
import pickle
import os

def capture_photo(save_path='photo.jpg'):
    """Capture photo using the local webcam and save it to the specified path."""
    video_capture = cv2.VideoCapture(0)  # Use the first webcam device
    if not video_capture.isOpened():
        print("Error: Unable to access the webcam.")
        return None
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Unable to capture image from webcam.")
        return None
    cv2.imwrite(save_path, frame)
    video_capture.release()
    return save_path

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

def recognize_faces(image_path):
    """Recognize faces in the specified image."""
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    print(f"Detected {len(face_locations)} faces in the image.")

    if not face_locations:
        print("No faces were detected in the image.")
        return

    face_encodings = face_recognition.face_encodings(image, face_locations)
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Draw a rectangle around the face
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image, name, (left + 6, bottom - 15), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

    # Display the resulting image with faces
    cv2.imshow('Face Recognition', image)
    cv2.waitKey(0)  # Wait until a key is pressed
    cv2.destroyAllWindows()

# Capture an image from the webcam
photo_path = capture_photo()
if photo_path:
    recognize_faces(photo_path)
