import cv2
import face_recognition
import pickle
import os

def capture_photo():
    """Open a live preview of the webcam and capture a photo when a key is pressed."""
    video_capture = cv2.VideoCapture(0)  # Use the first webcam device
    if not video_capture.isOpened():
        print("Error: Unable to access the webcam.")
        return None
    
    print("Press 'c' to capture the photo, or 'q' to quit.")
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Unable to capture image from webcam.")
            break
        
        # Display the live video frame
        cv2.imshow('Live Preview', frame)
        
        # Wait for user to press 'c' to capture or 'q' to quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            # Save the captured image
            cv2.imwrite('photo.jpg', frame)
            print("Photo captured successfully.")
            break
        elif key == ord('q'):
            print("Exiting without capturing.")
            frame = None
            break
    
    video_capture.release()
    cv2.destroyAllWindows()
    return 'photo.jpg' if frame is not None else None

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
    image = cv2.imread(image_path)  # Load the image from file
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert it from BGR to RGB

    face_locations = face_recognition.face_locations(rgb_image)
    print(f"Detected {len(face_locations)} faces in the image.")

    if not face_locations:
        print("No faces were detected in the image.")
        return

    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Draw a rectangle around the face in the original BGR image
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image, name, (left + 6, bottom - 15), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

    # Display the resulting image with faces
    cv2.imshow('Face Recognition', image)
    cv2.waitKey(0)  # Wait until a key is pressed
    cv2.destroyAllWindows()

# Open live preview, capture a photo on command
photo_path = capture_photo()
if photo_path:
    recognize_faces(photo_path)
