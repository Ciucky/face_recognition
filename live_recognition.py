import face_recognition
import cv2
import pickle

# Load face encodings and names from disk
try:
    with open("encodings.pickle", "rb") as ef:
        known_face_encodings = pickle.load(ef)
    with open("names.pickle", "rb") as nf:
        known_face_names = pickle.load(nf)
except Exception as e:
    print(f"Failed to load encodings or names: {e}")
    exit(1)

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error: Webcam could not be accessed.")
    exit(1)

try:
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame")
            break

        try:
            # Convert the image from BGR color (which OpenCV uses) to RGB color
            rgb_frame = frame[:, :, ::-1]

            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_frame)
            print(f"Detected {len(face_locations)} faces in the frame.")

            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            # Display the results
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

                name = "Unknown"
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]

                # Draw a box around the face and label with a name
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

            # Display the resulting image
            cv2.imshow('Video', frame)
            if cv2.waitKey(1000) & 0xFF == ord('q'):  # Delay increased to 1 second per frame
                break
        except Exception as e:
            print(f"Error during face processing: {e}")
            continue

finally:
    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()
