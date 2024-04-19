import face_recognition
import os
import pickle

def encode_faces(image_directory, encodings_file, names_file):
    known_face_encodings = []
    known_face_names = []

    # Loop over each person's folder and each image in the folder
    for person_name in os.listdir(image_directory):
        person_folder = os.path.join(image_directory, person_name)
        if not os.path.isdir(person_folder):
            continue

        for image_file in os.listdir(person_folder):
            if image_file.endswith(".jpg") or image_file.endswith(".png"):
                image_path = os.path.join(person_folder, image_file)
                print(f"Processing {image_path}...")
                face_image = face_recognition.load_image_file(image_path)
                face_encodings = face_recognition.face_encodings(face_image)
                
                if face_encodings:
                    known_face_encodings.append(face_encodings[0])
                    known_face_names.append(person_name)

    # Save the encodings and the names to disk
    with open(encodings_file, "wb") as ef:
        pickle.dump(known_face_encodings, ef)
    with open(names_file, "wb") as nf:
        pickle.dump(known_face_names, nf)

    print("Encoding complete. Data saved to disk.")

if __name__ == "__main__":
    encode_faces("faces", "encodings.pickle", "names.pickle")
