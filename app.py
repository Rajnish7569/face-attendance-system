import cv2
import face_recognition
import os
import pickle


if os.path.exists("encodings.pkl"):
    with open("encodings.pkl", "rb") as f:
        known_face_encodings, known_face_names = pickle.load(f)
else:
    known_face_encodings = []
    known_face_names = []

video_capture = cv2.VideoCapture(0)

print("Press 'n' to capture and save a new face.")
print("Press 'q' to quit.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    
    face_locations = face_recognition.face_locations(rgb_small_frame)
    if face_locations:
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            cv2.putText(frame, name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Video', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('n'):
        name = input("Enter name: ")
        if face_locations:
            for face_encoding in face_encodings:
                known_face_encodings.append(face_encoding)
                known_face_names.append(name)
            with open("encodings.pkl", "wb") as f:
                pickle.dump((known_face_encodings, known_face_names), f)
            print(f"Saved face for {name}")
        else:
            print("No face detected. Try again.")

    elif key == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
