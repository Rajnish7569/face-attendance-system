import cv2
import face_recognition

def get_face_encodings_from_camera():
    video_capture = cv2.VideoCapture(0)
    ret, frame = video_capture.read()
    if not ret:
        print("Camera not accessible")
        return None

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    if face_locations:
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        return face_encodings, face_locations, frame
    else:
        print("No face detected.")
        return None, None, frame


if __name__ == "__main__":
    encodings, locations, frame = get_face_encodings_from_camera()
    if encodings:
        print("Face encodings obtained.")
