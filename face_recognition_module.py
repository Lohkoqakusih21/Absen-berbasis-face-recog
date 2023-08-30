import face_recognition
import os

class FaceRecognition:
    def __init__(self, known_faces_path):
        self.known_faces = []
        self.known_names = []

        for name in os.listdir(known_faces_path):
            image = face_recognition.load_image_file(os.path.join(known_faces_path, name))
            encoding = face_recognition.face_encodings(image)[0]
            self.known_faces.append(encoding)
            self.known_names.append(os.path.splitext(name)[0])

    def recognize_faces(self, frame):
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        recognized_names = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_faces, face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_names[first_match_index]
                recognized_names.append(name)

        return recognized_names
