import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.QtGui import QPixmap, QImage
import cv2
from face_recognition_module import FaceRecognition

class AbsenceTrackerUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.face_recognition = FaceRecognition("known_faces")

        self.camera = cv2.VideoCapture(0)
        self.central_widget = QLabel(self)
        self.setCentralWidget(self.central_widget)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.camera.read()
        recognized_names = self.face_recognition.recognize_faces(frame)

        for name in recognized_names:
            self.log_absence(name)

        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(img)
            self.central_widget.setPixmap(pixmap)

    def log_absence(self, name):
        with open("logs/absence_log.txt", "a") as log_file:
            log_file.write(f"{name} - {datetime.datetime.now()}\n")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AbsenceTrackerUI()
    window.show()
    sys.exit(app.exec_())
