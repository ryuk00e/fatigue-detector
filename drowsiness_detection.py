import cv2
import imutils
from imutils import face_utils
import dlib
from scipy.spatial import distance
from pygame import mixer
import pyttsx3
import numpy as np
from collections import deque

# Initialize TTS and sound
engine = pyttsx3.init()
engine.setProperty('rate', 150)
mixer.init()
mixer.music.load("music.wav")

# EAR Calculation
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Improved Gaze Detection using Intensity Center
def is_looking_forward(eye_points, gray):
    x, y, w, h = cv2.boundingRect(eye_points)
    if w == 0 or h == 0:
        return True
    
    eye_frame = gray[y:y + h, x:x + w]
    if eye_frame.size == 0:
        return True

    eye_frame = cv2.equalizeHist(eye_frame)
    eye_blur = cv2.GaussianBlur(eye_frame, (7, 7), 0)
    eye_thresh = cv2.threshold(eye_blur, 50, 255, cv2.THRESH_BINARY_INV)[1]
    M = cv2.moments(eye_thresh)

    if M['m00'] == 0:
        return True  # Cannot determine, assume looking forward

    cx = int(M['m10'] / M['m00'])
    pupil_ratio = cx / float(w)

    return 0.3 < pupil_ratio < 0.7

# Alert system to avoid repeated announcements
class AlertManager:
    def __init__(self):
        self.sleep_alert_given = False
        self.gaze_alert_given = False

    def sleep_alert(self):
        if not self.sleep_alert_given:
            print("DROWSINESS ALERT")
            mixer.music.play()
            engine.say("Sleep Alert")
            engine.runAndWait()
            self.sleep_alert_given = True

    def gaze_alert(self):
        if not self.gaze_alert_given:
            print("NOT LOOKING FORWARD ALERT")
            engine.say("Not looking forward")
            engine.runAndWait()
            self.gaze_alert_given = True

    def reset(self):
        self.sleep_alert_given = False
        self.gaze_alert_given = False

# Constants
EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 20
GAZE_CONSEC_FRAMES = 60

# EAR smoothing with deque
ear_buffer = deque(maxlen=5)

# Flags
ear_counter = 0
gaze_counter = 0

# Dlib setup
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS['right_eye']

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

alerts = AlertManager()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    alerts.reset()

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        ear_buffer.append(ear)
        avg_ear = np.mean(ear_buffer)

        # Draw eyes
        cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)

        # Drowsiness Detection
        if avg_ear < EAR_THRESHOLD:
            ear_counter += 1
            if ear_counter >= EAR_CONSEC_FRAMES:
                cv2.putText(frame, "****** DROWSINESS ALERT ******", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                alerts.sleep_alert()
        else:
            ear_counter = 0

        # Gaze Detection
        if not (is_looking_forward(leftEye, gray) and is_looking_forward(rightEye, gray)):
            gaze_counter += 1
            if gaze_counter >= GAZE_CONSEC_FRAMES:
                cv2.putText(frame, "****** NOT LOOKING FORWARD ******", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                alerts.gaze_alert()
        else:
            gaze_counter = 0

    cv2.imshow("Driver Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
