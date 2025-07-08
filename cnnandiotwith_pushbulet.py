import cv2
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from collections import deque
import dlib
from scipy.spatial import distance as dist
import time
import pyttsx3  # Voice alert module

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# ESP32-CAM URL
ESP32_CAM_URL = "http://192.168.9.20/capture"

# Pushbullet API details
PUSHBULLET_ACCESS_TOKEN = "o.7pRUk9gGosXpwCBw4WuaLDLYgcePlLi0"
PUSHBULLET_URL = "https://api.pushbullet.com/v2/pushes"

# Load YOLO for face detection
yolo_model = YOLO("yolov8n-face-lindevs.pt")

# Load CNN model for drowsiness classification
drowsiness_model = load_model('drowsiness_model.h5')

# Load dlib for facial landmarks
predictor_path = "D:/Cap project/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# EAR & MAR Thresholds
EAR_THRESHOLD = 0.28  
MAR_THRESHOLD = 0.5   
CNN_THRESHOLD = 0.50  
EYE_CLOSURE_TIME_THRESHOLD = 2.5  
DROWSINESS_ALERT_THRESHOLD = 1.0  # Voice alert & Pushbullet trigger

# Function to send Pushbullet notification
def send_pushbullet_notification():
    headers = {
        "Access-Token": PUSHBULLET_ACCESS_TOKEN,
        "Content-Type": "application/json"
    }
    data = {
        "type": "note",
        "title": "🚨 Drowsiness Alert!",
        "body": "Driver is drowsy! Please take action."
    }
    response = requests.post(PUSHBULLET_URL, json=data, headers=headers)
    if response.status_code == 200:
        print("✅ Pushbullet notification sent!")
    else:
        print(f"❌ Failed to send notification. HTTP {response.status_code}: {response.text}")

# Function to Capture Image from ESP32-CAM
def get_frame():
    response = requests.get(ESP32_CAM_URL, timeout=10)
    img = Image.open(BytesIO(response.content))
    frame = np.array(img)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame

# Functions to calculate EAR & MAR
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10])  
    B = dist.euclidean(mouth[4], mouth[8])   
    C = dist.euclidean(mouth[0], mouth[6])   
    return (A + B) / (2.0 * C)

# Drowsiness Detection Class
class DrowsinessClassifier:
    def __init__(self):
        self.prediction_history = deque(maxlen=8)  
        self.drowsy_time = 0.0  
        self.total_drowsy_time = 0.0  
        self.eye_closed = False
        self.eye_closure_start_time = None
        self.eye_closure_duration = 0.0
        self.last_frame_time = time.time()
        self.alert_sent = False  # Ensure notification is sent only once per drowsiness event

    def classify_drowsiness(self, face_crop, ear, mar):
        # Preprocess face for CNN
        face_resized = cv2.resize(face_crop, (128, 128))
        face_normalized = face_resized.astype("float32") / 255.0
        face_expanded = np.expand_dims(face_normalized, axis=0)
        
        # CNN Prediction
        cnn_prediction = float(drowsiness_model.predict(face_expanded, verbose=0)[0])
        self.prediction_history.append(cnn_prediction)
        avg_prediction = np.mean(self.prediction_history)

        # Track eye closure duration
        current_time = time.time()
        time_delta = current_time - self.last_frame_time
        self.last_frame_time = current_time

        if ear < EAR_THRESHOLD:
            if not self.eye_closed:
                self.eye_closed = True
                self.eye_closure_start_time = current_time
            else:
                self.eye_closure_duration = current_time - self.eye_closure_start_time
        else:
            self.eye_closed = False
            self.eye_closure_duration = 0.0

        # Determine Drowsiness State
        if self.eye_closure_duration >= EYE_CLOSURE_TIME_THRESHOLD or (avg_prediction > CNN_THRESHOLD and ear < EAR_THRESHOLD):
            state = 'DROWSY'
            self.drowsy_time += time_delta
            self.total_drowsy_time += time_delta
            color = (0, 0, 255)

            # 🚨 Trigger Pushbullet Notification & Voice Alert
            if self.drowsy_time >= DROWSINESS_ALERT_THRESHOLD and not self.alert_sent:
                send_pushbullet_notification()
                engine.say("Drowsiness detected, please wake up!")
                engine.runAndWait()
                self.alert_sent = True  # Avoid sending repeated alerts
            
        else:
            state = 'AWAKE'
            color = (0, 255, 0)
            self.drowsy_time = 0  
            self.alert_sent = False  # Reset alert flag when awake

        return state, color, avg_prediction, ear, mar, self.drowsy_time, self.total_drowsy_time

# Main Function
def main():
    classifier = DrowsinessClassifier()
    
    while True:
        try:
            frame = get_frame()
            display_frame = frame.copy()
            results = yolo_model(frame)

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = box.conf[0]

                    if conf > 0.25:
                        face_crop = frame[y1:y2, x1:x2]
                        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                        faces = detector(gray)

                        if len(faces) > 0:
                            shape = predictor(gray, faces[0])
                            landmarks = np.array([(shape.part(n).x, shape.part(n).y) for n in range(68)])
                            left_eye = landmarks[36:42]
                            right_eye = landmarks[42:48]
                            mouth = landmarks[48:68]

                            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
                            mar = mouth_aspect_ratio(mouth)

                            state, color, confidence, ear, mar, drowsy_time, total_drowsy_time = classifier.classify_drowsiness(face_crop, ear, mar)
                            
                            # 📍 Display Status, Bounding Box & Drowsy Time
                            box_color = (0, 255, 0) if state == "AWAKE" else (0, 0, 255)
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 2)
                            cv2.putText(display_frame, f"State: {state}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
                            cv2.putText(display_frame, f"Total Drowsy Time: {total_drowsy_time:.1f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Drowsiness Detection", display_frame)
        except Exception as e:
            print(f"Error: {e}")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
