import cv2
import os
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from collections import deque
import dlib
from scipy.spatial import distance as dist
import time

# Load YOLO for face detection
yolo_model = YOLO("yolov8n-face-lindevs.pt")

# Load CNN model for drowsiness classification
drowsiness_model = load_model('drowsiness_model.h5')

# Load dlib for facial landmarks
predictor_path = "D:/Cap project/shape_predictor_68_face_landmarks.dat"  
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# EAR & MAR threshold values - Increased EAR threshold for sensitivity
EAR_THRESHOLD = 0.25  # Increased from 0.22 to make more sensitive
MAR_THRESHOLD = 0.5   
CNN_THRESHOLD = 0.55 
EYE_CLOSURE_TIME_THRESHOLD = 3.0  # Time threshold in seconds for eye closure

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

# Drowsiness classifier with prediction smoothing
class DrowsinessClassifier:
    def __init__(self):
        self.prediction_history = deque(maxlen=10)
        self.drowsy_time = 0.0  
        self.total_drowsy_time = 0.0  # Tracks overall drowsy time
        self.consecutive_drowsy = 0
        self.state_confidence = 0.0
        self.mar_history = deque(maxlen=30)  # Track MAR values to establish baseline
        self.mar_baseline = None
        
        # Eye closure tracking
        self.eye_closed = False
        self.eye_closure_start_time = None
        self.eye_closure_duration = 0.0
        self.last_frame_time = time.time()
        
        # Baseline EAR for the person
        self.ear_history = deque(maxlen=100)
        self.ear_baseline = None
        self.ear_threshold = EAR_THRESHOLD  # Start with default, will be personalized

    def classify_drowsiness(self, face_crop, ear, mar):
        # Process face for CNN
        face_resized = cv2.resize(face_crop, (128, 128))
        face_normalized = face_resized.astype("float32") / 255.0
        face_expanded = np.expand_dims(face_normalized, axis=0)
        
        # CNN prediction
        cnn_prediction = float(drowsiness_model.predict(face_expanded, verbose=0)[0])
        self.prediction_history.append(cnn_prediction)
        avg_prediction = np.mean(self.prediction_history)
        
        # Update MAR history and establish baseline
        self.mar_history.append(mar)
        if len(self.mar_history) >= 20 and self.mar_baseline is None:
            # Initialize baseline after collecting enough samples
            self.mar_baseline = np.percentile(self.mar_history, 25)  # Use lower quartile as baseline
        
        # Update EAR history and establish baseline
        if ear > 0.15:  # Only collect when eyes are reasonably open (avoid collecting closed eyes as baseline)
            self.ear_history.append(ear)
        
        # After collecting enough samples, establish EAR baseline
        if len(self.ear_history) >= 50 and self.ear_baseline is None:
            self.ear_baseline = np.percentile(self.ear_history, 80)  # Use upper values as baseline for open eyes
            # Set personalized threshold at 80% of baseline (this can be tuned)
            self.ear_threshold = self.ear_baseline * 0.8
        
        # Adjust MAR threshold based on baseline if available
        adjusted_mar_threshold = MAR_THRESHOLD
        if self.mar_baseline is not None:
            # Add margin to baseline for personalized threshold
            adjusted_mar_threshold = self.mar_baseline + 0.15
        
        # Determine if mouth is significantly open (indicating yawning)
        mouth_open = mar > adjusted_mar_threshold
        
        # Calculate time since last frame
        current_time = time.time()
        time_delta = current_time - self.last_frame_time
        self.last_frame_time = current_time
        
        # Track eye closure duration using personalized threshold if available
        effective_ear_threshold = self.ear_threshold if self.ear_baseline is not None else EAR_THRESHOLD
        
        if ear < effective_ear_threshold:  # Eyes are closed or drooping
            if not self.eye_closed:  # Just started closing
                self.eye_closed = True
                self.eye_closure_start_time = current_time
            else:  # Continuing to be closed
                self.eye_closure_duration = current_time - self.eye_closure_start_time
        else:  # Eyes are open
            self.eye_closed = False
            self.eye_closure_duration = 0.0
        
        # Decision based on CNN + EAR + MAR + eye closure duration
        if self.eye_closure_duration >= EYE_CLOSURE_TIME_THRESHOLD:
            # Eyes closed for more than threshold time - definite drowsiness
            state = 'DROWSY'
            self.drowsy_time += time_delta
            self.total_drowsy_time += time_delta
            color = (0, 0, 255)  # Red
        elif avg_prediction > CNN_THRESHOLD and ear < effective_ear_threshold:
            # High drowsiness confidence from CNN and eyes are closing
            state = 'DROWSY'
            self.drowsy_time += time_delta
            self.total_drowsy_time += time_delta
            color = (0, 0, 255)  # Red
        elif ear < effective_ear_threshold and mouth_open:
            # Eyes closed and yawning - strong drowsiness indicator
            state = 'DROWSY'
            self.drowsy_time += time_delta
            self.total_drowsy_time += time_delta
            color = (0, 0, 255)  # Red
        elif ear < effective_ear_threshold * 0.9:  # More sensitive check for drooping eyelids
            # Eyelids drooping significantly
            state = 'DROWSY'
            self.drowsy_time += time_delta
            self.total_drowsy_time += time_delta
            color = (0, 0, 255)  # Red
        elif avg_prediction < 0.35 and ear > effective_ear_threshold:
            # Low drowsiness confidence and eyes are open
            state = 'AWAKE'
            color = (0, 255, 0)  # Green
            self.drowsy_time = 0  
        else:
            state = 'UNCERTAIN'
            color = (255, 165, 0)  # Orange
        
        return state, color, avg_prediction, ear, mar, self.total_drowsy_time, self.eye_closure_duration, effective_ear_threshold

# Main function
def main():
    #ESP32_CAM_URL = "http://192.168.95.20:81/stream"
    ESP32_CAM_URL = "http://192.168.95.20/capture"
    cap = cv2.VideoCapture(ESP32_CAM_URL)


    #cap = cv2.VideoCapture(0)
    classifier = DrowsinessClassifier()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        display_frame = frame.copy()
        results = yolo_model(frame)
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                
                if conf > 0.5:
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
                        
                        state, color, confidence, ear, mar, total_drowsy_time, eye_closure_duration, current_ear_threshold = classifier.classify_drowsiness(face_crop, ear, mar)
                        
                        # Draw bounding box & text
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(display_frame, f"State: {state}", (x1, y1 - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        cv2.putText(display_frame, f"EAR: {ear:.2f} | MAR: {mar:.2f}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        
                        # Display total drowsy time
                        cv2.putText(display_frame, f"Drowsy Time: {total_drowsy_time:.1f}s", 
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                        
                        # Display eye closure duration
                        cv2.putText(display_frame, f"Eyes Closed: {eye_closure_duration:.1f}s", 
                                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        
                        # Display thresholds
                        cv2.putText(display_frame, f"EAR Threshold: {current_ear_threshold:.2f}", 
                                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        
                        # Display MAR baseline if available
                        if classifier.mar_baseline is not None:
                            cv2.putText(display_frame, f"MAR Baseline: {classifier.mar_baseline:.2f}", 
                                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow("Drowsiness Detection", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()