# Drowsiness_Detection_System
An IoT-based Drowsiness Detection System using Machine Learning to monitor driver alertness in real-time. It detects drowsiness through eye closure patterns and alerts via buzzer and water spray, ensuring road safety. Built using OpenCV, Haar cascades, and CNN.

# Drowsiness Detection System using Machine Learning in IoT

This project presents a real-time drowsiness detection system that uses machine learning and IoT to monitor driver alertness and prevent road accidents. It identifies signs of fatigue through eye closure detection using image processing and CNN models, and activates a buzzer or water spray as an alert mechanism.

## Objective

To design an effective system that detects drowsiness in drivers using camera-based monitoring, computer vision, and machine learning, aiming to reduce road accidents caused by fatigue.

## Key Features

- Real-time eye detection using Haar Cascade Classifier
- Drowsiness detection using CNN (Convolutional Neural Networks)
- Alerts via buzzer and water spray
- Data transmission to cloud using IoT (ThingSpeak)
- Works in low-light and varying face positions

## Technologies & Tools Used

- Python  
- OpenCV  
- Haar Cascade Classifier for eye detection  
- TensorFlow / Keras for CNN model  
- Arduino UNO / NodeMCU (for hardware integration)  
- ThingSpeak (for real-time cloud monitoring)  
- Buzzer and Water Spray as alert mechanisms  

## Hardware Requirements

- USB Camera
- Buzzer
- Water Pump / Spray
- NodeMCU or Arduino
- Relay Module
- Jumper Wires and Power Supply

## ML Model

A CNN is trained to detect eye state (open/closed) from webcam frames. If eyes are detected closed for a defined threshold duration, the alert system is triggered.

## IoT Integration

Sensor and alert data is sent to the cloud using ThingSpeak, enabling real-time monitoring and data analysis.

## Future Improvements

- Integrating GPS tracking
- Enhancing CNN accuracy with larger datasets
- Mobile app for live alert notifications



