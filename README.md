# People Detection System

A real-time people detection system using YOLO (MobileNet SSD) and background subtraction, with MQTT integration for IoT-based monitoring.

## Overview

This project implements a sophisticated computer vision system that:
- Detects people in video streams using YOLO (MobileNet SSD)
- Employs background subtraction (MOG2 or KNN) for motion detection
- Achieves 80% improvement over traditional methods through combined movement and object detection
- Publishes detection alerts via MQTT for IoT integration
- Optimized for deployment on Raspberry Pi 4

## Features

- Real-time human detection with YOLO
- Configurable video source (webcam or IP camera)
- MQTT-based alert system for automatic notifications
- IoT-ready with Raspberry Pi 4 compatibility
- Adjustable detection parameters via config file
- Real-time FPS display
- Video recording of detections
- Background subtraction for enhanced motion detection

## Prerequisites

- Python 3.8+
- OpenCV
- MobileNet SSD model files
- MQTT broker access
- (Optional) Raspberry Pi 4 for IoT deployment

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/people-detection-system.git
cd people-detection-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Download MobileNet SSD model files
4. Configure the system: Edit config.jason with your settings

## Usage
Run the directory 
```bash
python main.py
```
- press 'q' to quit
- Video recordings are saved with timestamp in filename
- Detection alerts are sent via MQTT to configured broker

For Raspberry Pi deployment:
- Install Raspberry Pi OS
- Connect camera module or USB webcam
- Follow installation steps above
- Run the script with appropriate config settings

## Performance
- Achieves 80% improvement over traditional motion detection methods
- Combines YOLO object detection with background subtraction
- Optimized for real-time performance on Raspberry Pi 4

## Image
This image shows a Raspberry Pi 4 mounted in a case. The setup is intended to be connected to a CCTV camera installed on a 3-meter pole, where it will perform object detection tasks.

<img src="https://github.com/user-attachments/assets/c3eaf4b1-5cb7-4c17-bbbe-d4cc072f30eb" width="350" height="300" >
<img src="https://github.com/user-attachments/assets/4c28e0a7-33c6-44c0-8332-ec6d14e5aeea" width="200" height="350">

