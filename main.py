from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import cv2
import numpy as np
import json
import logging
from paho.mqtt import client as mqtt_client
import socket
from typing import Dict, Tuple
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
          "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
          "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
          "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

class PeopleDetector:
    def __init__(self, config_path: str = 'config.json'):
        """Initialize the PeopleDetector with configuration."""
        self.config = self._load_config(config_path)
        self._setup_background_model()
        self._setup_video_capture()
        self._setup_neural_network()
        self._setup_mqtt()
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.prev_frame_time = 0
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as config_file:
                return json.load(config_file)
        except FileNotFoundError:
            logger.error(f"Config file {config_path} not found")
            raise
        except json.JSONDecodeError:
            logger.error("Invalid JSON in config file")
            raise

    def _setup_background_model(self) -> None:
        """Initialize background subtraction model."""
        if self.config["background_model"] == "mog2":
            self.background_model = cv2.createBackgroundSubtractorMOG2()
        elif self.config["background_model"] == "knn":
            self.background_model = cv2.createBackgroundSubtractorKNN()
        else:
            raise ValueError(f"Unsupported background model: {self.config['background_model']}")

    def _setup_video_capture(self) -> None:
        """Initialize video capture source."""
        if self.config["capture_camera"] == 'webcam':
            self.vs = VideoStream(src=0).start()
            time.sleep(2.0)
        elif self.config["capture_camera"] == 'ip_camera':
            self.vs = cv2.VideoCapture(self.config["camera_ip"])
        else:
            raise ValueError(f"Unsupported capture source: {self.config['capture_camera']}")

    def _setup_neural_network(self) -> None:
        """Initialize the neural network for object detection."""
        try:
            self.net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 
                                              'MobileNetSSD_deploy.caffemodel')
        except cv2.error as e:
            logger.error(f"Failed to load neural network: {e}")
            raise

    def _setup_mqtt(self) -> None:
        """Setup MQTT client configuration."""
        self.broker = '91.92.231.159'
        self.port = 1883
        self.topic = "python/mqtt"
        self.client_id = f'NG-Camera-{socket.gethostname()}'

    def send_mqtt(self, number_of_people: int) -> None:
        """Send detection data via MQTT."""
        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                logger.info("Connected to MQTT Broker!")
            else:
                logger.error(f"Failed to connect, return code {rc}")

        client = mqtt_client.Client(mqtt_client.CallbackAPIVersion.VERSION1, self.client_id)
        client.on_connect = on_connect
        
        try:
            client.connect(self.broker, self.port)
            client.loop_start()
            
            msg = {
                'DateTime': datetime.datetime.now().isoformat(),
                'NumberOfHumans': number_of_people,
                'DeviceName': socket.gethostname()
            }
            result = client.publish(self.topic, json.dumps(msg))
            
            if result[0] == 0:
                logger.info(f"Sent message to topic {self.topic}: {msg}")
            else:
                logger.error(f"Failed to send message to topic {self.topic}")
                
        except Exception as e:
            logger.error(f"MQTT connection error: {e}")
        finally:
            client.loop_stop()

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, bool, int]:
        """Process a single video frame and detect people."""
        people_count = 0
        yolo_active = False
        
        frame = imutils.resize(frame, width=self.config["resize_width"], 
                             height=self.config["resize_height"])
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (self.config["gaussian_blur_amount"], 
                                      self.config["gaussian_blur_amount"]), 0)
        
        frameDelta = self.background_model.apply(gray)
        thresh = cv2.threshold(frameDelta, 10, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        for c in cnts:
            if cv2.contourArea(c) < self.config["minimum_area"]:
                continue
                
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 
                                       0.007843, (300, 300), 127.5)
            self.net.setInput(blob)
            detections = self.net.forward()
            yolo_active = True
            
            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > self.config["confidence_interval"]:
                    idx = int(detections[0, 0, i, 1])
                    if CLASSES[idx] == "person":
                        people_count += 1
                        box = detections[0, 0, i, 3:7] * np.array([self.config["resize_width"], 
                                                                 self.config["resize_height"],
                                                                 self.config["resize_width"],
                                                                 self.config["resize_height"]])
                        (startX, startY, endX, endY) = box.astype("int")
                        label = f"{CLASSES[idx]}: {confidence * 100:.2f}%"
                        cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
                        y = startY - 15 if startY - 15 > 15 else startY + 15
                        cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.5, COLORS[idx], 2)
            break

        return frame, yolo_active, people_count

    def run(self) -> None:
        """Main loop for video processing."""
        try:
            while True:
                if self.config["capture_camera"] == "webcam":
                    frame = self.vs.read()
                else:
                    ret, frame = self.vs.read()
                    if not ret:
                        logger.error("Failed to read frame from video source")
                        break

                frame, yolo_active, people_count = self.process_frame(frame)
                
                # Calculate and display FPS
                new_frame_time = time.time()
                fps = 1 / (new_frame_time - self.prev_frame_time)
                self.prev_frame_time = new_frame_time
                
                # Add overlays
                cv2.putText(frame, f"FPS: {round(fps, 2)}", (7, 70), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                cv2.putText(frame, f"Yolo: {yolo_active}", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"), 
                          (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.35, (0, 0, 255), 1)

                cv2.imshow("Security Feed", frame)
                
                if people_count > 0:
                    self.send_mqtt(people_count)
                    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
                    filename = f"output_{timestamp}.avi"
                    out = cv2.VideoWriter(filename, self.fourcc, 20.0, 
                                        (self.config["resize_width"], self.config["resize_height"]))
                    out.write(frame)
                    out.release()

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                    
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Clean up resources."""
        if self.config["capture_camera"] == "webcam":
            self.vs.stop()
        else:
            self.vs.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = PeopleDetector()
    detector.run()