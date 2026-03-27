import numpy as np
import tensorflow as tf
import cv2
from scr.pose_extractor import PoseExtractor
from scr.utils import draw_status

class FallDetector:
    def __init__(self, 
                 pose_model='yolov8n-pose.pt',
                 lstm_model_path='models/fall_lstm.h5',
                 sequence_length=30,
                 fall_threshold=0.7):
        
        # Initialize pose extractor
        self.pose_extractor = PoseExtractor(model_name=pose_model)
        
        # Load LSTM model
        try:
            self.lstm_model = tf.keras.models.load_model(lstm_model_path)
            print(f"✓ Loaded LSTM model from {lstm_model_path}")
        except:
            self.lstm_model = None
            print("⚠ Warning: LSTM model not found.")
        
        self.sequence_length = sequence_length
        self.fall_threshold = fall_threshold
        self.frame_buffer = []
        self.fall_counter = 0
        self.is_fall_detected = False
        
    def add_frame(self, frame):
        """
        Add frame to buffer and extract keypoints
        """
        keypoints = self.pose_extractor.extract(frame)
        
        if keypoints is not None:
            features = self.pose_extractor.keypoints_to_features(keypoints)
            self.frame_buffer.append(features)
            
            if len(self.frame_buffer) > self.sequence_length:
                self.frame_buffer.pop(0)
                
            return keypoints
        return None
    
    def predict(self):
        """
        Predict if fall occurred
        """
        if self.lstm_model is None or len(self.frame_buffer) < self.sequence_length:
            return "No Model", 0.0
        
        sequence = np.array(self.frame_buffer[-self.sequence_length:])
        sequence = sequence.reshape(1, self.sequence_length, -1)
        
        prediction = self.lstm_model.predict(sequence, verbose=0)[0]
        fall_prob = prediction[1]
        
        if fall_prob > self.fall_threshold:
            self.fall_counter += 1
        else:
            self.fall_counter = max(0, self.fall_counter - 1)
        
        if self.fall_counter >= 15:
            self.is_fall_detected = True
            return "FALL DETECTED!", fall_prob
        else:
            self.is_fall_detected = False
            return "Normal", fall_prob
    
    def process_frame(self, frame):
        """
        Process a single frame
        """
        keypoints = self.add_frame(frame)
        
        if keypoints is not None:
            frame = self.pose_extractor.draw_pose(frame, keypoints)
        
        status, probability = self.predict()
        frame = draw_status(frame, status, probability)
        
        if self.is_fall_detected and status == "FALL DETECTED!":
            frame = cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 3)
        
        return frame, keypoints, status, probability
    
    def reset(self):
        """Reset detector state"""
        self.frame_buffer = []
        self.fall_counter = 0
        self.is_fall_detected = False