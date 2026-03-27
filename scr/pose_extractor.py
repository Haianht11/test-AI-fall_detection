import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

class PoseExtractor:
    def __init__(self, model_name='yolov8n-pose.pt'):
        """
        Khởi tạo YOLOv8-Pose model
        """
        self.model = YOLO(model_name)
        self.conf_threshold = 0.5
        
    def extract(self, frame):
        """
        Trích xuất keypoints từ khung hình
        Returns: numpy array chứa tọa độ các keypoints (17 điểm)
        """
        results = self.model(frame, verbose=False, conf=self.conf_threshold)
        
        if results[0].keypoints is not None:
            keypoints = results[0].keypoints.xy.cpu().numpy()
            
            # Lấy người có confidence cao nhất
            if len(keypoints) > 0:
                return keypoints[0]
        
        return None
    
    def keypoints_to_features(self, keypoints):
        """
        Chuyển keypoints thành feature vector
        """
        if keypoints is None:
            return None
            
        # Flatten keypoints thành 1 vector (17 points x 2 coords = 34 values)
        return keypoints.flatten()
    
    def draw_pose(self, frame, keypoints):
        """
        Vẽ pose lên khung hình
        """
        if keypoints is None:
            return frame
            
        # Vẽ các điểm keypoints
        for i, (x, y) in enumerate(keypoints):
            x, y = int(x), int(y)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            
        return frame