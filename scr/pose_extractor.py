import cv2
import numpy as np
from ultralytics import YOLO

class PoseExtractor:
    def __init__(self, model_name='yolov8n-pose.pt'):
        """
        Khởi tạo YOLOv8-Pose model
        Models available: yolov8n-pose.pt, yolov8s-pose.pt, yolov8m-pose.pt, yolov8l-pose.pt
        """
        self.model = YOLO(model_name)
        self.conf_threshold = 0.5
        
    def extract(self, frame):
        """
        Trích xuất keypoints từ khung hình
        Returns: numpy array chứa tọa độ các keypoints (17 điểm của COCO format)
        """
        results = self.model(frame, verbose=False, conf=self.conf_threshold)
        
        if results[0].keypoints is not None:
            keypoints = results[0].keypoints.xy.cpu().numpy()  # Shape: (num_persons, 17, 2)
            
            # Lấy người có confidence cao nhất
            if len(keypoints) > 0:
                return keypoints[0]  # Trả về 17 keypoints của người đầu tiên
        
        return None
    
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
            
        # Vẽ các connections (skeleton)
        connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head and arms
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Legs
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]
        
        for idx1, idx2 in connections:
            if idx1 < len(keypoints) and idx2 < len(keypoints):
                pt1 = tuple(keypoints[idx1].astype(int))
                pt2 = tuple(keypoints[idx2].astype(int))
                cv2.line(frame, pt1, pt2, (255, 0, 0), 2)
                
        return frame
    
    def keypoints_to_features(self, keypoints):
        """
        Chuyển keypoints thành feature vector
        COCO format: 0-nose, 1-LEye, 2-REye, 3-LEar, 4-REar, 
                     5-LSho, 6-RSho, 7-LElb, 8-RElb, 9-LWri, 10-RWri,
                     11-LHip, 12-RHip, 13-LKne, 14-RKne, 15-LAnk, 16-RAnk
        """
        if keypoints is None:
            return None
            
        # Flatten keypoints thành 1 vector (17 points x 2 coords = 34 values)
        return keypoints.flatten()