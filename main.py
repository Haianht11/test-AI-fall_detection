import cv2
from scr.detector import FallDetector

def main():
    detector = FallDetector(
        pose_model='yolov8n-pose.pt',
        lstm_model_path='models/fall_lstm.h5',
        sequence_length=30,
        fall_threshold=0.7
    )
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    print("📹 Starting fall detection... Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame, keypoints, status, prob = detector.process_frame(frame)
        
        cv2.imshow('Fall Detection', processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Stopped")

if __name__ == "__main__":
    main()