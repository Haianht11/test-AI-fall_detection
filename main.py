import cv2
import argparse
from src.detector import FallDetector

def main(args):
    """
    Main function to run fall detection
    """
    # Initialize detector
    detector = FallDetector(
        pose_model=args.pose_model,
        lstm_model_path=args.model,
        sequence_length=args.seq_length,
        fall_threshold=args.threshold
    )
    
    # Open video source
    if args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source))
    else:
        cap = cv2.VideoCapture(args.source)
    
    if not cap.isOpened():
        print("Error: Cannot open video source")
        return
    
    print(f"📹 Video source: {args.source}")
    print("Press 'q' to quit, 'r' to reset detector")
    
    # Process video
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video")
            break
        
        # Process frame
        processed_frame, keypoints, status, prob = detector.process_frame(frame)
        
        # Display
        cv2.imshow('Fall Detection - YOLOv8-Pose + LSTM', processed_frame)
        
        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            detector.reset()
            print("Detector reset")
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Detection stopped")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fall Detection System with YOLOv8-Pose')
    
    parser.add_argument('--source', type=str, default='0',
                       help='Video source (0 for webcam, or video file path)')
    parser.add_argument('--model', type=str, default='models/fall_lstm.h5',
                       help='Path to trained LSTM model')
    parser.add_argument('--pose-model', type=str, default='yolov8n-pose.pt',
                       help='YOLOv8 pose model (yolov8n/s/m/l-pose.pt)')
    parser.add_argument('--seq-length', type=int, default=30,
                       help='Sequence length for LSTM')
    parser.add_argument('--threshold', type=float, default=0.7,
                       help='Fall detection threshold')
    
    args = parser.parse_args()
    main(args)