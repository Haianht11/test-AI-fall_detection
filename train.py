import cv2
import numpy as np
import os
import tensorflow as tf
from scr.pose_extractor import PoseExtractor
from scr.model import create_lstm_model
from tensorflow.keras.utils import to_categorical
import shutil

def load_and_process_dataset(data_dir='data', sequence_length=30):
    """
    Load dataset từ cấu trúc mới:
    data/
    ├── Fall/
    │   └── Raw_Video/
    └── No_Fall/
        └── Raw_Video/
    """
    pose_extractor = PoseExtractor()
    
    all_keypoints = []
    all_labels = []
    
    # Process FALL videos (Label = 1)
    fall_dir = os.path.join(data_dir, 'Fall', 'Raw_Video')
    if os.path.exists(fall_dir):
        print(f"📂 Processing FALL videos from: {fall_dir}")
        for video_file in os.listdir(fall_dir):
            if video_file.endswith(('.mp4', '.avi', '.mov')):
                print(f"  🎬 Processing: {video_file}")
                video_path = os.path.join(fall_dir, video_file)
                keypoints_seq = extract_keypoints_from_video(video_path, pose_extractor)
                
                # Label: 1 for fall
                labels = [1] * len(keypoints_seq)
                all_keypoints.extend(keypoints_seq)
                all_labels.extend(labels)
    else:
        print(f"⚠️ Warning: Fall directory not found: {fall_dir}")
    
    # Process NO_FALL videos (Label = 0)
    normal_dir = os.path.join(data_dir, 'No_Fall', 'Raw_Video')
    if os.path.exists(normal_dir):
        print(f"📂 Processing NO_FALL videos from: {normal_dir}")
        for video_file in os.listdir(normal_dir):
            if video_file.endswith(('.mp4', '.avi', '.mov')):
                print(f"  🎬 Processing: {video_file}")
                video_path = os.path.join(normal_dir, video_file)
                keypoints_seq = extract_keypoints_from_video(video_path, pose_extractor)
                
                # Label: 0 for normal
                labels = [0] * len(keypoints_seq)
                all_keypoints.extend(keypoints_seq)
                all_labels.extend(labels)
    else:
        print(f"⚠️ Warning: No_Fall directory not found: {normal_dir}")
    
    if len(all_keypoints) == 0:
        print("❌ ERROR: No videos found! Check your dataset structure.")
        return None, None, None, None
    
    # Create sequences
    print(f"\n📊 Creating sequences (length={sequence_length})...")
    X, y = prepare_sequences(all_keypoints, all_labels, sequence_length)
    
    if len(X) == 0:
        print("❌ ERROR: Could not create sequences. Not enough data?")
        return None, None, None, None
    
    # Convert labels to categorical
    y_categorical = to_categorical(y, num_classes=2)
    
    # Split train/val
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y_categorical[:split_idx], y_categorical[split_idx:]
    
    print(f"\n✅ Dataset prepared:")
    print(f"  Train samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Sequence shape: {X_train.shape}")
    
    return X_train, y_train, X_val, y_val

def extract_keypoints_from_video(video_path, pose_extractor):
    """
    Extract keypoints from all frames in a video
    """
    cap = cv2.VideoCapture(video_path)
    keypoints_list = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        keypoints = pose_extractor.extract(frame)
        if keypoints is not None:
            features = pose_extractor.keypoints_to_features(keypoints)
            keypoints_list.append(features)
            frame_count += 1
    
    cap.release()
    print(f"    Extracted {frame_count} frames with keypoints")
    return keypoints_list

def prepare_sequences(keypoints_data, labels, sequence_length=30):
    """
    Chuẩn bị dữ liệu sequence cho LSTM
    """
    X, y = [], []
    
    for i in range(len(keypoints_data) - sequence_length):
        # Tạo sequence
        sequence = keypoints_data[i:i + sequence_length]
        X.append(sequence)
        y.append(labels[i + sequence_length - 1])
    
    return np.array(X), np.array(y)

if __name__ == "__main__":
    print("="*60)
    print("🚀 FALL DETECTION MODEL TRAINING")
    print("="*60)
    
    # Parameters
    SEQUENCE_LENGTH = 30
    NUM_FEATURES = 34  # 17 keypoints x 2 coordinates
    
    # Load and process dataset
    X_train, y_train, X_val, y_val = load_and_process_dataset(
        data_dir='data',
        sequence_length=SEQUENCE_LENGTH
    )
    
    if X_train is None:
        print("\n❌ Training aborted due to errors above.")
    else:
        # Train model
        print("\n" + "="*60)
        print("📊 TRAINING LSTM MODEL...")
        print("="*60)
        
        input_shape = (SEQUENCE_LENGTH, NUM_FEATURES)
        model = create_lstm_model(input_shape)
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint('models/fall_lstm.h5', monitor='val_accuracy', save_best_only=True, mode='max')
        ]
        
        # Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Save model
        model.save('models/fall_lstm.h5')
        print(f"\n✅ Model saved to: models/fall_lstm.h5")
        
        # Save to Google Drive if available
        if os.path.exists('/content/drive'):
            drive_path = '/content/drive/MyDrive/FallDetection/models'
            os.makedirs(drive_path, exist_ok=True)
            shutil.copy('models/fall_lstm.h5', f'{drive_path}/fall_lstm.h5')
            print(f"💾 Backup saved to Google Drive: {drive_path}/fall_lstm.h5")
        
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETED!")
        print("="*60)