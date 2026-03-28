import numpy as np
import os
import tensorflow as tf
from scr.model import create_lstm_model
from tensorflow.keras.utils import to_categorical
import shutil
import pandas as pd

def load_keypoints_from_csv(csv_path):
    """
    Load keypoints từ 1 file CSV
    Returns: List of feature arrays (mỗi frame = 1 array 34 values)
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Chuyển DataFrame thành list of arrays
        keypoints_list = df.values.tolist()
        
        print(f"   ✓ Loaded {os.path.basename(csv_path)}: {len(keypoints_list)} frames")
        return keypoints_list
    
    except Exception as e:
        print(f"   ✗ Error loading {csv_path}: {e}")
        return None


def load_and_process_dataset(data_dir='data', sequence_length=30):
    """
    Load dataset từ Keypoints_CSV (NHANH HƠN - KHÔNG CẦN EXTRACT)
    data/
    ├── Fall/
    │   └── Keypoints_CSV/
    └── No_Fall/
        └── Keypoints_CSV/
    """
    all_keypoints = []
    all_labels = []
    
    # Process FALL keypoints (Label = 1)
    fall_csv_dir = os.path.join(data_dir, 'Fall', 'Keypoints_CSV')
    if os.path.exists(fall_csv_dir):
        print(f"📂 Loading FALL keypoints from: {fall_csv_dir}")
        csv_files = [f for f in os.listdir(fall_csv_dir) if f.endswith('.csv')]
        print(f"   Found {len(csv_files)} CSV files")
        
        for csv_file in csv_files:
            csv_path = os.path.join(fall_csv_dir, csv_file)
            keypoints_data = load_keypoints_from_csv(csv_path)
            
            if keypoints_data is not None and len(keypoints_data) > 0:
                labels = [1] * len(keypoints_data)
                all_keypoints.extend(keypoints_data)
                all_labels.extend(labels)
    else:
        print(f"⚠️ Warning: Fall CSV directory not found: {fall_csv_dir}")
    
    # Process NO_FALL keypoints (Label = 0)
    normal_csv_dir = os.path.join(data_dir, 'No_Fall', 'Keypoints_CSV')
    if os.path.exists(normal_csv_dir):
        print(f"📂 Loading NO_FALL keypoints from: {normal_csv_dir}")
        csv_files = [f for f in os.listdir(normal_csv_dir) if f.endswith('.csv')]
        print(f"   Found {len(csv_files)} CSV files")
        
        for csv_file in csv_files:
            csv_path = os.path.join(normal_csv_dir, csv_file)
            keypoints_data = load_keypoints_from_csv(csv_path)
            
            if keypoints_data is not None and len(keypoints_data) > 0:
                labels = [0] * len(keypoints_data)
                all_keypoints.extend(keypoints_data)
                all_labels.extend(labels)
    else:
        print(f"⚠️ Warning: No_Fall CSV directory not found: {normal_csv_dir}")
    
    if len(all_keypoints) == 0:
        print("❌ ERROR: No keypoints found! Check your CSV files.")
        print("💡 Make sure Keypoints_CSV folders contain .csv files")
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
    print("📊 Using pre-extracted Keypoints (FAST!)")
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
        print("\n💡 Troubleshooting:")
        print("   1. Check if data/Fall/Keypoints_CSV/ exists")
        print("   2. Check if data/No_Fall/Keypoints_CSV/ exists")
        print("   3. Make sure CSV files are present")
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