import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional

def create_lstm_model(input_shape, num_classes=2):
    """
    Tạo mô hình LSTM cho fall detection
    
    Args:
        input_shape: (sequence_length, num_features)
        num_classes: 2 (fall/not fall)
    """
    model = Sequential([
        # Bidirectional LSTM
        Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape),
        Dropout(0.3),
        
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        
        Bidirectional(LSTM(32)),
        Dropout(0.3),
        
        # Fully connected layers
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model