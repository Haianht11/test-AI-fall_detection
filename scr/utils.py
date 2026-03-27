import cv2
import numpy as np
from datetime import datetime

def draw_status(frame, status, probability):
    """
    Draw status text on frame
    """
    if "FALL" in status:
        color = (0, 0, 255)
        bg_color = (200, 200, 255)
    else:
        color = (0, 255, 0)
        bg_color = (200, 255, 200)
    
    text = f"Status: {status}"
    prob_text = f"Probability: {probability:.2%}"
    
    cv2.rectangle(frame, (10, 10), (400, 80), bg_color, -1)
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, prob_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return frame