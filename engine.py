import cv2
import mediapipe as mp
import time
from PyQt6.QtCore import QThread, pyqtSignal, PyObject

class HandTrackingEngine(QThread):
    # Signals to send data back to the PyQt UI safely
    change_pixmap_signal = pyqtSignal(object)  # For the Live Webcam Preview
    status_signal = pyqtSignal(bool, float)     # (Hand Detected, FPS)
    gesture_signal = pyqtSignal(str)           # Triggered gesture name

    def __init__(self):
        super().__init__()
        self._run_flag = True
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Configuration Variables (Mappable from UI)
        self.tracking_enabled = True
        self.sensitivity = 1.5

    def run(self):
        cap = cv2.VideoCapture(0)
        prev_frame_time = 0

        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                continue

            # Flip frame horizontally for a mirror effect
            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            hand_detected = False
            fps = 0.0

            # Calculate FPS
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0
            prev_frame_time = new_frame_time

            if results.multi_hand_landmarks:
                hand_detected = True
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks on the frame for the UI preview
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    # Extract Index Finger Tip (Landmark 8) for cursor movement
                    index_tip = hand_landmarks.landmark[8]
                    cx, cy = int(index_tip.x * w), int(index_tip.y * h)
                    
                    # Core Logic: Simple Gesture Processing Example
                    # (In production, map these coordinates to screen size and trigger automation)
                    if self.tracking_enabled:
                        self.process_cursor_and_gestures(hand_landmarks, w, h)

            # Emit updates to the UI
            self.change_pixmap_signal.emit(frame)
            self.status_signal.emit(hand_detected, round(fps, 1))

        cap.release()

    def process_cursor_and_gestures(self, landmarks, w, h):
        """
        Placeholder for gesture math logic.
        E.g., comparing distance between thumb tip (4) and index tip (8) for a click.
        """
        # Example: thumb_tip = landmarks.landmark[4]
        # Calculate distance -> if distance < threshold: self.gesture_signal.emit("Left Click")
        pass

    def stop(self):
        self._run_flag = False
        self.wait()