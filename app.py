import sys
import cv2
import mediapipe as mp
import time
import pyautogui
import math
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, 
                             QVBoxLayout, QLabel, QPushButton, QSlider, QStackedWidget)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap

# CRITICAL SYSTEM OVERRIDES FOR RAW PERFORMANCE
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

# ==========================================
# 1. THE AI ENGINE THREAD
# ==========================================
class HandTrackingEngine(QThread):
    change_pixmap_signal = pyqtSignal(object)  
    status_signal = pyqtSignal(bool, float)     

    def __init__(self):
        super().__init__()
        self._run_flag = True
        
        # Initialize MediaPipe Hand Tracker
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,       # Performance optimized
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        
        # UI Mapped Settings
        self.tracking_enabled = False
        self.sensitivity = 1.7        # Slightly increased to reduce hand strain
        self.screen_width, self.screen_height = pyautogui.size()

        # Dynamic Tracking Variables (Removes input lag while keeping stability)
        self.prev_x, self.prev_y = 0, 0
        
        # State Holding Flags for Drag & Drop / Continuous holding
        self.left_button_held = False
        self.is_right_clicking = False
        
        # Standardized 2D Hand Space Thresholds
        self.click_threshold = 0.04  

    def run(self):
        cap = cv2.VideoCapture(1)
        
        # Force lower hardware capture resolution to eliminate processing backlog lag
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        prev_frame_time = 0

        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            hand_detected = False

            # Calculate FPS
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0
            prev_frame_time = new_frame_time

            if results.multi_hand_landmarks:
                hand_detected = True
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw visual skeletal map
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    # Target tracked landmarks
                    thumb_tip = hand_landmarks.landmark[4]
                    index_tip = hand_landmarks.landmark[8]
                    middle_tip = hand_landmarks.landmark[12]
                    
                    if self.tracking_enabled:
                        # 1. ADAPTIVE CURSOR SMOOTHING (Fixes both jitter AND ice gliding)
                        raw_x = int(index_tip.x * self.screen_width * self.sensitivity)
                        raw_y = int(index_tip.y * self.screen_height * self.sensitivity)
                        
                        if self.prev_x == 0 and self.prev_y == 0:
                            smooth_x, smooth_y = raw_x, raw_y
                        else:
                            # Calculate frame-by-frame physical movement speed
                            movement_distance = math.hypot(raw_x - self.prev_x, raw_y - self.prev_y)
                            
                            # Dynamic smoothing engine step
                            if movement_distance > 30:
                                # Quick motion: reduce smoothing completely to eliminate cursor lag/gliding
                                current_smoothing = 0.60
                            else:
                                # Slow motion / hovering: apply heavy smoothing to lock out jitter
                                current_smoothing = 0.15
                                
                            smooth_x = int(self.prev_x + current_smoothing * (raw_x - self.prev_x))
                            smooth_y = int(self.prev_y + current_smoothing * (raw_y - self.prev_y))
                        
                        # Apply desktop boundary snapping
                        smooth_x = max(0, min(smooth_x, self.screen_width - 1))
                        smooth_y = max(0, min(smooth_y, self.screen_height - 1))
                        
                        pyautogui.moveTo(smooth_x, smooth_y)
                        self.prev_x, self.prev_y = smooth_x, smooth_y

                        # 2. UPDATED MOUSE DOWN / UP LEFT CLICK MECHANIC (Thumb + Index)
                        left_dist = math.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)
                        
                        if left_dist < self.click_threshold:
                            if not self.left_button_held:
                                pyautogui.mouseDown(button='left')
                                self.left_button_held = True
                        else:
                            if self.left_button_held:
                                pyautogui.mouseUp(button='left')
                                self.left_button_held = False

                        # 3. RIGHT CLICK LOGIC (Thumb + Middle Finger)
                        right_dist = math.hypot(middle_tip.x - thumb_tip.x, middle_tip.y - thumb_tip.y)
                        
                        if right_dist < self.click_threshold:
                            if not self.is_right_clicking:
                                pyautogui.click(button='right')
                                self.is_right_clicking = True  
                        else:
                            self.is_right_clicking = False  

            # Safely release physical mouse state locks if your hand leaves the camera frame completely
            elif self.left_button_held:
                pyautogui.mouseUp(button='left')
                self.left_button_held = False

            # Push visual update data array layers up to the PyQt application dashboard
            self.change_pixmap_signal.emit(frame)
            self.status_signal.emit(hand_detected, round(fps, 1))

        cap.release()

    def stop(self):
        # Clean up mouse click state before quitting application
        if self.left_button_held:
            pyautogui.mouseUp(button='left')
        self._run_flag = False
        self.wait()


# ==========================================
# 2. THE PREMIUM INTERFACE DISPLAY
# ==========================================
class ZeroTouchApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ZeroTouch - AI Productivity Tool")
        self.setMinimumSize(1100, 700)
        
        self.engine = HandTrackingEngine()
        self.engine.change_pixmap_signal.connect(self.update_image)
        self.engine.status_signal.connect(self.update_status)
        
        self.init_ui()
        self.engine.start()

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Left Sidebar View Panel
        self.sidebar = QWidget()
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setFixedWidth(220)
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(10, 30, 10, 10)
        
        app_logo = QLabel("<h2>Ø ZeroTouch</h2>")
        app_logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sidebar_layout.addWidget(app_logo)
        sidebar_layout.addSpacing(30)
        
        dash_btn = QPushButton(" 📊 Dashboard")
        gesture_btn = QPushButton(" 🖐️ Gestures")
        settings_btn = QPushButton(" ⚙️ Settings")
        
        sidebar_layout.addWidget(dash_btn)
        sidebar_layout.addWidget(gesture_btn)
        sidebar_layout.addWidget(settings_btn)
        sidebar_layout.addStretch()

        # Right Workspace View Container
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(25, 25, 25, 25)

        # Header Action / Status Display Bar
        header_layout = QHBoxLayout()
        self.title_lbl = QLabel("<h1>Dashboard</h1>")
        self.status_lbl = QLabel("Status: Ready")
        self.fps_lbl = QLabel("FPS: 0")
        
        header_layout.addWidget(self.title_lbl)
        header_layout.addStretch()
        header_layout.addWidget(self.status_lbl)
        header_layout.addWidget(self.fps_lbl)
        right_layout.addLayout(header_layout)
        right_layout.addSpacing(20)

        self.content_stack = QStackedWidget()
        self.content_stack.addWidget(self.create_dashboard_view())
        right_layout.addWidget(self.content_stack)

        main_layout.addWidget(self.sidebar)
        main_layout.addWidget(right_container)
        self.setCentralWidget(main_widget)
        
        self.apply_premium_stylesheet()

    def create_dashboard_view(self):
        dash_page = QWidget()
        layout = QHBoxLayout(dash_page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(20)
        
        # Camera Feed Box Card Setup
        preview_card = QWidget()
        preview_card.setStyleSheet("background-color: #121214; border-radius: 12px; border: 1px solid #2d2d34;")
        preview_layout = QVBoxLayout(preview_card)
        
        self.video_label = QLabel("Starting camera system...")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_layout.addWidget(self.video_label)
        
        # Dynamic Sliders & Action Toggle Buttons Panel
        controls_card = QWidget()
        controls_card.setFixedWidth(320)
        controls_card.setStyleSheet("background-color: #121214; border-radius: 12px; border: 1px solid #2d2d34; padding: 15px;")
        controls_layout = QVBoxLayout(controls_card)
        
        self.toggle_tracking_btn = QPushButton("▶ Start Tracking Engine")
        self.toggle_tracking_btn.setStyleSheet("""
            QPushButton {
                background-color: #007ACC; color: white; border-radius: 6px; 
                padding: 12px; font-weight: bold; font-size: 14px; text-align: center;
            }
            QPushButton:hover { background-color: #0098FF; }
        """)
        self.toggle_tracking_btn.clicked.connect(self.toggle_tracking)
        controls_layout.addWidget(self.toggle_tracking_btn)
        controls_layout.addSpacing(20)
        
        sensitivity_lbl = QLabel("<b>Cursor Sensitivity</b>")
        controls_layout.addWidget(sensitivity_lbl)
        
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(10, 30)  
        self.slider.setValue(17)
        self.slider.valueChanged.connect(self.change_sensitivity)
        controls_layout.addWidget(self.slider)
        
        controls_layout.addStretch()

        layout.addWidget(preview_card, stretch=2)
        layout.addWidget(controls_card, stretch=1)
        return dash_page

    def toggle_tracking(self):
        if not self.engine.tracking_enabled:
            self.engine.tracking_enabled = True
            self.toggle_tracking_btn.setText("⏸ Pause Tracking Engine")
            self.toggle_tracking_btn.setStyleSheet("background-color: #D32F2F; color: white; border-radius: 6px; padding: 12px; font-weight: bold; text-align: center;")
        else:
            self.engine.tracking_enabled = False
            self.toggle_tracking_btn.setText("▶ Start Tracking Engine")
            self.toggle_tracking_btn.setStyleSheet("background-color: #007ACC; color: white; border-radius: 6px; padding: 12px; font-weight: bold; text-align: center;")

    def change_sensitivity(self, value):
        self.engine.sensitivity = value / 10.0

    def update_image(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        p = convert_to_Qt_format.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)
        self.video_label.setPixmap(QPixmap.fromImage(p))

    def update_status(self, hand_detected, fps):
        self.fps_lbl.setText(f"FPS: {fps}")
        if hand_detected:
            self.status_lbl.setText("Status: Hand Tracked")
            self.status_lbl.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 14px;")
        else:
            self.status_lbl.setText("Status: No Hand Detected")
            self.status_lbl.setStyleSheet("color: #F44336; font-size: 14px;")

    def apply_premium_stylesheet(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #1a1a1e; }
            QWidget#Sidebar { background-color: #111113; border-right: 1px solid #2d2d34; }
            QPushButton { 
                background-color: transparent; color: #b3b3b3; 
                border: none; padding: 12px 15px; text-align: left; font-size: 14px;
            }
            QPushButton:hover { background-color: #222226; color: #ffffff; border-radius: 6px; }
            QLabel { color: #ffffff; font-family: 'Segoe UI', sans-serif; border: none; background: transparent;}
        """)

    def closeEvent(self, event):
        self.engine.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ZeroTouchApp()
    window.show()
    sys.exit(app.exec())