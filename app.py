import sys
import cv2
import mediapipe as mp
import time
import pyautogui
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, 
                             QVBoxLayout, QLabel, QPushButton, QSlider, QStackedWidget)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap  # Fixed: Imported QPixmap and QImage here

# Disable PyAutoGUI fail-safe pause to make moving the pointer smoother
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
        
        # Initialize MediaPipe Hand Tracker using the fixed library version
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Settings controlled by the UI
        self.tracking_enabled = False
        self.sensitivity = 1.5
        self.screen_width, self.screen_height = pyautogui.size()

    def run(self):
        cap = cv2.VideoCapture(1)
        prev_frame_time = 0

        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                continue

            # Mirror the frame horizontally so your movements match what you see
            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape
            
            # MediaPipe requires RGB images, but OpenCV reads in BGR
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            hand_detected = False

            # Calculate Frames Per Second (FPS)
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0
            prev_frame_time = new_frame_time

            if results.multi_hand_landmarks:
                hand_detected = True
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw skeletal lines over your hand in the preview
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    # Track Index Finger Tip (Landmark #8)
                    index_tip = hand_landmarks.landmark[8]
                    
                    if self.tracking_enabled:
                        # Convert hand normalized coordinates (0.0 to 1.0) to screen pixel targets
                        target_x = int(index_tip.x * self.screen_width * self.sensitivity)
                        target_y = int(index_tip.y * self.screen_height * self.sensitivity)
                        
                        # Prevent pointer from leaving monitor boundaries
                        target_x = max(0, min(target_x, self.screen_width - 1))
                        target_y = max(0, min(target_y, self.screen_height - 1))
                        
                        # Command the OS mouse to move smoothly
                        pyautogui.moveTo(target_x, target_y, duration=0.01)

            # Send the updated frame and data up to the dashboard interface
            self.change_pixmap_signal.emit(frame)
            self.status_signal.emit(hand_detected, round(fps, 1))

        cap.release()

    def stop(self):
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
        
        # Build and link the background tracking thread
        self.engine = HandTrackingEngine()
        self.engine.change_pixmap_signal.connect(self.update_image)
        self.engine.status_signal.connect(self.update_status)
        
        self.init_ui()
        self.engine.start()

    def init_ui(self):
        # Core master layout layout splits Left (Sidebar) and Right (Main Window Content)
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Left Sidebar View
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

        # Right View Pane (Header + Dynamic Stack)
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(25, 25, 25, 25)

        # Dashboard Top Action / Metric Bar
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

        # Create our workspace stack where different screens can swap
        self.content_stack = QStackedWidget()
        self.content_stack.addWidget(self.create_dashboard_view())
        right_layout.addWidget(self.content_stack)

        # Tie everything together to build the framework
        main_layout.addWidget(self.sidebar)
        main_layout.addWidget(right_container)
        self.setCentralWidget(main_widget)
        
        self.apply_premium_stylesheet()

    def create_dashboard_view(self):
        """Generates the Main Layout Grid holding the Video Feed and Settings Controls"""
        dash_page = QWidget()
        layout = QHBoxLayout(dash_page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(20)
        
        # Left Dashboard Side: AI Live Feed Preview Card
        preview_card = QWidget()
        preview_card.setStyleSheet("background-color: #121214; border-radius: 12px; border: 1px solid #2d2d34;")
        preview_layout = QVBoxLayout(preview_card)
        
        self.video_label = QLabel("Starting camera system...")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_layout.addWidget(self.video_label)
        
        # Right Dashboard Side: System Adjustments / Switches Panel
        controls_card = QWidget()
        controls_card.setFixedWidth(320)
        controls_card.setStyleSheet("background-color: #121214; border-radius: 12px; border: 1px solid #2d2d34; padding: 15px;")
        controls_layout = QVBoxLayout(controls_card)
        
        # Quick Control Master Action Button
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
        
        # Configuration Modifier Sliders
        sensitivity_lbl = QLabel("<b>Cursor Sensitivity</b>")
        controls_layout.addWidget(sensitivity_lbl)
        
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(10, 30)  # Maps to 1.0x - 3.0x sensitivity
        self.slider.setValue(15)
        self.slider.valueChanged.connect(self.change_sensitivity)
        controls_layout.addWidget(self.slider)
        
        controls_layout.addStretch()

        layout.addWidget(preview_card, stretch=2)
        layout.addWidget(controls_card, stretch=1)
        return dash_page

    def toggle_tracking(self):
        """Turn actual system mouse movement tracking on or off"""
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
        """Accepts raw tracking output video feeds, maps it to Qt frames, and prints it"""
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
        """Infuses global CSS rules to create an ultra-modern dark UI palette"""
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