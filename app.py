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

# Dynamic import helper for cross-platform hardware brightness control
try:
    import screen_brightness_control as sbc
    HAS_SBC = True
except ImportError:
    HAS_SBC = False

# SYSTEM OVERRIDES FOR PERFORMANCE
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
        
        # Initialize MediaPipe Hand Tracker for 2 hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,          # Track both hands simultaneously
            model_complexity=0,       # Fast processing mode
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        
        # Mapped UI Settings
        self.tracking_enabled = False
        self.sensitivity = 1.7        
        self.screen_width, self.screen_height = pyautogui.size()

        # Dynamic Tracking Filter Position Trackers (Right Hand)
        self.prev_x, self.prev_y = 0, 0
        
        # Gesture Hold State Controls
        self.left_button_held = False
        self.is_right_clicking = False
        
        # Standardized Gesturing Limits
        self.click_threshold = 0.04  

        # Angle Tracking Variables for Right Hand Rotation (Volume & Brightness Modes)
        self.prev_angle = None
        self.angle_accumulator = 0.0
        self.rotation_threshold = 5.0  

        # Scroll Throttle Timer (prevents runaway scrolling)
        self.last_scroll_time = 0

        # State Tracker for Copy/Paste Trigger
        self.right_was_fist = False  

    def run(self):
        cap = cv2.VideoCapture(1)
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
            left_freeze = False
            left_volume_mode = False
            left_brightness_mode = False
            left_copy_mode = False
            left_paste_mode = False
            
            # Scroll Mode variables
            left_scroll_speed_active = False
            scroll_multiplier = 1.0  # Speed based on vertical height
            
            right_hand_landmarks = None

            # Dynamic Loop FPS Reader
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0
            prev_frame_time = new_frame_time

            if results.multi_hand_landmarks and results.multi_handedness:
                hand_detected = True
                
                # FIRST PASS: Find the Left Hand state
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    handedness = results.multi_handedness[idx].classification[0].label
                    
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    if handedness == "Left":
                        wrist = hand_landmarks.landmark[0]
                        
                        # Get finger curl statuses
                        tips = [8, 12, 16, 20]
                        mcps = [5, 9, 13, 17]
                        fingers_curled = []
                        
                        for tip_idx, mcp_idx in zip(tips, mcps):
                            tip = hand_landmarks.landmark[tip_idx]
                            mcp = hand_landmarks.landmark[mcp_idx]
                            
                            dist_to_wrist = math.hypot(tip.x - wrist.x, tip.y - wrist.y)
                            mcp_to_wrist = math.hypot(mcp.x - wrist.x, mcp.y - wrist.y)
                            fingers_curled.append(dist_to_wrist < mcp_to_wrist)
                        
                        # GESTURE 1: Left index pointing up only (VOLUME MODE)
                        if not fingers_curled[0] and fingers_curled[1] and fingers_curled[2] and fingers_curled[3]:
                            left_volume_mode = True
                            
                        # GESTURE 2: Left index and middle pointing up (BRIGHTNESS MODE)
                        elif not fingers_curled[0] and not fingers_curled[1] and fingers_curled[2] and fingers_curled[3]:
                            left_brightness_mode = True

                        # GESTURE 3: COPY MODE (Fist and raise ONLY pinky finger)
                        elif fingers_curled[0] and fingers_curled[1] and fingers_curled[2] and not fingers_curled[3]:
                            left_copy_mode = True
                            left_freeze = True

                        # GESTURE 4: PASTE MODE (Rock sign: Index + Pinky up, Middle + Ring curled)
                        elif not fingers_curled[0] and fingers_curled[1] and fingers_curled[2] and not fingers_curled[3]:
                            left_paste_mode = True
                            left_freeze = True
                            
                        # GESTURE 5: Open Palm (All fingers extended -> SCROLL SPEED CONTROLLER)
                        elif sum(fingers_curled) == 0:
                            left_scroll_speed_active = True
                            # Calculate speed based on how high the hand is in the frame (normalized 0.0 to 1.0)
                            # landmark coordinates are 0 at top, 1 at bottom. So (1.0 - y) makes high hand = high speed.
                            hand_height = 1.0 - wrist.y
                            # Map vertical range roughly from a baseline to max sensitivity
                            scroll_multiplier = max(0.5, min(5.0, hand_height * 6.0))
                            
                        # GESTURE 6: Full Box / Fist (3 or more fingers curled)
                        elif sum(fingers_curled) >= 3:
                            left_freeze = True
                            
                    elif handedness == "Right":
                        right_hand_landmarks = hand_landmarks

                # SECOND PASS: Process Right Hand actions
                if right_hand_landmarks and self.tracking_enabled:
                    thumb_tip = right_hand_landmarks.landmark[4]
                    thumb_mcp = right_hand_landmarks.landmark[2]
                    index_tip = right_hand_landmarks.landmark[8]
                    middle_tip = right_hand_landmarks.landmark[12]
                    wrist_r = right_hand_landmarks.landmark[0]

                    # Analyze finger curl states for Right Hand
                    r_tips = [8, 12, 16, 20]
                    r_mcps = [5, 9, 13, 17]
                    r_fingers_curled = []
                    for tip_idx, mcp_idx in zip(r_tips, r_mcps):
                        t = right_hand_landmarks.landmark[tip_idx]
                        m = right_hand_landmarks.landmark[mcp_idx]
                        r_fingers_curled.append(math.hypot(t.x - wrist_r.x, t.y - wrist_r.y) < math.hypot(m.x - wrist_r.x, m.y - wrist_r.y))
                    
                    is_right_thumbs_up = sum(r_fingers_curled) >= 3
                    right_is_fist = sum(r_fingers_curled) >= 3
                    right_is_open = sum(r_fingers_curled) == 0

                    # MODE A: COPY OR PASTE COMMAND EXECUTION
                    if left_copy_mode or left_paste_mode:
                        if left_copy_mode:
                            cv2.putText(frame, "COPY MODE (MOUSE LOCKED)", (10, 30), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                        else:
                            cv2.putText(frame, "PASTE MODE (MOUSE LOCKED)", (10, 30), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                        # Arming step: detect if right hand becomes a fist
                        if right_is_fist:
                            self.right_was_fist = True
                            cv2.putText(frame, "TRIGGER ARMED (FIST DETECTED)", (10, 60), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        
                        # Fire step: detect transition from fist to open palm
                        elif right_is_open and self.right_was_fist:
                            if left_copy_mode:
                                pyautogui.hotkey('ctrl', 'c')
                                cv2.putText(frame, "!!! COPIED !!!", (10, 90), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
                            elif left_paste_mode:
                                pyautogui.hotkey('ctrl', 'v')
                                cv2.putText(frame, "!!! PASTED !!!", (10, 90), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
                            
                            self.right_was_fist = False  # Reset trigger state
                        else:
                            cv2.putText(frame, "Right Hand: Fist then Open to trigger", (10, 60), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                    # MODE B: SCROLL ENGINE ACTIVE (Left hand open palm + Right hand thumbs up)
                    elif left_scroll_speed_active and is_right_thumbs_up:
                        self.right_was_fist = False # Clear active copy/paste latch states
                        # Determine direction of the thumb relative to its base knuckle
                        dx = thumb_tip.x - thumb_mcp.x
                        dy = thumb_tip.y - thumb_mcp.y
                        
                        # Calculate primary axis of the thumb pointing direction
                        direction = "NONE"
                        if abs(dx) > abs(dy):
                            direction = "RIGHT" if dx > 0.05 else ("LEFT" if dx < -0.05 else "NONE")
                        else:
                            direction = "DOWN" if dy > 0.05 else ("UP" if dy < -0.05 else "NONE")
                        
                        cv2.putText(frame, f"SCROLL ACTIVE: {direction} (Speed: {round(scroll_multiplier, 1)}x)", 
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Apply scrolling based on direction and left-hand speed multiplier
                        current_time = time.time()
                        # Dynamic delay: faster scroll speed = less waiting between scroll ticks
                        scroll_interval = max(0.02, 0.15 / scroll_multiplier)
                        
                        if current_time - self.last_scroll_time > scroll_interval:
                            scroll_amount = int(1 * scroll_multiplier)
                            if scroll_amount < 1:
                                scroll_amount = 1
                                
                            if direction == "UP":
                                pyautogui.scroll(scroll_amount * 10)
                            elif direction == "DOWN":
                                pyautogui.scroll(-scroll_amount * 10)
                            elif direction == "LEFT":
                                pyautogui.hscroll(-scroll_amount * 10)
                            elif direction == "RIGHT":
                                pyautogui.hscroll(scroll_amount * 10)
                                
                            self.last_scroll_time = current_time

                    # MODE C: VOLUME KNOB OR BRIGHTNESS KNOB ACTIVE
                    elif (left_volume_mode or left_brightness_mode) and not left_scroll_speed_active:
                        self.right_was_fist = False # Clear active copy/paste latch states
                        if left_volume_mode:
                            cv2.putText(frame, "VOLUME CONTROL ACTIVE", (10, 30), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        else:
                            cv2.putText(frame, "BRIGHTNESS CONTROL ACTIVE", (10, 30), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                        
                        # Compute angle of rotation
                        dx_rot = index_tip.x - wrist_r.x
                        dy_rot = index_tip.y - wrist_r.y
                        current_angle = math.degrees(math.atan2(dy_rot, dx_rot))
                        
                        if self.prev_angle is not None:
                            delta_angle = current_angle - self.prev_angle
                            if delta_angle > 180:
                                delta_angle -= 360
                            elif delta_angle < -180:
                                delta_angle += 360
                                
                            self.angle_accumulator += delta_angle
                            
                            if abs(self.angle_accumulator) >= self.rotation_threshold:
                                steps = int(abs(self.angle_accumulator) // self.rotation_threshold)
                                is_clockwise = self.angle_accumulator > 0
                                
                                if left_volume_mode:
                                    key_to_press = "volumeup" if is_clockwise else "volumedown"
                                    for _ in range(steps):
                                        pyautogui.press(key_to_press)
                                
                                elif left_brightness_mode:
                                    if HAS_SBC:
                                        try:
                                            curr_bright = sbc.get_brightness(display=0)[0]
                                            change = 4 * steps if is_clockwise else -4 * steps
                                            new_bright = max(0, min(100, curr_bright + change))
                                            sbc.set_brightness(new_bright, display=0)
                                        except Exception:
                                            key_to_press = "displaybrightnessup" if is_clockwise else "displaybrightnessdown"
                                            for _ in range(steps):
                                                pyautogui.press(key_to_press)
                                    else:
                                        key_to_press = "displaybrightnessup" if is_clockwise else "displaybrightnessdown"
                                        for _ in range(steps):
                                            pyautogui.press(key_to_press)
                                            
                                self.angle_accumulator = 0.0
                        
                        self.prev_angle = current_angle
                    
                    # MODE D: NORMAL CURSOR MOVEMENT
                    else:
                        self.prev_angle = None
                        self.angle_accumulator = 0.0
                        self.right_was_fist = False # Clear active copy/paste latch states

                        if not left_freeze:
                            raw_x = int(index_tip.x * self.screen_width * self.sensitivity)
                            raw_y = int(index_tip.y * self.screen_height * self.sensitivity)
                            
                            if self.prev_x == 0 and self.prev_y == 0:
                                smooth_x, smooth_y = raw_x, raw_y
                            else:
                                movement_distance = math.hypot(raw_x - self.prev_x, raw_y - self.prev_y)
                                
                                if movement_distance > 30:
                                    current_smoothing = 0.60
                                else:
                                    current_smoothing = 0.15
                                    
                                smooth_x = int(self.prev_x + current_smoothing * (raw_x - self.prev_x))
                                smooth_y = int(self.prev_y + current_smoothing * (raw_y - self.prev_y))
                            
                            smooth_x = max(0, min(smooth_x, self.screen_width - 1))
                            smooth_y = max(0, min(smooth_y, self.screen_height - 1))
                            
                            pyautogui.moveTo(smooth_x, smooth_y)
                            self.prev_x, self.prev_y = smooth_x, smooth_y
                        else:
                            cv2.putText(frame, "TRACKING LOCKED (Left Hand Command)", (10, 30), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

                        # Click processing (Disabled during control/scroll modes)
                        left_dist = math.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)
                        if left_dist < self.click_threshold:
                            if not self.left_button_held:
                                pyautogui.mouseDown(button='left')
                                self.left_button_held = True
                        else:
                            if self.left_button_held:
                                pyautogui.mouseUp(button='left')
                                self.left_button_held = False

                        right_dist = math.hypot(middle_tip.x - thumb_tip.x, middle_tip.y - thumb_tip.y)
                        if right_dist < self.click_threshold:
                            if not self.is_right_clicking:
                                pyautogui.click(button='right')
                                self.is_right_clicking = True  
                        else:
                            self.is_right_clicking = False  
            else:
                self.prev_angle = None
                self.angle_accumulator = 0.0
                self.right_was_fist = False
                if self.left_button_held:
                    pyautogui.mouseUp(button='left')
                    self.left_button_held = False

            self.change_pixmap_signal.emit(frame)
            self.status_signal.emit(hand_detected, round(fps, 1))

        cap.release()

    def stop(self):
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

        # Right Workspace Layout Panel
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(25, 25, 25, 25)

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
        
        preview_card = QWidget()
        preview_card.setStyleSheet("background-color: #121214; border-radius: 12px; border: 1px solid #2d2d34;")
        preview_layout = QVBoxLayout(preview_card)
        
        self.video_label = QLabel("Starting camera system...")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_layout.addWidget(self.video_label)
        
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