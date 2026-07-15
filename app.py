import sys
import cv2
import mediapipe as mp
import time
import pyautogui
import math
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, 
                             QVBoxLayout, QLabel, QPushButton, QSlider, QStackedWidget,
                             QScrollArea, QGridLayout, QFrame)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QImage, QPixmap, QColor, QFont

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
# HELPER: Calculate Angle of Three Points
# ==========================================
def calculate_angle(p1, p2, p3):
    """
    Calculates the 2D angle (in degrees) at point p2 (the vertex)
    formed by vectors p2->p1 and p2->p3.
    """
    a = (p1.x - p2.x, p1.y - p2.y)
    b = (p3.x - p2.x, p3.y - p2.y)
    
    dot_product = a[0] * b[0] + a[1] * b[1]
    norm_a = math.hypot(a[0], a[1])
    norm_b = math.hypot(b[0], b[1])
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    cos_angle = dot_product / (norm_a * norm_b)
    # Clamp to avoid floating point errors out of range [-1.0, 1.0]
    cos_angle = max(-1.0, min(1.0, cos_angle))
    
    angle = math.degrees(math.acos(cos_angle))
    return angle


# ==========================================
# 1. THE AI ENGINE THREAD (Background Logic)
# ==========================================
class HandTrackingEngine(QThread):
    change_pixmap_signal = pyqtSignal(object)  
    status_signal = pyqtSignal(bool, float)     
    performance_signal = pyqtSignal(float, float) # (Confidence, Latency)

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

        # State Tracker for Copy/Paste/Screenshot Trigger
        self.right_was_fist = False 

    def run(self):
        # We try opening camera 0 first. Modify index if needed.
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
            
            start_inference = time.time()
            results = self.hands.process(rgb_frame)
            latency = (time.time() - start_inference) * 1000  # in ms

            hand_detected = False
            left_freeze = False
            left_volume_mode = False
            left_brightness_mode = False
            left_copy_mode = False
            left_paste_mode = False
            left_screenshot_mode = False
            
            # Scroll Mode variables
            left_scroll_speed_active = False
            scroll_multiplier = 1.0  # Speed based on vertical height
            
            right_hand_landmarks = None

            # Dynamic Loop FPS Reader
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0
            prev_frame_time = new_frame_time

            confidence = 0.0

            if results.multi_hand_landmarks and results.multi_handedness:
                hand_detected = True
                confidence = 0.85 # Approximation of ML model confidence for UI
                
                # FIRST PASS: Find the Left Hand state
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    handedness = results.multi_handedness[idx].classification[0].label
                    
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    if handedness == "Left":
                        wrist = hand_landmarks.landmark[0]
                        
                        # Get finger curl statuses (Index, Middle, Ring, Pinky)
                        tips = [8, 12, 16, 20]
                        mcps = [5, 9, 13, 17]
                        fingers_curled = []
                        
                        for tip_idx, mcp_idx in zip(tips, mcps):
                            tip = hand_landmarks.landmark[tip_idx]
                            mcp = hand_landmarks.landmark[mcp_idx]
                            
                            dist_to_wrist = math.hypot(tip.x - wrist.x, tip.y - wrist.y)
                            mcp_to_wrist = math.hypot(mcp.x - wrist.x, mcp.y - wrist.y)
                            fingers_curled.append(dist_to_wrist < mcp_to_wrist)

                        # THUMB ANGLE DETECTION (Points 2, 3, 4)
                        thumb_mcp_l = hand_landmarks.landmark[2]
                        thumb_ip_l = hand_landmarks.landmark[3]
                        thumb_tip_l = hand_landmarks.landmark[4]
                        
                        thumb_angle = calculate_angle(thumb_mcp_l, thumb_ip_l, thumb_tip_l)
                        # An angle > 145 degrees represents a straight/extended thumb
                        thumb_extended_l = thumb_angle > 155.0
                        
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

                        # GESTURE 5: SCREENSHOT MODE (Fist with ONLY the thumb out/extended)
                        elif fingers_curled[0] and fingers_curled[1] and fingers_curled[2] and fingers_curled[3] and thumb_extended_l:
                            left_screenshot_mode = True
                            left_freeze = True
                            
                        # GESTURE 6: Open Palm (All fingers extended -> SCROLL SPEED CONTROLLER)
                        elif sum(fingers_curled) == 0:
                            left_scroll_speed_active = True
                            # Calculate speed based on how high the hand is in the frame
                            hand_height = 1.0 - wrist.y
                            # Map vertical range roughly from a baseline to max sensitivity
                            scroll_multiplier = max(0.5, min(5.0, hand_height * 6.0))
                            
                        # GESTURE 7: Full Box / Fist (3 or more fingers curled)
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

                    # MODE A: COPY / PASTE / SCREENSHOT COMMAND EXECUTION
                    if left_copy_mode or left_paste_mode or left_screenshot_mode:
                        if left_copy_mode:
                            cv2.putText(frame, "COPY MODE (MOUSE LOCKED)", (10, 30), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 240, 255), 2)
                        elif left_paste_mode:
                            cv2.putText(frame, "PASTE MODE (MOUSE LOCKED)", (10, 30), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 240, 255), 2)
                        elif left_screenshot_mode:
                            cv2.putText(frame, "SCREENSHOT MODE (MOUSE LOCKED)", (10, 30), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 240, 255), 2)

                        # Arming step: detect if right hand becomes a fist
                        if right_is_fist:
                            self.right_was_fist = True
                            cv2.putText(frame, "TRIGGER ARMED (FIST DETECTED)", (10, 60), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        
                        # Fire step: detect transition from fist to open palm (release)
                        elif right_is_open and self.right_was_fist:
                            if left_copy_mode:
                                pyautogui.hotkey('ctrl', 'c')
                                cv2.putText(frame, "!!! COPIED !!!", (10, 90), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
                            elif left_paste_mode:
                                pyautogui.hotkey('ctrl', 'v')
                                cv2.putText(frame, "!!! PASTED !!!", (10, 90), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
                            elif left_screenshot_mode:
                                file_path = f"screenshot_{int(time.time())}.png"
                                pyautogui.screenshot(file_path)
                                cv2.putText(frame, "!!! SCREENSHOT TAKEN !!!", (10, 90), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
                            
                            self.right_was_fist = False  # Reset trigger state
                        else:
                            cv2.putText(frame, "Right Hand: Fist then Open to trigger", (10, 60), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                    # MODE B: SCROLL ENGINE ACTIVE (Left hand open palm + Right hand thumbs up)
                    elif left_scroll_speed_active and is_right_thumbs_up:
                        self.right_was_fist = False # Clear active latch states
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
                        self.right_was_fist = False # Clear active latch states
                        if left_volume_mode:
                            cv2.putText(frame, "VOLUME CONTROL ACTIVE", (10, 30), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        else:
                            cv2.putText(frame, "BRIGHTNESS CONTROL ACTIVE", (10, 30), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                        
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
                        self.right_was_fist = False # Clear active latch states

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
                            cv2.putText(frame, "TRACKING LOCKED (Left Hand)", (10, 30), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

                        # Click processing (Disabled during control/scroll/freeze modes)
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
            self.performance_signal.emit(confidence, latency)

        cap.release()

    def stop(self):
        if self.left_button_held:
            pyautogui.mouseUp(button='left')
        self._run_flag = False
        self.wait()


# ==========================================
# 2. THE UPGRADED HIGH-FIDELITY USER INTERFACE
# ==========================================
class ZeroTouchApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AirMouse AI - Touchless Desktop Control")
        self.setMinimumSize(1280, 820)
        
        # Engine thread config
        self.engine = HandTrackingEngine()
        self.engine.change_pixmap_signal.connect(self.update_image)
        self.engine.status_signal.connect(self.update_status)
        self.engine.performance_signal.connect(self.update_performance)

        self.init_ui()
        self.engine.start()

    def init_ui(self):
        # Master Base Layout
        main_widget = QWidget()
        main_widget.setObjectName("BaseContainer")
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ----------------------------------------
        # Side Collapsible Navigation Panel
        # ----------------------------------------
        self.sidebar = QWidget()
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setFixedWidth(240)
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(10)

        # Logo Header
        logo_container = QWidget()
        logo_container.setObjectName("LogoContainer")
        logo_container.setFixedHeight(75)
        logo_layout = QHBoxLayout(logo_container)
        logo_layout.setContentsMargins(20, 0, 20, 0)
        
        logo_label = QLabel("⚡ AirMouse AI")
        logo_label.setObjectName("AppName")
        logo_layout.addWidget(logo_label)
        sidebar_layout.addWidget(logo_container)

        # Nav Elements
        self.btn_dash = QPushButton("  📊   Dashboard")
        self.btn_gestures = QPushButton("  🖐️   Gestures")
        self.btn_settings = QPushButton("  ⚙️   Settings")

        # Style Navigation Buttons
        for btn in [self.btn_dash, self.btn_gestures, self.btn_settings]:
            btn.setObjectName("NavBtn")
            btn.setFixedHeight(50)
            sidebar_layout.addWidget(btn)

        self.btn_dash.setProperty("active", True) # Set initial active view marker
        
        self.btn_dash.clicked.connect(lambda: self.switch_view(0, self.btn_dash))
        self.btn_gestures.clicked.connect(lambda: self.switch_view(1, self.btn_gestures))
        self.btn_settings.clicked.connect(lambda: self.switch_view(2, self.btn_settings))

        sidebar_layout.addStretch()

        # Footer branding
        foot_br = QLabel("v1.2.0-PRO Build")
        foot_br.setObjectName("SidebarFooter")
        sidebar_layout.addWidget(foot_br)

        # ----------------------------------------
        # Right workspace Container & Header
        # ----------------------------------------
        right_container = QWidget()
        right_container.setObjectName("RightWorkspace")
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(25, 10, 25, 20)
        right_layout.setSpacing(15)

        # Global Sticky App Header bar
        top_header_bar = QWidget()
        top_header_bar.setObjectName("TopHeader")
        top_header_bar.setFixedHeight(65)
        top_layout = QHBoxLayout(top_header_bar)
        top_layout.setContentsMargins(0, 0, 0, 0)

        self.header_title = QLabel("Dashboard")
        self.header_title.setObjectName("HeaderTitle")
        top_layout.addWidget(self.header_title)
        top_layout.addStretch()

        # System telemetry displays
        self.tracker_status_chip = QLabel("OFFLINE")
        self.tracker_status_chip.setObjectName("StatusChip")
        self.tracker_status_chip.setFixedSize(140, 32)
        self.tracker_status_chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.system_fps_badge = QLabel("FPS: --")
        self.system_fps_badge.setObjectName("FPSBadge")
        self.system_fps_badge.setFixedSize(85, 32)
        self.system_fps_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)

        top_layout.addWidget(self.tracker_status_chip)
        top_layout.addWidget(self.system_fps_badge)
        right_layout.addWidget(top_header_bar)

        # Dynamic screen stack pages
        self.content_stack = QStackedWidget()
        self.content_stack.setObjectName("StackArea")
        
        # Build views
        self.content_stack.addWidget(self.create_dashboard_view())
        self.content_stack.addWidget(self.create_gestures_view())
        self.content_stack.addWidget(self.create_settings_view())

        right_layout.addWidget(self.content_stack)

        # Assembly
        main_layout.addWidget(self.sidebar)
        main_layout.addWidget(right_container)
        self.setCentralWidget(main_widget)
        
        # Apply premium Logitech G HUB styled stylesheet
        self.apply_premium_stylesheet()

    # ----------------------------------------
    # SCREEN VIEW 1: Premium Dashboard
    # ----------------------------------------
    def create_dashboard_view(self):
        dash_page = QWidget()
        main_grid = QGridLayout(dash_page)
        main_grid.setContentsMargins(0, 0, 0, 0)
        main_grid.setSpacing(20)

        # Left Column: Stream Panel
        stream_card = QFrame()
        stream_card.setObjectName("PanelCard")
        stream_layout = QVBoxLayout(stream_card)
        stream_layout.setContentsMargins(15, 15, 15, 15)

        stream_header = QHBoxLayout()
        stream_title = QLabel("AI Engine Live Telemetry")
        stream_title.setObjectName("CardTitle")
        stream_header.addWidget(stream_title)
        stream_header.addStretch()
        stream_layout.addLayout(stream_header)

        self.video_label = QLabel("Initializing camera pipeline...")
        self.video_label.setObjectName("VideoFeed")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        stream_layout.addWidget(self.video_label, stretch=1)

        # Quick Control Panel
        quick_action_bar = QHBoxLayout()
        self.btn_toggle_tracker = QPushButton("▶ Start Tracking")
        self.btn_toggle_tracker.setObjectName("TrackingButton")
        self.btn_toggle_tracker.clicked.connect(self.toggle_tracking)

        self.btn_calibrate = QPushButton("⚡ Calibrate")
        self.btn_calibrate.setObjectName("ControlBtn")
        
        self.btn_reset = QPushButton("↺ Reset")
        self.btn_reset.setObjectName("ControlBtn")

        quick_action_bar.addWidget(self.btn_toggle_tracker, stretch=2)
        quick_action_bar.addWidget(self.btn_calibrate, stretch=1)
        quick_action_bar.addWidget(self.btn_reset, stretch=1)
        stream_layout.addLayout(quick_action_bar)

        # Right Column: Custom Settings & Metrics
        control_sidebar_card = QFrame()
        control_sidebar_card.setObjectName("PanelCard")
        control_sidebar_card.setFixedWidth(340)
        cs_layout = QVBoxLayout(control_sidebar_card)
        cs_layout.setContentsMargins(20, 20, 20, 20)
        cs_layout.setSpacing(15)

        cs_title = QLabel("Device Parameters")
        cs_title.setObjectName("CardTitle")
        cs_layout.addWidget(cs_title)

        # Divider
        div = QFrame()
        div.setFrameShape(QFrame.Shape.HLine)
        div.setObjectName("PanelDivider")
        cs_layout.addWidget(div)

        # Slider 1: Cursor Sensitivity
        sens_head = QHBoxLayout()
        sens_head.addWidget(QLabel("Cursor Sensitivity"))
        self.lbl_sens_val = QLabel("1.7x")
        self.lbl_sens_val.setObjectName("AccentValue")
        sens_head.addStretch()
        sens_head.addWidget(self.lbl_sens_val)
        cs_layout.addLayout(sens_head)

        self.sens_slider = QSlider(Qt.Orientation.Horizontal)
        self.sens_slider.setObjectName("FancySlider")
        self.sens_slider.setRange(10, 30)  
        self.sens_slider.setValue(17)
        self.sens_slider.valueChanged.connect(self.change_sensitivity)
        cs_layout.addWidget(self.sens_slider)

        # Slider 2: Cursor Smoothing
        smooth_head = QHBoxLayout()
        smooth_head.addWidget(QLabel("Cursor Jitter Smooth Filter"))
        self.lbl_smooth_val = QLabel("0.15s")
        self.lbl_smooth_val.setObjectName("AccentValue")
        smooth_head.addStretch()
        smooth_head.addWidget(self.lbl_smooth_val)
        cs_layout.addLayout(smooth_head)

        self.smooth_slider = QSlider(Qt.Orientation.Horizontal)
        self.smooth_slider.setObjectName("FancySlider")
        self.smooth_slider.setRange(5, 50)
        self.smooth_slider.setValue(15)
        cs_layout.addWidget(self.smooth_slider)

        cs_layout.addSpacing(10)
        cs_layout.addWidget(QLabel("Diagnostic Performance Analytics"))

        # Performance Grid indicators
        self.metric_latency = self.create_metric_widget("Inference Time", "0 ms")
        self.metric_conf = self.create_metric_widget("Confidence Index", "0.0%")
        self.metric_res = self.create_metric_widget("Input Frame", "640 x 480")

        cs_layout.addWidget(self.metric_latency)
        cs_layout.addWidget(self.metric_conf)
        cs_layout.addWidget(self.metric_res)

        cs_layout.addStretch()

        # Grid Assembly
        main_grid.addWidget(stream_card, 0, 0, 1, 1)
        main_grid.addWidget(control_sidebar_card, 0, 1, 1, 1)
        return dash_page

    # Helper function to generate clean indicators
    def create_metric_widget(self, name, val):
        container = QFrame()
        container.setObjectName("MetricFrame")
        lyt = QHBoxLayout(container)
        lyt.setContentsMargins(15, 10, 15, 10)
        
        lbl_name = QLabel(name)
        lbl_name.setObjectName("MetricLabel")
        lbl_val = QLabel(val)
        lbl_val.setObjectName("MetricValue")

        lyt.addWidget(lbl_name)
        lyt.addStretch()
        lyt.addWidget(lbl_val)
        return container

    # ----------------------------------------
    # SCREEN VIEW 2: Gesture Control Panel Directory
    # ----------------------------------------
    def create_gestures_view(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(15)

        sub_head = QLabel("Bind hand landmarks movements to execution sequences")
        sub_head.setObjectName("PageSubHeader")
        layout.addWidget(sub_head)

        scroll = QScrollArea()
        scroll.setObjectName("DashboardScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        scroll_widget = QWidget()
        scroll_widget.setObjectName("TransparentBase")
        grid = QGridLayout(scroll_widget)
        grid.setSpacing(15)
        grid.setContentsMargins(0, 0, 10, 0)

        # Active hand gesture inventory map
        gestures_data = [
            ("Move Cursor", "Pointer Action", "Right Hand movement tracking (Neutral index)", "🔗 Dynamic Mapping"),
            ("Left Click", "Primary Click", "Index Tip & Thumb Tip pinch (d < 0.04)", "🖱️ Press Toggle"),
            ("Right Click", "Secondary Click", "Middle Tip & Thumb Tip pinch (d < 0.04)", "🖱️ Double Click bind"),
            ("Scroll Mode", "Document Scroll", "Left Hand Open Palm + Right Hand Thumbs up pointing direction", "🔄 Throttle Enabled"),
            ("Volume Control", "Audio Adjuster", "Left Index up + Rotate Right index clockwise/counter-clockwise", "🔊 Sys Bind"),
            ("Brightness Control", "Hardware Control", "Left Index+Middle up + Rotate Right index relative to wrist", "🔆 SBC Bridge"),
            ("Copy Macro", "Keyboard Event", "Left Pinky up + Right Hand transition: Fist to Open Palm", "📋 CTRL + C Bind"),
            ("Paste Macro", "Keyboard Event", "Left Rock Sign + Right Hand transition: Fist to Open Palm", "📋 CTRL + V Bind"),
            ("Screenshot Macro", "OS Action Sequence", "Left Thumb extended up + Right Hand Fist to Open Palm", "📸 Auto Save Local PNG")
        ]

        row = 0
        col = 0
        for name, category, logic, bind in gestures_data:
            card = self.create_gesture_card(name, category, logic, bind)
            grid.addWidget(card, row, col)
            col += 1
            if col > 1: # 2 cards per row
                col = 0
                row += 1

        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        return page

    def create_gesture_card(self, name, category, desc, mapping):
        card = QFrame()
        card.setObjectName("GestureCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(20, 20, 20, 20)
        card_layout.setSpacing(10)

        top_bar = QHBoxLayout()
        name_lbl = QLabel(name)
        name_lbl.setObjectName("GestureName")
        cat_lbl = QLabel(category)
        cat_lbl.setObjectName("GestureCategory")
        
        top_bar.addWidget(name_lbl)
        top_bar.addStretch()
        top_bar.addWidget(cat_lbl)
        card_layout.addLayout(top_bar)

        desc_lbl = QLabel(desc)
        desc_lbl.setObjectName("GestureDesc")
        desc_lbl.setWordWrap(True)
        card_layout.addWidget(desc_lbl)

        bottom_bar = QHBoxLayout()
        map_lbl = QLabel(mapping)
        map_lbl.setObjectName("GestureMapping")
        
        btn_edit = QPushButton("Edit Action")
        btn_edit.setObjectName("SmallEditButton")
        btn_edit.setFixedSize(90, 28)

        bottom_bar.addWidget(map_lbl)
        bottom_bar.addStretch()
        bottom_bar.addWidget(btn_edit)
        card_layout.addLayout(bottom_bar)

        return card

    # ----------------------------------------
    # SCREEN VIEW 3: Settings Panel
    # ----------------------------------------
    def create_settings_view(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(10, 0, 10, 0)
        layout.setSpacing(20)

        # Setting items list
        scroll = QScrollArea()
        scroll.setObjectName("DashboardScroll")
        scroll.setWidgetResizable(True)

        scroll_widget = QWidget()
        scroll_widget.setObjectName("TransparentBase")
        items_layout = QVBoxLayout(scroll_widget)
        items_layout.setSpacing(20)
        items_layout.setContentsMargins(0, 0, 10, 0)

        # Box 1: Hardware configuration
        hw_box = QFrame()
        hw_box.setObjectName("PanelCard")
        hw_layout = QVBoxLayout(hw_box)
        hw_layout.setContentsMargins(20, 20, 20, 20)
        
        title = QLabel("Webcam Pipeline Preferences")
        title.setObjectName("CardTitle")
        hw_layout.addWidget(title)
        hw_layout.addSpacing(15)

        hw_layout.addLayout(self.create_settings_row("Default Camera Sensor index", "Device Port: [0] Primary Integrated webcam"))
        hw_layout.addLayout(self.create_settings_row("MediaPipe ML Model Complexity", "Standard Performance (Low GPU usage)"))

        items_layout.addWidget(hw_box)

        # Box 2: System Hooks and Automations
        system_box = QFrame()
        system_box.setObjectName("PanelCard")
        sys_layout = QVBoxLayout(system_box)
        sys_layout.setContentsMargins(20, 20, 20, 20)
        
        sys_title = QLabel("System Integration & Overrides")
        sys_title.setObjectName("CardTitle")
        sys_layout.addWidget(sys_title)
        sys_layout.addSpacing(15)

        sys_layout.addLayout(self.create_settings_row("PyAutoGUI Safety Kill-Switch", "Disabled (Safe margins enabled)"))
        sys_layout.addLayout(self.create_settings_row("Automatic Frame Inversion mapping", "Horizontal Inversion enabled"))

        items_layout.addWidget(system_box)
        items_layout.addStretch()

        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        return page

    def create_settings_row(self, title, val):
        row = QHBoxLayout()
        row.setContentsMargins(0, 5, 0, 5)
        
        t = QLabel(title)
        t.setObjectName("SettingsText")
        
        v = QLabel(val)
        v.setObjectName("AccentValue")

        row.addWidget(t)
        row.addStretch()
        row.addWidget(v)
        return row

    # ----------------------------------------
    # Control Actions & Slots
    # ----------------------------------------
    def switch_view(self, idx, active_btn):
        # Reset navigation properties
        for btn in [self.btn_dash, self.btn_gestures, self.btn_settings]:
            btn.setProperty("active", False)
            btn.style().polish(btn)

        active_btn.setProperty("active", True)
        active_btn.style().polish(active_btn)

        self.content_stack.setCurrentIndex(idx)
        titles = ["Dashboard", "Gesture Settings Directory", "Global Configuration Preferences"]
        self.header_title.setText(titles[idx])

    def toggle_tracking(self):
        if not self.engine.tracking_enabled:
            self.engine.tracking_enabled = True
            self.btn_toggle_tracker.setText("⏸ Pause Core Tracking")
            self.btn_toggle_tracker.setStyleSheet("""
                QPushButton#TrackingButton {
                    background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 #E53935, stop:1 #B71C1C);
                    color: #FFFFFF;
                }
            """)
        else:
            self.engine.tracking_enabled = False
            self.btn_toggle_tracker.setText("▶ Start Tracking Engine")
            self.btn_toggle_tracker.setStyleSheet("") # Restores stylesheet style

    def change_sensitivity(self, value):
        self.engine.sensitivity = value / 10.0
        self.lbl_sens_val.setText(f"{round(self.engine.sensitivity, 1)}x")

    def update_image(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        p = convert_to_Qt_format.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)
        self.video_label.setPixmap(QPixmap.fromImage(p))

    def update_status(self, hand_detected, fps):
        self.system_fps_badge.setText(f"FPS: {int(fps)}")
        if hand_detected:
            self.tracker_status_chip.setText("HAND TRACKED")
            self.tracker_status_chip.setStyleSheet("background-color: rgba(0, 240, 255, 0.15); color: #00F0FF; border: 1px solid #00F0FF;")
        else:
            self.tracker_status_chip.setText("NO SOURCE DETECTED")
            self.tracker_status_chip.setStyleSheet("background-color: rgba(244, 67, 54, 0.15); color: #F44336; border: 1px solid #F44336;")

    def update_performance(self, confidence, latency):
        self.metric_latency.findChild(QLabel, "MetricValue").setText(f"{int(latency)} ms")
        self.metric_conf.findChild(QLabel, "MetricValue").setText(f"{int(confidence*100)}%")

    # ----------------------------------------
    # Premium G HUB / Dark Style Definition
    # ----------------------------------------
    def apply_premium_stylesheet(self):
        self.setStyleSheet("""
            /* Main Application Foundations */
            QWidget#BaseContainer {
                background-color: #0F0F11;
            }
            
            QWidget#RightWorkspace {
                background-color: #121214;
            }

            /* Collapsible Left Sidebar */
            QWidget#Sidebar {
                background-color: #0B0B0C;
                border-right: 1px solid #1E1E22;
            }
            
            QWidget#LogoContainer {
                border-bottom: 1px solid #1E1E22;
            }

            QLabel#AppName {
                color: #FFFFFF;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 18px;
                font-weight: 800;
                letter-spacing: 0.5px;
            }

            /* Navigation Buttons */
            QPushButton#NavBtn {
                background-color: transparent;
                color: #8C8C96;
                border: none;
                border-left: 3px solid transparent;
                padding-left: 20px;
                text-align: left;
                font-family: 'Segoe UI', sans-serif;
                font-size: 14px;
                font-weight: 600;
            }
            
            QPushButton#NavBtn:hover {
                color: #00F0FF;
                background-color: #16161A;
            }

            QPushButton#NavBtn[active="true"] {
                color: #00F0FF;
                background-color: #131A22;
                border-left: 3px solid #00F0FF;
            }

            QLabel#SidebarFooter {
                color: #4C4C54;
                font-size: 11px;
                margin-bottom: 15px;
                margin-left: 20px;
            }

            /* Global App Header */
            QLabel#HeaderTitle {
                color: #FFFFFF;
                font-family: 'Segoe UI', sans-serif;
                font-size: 26px;
                font-weight: 700;
            }

            QLabel#StatusChip {
                border-radius: 6px;
                font-family: 'Segoe UI', sans-serif;
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 0.8px;
            }

            QLabel#FPSBadge {
                background-color: #1E1E22;
                color: #D1D1D6;
                border: 1px solid #2D2D34;
                border-radius: 6px;
                font-family: 'Segoe UI', sans-serif;
                font-size: 11px;
                font-weight: 700;
            }

            /* Premium Panel Cards */
            QFrame#PanelCard {
                background-color: #17171A;
                border: 1px solid #232328;
                border-radius: 12px;
            }

            QLabel#CardTitle {
                color: #FFFFFF;
                font-family: 'Segoe UI', sans-serif;
                font-size: 16px;
                font-weight: 700;
            }

            QFrame#PanelDivider {
                color: #232328;
                max-height: 1px;
            }

            /* Live Video Panel */
            QLabel#VideoFeed {
                background-color: #0B0B0C;
                border: 1px solid #232328;
                border-radius: 8px;
                color: #8C8C96;
                font-family: 'Segoe UI', sans-serif;
            }

            /* Tracking Engine Power Button */
            QPushButton#TrackingButton {
                background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 #007ACC, stop:1 #00F0FF);
                color: #0F0F11;
                border-radius: 8px;
                padding: 14px;
                font-family: 'Segoe UI', sans-serif;
                font-size: 14px;
                font-weight: 700;
            }
            
            QPushButton#TrackingButton:hover {
                background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 #0098FF, stop:1 #52FAFF);
            }

            /* Secondary System Controllers */
            QPushButton#ControlBtn {
                background-color: #232328;
                color: #E1E1E6;
                border: 1px solid #2D2D34;
                border-radius: 8px;
                padding: 14px;
                font-family: 'Segoe UI', sans-serif;
                font-weight: 600;
            }

            QPushButton#ControlBtn:hover {
                background-color: #2D2D34;
                color: #FFFFFF;
                border: 1px solid #00F0FF;
            }

            /* Modern Sliding Handles */
            QSlider#FancySlider::groove:horizontal {
                height: 6px;
                background: #232328;
                border-radius: 3px;
            }
            
            QSlider#FancySlider::handle:horizontal {
                background: #00F0FF;
                border: 1px solid #007ACC;
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }

            QSlider#FancySlider::handle:horizontal:hover {
                background: #FFFFFF;
                box-shadow: 0 0 10px #00F0FF;
            }

            /* Telemetry Data Nodes */
            QFrame#MetricFrame {
                background-color: #1C1C21;
                border: 1px solid #232328;
                border-radius: 8px;
            }

            QLabel#MetricLabel {
                color: #8C8C96;
                font-family: 'Segoe UI', sans-serif;
                font-size: 13px;
                font-weight: 500;
            }

            QLabel#MetricValue {
                color: #00F0FF;
                font-family: 'Segoe UI', sans-serif;
                font-size: 14px;
                font-weight: 700;
            }

            QLabel#AccentValue {
                color: #00F0FF;
                font-weight: 700;
            }

            /* Standard Texts styling */
            QLabel {
                color: #E1E1E6;
                font-family: 'Segoe UI', sans-serif;
                font-size: 13px;
            }

            QLabel#PageSubHeader {
                color: #8C8C96;
                font-size: 14px;
                margin-bottom: 5px;
            }

            /* Scroll Area configurations */
            QScrollArea#DashboardScroll {
                border: none;
                background-color: transparent;
            }

            QWidget#TransparentBase {
                background-color: transparent;
            }

            /* Gesture Cards layout directory */
            QFrame#GestureCard {
                background-color: #17171A;
                border: 1px solid #232328;
                border-radius: 12px;
            }
            
            QFrame#GestureCard:hover {
                border: 1px solid #00F0FF;
            }

            QLabel#GestureName {
                color: #FFFFFF;
                font-size: 15px;
                font-weight: 700;
            }

            QLabel#GestureCategory {
                color: #00F0FF;
                background-color: rgba(0, 240, 255, 0.08);
                border: 1px solid rgba(0, 240, 255, 0.2);
                border-radius: 4px;
                padding: 2px 6px;
                font-size: 11px;
                font-weight: 700;
            }

            QLabel#GestureDesc {
                color: #8C8C96;
                font-size: 13px;
            }

            QLabel#GestureMapping {
                color: #A1A1AA;
                font-weight: 600;
                font-size: 12px;
            }

            QPushButton#SmallEditButton {
                background-color: #232328;
                color: #E1E1E6;
                border: 1px solid #2D2D34;
                border-radius: 6px;
                font-size: 11px;
                font-weight: 600;
            }

            QPushButton#SmallEditButton:hover {
                background-color: #007ACC;
                color: #FFFFFF;
                border: 1px solid #00F0FF;
            }

            QLabel#SettingsText {
                font-size: 14px;
                font-weight: 500;
            }
        """)

    def closeEvent(self, event):
        self.engine.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ZeroTouchApp()
    window.show()
    sys.exit(app.exec())