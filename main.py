import sys
import cv2
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, 
                             QVBoxLayout, QLabel, QPushButton, QSlider, QStackedWidget)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QImage, QPixmap
from engine import HandTrackingEngine

class ZeroTouchApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ZeroTouch - AI Productivity Tool")
        self.setMinimumSize(1100, 700)
        
        # Initialize Core tracking engine
        self.engine = HandTrackingEngine()
        self.engine.change_pixmap_signal.connect(self.update_image)
        self.engine.status_signal.connect(self.update_status)
        
        self.init_ui()
        self.engine.start()

    def init_ui(self):
        # Main Window Layout Split: Sidebar + Right Content Pane
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 1. Sidebar Setup
        self.sidebar = QWidget()
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setFixedWidth(200)
        sidebar_layout = QVBoxLayout(self.sidebar)
        
        # Sidebar Menu Items
        dash_btn = QPushButton(" Dashboard")
        gesture_btn = QPushButton(" Gestures")
        settings_btn = QPushButton(" Settings")
        
        sidebar_layout.addWidget(dash_btn)
        sidebar_layout.addWidget(gesture_btn)
        sidebar_layout.addWidget(settings_btn)
        sidebar_layout.addStretch()

        # 2. Right Content Pane (Header + Dynamic Content Screen)
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(20, 20, 20, 20)

        # Top Header
        header_layout = QHBoxLayout()
        self.title_lbl = QLabel("<h1>ZeroTouch Dashboard</h1>")
        self.fps_lbl = QLabel("FPS: 0")
        self.status_lbl = QLabel("Status: No Hand Detected")
        
        header_layout.addWidget(self.title_lbl)
        header_layout.addStretch()
        header_layout.addWidget(self.status_lbl)
        header_layout.addWidget(self.fps_lbl)
        
        right_layout.addLayout(header_layout)

        # Dynamic Content Panel via QStackedWidget
        self.content_stack = QStackedWidget()
        self.content_stack.addWidget(self.create_dashboard_view())
        
        right_layout.addWidget(self.content_stack)

        # Final Layout Assembly
        main_layout.addWidget(self.sidebar)
        main_layout.addWidget(right_container)
        self.setCentralWidget(main_widget)
        
        self.apply_premium_stylesheet()

    def create_dashboard_view(self):
        """Generates the Main Dashboard Grid (Live Preview + Quick Controls)"""
        dash_page = QWidget()
        layout = QHBoxLayout(dash_page)
        
        # Live Preview Box
        preview_box = QWidget()
        preview_layout = QVBoxLayout(preview_box)
        self.video_label = QLabel("Initializing Camera...")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: #121214; border-radius: 12px;")
        preview_layout.addWidget(self.video_label)
        
        # Controls Panel (Sliders & Quick Controls)
        controls_box = QWidget()
        controls_layout = QVBoxLayout(controls_box)
        
        sensitivity_lbl = QLabel("Cursor Sensitivity")
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(1, 5)
        
        controls_layout.addWidget(sensitivity_lbl)
        controls_layout.addWidget(self.slider)
        controls_layout.addStretch()

        layout.addWidget(preview_box, stretch=2)
        layout.addWidget(controls_box, stretch=1)
        return dash_page

    def update_image(self, cv_img):
        """Updates the video_label with a new opencv frame cleanly"""
        qt_img = self.convert_cv_qt(cv_img)
        self.video_label.setPixmap(qt_img)

    def update_status(self, hand_detected, fps):
        self.fps_lbl.setText(f"FPS: {fps}")
        if hand_detected:
            self.status_lbl.setText("Status: Hand Detected")
            self.status_lbl.setStyleSheet("color: #4CAF50; font-weight: bold;")
        else:
            self.status_lbl.setText("Status: No Hand")
            self.status_lbl.setStyleSheet("color: #F44336;")

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        p = convert_to_Qt_format.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def apply_premium_stylesheet(self):
        """A dark, modern aesthetic matching premium SaaS tools"""
        self.setStyleSheet("""
            QMainWindow { background-color: #1a1a1e; }
            QWidget#Sidebar { background-color: #111113; border-right: 1px solid #2d2d34; }
            QPushButton { 
                background-color: transparent; color: #b3b3b3; 
                border: none; padding: 12px; text-align: left; font-size: 14px;
            }
            QPushButton:hover { background-color: #222226; color: #ffffff; border-radius: 6px; }
            QLabel { color: #ffffff; font-family: 'Segoe UI', sans-serif; }
        """)

    def closeEvent(self, event):
        self.engine.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ZeroTouchApp()
    window.show()
    sys.exit(app.exec())