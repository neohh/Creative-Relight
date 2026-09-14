"""
Splash screen for Creative Relight
Shows branding and loading status during startup
"""
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QProgressBar, QHBoxLayout
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QPainter, QLinearGradient, QColor, QFont
from pathlib import Path
import sys


class SplashScreen(QWidget):
    """Splash screen with branding and loading status"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # Set fixed size
        self.setFixedSize(800, 500)
        
        # Center on screen
        from PyQt6.QtGui import QGuiApplication
        screen = QGuiApplication.primaryScreen().geometry()
        x = (screen.width() - self.width()) // 2
        y = (screen.height() - self.height()) // 2
        self.move(x, y)
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        """Setup splash screen UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Main container with dark background
        container = QWidget()
        container.setStyleSheet("""
            QWidget {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1a1a2e,
                    stop:0.5 #16213e,
                    stop:1 #0f1419
                );
                border-radius: 20px;
            }
        """)
        
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(40, 40, 40, 40)
        container_layout.setSpacing(20)
        
        # Top spacer
        container_layout.addStretch(1)
        
        # Logo/Brand area
        brand_layout = QVBoxLayout()
        brand_layout.setSpacing(10)
        
        # Try to load brand image (lightbulb or icon)
        try:
            # Get asset path dynamically
            if getattr(sys, 'frozen', False):
                # Running as compiled
                base_path = Path(sys._MEIPASS)
            else:
                # Running from source
                base_path = Path(__file__).parent.parent
            
            logo_path = base_path / "assets" / "Lightbulb.png"
            if logo_path.exists():
                logo_label = QLabel()
                logo_pixmap = QPixmap(str(logo_path))
                scaled_logo = logo_pixmap.scaled(80, 80, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                logo_label.setPixmap(scaled_logo)
                logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                brand_layout.addWidget(logo_label)
        except Exception as e:
            print(f"[SPLASH] Could not load logo: {e}")
        
        # App name
        title_label = QLabel("Creative Relight")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-size: 36px;
                font-weight: bold;
                background: transparent;
            }
        """)
        brand_layout.addWidget(title_label)
        
        # Subtitle
        subtitle_label = QLabel("VFX Passes")
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle_label.setStyleSheet("""
            QLabel {
                color: #888888;
                font-size: 18px;
                background: transparent;
            }
        """)
        brand_layout.addWidget(subtitle_label)
        
        container_layout.addLayout(brand_layout)
        container_layout.addSpacing(30)
        
        # Status message
        self.status_label = QLabel("Initializing...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                color: #4caf50;
                font-size: 14px;
                font-weight: bold;
                background: transparent;
            }
        """)
        container_layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")  # Show percentage
        self.progress_bar.setFixedHeight(30)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 15px;
                text-align: center;
                color: #ffffff;
                font-weight: bold;
                font-size: 12px;
            }
            QProgressBar::chunk {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4caf50,
                    stop:1 #66bb6a
                );
                border-radius: 14px;
            }
        """)
        container_layout.addWidget(self.progress_bar)
        
        # Version and copyright
        info_layout = QVBoxLayout()
        info_layout.setSpacing(5)
        
        version_label = QLabel("Version 1.0.0")
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        version_label.setStyleSheet("""
            QLabel {
                color: #666666;
                font-size: 11px;
                background: transparent;
            }
        """)
        info_layout.addWidget(version_label)
        
        copyright_label = QLabel("© 2025 Creative Twins")
        copyright_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        copyright_label.setStyleSheet("""
            QLabel {
                color: #666666;
                font-size: 11px;
                background: transparent;
            }
        """)
        info_layout.addWidget(copyright_label)
        
        container_layout.addSpacing(20)
        container_layout.addLayout(info_layout)
        
        # Bottom spacer
        container_layout.addStretch(1)
        
        layout.addWidget(container)
    
    def update_status(self, message, progress=None):
        """Update status message and progress"""
        self.status_label.setText(message)
        if progress is not None:
            self.progress_bar.setValue(int(progress))
    
    def set_indeterminate(self):
        """Set progress bar to indeterminate (pulsing) mode"""
        self.progress_bar.setMaximum(0)
        self.progress_bar.setMinimum(0)
    
    def set_determinate(self):
        """Set progress bar to determinate mode"""
        self.progress_bar.setMaximum(100)
        self.progress_bar.setMinimum(0)
    
    def paintEvent(self, event):
        """Custom paint for rounded corners and shadow"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw shadow (optional, for depth)
        # Could add QGraphicsDropShadowEffect instead
        
        super().paintEvent(event)
