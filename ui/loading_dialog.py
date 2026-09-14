"""
Loading Dialog for Creative Relight
Shows loading progress when initializing AI models
"""

from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *


class LoadingDialog(QDialog):
    """Simple loading dialog with progress indication"""
    
    def __init__(self, parent=None, title="Loading", message="Please wait..."):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setFixedSize(400, 180)
        
        # Remove window frame decorations for clean look
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
        
        self.setup_ui(message)
    
    def setup_ui(self, message):
        """Setup the dialog UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # Title with icon
        title_layout = QHBoxLayout()
        title_layout.addStretch()
        
        # Loading icon/text
        title_label = QLabel("🤖 Initializing AI Models")
        title_label.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: #42a5f5;
        """)
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        layout.addLayout(title_layout)
        
        # Message
        self.message_label = QLabel(message)
        self.message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.message_label.setWordWrap(True)
        self.message_label.setStyleSheet("""
            font-size: 13px;
            color: #cccccc;
        """)
        layout.addWidget(self.message_label)
        
        # Progress bar (indeterminate)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(0)  # Indeterminate mode
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                border-radius: 4px;
                background-color: #3c3c3c;
            }
            QProgressBar::chunk {
                background-color: #42a5f5;
                border-radius: 4px;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        layout.addStretch()
        
        # Apply dark theme styling
        self.setStyleSheet("""
            QDialog {
                background-color: #2b2b2b;
                border: 2px solid #42a5f5;
                border-radius: 10px;
            }
        """)
    
    def update_message(self, message):
        """Update the loading message"""
        self.message_label.setText(message)
    
    def close_dialog(self):
        """Close the dialog gracefully"""
        self.accept()


class ModelInitWorker(QThread):
    """Worker thread for initializing AI models in background"""
    
    progress = pyqtSignal(str, int)  # status message, percentage
    finished = pyqtSignal(bool, str)  # success, error_message
    
    def __init__(self, parent_window):
        super().__init__()
        self.parent_window = parent_window
    
    def run(self):
        """Initialize models in background thread"""
        try:
            self.progress.emit("Preparing AI models...", 5)
            QThread.msleep(100)  # Small delay for UI update
            
            self.progress.emit("Loading Lighting Models...", 15)
            try:
                # Try direct import first (dev mode - processors/ is in sys.path)
                from image_processor import ImageProcessor
                print("[LOADING] ✅ Imported ImageProcessor directly")
            except ImportError as e1:
                print(f"[LOADING] Failed direct import of ImageProcessor: {e1}")
                try:
                    # Fallback to package import (frozen mode)
                    from processors.image_processor import ImageProcessor
                    print("[LOADING] ✅ Imported ImageProcessor from processors package")
                except ImportError as e2:
                    print(f"[LOADING] Failed package import of ImageProcessor: {e2}")
                    # Last resort - try src prefix
                    from src.processors.image_processor import ImageProcessor
                    print("[LOADING] ✅ Imported ImageProcessor from src.processors")
            
            self.progress.emit("Loading Lighting Models...", 25)
            QThread.msleep(100)
            
            self.progress.emit("Loading Video Processor...", 40)
            try:
                # Try direct import first (dev mode - processors/ is in sys.path)
                from video_processor import VideoProcessor
                print("[LOADING] ✅ Imported VideoProcessor directly")
            except ImportError as e1:
                print(f"[LOADING] Failed direct import of VideoProcessor: {e1}")
                try:
                    # Fallback to package import (frozen mode)
                    from processors.video_processor import VideoProcessor
                    print("[LOADING] ✅ Imported VideoProcessor from processors package")
                except ImportError as e2:
                    print(f"[LOADING] Failed package import of VideoProcessor: {e2}")
                    # Last resort - try src prefix
                    from src.processors.video_processor import VideoProcessor
                    print("[LOADING] ✅ Imported VideoProcessor from src.processors")
            
            self.progress.emit("Loading Video Processor...", 50)
            QThread.msleep(100)
            
            self.progress.emit("Loading Sequence Processor...", 60)
            try:
                # Try direct import first (dev mode - processors/ is in sys.path)
                from sequence_processor import SequenceProcessor
                print("[LOADING] ✅ Imported SequenceProcessor directly")
            except ImportError as e1:
                print(f"[LOADING] Failed direct import of SequenceProcessor: {e1}")
                try:
                    # Fallback to package import (frozen mode)
                    from processors.sequence_processor import SequenceProcessor
                    print("[LOADING] ✅ Imported SequenceProcessor from processors package")
                except ImportError as e2:
                    print(f"[LOADING] Failed package import of SequenceProcessor: {e2}")
                    # Last resort - try src prefix
                    from src.processors.sequence_processor import SequenceProcessor
                    print("[LOADING] ✅ Imported SequenceProcessor from src.processors")
            
            self.progress.emit("Loading Sequence Processor...", 70)
            QThread.msleep(100)
            
            # Get device from parent
            device = self.parent_window.device
            
            self.progress.emit("Loading Geometry Models...", 80)
            QThread.msleep(100)
            
            self.progress.emit("Accelerating Hardware...", 85)
            
            # Initialize processors
            self.parent_window.image_processor = ImageProcessor(device=device)
            self.progress.emit("Accelerating Hardware...", 90)
            
            # Initialize VideoProcessor WITHOUT temporal consistency by default
            # Temporal consistency will be enabled only when explicitly requested by user
            self.parent_window.video_processor = VideoProcessor(device=device, use_temporal_consistency=False)
            self.progress.emit("Accelerating Hardware...", 94)
            
            self.parent_window.sequence_processor = SequenceProcessor(device=device)
            self.progress.emit("Accelerating Hardware...", 98)
            
            self.progress.emit("Ready!", 100)
            
            self.finished.emit(True, "")
            
        except Exception as e:
            error_msg = f"Error initializing models: {str(e)}"
            self.progress.emit(f"✗ {error_msg}", 0)
            self.finished.emit(False, error_msg)
