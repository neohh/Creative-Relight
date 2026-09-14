"""
Model download dialog for Creative Relight
Shows when models need to be downloaded on first launch
"""
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar, QPushButton, QWidget, QGridLayout
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QPixmap, QFont
from pathlib import Path


class ModelDownloadDialog(QDialog):
    """Dialog for downloading AI models with example showcase"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Creative Relight - Downloading Models")
        self.setModal(True)
        self.setFixedSize(800, 550)  # Larger to accommodate example images
        
        self.current_example_index = 0  # For cycling through examples
        
        # Window flags: frameless + stay on top
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | 
            Qt.WindowType.Dialog |
            Qt.WindowType.WindowStaysOnTopHint
        )
        
        # Center on screen
        from PyQt6.QtGui import QGuiApplication
        screen = QGuiApplication.primaryScreen().geometry()
        x = (screen.width() - self.width()) // 2
        y = (screen.height() - self.height()) // 2
        self.move(x, y)
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup download dialog UI with example images"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Main container
        container = QWidget()
        container.setStyleSheet("""
            QWidget {
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1e1e2e,
                    stop:1 #151520
                );
            }
        """)
        
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(40, 30, 40, 30)
        container_layout.setSpacing(20)
        
        # Top section with branding
        brand_layout = QHBoxLayout()
        brand_layout.setSpacing(15)
        
        # Try to load logo
        try:
            import sys
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
                scaled_logo = logo_pixmap.scaled(50, 50, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                logo_label.setPixmap(scaled_logo)
                logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                brand_layout.addWidget(logo_label)
        except Exception as e:
            print(f"[DOWNLOAD] Could not load logo: {e}")
        
        # Title and subtitle in vertical layout
        title_layout = QVBoxLayout()
        title_layout.setSpacing(5)
        
        title_label = QLabel("Creative Relight")
        title_label.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-size: 24px;
                font-weight: bold;
                background: transparent;
            }
        """)
        title_layout.addWidget(title_label)
        
        subtitle_label = QLabel("VFX Pass Extraction")
        subtitle_label.setStyleSheet("""
            QLabel {
                color: #888888;
                font-size: 13px;
                background: transparent;
            }
        """)
        title_layout.addWidget(subtitle_label)
        
        brand_layout.addLayout(title_layout)
        brand_layout.addStretch()
        
        container_layout.addLayout(brand_layout)
        
        # Example images showcase section
        examples_section = self.create_examples_section()
        container_layout.addWidget(examples_section)
        
        container_layout.addSpacing(10)
        
        # Main message
        message_label = QLabel("Downloading AI Models")
        message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        message_label.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-size: 16px;
                font-weight: bold;
                background: transparent;
            }
        """)
        container_layout.addWidget(message_label)
        
        # Info message
        info_label = QLabel("This only happens on first launch • ~4 GB download")
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_label.setStyleSheet("""
            QLabel {
                color: #888888;
                font-size: 12px;
                background: transparent;
            }
        """)
        container_layout.addWidget(info_label)
        
        container_layout.addSpacing(10)
        
        # Current file label
        self.file_label = QLabel("Preparing download...")
        self.file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.file_label.setStyleSheet("""
            QLabel {
                color: #42a5f5;
                font-size: 11px;
                background: transparent;
            }
        """)
        container_layout.addWidget(self.file_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: rgba(255, 255, 255, 0.1);
                border: none;
                border-radius: 3px;
            }
            QProgressBar::chunk {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4caf50,
                    stop:0.5 #66bb6a,
                    stop:1 #81c784
                );
                border-radius: 3px;
            }
        """)
        container_layout.addWidget(self.progress_bar)
        
        # Progress text (e.g., "45%")
        self.progress_label = QLabel("0%")
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_label.setStyleSheet("""
            QLabel {
                color: #666666;
                font-size: 10px;
                background: transparent;
            }
        """)
        container_layout.addWidget(self.progress_label)
        
        # Note at bottom
        note_label = QLabel("Please keep the application running during download")
        note_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        note_label.setStyleSheet("""
            QLabel {
                color: #555555;
                font-size: 10px;
                font-style: italic;
                background: transparent;
            }
        """)
        container_layout.addWidget(note_label)
        
        layout.addWidget(container)
    
    def create_examples_section(self):
        """Create example images showcase section"""
        examples_widget = QWidget()
        examples_widget.setStyleSheet("QWidget { background: transparent; }")
        examples_layout = QHBoxLayout(examples_widget)
        examples_layout.setSpacing(15)
        examples_layout.setContentsMargins(0, 0, 0, 0)
        
        # Get base path for assets
        import sys
        if getattr(sys, 'frozen', False):
            base_path = Path(sys._MEIPASS)
        else:
            base_path = Path(__file__).parent.parent
        
        # Example images with labels
        examples = [
            ("normal_000000.png", "Normal Map"),
            ("Albedo.jpg", "Albedo (Diffuse)"),
            ("depth.jpg", "Depth Map")
        ]
        
        for image_file, label_text in examples:
            example_container = QWidget()
            example_container.setStyleSheet("""
                QWidget {
                    background-color: rgba(255, 255, 255, 0.05);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    border-radius: 8px;
                    padding: 8px;
                }
            """)
            example_layout = QVBoxLayout(example_container)
            example_layout.setSpacing(8)
            example_layout.setContentsMargins(8, 8, 8, 8)
            
            # Image
            image_path = base_path / "assets" / image_file
            if image_path.exists():
                image_label = QLabel()
                image_pixmap = QPixmap(str(image_path))
                # Scale to consistent size (180x120 for good showcase)
                scaled_image = image_pixmap.scaled(180, 120, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                image_label.setPixmap(scaled_image)
                image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                image_label.setStyleSheet("background: transparent; border: none;")
                example_layout.addWidget(image_label)
            else:
                print(f"[DOWNLOAD] Example image not found: {image_path}")
            
            # Label
            text_label = QLabel(label_text)
            text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            text_label.setStyleSheet("""
                QLabel {
                    color: #aaaaaa;
                    font-size: 11px;
                    font-weight: bold;
                    background: transparent;
                    border: none;
                }
            """)
            example_layout.addWidget(text_label)
            
            examples_layout.addWidget(example_container)
        
        return examples_widget
    
    def update_progress(self, file_name, downloaded, total):
        """Update download progress"""
        if total > 0:
            percent = int((downloaded / total) * 100)
            self.progress_bar.setValue(percent)
            self.progress_label.setText(f"{percent}%")
            
            # Show generic user-friendly message (hide technical details)
            downloaded_mb = downloaded / (1024 * 1024)
            total_mb = total / (1024 * 1024)
            
            # Only show size info, not file names or technical details
            if downloaded_mb < 1024:
                self.file_label.setText(f"Downloading: {downloaded_mb:.1f} / {total_mb:.1f} MB")
            else:
                # Show in GB for larger downloads
                downloaded_gb = downloaded_mb / 1024
                total_gb = total_mb / 1024
                self.file_label.setText(f"Downloading: {downloaded_gb:.2f} / {total_gb:.2f} GB")
    
    def set_completed(self):
        """Show download completion"""
        self.progress_bar.setValue(100)
        self.progress_label.setText("100%")
        self.file_label.setText("Download complete!")
        self.file_label.setStyleSheet("""
            QLabel {
                color: #4caf50;
                font-size: 12px;
                background: transparent;
            }
        """)
