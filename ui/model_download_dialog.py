"""
Model Download Dialog for Creative Relight
PyQt6 dialog showing model download progress
"""

from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.model_manager import ModelManager


class ModelDownloadWorker(QThread):
    """Worker thread for downloading models"""
    
    progress = pyqtSignal(str, int, int)  # model_name, downloaded, total
    status = pyqtSignal(str)  # status message
    finished = pyqtSignal(bool)  # success
    
    def __init__(self, model_manager: ModelManager, model_names: list):
        super().__init__()
        self.model_manager = model_manager
        self.model_names = model_names
        self.should_stop = False
    
    def run(self):
        """Download models in background thread"""
        try:
            self.status.emit("Starting model download...")
            
            results = self.model_manager.download_models(
                model_names=self.model_names,
                progress_callback=self.progress_callback
            )
            
            success = all(results.values())
            
            if success:
                self.status.emit("✓ All models downloaded successfully!")
            else:
                failed = [name for name, result in results.items() if not result]
                self.status.emit(f"✗ Failed to download: {', '.join(failed)}")
            
            self.finished.emit(success)
            
        except Exception as e:
            self.status.emit(f"✗ Error: {str(e)}")
            self.finished.emit(False)
    
    def progress_callback(self, model_name: str, downloaded: int, total: int):
        """Called when download progress updates"""
        if not self.should_stop:
            self.progress.emit(model_name, downloaded, total)
    
    def stop(self):
        """Stop the download"""
        self.should_stop = True


class ModelDownloadDialog(QDialog):
    """Dialog for downloading AI models"""
    
    def __init__(self, parent=None, pass_types=None):
        super().__init__(parent)
        self.pass_types = pass_types or ['albedo', 'specular', 'depth', 'normal']
        self.model_manager = ModelManager(cache_dir="models")
        self.worker = None
        
        self.setup_ui()
        self.check_models()
    
    def setup_ui(self):
        """Setup the dialog UI"""
        self.setWindowTitle("Creative Relight - Model Download")
        self.setModal(True)
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)
        
        # Ensure dialog is always visible and on top
        self.setWindowFlags(
            Qt.WindowType.Dialog |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.WindowTitleHint |
            Qt.WindowType.CustomizeWindowHint
        )
        
        # Activate and raise to ensure visibility
        self.activateWindow()
        self.raise_()
        
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title = QLabel("🤖 AI Models Required")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Description
        desc = QLabel(
            "Creative Relight requires AI models to process your images.\n"
            "Models will be downloaded once and cached locally for future use."
        )
        desc.setWordWrap(True)
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setStyleSheet("color: #888; margin-bottom: 10px;")
        layout.addWidget(desc)
        
        # Model status group
        status_group = QGroupBox("Model Status")
        status_layout = QVBoxLayout(status_group)
        
        self.model_status_text = QTextEdit()
        self.model_status_text.setReadOnly(True)
        self.model_status_text.setMaximumHeight(150)
        status_layout.addWidget(self.model_status_text)
        
        layout.addWidget(status_group)
        
        # Download info
        self.download_info_label = QLabel()
        self.download_info_label.setStyleSheet("font-weight: bold; color: #42a5f5;")
        self.download_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.download_info_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")  # Show percentage
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #555;
                border-radius: 5px;
                text-align: center;
                background-color: #3c3c3c;
                color: #FFFFFF;
                font-weight: bold;
                min-height: 30px;
            }
            QProgressBar::chunk {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4caf50,
                    stop:1 #66bb6a
                );
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        # Current file label
        self.current_file_label = QLabel()
        self.current_file_label.setStyleSheet("color: #888; font-size: 11px;")
        self.current_file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.current_file_label)
        
        # Status label
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #888; margin-top: 10px;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.download_btn = QPushButton("📥 Download Models")
        self.download_btn.setMinimumWidth(150)
        self.download_btn.setStyleSheet("""
            QPushButton {
                background-color: #42a5f5;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976d2;
            }
            QPushButton:disabled {
                background-color: #666;
            }
        """)
        self.download_btn.clicked.connect(self.start_download)
        button_layout.addWidget(self.download_btn)
        
        self.skip_btn = QPushButton("Skip")
        self.skip_btn.setMinimumWidth(100)
        self.skip_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.skip_btn)
        
        self.close_btn = QPushButton("Close")
        self.close_btn.setMinimumWidth(100)
        self.close_btn.clicked.connect(self.accept)
        self.close_btn.setVisible(False)
        button_layout.addWidget(self.close_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        # Help text
        help_text = QLabel(
            '<a href="https://huggingface.co/creative-twins/creative-relight-models" '
            'style="color: #42a5f5;">View models on Hugging Face</a>'
        )
        help_text.setOpenExternalLinks(True)
        help_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        help_text.setStyleSheet("margin-top: 10px; font-size: 11px;")
        layout.addWidget(help_text)
    
    def check_models(self):
        """Check which models are available and which need downloading"""
        # Check what's missing
        missing = self.model_manager.get_missing_models(self.pass_types)
        
        status_text = "📊 Model Status:\n\n"
        
        if not missing:
            status_text += "✓ All required models are already available!\n\n"
            self.download_info_label.setText("All models ready to use")
            self.download_info_label.setStyleSheet("font-weight: bold; color: #4caf50;")
            self.download_btn.setEnabled(False)
            self.download_btn.setText("✓ Models Ready")
            self.skip_btn.setText("Continue")
        else:
            total_size = self.model_manager.calculate_download_size(missing)
            
            for model_name in missing:
                model_config = self.model_manager.config['models'][model_name]
                size = model_config.get('size_mb', 0)
                required_for = model_config.get('required_for', [])
                status_text += f"✗ {model_name}: {size} MB\n"
                status_text += f"   Required for: {', '.join(required_for)}\n\n"
            
            status_text += f"\nTotal download size: ~{total_size} MB ({total_size/1024:.2f} GB)"
            
            self.download_info_label.setText(
                f"Need to download {len(missing)} model(s) (~{total_size} MB)"
            )
            self.missing_models = missing
        
        self.model_status_text.setText(status_text)
    
    def start_download(self):
        """Start downloading models"""
        if not hasattr(self, 'missing_models') or not self.missing_models:
            return
        
        # Disable buttons
        self.download_btn.setEnabled(False)
        self.skip_btn.setEnabled(False)
        
        # Update status
        self.status_label.setText("Downloading models...")
        
        # Create and start worker thread
        self.worker = ModelDownloadWorker(self.model_manager, self.missing_models)
        self.worker.progress.connect(self.on_progress)
        self.worker.status.connect(self.on_status)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()
    
    def on_progress(self, model_name: str, downloaded: int, total: int):
        """Update progress bar with real-time updates"""
        if total > 0:
            percent = int((downloaded / total) * 100)
            self.progress_bar.setValue(percent)
            
            # Format sizes
            downloaded_mb = downloaded / 1024 / 1024
            total_mb = total / 1024 / 1024
            
            # Update file label
            self.current_file_label.setText(
                f"Downloading: {model_name} ({downloaded_mb:.1f} / {total_mb:.1f} MB)"
            )
            
            # Force UI update for real-time display
            self.progress_bar.repaint()
            QApplication.processEvents()
    
    def on_status(self, message: str):
        """Update status message"""
        self.status_label.setText(message)
    
    def on_finished(self, success: bool):
        """Handle download completion"""
        if success:
            self.progress_bar.setValue(100)
            self.download_info_label.setText("✓ All models downloaded successfully!")
            self.download_info_label.setStyleSheet("font-weight: bold; color: #4caf50;")
            self.close_btn.setVisible(True)
            self.skip_btn.setVisible(False)
        else:
            self.download_info_label.setText("✗ Download failed")
            self.download_info_label.setStyleSheet("font-weight: bold; color: #f44336;")
            self.skip_btn.setEnabled(True)
            self.download_btn.setEnabled(True)
            self.download_btn.setText("🔄 Retry Download")
    
    def closeEvent(self, event):
        """Handle dialog close"""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Download in Progress",
                "Download is still in progress. Cancel download?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.worker.stop()
                self.worker.wait()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


# Test the dialog
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Apply dark theme
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(43, 43, 43))
    palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
    app.setPalette(palette)
    
    dialog = ModelDownloadDialog()
    result = dialog.exec()
    
    print(f"Dialog result: {'Accepted' if result == QDialog.DialogCode.Accepted else 'Rejected'}")
