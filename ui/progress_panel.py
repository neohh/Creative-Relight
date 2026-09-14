"""
Progress Panel Widget for Creative Relight
Advanced progress tracking with detailed status information
"""

import time
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *


class ProgressPanel(QGroupBox):
    """Advanced progress panel with multiple progress indicators"""
    
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__("Processing Progress", parent)
        self.setup_ui()
        self.reset_progress()
        
    def setup_ui(self):
        """Setup the progress panel interface"""
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        
        # Overall progress
        overall_layout = QVBoxLayout()
        overall_layout.addWidget(QLabel("Overall Progress:"))
        
        self.overall_progress = QProgressBar()
        self.overall_progress.setMinimum(0)
        self.overall_progress.setMaximum(100)
        self.overall_progress.setValue(0)
        self.overall_progress.setTextVisible(True)
        # Set green color for progress bar with black background and bright text
        self.overall_progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #999;
                border-radius: 3px;
                text-align: center;
                background-color: #3c3c3c;
                color: #FFFFFF;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
        """)
        overall_layout.addWidget(self.overall_progress)
        
        layout.addLayout(overall_layout)
        
        # Current task progress
        task_layout = QVBoxLayout()
        self.task_label = QLabel("Current Task: Idle")
        task_layout.addWidget(self.task_label)
        
        self.task_progress = QProgressBar()
        self.task_progress.setMinimum(0)
        self.task_progress.setMaximum(100)
        self.task_progress.setValue(0)
        self.task_progress.setTextVisible(True)
        # Set green color for progress bar with black background and bright text
        self.task_progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #999;
                border-radius: 3px;
                text-align: center;
                background-color: #3c3c3c;
                color: #FFFFFF;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
        """)
        task_layout.addWidget(self.task_progress)
        
        layout.addLayout(task_layout)
        
        # Processing statistics
        stats_layout = QGridLayout()
        
        # Time elapsed
        stats_layout.addWidget(QLabel("Time Elapsed:"), 0, 0)
        self.time_elapsed_label = QLabel("00:00:00")
        stats_layout.addWidget(self.time_elapsed_label, 0, 1)
        
        # Time remaining
        stats_layout.addWidget(QLabel("Time Remaining:"), 1, 0)
        self.time_remaining_label = QLabel("--:--:--")
        stats_layout.addWidget(self.time_remaining_label, 1, 1)
        
        # Processing speed
        stats_layout.addWidget(QLabel("Processing Speed:"), 2, 0)
        self.speed_label = QLabel("-- units/sec")
        stats_layout.addWidget(self.speed_label, 2, 1)
        
        # Files processed
        stats_layout.addWidget(QLabel("Files Processed:"), 3, 0)
        self.files_label = QLabel("0 / 0")
        stats_layout.addWidget(self.files_label, 3, 1)
        
        layout.addLayout(stats_layout)
        
        # Status information
        status_layout = QVBoxLayout()
        status_layout.addWidget(QLabel("Status:"))
        
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(100)
        self.status_text.setReadOnly(True)
        status_layout.addWidget(self.status_text)
        
        layout.addLayout(status_layout)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.pause_button = QPushButton("Pause")
        self.pause_button.setEnabled(False)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        
        button_layout.addWidget(self.pause_button)
        button_layout.addWidget(self.cancel_button)
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
        
        # Timer for elapsed time updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_elapsed_time)
        
        # Progress tracking variables
        self.start_time = None
        self.paused_time = 0
        self.is_paused = False
        self.total_files = 0
        self.processed_files = 0
        self.current_speed = 0.0  # Current processing speed (units/sec)
        self.speed_samples = []   # Recent speed samples for averaging
        self.max_speed_samples = 10  # Keep last 10 samples for rolling average
        
    def start_progress(self, total_files=1):
        """Start progress tracking"""
        self.total_files = total_files
        self.processed_files = 0
        self.start_time = time.time()
        self.paused_time = 0
        self.is_paused = False
        
        self.overall_progress.setValue(0)
        self.task_progress.setValue(0)
        self.files_label.setText(f"0 / {total_files}")
        
        self.pause_button.setEnabled(True)
        self.cancel_button.setEnabled(True)
        
        self.timer.start(1000)  # Update every second
        self.add_status_message("Processing started...")
        
    def stop_progress(self):
        """Stop progress tracking"""
        self.timer.stop()
        self.pause_button.setEnabled(False)
        self.cancel_button.setEnabled(False)
        self.is_paused = False
        
        self.add_status_message("Processing stopped.")
        
    def complete_progress(self):
        """Mark progress as complete"""
        self.overall_progress.setValue(100)
        self.task_progress.setValue(100)
        self.timer.stop()  # Stop the elapsed time timer
        
        self.pause_button.setEnabled(False)
        self.cancel_button.setEnabled(False)
        
        # Set time remaining to 00:00:00 since we're done
        self.time_remaining_label.setText("00:00:00")
        
        self.add_status_message("Processing completed successfully!")
        
    def set_overall_progress(self, value):
        """Set overall progress percentage (0-100)"""
        self.overall_progress.setValue(max(0, min(100, value)))
        self.progress_updated.emit(value)
        
    def set_task_progress(self, value, task_name=None):
        """Set current task progress percentage (0-100)"""
        self.task_progress.setValue(max(0, min(100, value)))
        
        if task_name:
            self.task_label.setText(f"Current Task: {task_name}")
            
    def update_file_progress(self, processed, total=None):
        """Update file processing progress"""
        if total is not None:
            self.total_files = total
            
        self.processed_files = processed
        self.files_label.setText(f"{processed} / {self.total_files}")
        
        # Update overall progress based on files
        if self.total_files > 0:
            overall_percent = int((processed / self.total_files) * 100)
            self.set_overall_progress(overall_percent)
            
    def add_status_message(self, message):
        """Add a status message with timestamp"""
        timestamp = time.strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        self.status_text.append(formatted_message)
        self.status_updated.emit(message)
        
        # Auto-scroll to bottom
        scrollbar = self.status_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
    def update_elapsed_time(self):
        """Update elapsed time display and calculate time remaining"""
        if not self.start_time:
            return
            
        current_time = time.time()
        
        if self.is_paused:
            elapsed = self.paused_time
        else:
            elapsed = current_time - self.start_time
            
        self.time_elapsed_label.setText(self.format_time(elapsed))
        
        # Calculate time remaining based on current speed and remaining work
        self.update_time_remaining()
        
    def update_time_remaining(self, seconds=None):
        """Update the time remaining label"""
        if seconds is not None:
            # Use the provided time remaining value
            self.time_remaining_label.setText(self.format_time(seconds))
        else:
            # Calculate based on current progress and speed
            if self.current_speed > 0 and self.total_files > 0:
                remaining_files = self.total_files - self.processed_files
                if remaining_files > 0:
                    estimated_seconds = remaining_files / self.current_speed
                    self.time_remaining_label.setText(self.format_time(estimated_seconds))
                else:
                    self.time_remaining_label.setText("00:00:00")
            # Don't change the label if we don't have speed data yet (prevents flickering)
            
    def pause_processing(self):
        """Pause/resume processing"""
        if not self.is_paused:
            # Pause
            self.is_paused = True
            current_time = time.time()
            self.paused_time = current_time - self.start_time
            self.pause_button.setText("Resume")
            self.add_status_message("Processing paused.")
        else:
            # Resume
            self.is_paused = False
            current_time = time.time()
            self.start_time = current_time - self.paused_time
            self.pause_button.setText("Pause")
            self.add_status_message("Processing resumed.")
            
    def cancel_processing(self):
        """Cancel current processing"""
        self.stop_progress()
        self.add_status_message("Processing cancelled by user.")
        
    def reset_progress(self):
        """Reset all progress indicators"""
        self.overall_progress.setValue(0)
        self.task_progress.setValue(0)
        self.task_label.setText("Current Task: Idle")
        self.time_elapsed_label.setText("00:00:00")
        self.time_remaining_label.setText("--:--:--")
        self.speed_label.setText("-- files/sec")
        self.files_label.setText("0 / 0")
        self.status_text.clear()
        
        self.pause_button.setEnabled(False)
        self.cancel_button.setEnabled(False)
        self.pause_button.setText("Pause")
        
        self.timer.stop()
        self.start_time = None
        self.paused_time = 0
        self.is_paused = False
        self.total_files = 0
        self.processed_files = 0
        self.current_speed = 0.0
        self.speed_samples = []
        
    def format_time(self, seconds):
        """Format time in HH:MM:SS format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
    def update_processing_speed(self, files_per_second):
        """Update processing speed from external source"""
        self.speed_label.setText(f"{files_per_second:.2f} files/sec")
        
    def get_progress_info(self):
        """Get current progress information"""
        return {
            'overall_progress': self.overall_progress.value(),
            'task_progress': self.task_progress.value(),
            'processed_files': self.processed_files,
            'total_files': self.total_files,
            'is_paused': self.is_paused,
            'elapsed_time': self.paused_time if self.is_paused else (time.time() - self.start_time if self.start_time else 0)
        }
