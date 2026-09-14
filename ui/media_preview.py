"""
Media Preview Widget for Creative Relight
Advanced video player with QMediaPlayer and OpenCV fallback
"""

import os
import sys
from pathlib import Path
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *

# Enable OpenCV's OpenEXR codec before any EXR decode happens in this process
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')

# Try to import multimedia components
try:
    from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
    from PyQt6.QtMultimediaWidgets import QVideoWidget
    HAS_MULTIMEDIA = True
    print("✅ QMediaPlayer available for video playback")
except ImportError:
    HAS_MULTIMEDIA = False
    print("⚠️ QMediaPlayer not available, using OpenCV fallback only")


class MediaPreviewWidget(QWidget):
    """Media preview with QMediaPlayer and OpenCV fallback"""
    
    media_loaded = pyqtSignal(str)
    frame_changed = pyqtSignal(int)
    files_dropped = pyqtSignal(list, str)  # Signal: (files, file_type) where file_type is 'image', 'video', or 'sequence'
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        
        # Setup main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create stacked widget for different display modes
        self.display_stack = QStackedWidget()
        self.display_stack.setAcceptDrops(True)  # Enable drops on stacked widget
        layout.addWidget(self.display_stack)
        
        # Image display widget
        self.image_widget = QLabel()
        self.image_widget.setStyleSheet("border: 2px solid #555555; background-color: #333333;")
        self.image_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_widget.setText("No Media Loaded\nDrag and drop a file here")  # Fixed newline
        self.image_widget.setScaledContents(False)
        self.image_widget.setAcceptDrops(True)  # Enable drops on image widget
        self.display_stack.addWidget(self.image_widget)
        
        # Video display widget (QMediaPlayer)
        if HAS_MULTIMEDIA:
            self.video_widget = QVideoWidget()
            self.video_widget.setStyleSheet("border: 2px solid #555555; background-color: #333333;")
            self.video_widget.setAcceptDrops(True)  # Enable drops on video widget
            self.display_stack.addWidget(self.video_widget)
            
            # Initialize media player
            self.media_player = QMediaPlayer()
            self.audio_output = QAudioOutput()
            self.media_player.setAudioOutput(self.audio_output)
            self.media_player.setVideoOutput(self.video_widget)
            
            # Connect media player signals
            self.media_player.positionChanged.connect(self.on_position_changed)
            self.media_player.durationChanged.connect(self.on_duration_changed)
            self.media_player.mediaStatusChanged.connect(self.on_media_status_changed)
            self.media_player.playbackStateChanged.connect(self.on_playback_state_changed)
            self.media_player.errorOccurred.connect(self.on_media_error)
        
        # Video controls
        self.create_video_controls()
        layout.addWidget(self.video_controls)
        
        # Initialize state variables
        self.current_media_path = None
        self.media_type = None  # 'image', 'video_native', 'video_fallback'
        self.duration = 0
        self.is_playing = False
        self.seeking = False
        
        # Fallback video data (OpenCV)
        self.video_frames = []
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30
        self.fallback_timer = QTimer()
        self.fallback_timer.timeout.connect(self.advance_fallback_frame)
        
        # Preview management
        self.current_previews = {}
        self.current_pass = 'original'
        
        # Enable drag and drop
        self.setAcceptDrops(True)
        
        # Hide controls initially
        self.video_controls.setVisible(False)
    
    def create_video_controls(self):
        """Create video control panel"""
        self.video_controls = QWidget()
        controls_layout = QVBoxLayout(self.video_controls)
        controls_layout.setContentsMargins(5, 5, 5, 5)
        
        # Timeline slider
        self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.setMaximum(1000)  # Use 1000 steps for smooth seeking
        self.timeline_slider.setValue(0)
        self.timeline_slider.sliderPressed.connect(self.on_slider_pressed)
        self.timeline_slider.sliderReleased.connect(self.on_slider_released)
        self.timeline_slider.valueChanged.connect(self.on_slider_moved)
        controls_layout.addWidget(self.timeline_slider)
        
        # Control buttons row
        buttons_layout = QHBoxLayout()
        
        # Play/Pause button
        self.play_pause_btn = QPushButton("▶️")
        self.play_pause_btn.setFixedSize(40, 30)
        self.play_pause_btn.setToolTip("Play/Pause")
        self.play_pause_btn.clicked.connect(self.toggle_playback)
        buttons_layout.addWidget(self.play_pause_btn)
        
        # Stop button
        self.stop_btn = QPushButton("⏹️")
        self.stop_btn.setFixedSize(40, 30)
        self.stop_btn.setToolTip("Stop")
        self.stop_btn.clicked.connect(self.stop_playback)
        buttons_layout.addWidget(self.stop_btn)
        
        # Separator
        buttons_layout.addWidget(QLabel(" | "))
        
        # Speed control
        buttons_layout.addWidget(QLabel("Speed:"))
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.25x", "0.5x", "0.75x", "1.0x", "1.25x", "1.5x", "2.0x"])
        self.speed_combo.setCurrentText("1.0x")
        self.speed_combo.currentTextChanged.connect(self.on_speed_changed)
        self.speed_combo.setMaximumWidth(80)
        buttons_layout.addWidget(self.speed_combo)
        
        buttons_layout.addStretch()
        
        # Time display
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setMinimumWidth(100)
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        buttons_layout.addWidget(self.time_label)
        
        buttons_layout.addStretch()
        
        # Volume control
        buttons_layout.addWidget(QLabel("🔊"))
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setMinimum(0)
        self.volume_slider.setMaximum(100)
        self.volume_slider.setValue(75)
        self.volume_slider.setMaximumWidth(100)
        self.volume_slider.valueChanged.connect(self.on_volume_changed)
        self.volume_slider.setToolTip("Volume")
        buttons_layout.addWidget(self.volume_slider)
        
        controls_layout.addLayout(buttons_layout)
    
    def load_media(self, media_path):
        """Load image or video with intelligent format detection"""
        try:
            if not os.path.exists(media_path):
                self.show_error("File not found")
                return False
            
            self.current_media_path = media_path
            file_ext = Path(media_path).suffix.lower()
            
            # Reset state
            self.stop_playback()
            
            # Detect media type
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v']
            image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif', '.webp', '.exr', '.hdr', '.pic']
            
            if file_ext in video_extensions:
                return self.load_video(media_path)
            elif file_ext in image_extensions:
                return self.load_image(media_path)
            else:
                self.show_error(f"Unsupported file format: {file_ext}")
                return False
                
        except Exception as e:
            self.show_error(f"Error loading media: {str(e)}")
            return False
    
    def load_image(self, image_path):
        """Load and display an image"""
        try:
            print(f"📸 Loading image: {Path(image_path).name}")
            
            if str(image_path).lower().endswith(('.exr', '.hdr', '.pic')):
                # Qt can't decode EXR/HDR - load via OpenCV and tonemap for preview
                pixmap = None
                try:
                    import cv2
                    import numpy as np
                    data = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                    if data is not None:
                        data = data.astype('float32')
                        if data.ndim == 2:
                            data = np.dstack([data] * 3)
                        rgb = data[:, :, :3][:, :, ::-1]  # BGR -> RGB
                        rgb = np.clip(rgb, 0.0, 1.0)
                        rgb = np.where(rgb <= 0.0031308, rgb * 12.92,
                                       1.055 * np.power(np.maximum(rgb, 1e-10), 1.0 / 2.4) - 0.055)
                        rgb8 = (np.clip(rgb, 0, 1) * 255).astype('uint8')
                        rgb8 = np.ascontiguousarray(rgb8)
                        h, w = rgb8.shape[:2]
                        from PyQt6.QtGui import QImage
                        qimg = QImage(rgb8.data, w, h, 3 * w, QImage.Format.Format_RGB888)
                        pixmap = QPixmap.fromImage(qimg)
                except Exception as exr_err:
                    print(f"⚠️ EXR/HDR preview failed: {exr_err}")
                if pixmap is None:
                    self.show_error("Unsupported file format: HDR image could not be decoded")
                    return False
            else:
                pixmap = QPixmap(image_path)
            if pixmap.isNull():
                self.show_error("Invalid image format")
                return False
            
            self.media_type = 'image'
            self.video_controls.setVisible(False)
            self.display_stack.setCurrentWidget(self.image_widget)
            
            # Scale image to fit preview area
            scaled_pixmap = pixmap.scaled(
                self.image_widget.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            
            self.image_widget.setPixmap(scaled_pixmap)
            self.media_loaded.emit(image_path)
            
            print(f"✅ Image loaded successfully")
            return True
            
        except Exception as e:
            self.show_error(f"Error loading image: {str(e)}")
            return False
    
    def load_video(self, video_path):
        """Load video with QMediaPlayer (primary) or OpenCV (fallback)"""
        try:
            print(f"🎬 Loading video: {Path(video_path).name}")
            
            if HAS_MULTIMEDIA:
                # Try QMediaPlayer first
                return self.load_video_native(video_path)
            else:
                # Use OpenCV fallback
                return self.load_video_fallback(video_path)
                
        except Exception as e:
            print(f"❌ Error in video loading: {e}")
            # Try fallback if native fails
            if HAS_MULTIMEDIA:
                print("🔄 Falling back to OpenCV...")
                return self.load_video_fallback(video_path)
            return False
    
    def load_video_native(self, video_path):
        """Load video using QMediaPlayer"""
        try:
            print("🎯 Using QMediaPlayer for video playback")
            
            self.media_type = 'video_native'
            self.video_controls.setVisible(True)
            self.display_stack.setCurrentWidget(self.video_widget)
            
            # Create media source
            media_url = QUrl.fromLocalFile(os.path.abspath(video_path))
            print(f"📁 Media URL: {media_url.toString()}")
            
            # Reset player state
            self.media_player.stop()
            self.duration = 0
            self.timeline_slider.setValue(0)
            self.time_label.setText("00:00 / 00:00")
            
            # Set source and load
            self.media_player.setSource(media_url)
            
            # Set up timeout to detect loading issues
            QTimer.singleShot(5000, self.check_loading_timeout)
            
            self.media_loaded.emit(video_path)
            return True
            
        except Exception as e:
            print(f"❌ QMediaPlayer error: {e}")
            return False
    
    def load_video_fallback(self, video_path):
        """Load video using OpenCV fallback"""
        try:
            print("🔄 Using OpenCV fallback for video playback")
            
            import cv2
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                self.show_error("Could not open video file")
                return False
            
            # Get video properties
            self.fps = cap.get(cv2.CAP_PROP_FPS) or 30
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"📊 Video info: {total_frames} frames at {self.fps:.1f} FPS")
            
            # Extract frames (limit for memory efficiency)
            max_frames = min(total_frames, 2000)  # Limit to 2000 frames
            frame_skip = max(1, total_frames // max_frames)
            
            self.video_frames = []
            frame_count = 0
            extracted = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_skip == 0:
                    # Convert to Qt format
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    height, width, channel = rgb_frame.shape
                    bytes_per_line = 3 * width
                    q_image = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                    
                    # Store as PNG data
                    ba = QByteArray()
                    buffer = QBuffer(ba)
                    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
                    q_image.save(buffer, "PNG")
                    
                    self.video_frames.append(ba.data())
                    extracted += 1
                
                frame_count += 1
            
            cap.release()
            
            self.total_frames = len(self.video_frames)
            self.current_frame = 0
            self.media_type = 'video_fallback'
            
            # Setup UI
            self.video_controls.setVisible(True)
            self.display_stack.setCurrentWidget(self.image_widget)
            self.timeline_slider.setValue(0)
            
            # Show first frame
            if self.video_frames:
                self.show_fallback_frame(0)
                self.update_time_display_fallback()
            
            print(f"✅ OpenCV fallback loaded: {self.total_frames} frames")
            self.media_loaded.emit(video_path)
            return True
            
        except ImportError:
            self.show_error("OpenCV not available for video processing")
            return False
        except Exception as e:
            self.show_error(f"Error in fallback video loading: {str(e)}")
            return False
    
    def check_loading_timeout(self):
        """Check if QMediaPlayer is stuck loading"""
        if self.media_type == 'video_native' and self.duration == 0:
            status = self.media_player.mediaStatus()
            if status in [QMediaPlayer.MediaStatus.LoadingMedia, 
                         QMediaPlayer.MediaStatus.BufferingMedia,
                         QMediaPlayer.MediaStatus.StalledMedia]:
                print("⏰ QMediaPlayer loading timeout, switching to fallback")
                if self.current_media_path:
                    self.load_video_fallback(self.current_media_path)
    
    def show_fallback_frame(self, frame_index):
        """Display a specific frame in fallback mode"""
        if not self.video_frames or frame_index >= len(self.video_frames):
            return
        
        try:
            frame_data = self.video_frames[frame_index]
            pixmap = QPixmap()
            pixmap.loadFromData(frame_data)
            
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(
                    self.image_widget.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.image_widget.setPixmap(scaled_pixmap)
                
        except Exception as e:
            print(f"❌ Error showing frame {frame_index}: {e}")
    
    def toggle_playback(self):
        """Toggle play/pause"""
        if self.media_type == 'image':
            return
        
        if self.media_type == 'video_native' and HAS_MULTIMEDIA:
            # QMediaPlayer mode
            state = self.media_player.playbackState()
            if state == QMediaPlayer.PlaybackState.PlayingState:
                self.media_player.pause()
            else:
                self.media_player.play()
                
        elif self.media_type == 'video_fallback':
            # OpenCV fallback mode
            if self.is_playing:
                self.fallback_timer.stop()
                self.is_playing = False
                self.play_pause_btn.setText("▶️")
            else:
                # Calculate interval based on FPS and speed
                speed = float(self.speed_combo.currentText().replace('x', ''))
                interval = int(1000 / (self.fps * speed))
                self.fallback_timer.start(interval)
                self.is_playing = True
                self.play_pause_btn.setText("⏸️")
    
    def stop_playback(self):
        """Stop playback and return to beginning"""
        if self.media_type == 'video_native' and HAS_MULTIMEDIA:
            self.media_player.stop()
            
        elif self.media_type == 'video_fallback':
            self.fallback_timer.stop()
            self.is_playing = False
            self.current_frame = 0
            self.timeline_slider.setValue(0)
            if self.video_frames:
                self.show_fallback_frame(0)
                self.update_time_display_fallback()
        
        self.play_pause_btn.setText("▶️")
        self.is_playing = False
    
    def advance_fallback_frame(self):
        """Advance to next frame in fallback mode"""
        if not self.video_frames or not self.is_playing:
            return
        
        self.current_frame += 1
        if self.current_frame >= self.total_frames:
            # Stop at end
            self.stop_playback()
            return
        
        # Update slider
        progress = int((self.current_frame / (self.total_frames - 1)) * 1000) if self.total_frames > 1 else 0
        self.timeline_slider.setValue(progress)
        
        self.show_fallback_frame(self.current_frame)
        self.update_time_display_fallback()
        self.frame_changed.emit(self.current_frame)
    
    def on_slider_pressed(self):
        """Handle slider press for seeking"""
        self.seeking = True
    
    def on_slider_released(self):
        """Handle slider release after seeking"""
        self.seeking = False
        
        if self.media_type == 'video_native' and HAS_MULTIMEDIA and self.duration > 0:
            # Seek in QMediaPlayer
            position = int((self.timeline_slider.value() / 1000.0) * self.duration)
            self.media_player.setPosition(position)
            
        elif self.media_type == 'video_fallback' and self.video_frames:
            # Seek in fallback mode
            frame_index = int((self.timeline_slider.value() / 1000.0) * (self.total_frames - 1))
            frame_index = max(0, min(frame_index, self.total_frames - 1))
            self.current_frame = frame_index
            self.show_fallback_frame(frame_index)
            self.update_time_display_fallback()
    
    def on_slider_moved(self, value):
        """Handle slider movement during seeking"""
        if self.seeking:
            if self.media_type == 'video_fallback' and self.video_frames:
                # Live preview during seeking in fallback mode
                frame_index = int((value / 1000.0) * (self.total_frames - 1))
                frame_index = max(0, min(frame_index, self.total_frames - 1))
                self.show_fallback_frame(frame_index)
    
    def on_speed_changed(self, speed_text):
        """Handle speed change"""
        speed = float(speed_text.replace('x', ''))
        
        if self.media_type == 'video_native' and HAS_MULTIMEDIA:
            self.media_player.setPlaybackRate(speed)
            
        elif self.media_type == 'video_fallback' and self.is_playing:
            # Update timer interval
            interval = int(1000 / (self.fps * speed))
            self.fallback_timer.setInterval(interval)
    
    def on_volume_changed(self, volume):
        """Handle volume change"""
        if HAS_MULTIMEDIA and hasattr(self, 'audio_output'):
            self.audio_output.setVolume(volume / 100.0)
    
    # QMediaPlayer signal handlers
    def on_position_changed(self, position):
        """Handle position updates from QMediaPlayer"""
        if not self.seeking and self.duration > 0:
            progress = int((position / self.duration) * 1000)
            self.timeline_slider.setValue(progress)
            self.update_time_display_native(position, self.duration)
    
    def on_duration_changed(self, duration):
        """Handle duration updates from QMediaPlayer"""
        self.duration = duration
        if duration > 0:
            print(f"🕐 Video duration: {self.format_time(duration / 1000)}")
    
    def on_media_status_changed(self, status):
        """Handle media status changes from QMediaPlayer"""
        status_names = {
            QMediaPlayer.MediaStatus.NoMedia: "NoMedia",
            QMediaPlayer.MediaStatus.LoadingMedia: "LoadingMedia",
            QMediaPlayer.MediaStatus.LoadedMedia: "LoadedMedia",
            QMediaPlayer.MediaStatus.StalledMedia: "StalledMedia", 
            QMediaPlayer.MediaStatus.BufferingMedia: "BufferingMedia",
            QMediaPlayer.MediaStatus.BufferedMedia: "BufferedMedia",
            QMediaPlayer.MediaStatus.EndOfMedia: "EndOfMedia",
            QMediaPlayer.MediaStatus.InvalidMedia: "InvalidMedia"
        }
        
        status_name = status_names.get(status, f"Unknown({status})")
        print(f"📺 Media status: {status_name}")
        
        if status == QMediaPlayer.MediaStatus.LoadedMedia:
            print("✅ QMediaPlayer ready for playback")
            
        elif status == QMediaPlayer.MediaStatus.InvalidMedia:
            print("❌ Invalid media, switching to fallback")
            if self.current_media_path:
                self.load_video_fallback(self.current_media_path)
    
    def on_playback_state_changed(self, state):
        """Handle playback state changes from QMediaPlayer"""
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.is_playing = True
            self.play_pause_btn.setText("⏸️")
        elif state == QMediaPlayer.PlaybackState.PausedState:
            self.is_playing = False
            self.play_pause_btn.setText("▶️")
        elif state == QMediaPlayer.PlaybackState.StoppedState:
            self.is_playing = False
            self.play_pause_btn.setText("▶️")
    
    def on_media_error(self, error, error_string):
        """Handle media player errors"""
        print(f"❌ QMediaPlayer error: {error} - {error_string}")
        if self.current_media_path:
            print("🔄 Switching to OpenCV fallback due to error")
            self.load_video_fallback(self.current_media_path)
    
    def update_time_display_native(self, position, duration):
        """Update time display for QMediaPlayer"""
        current_time = self.format_time(position / 1000)
        total_time = self.format_time(duration / 1000)
        self.time_label.setText(f"{current_time} / {total_time}")
    
    def update_time_display_fallback(self):
        """Update time display for fallback mode"""
        if self.total_frames == 0 or self.fps == 0:
            self.time_label.setText("00:00 / 00:00")
            return
        
        current_seconds = self.current_frame / self.fps
        total_seconds = self.total_frames / self.fps
        
        current_time = self.format_time(current_seconds)
        total_time = self.format_time(total_seconds)
        self.time_label.setText(f"{current_time} / {total_time}")
    
    def format_time(self, seconds):
        """Format seconds as MM:SS or HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"
    
    def show_error(self, message):
        """Display error message"""
        self.image_widget.setText(f"Error:\\n{message}")
        self.image_widget.setPixmap(QPixmap())
        self.display_stack.setCurrentWidget(self.image_widget)
        self.video_controls.setVisible(False)
        print(f"❌ Media preview error: {message}")
    
    def clear_preview(self):
        """Clear current preview"""
        self.stop_playback()
        
        self.current_media_path = None
        self.media_type = None
        self.video_controls.setVisible(False)
        
        if HAS_MULTIMEDIA:
            self.media_player.setSource(QUrl())
        
        self.image_widget.setText("No Media Loaded\\nDrag and drop a file here")
        self.image_widget.setPixmap(QPixmap())
        self.display_stack.setCurrentWidget(self.image_widget)
        
        # Clear fallback data
        self.video_frames = []
        self.current_previews = {}
        
        print("🗑️ Media preview cleared")
    
    def update_previews(self, previews):
        """Update preview with processed results"""
        self.current_previews = previews
        
        if self.current_pass in previews:
            self.load_media(previews[self.current_pass])
        elif 'original' in previews:
            self.load_media(previews['original'])
        elif previews:
            first_key = next(iter(previews))
            self.load_media(previews[first_key])
    
    def set_current_pass(self, pass_name):
        """Set current pass for preview"""
        self.current_pass = pass_name.lower()
        if self.current_previews and self.current_pass in self.current_previews:
            self.load_media(self.current_previews[self.current_pass])
    
    # Drag and drop support
    def dragEnterEvent(self, event):
        """Handle drag enter events"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event):
        """Handle drop events with intelligent file/folder detection"""
        urls = event.mimeData().urls()
        if not urls:
            return
        
        # Get the first dropped item
        first_path = urls[0].toLocalFile()
        
        # Check if it's a folder (for image sequences)
        if os.path.isdir(first_path):
            self.handle_folder_drop(first_path)
        else:
            # Handle file(s) drop
            self.handle_files_drop(urls)
    
    def handle_folder_drop(self, folder_path):
        """Handle folder drop for image sequences"""
        try:
            # Get all image files from the folder
            extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif', '.webp', '.exr', '.hdr', '.pic']
            files = []
            for ext in extensions:
                files.extend([str(p) for p in Path(folder_path).glob(f'*{ext}')])
            
            # Remove duplicates and sort
            files = sorted(list(set(files)))
            
            if files:
                print(f"📁 Folder dropped: {Path(folder_path).name} with {len(files)} images")
                # Emit signal with file list and type 'sequence'
                self.files_dropped.emit(files, 'sequence')
                # Load first image for preview
                self.load_media(files[0])
            else:
                self.show_error("No image files found in folder")
                
        except Exception as e:
            print(f"Error handling folder drop: {e}")
            self.show_error(f"Error loading folder: {str(e)}")
    
    def handle_files_drop(self, urls):
        """Handle file(s) drop with auto-detection"""
        try:
            files = [url.toLocalFile() for url in urls]
            
            if not files:
                return
            
            # Auto-detect file type from first file
            first_file = files[0]
            file_ext = Path(first_file).suffix.lower()
            
            # Define file types
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v']
            image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif', '.webp', '.exr', '.hdr', '.pic']
            
            if file_ext in video_extensions:
                # Video file(s) - only use first one
                file_type = 'video'
                files = [first_file]  # Only process first video
                print(f"🎬 Video dropped: {Path(first_file).name}")
            elif file_ext in image_extensions:
                # Image file(s)
                if len(files) > 1:
                    # Multiple images = sequence
                    file_type = 'sequence'
                    print(f"🖼️ Image sequence dropped: {len(files)} images")
                else:
                    # Single image
                    file_type = 'image'
                    print(f"🖼️ Single image dropped: {Path(first_file).name}")
            else:
                self.show_error(f"Unsupported file format: {file_ext}")
                return
            
            # Emit signal with detected type
            self.files_dropped.emit(files, file_type)
            
            # Load first file for preview
            self.load_media(first_file)
            
        except Exception as e:
            print(f"Error handling file drop: {e}")
            self.show_error(f"Error loading files: {str(e)}")

