"""
Creative Relight - Main Window
PyQt6 interface with advanced features
"""

import sys
import os
import webbrowser
from pathlib import Path
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import torch

# Lazy imports - these will be imported inside functions after sys.path is set up
# from src.processors... (imported in __init__)
# from processors... (imported in __init__)

# Import UI components
from ui.progress_panel import ProgressPanel
from ui.media_preview import MediaPreviewWidget
from ui.processing_worker import ProcessingWorker
from ui.loading_dialog import LoadingDialog, ModelInitWorker


def get_asset_path(relative_path):
    """
    Get absolute path to asset, works for dev and for PyInstaller frozen builds.
    
    Args:
        relative_path: Path relative to pyqt_app folder (e.g., "assets/icon.ico")
    
    Returns:
        Absolute path to the asset file
    """
    if getattr(sys, 'frozen', False):
        # Running as PyInstaller bundle
        # Assets are in pyqt_app/assets in the _MEIPASS extraction folder
        base_path = Path(sys._MEIPASS)
        asset_path = base_path / "pyqt_app" / relative_path
        if not asset_path.exists():
            # Fallback: try without pyqt_app prefix
            asset_path = base_path / relative_path
    else:
        # Running as script - __file__ is in pyqt_app/ui/main_window.py
        base_path = Path(__file__).parent.parent
        asset_path = base_path / relative_path
    
    return asset_path


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Set window properties
        self.setWindowTitle("Creative Relight v1.0.0")
        self.setMinimumSize(1400, 900)
        self.resize(1600, 1000)
        
        # Initialize processors
        self.image_processor = None
        self.video_processor = None
        self.sequence_processor = None
        
        # Detect device
        import torch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Variables
        self.current_files = []
        self.sequence_folder = None  # Store folder path for image sequences
        self.output_dir = None
        self.processing = False
        self.output_results = {}  # Cache for processed results from output directory
        self.current_video_frame_index = None  # Track current video frame for preview
        
        # Setup UI
        self.setup_ui()
        self.setup_menu_bar()
        self.setup_toolbar()
        self.setup_status_bar()
        
        # Apply custom styles
        self.apply_custom_styles()
        
        # Set application icon AFTER UI is ready (fixes taskbar icon loading issues)
        QTimer.singleShot(0, self.set_app_icon)
        
        # Load models into memory (models already downloaded by main.py startup)
        QTimer.singleShot(500, self.load_models_into_memory)
    
    def set_app_icon(self):
        """Set the application icon from ICO file"""
        try:
            # Windows: Set AppUserModelID to show icon in taskbar
            import sys
            if sys.platform == 'win32':
                try:
                    import ctypes
                    # Set AppUserModelID for Windows taskbar icon
                    myappid = 'CreativeTwins.CreativeRelight.VFXPasses.1.0'
                    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
                    print(f"[OK] AppUserModelID set: {myappid}")
                except Exception as e:
                    print(f"[WARNING] Could not set AppUserModelID: {e}")
            
            # Load Windows ICO file for best compatibility
            icon_path = get_asset_path("assets/icon_48x48.ico")
            
            if not icon_path.exists():
                print(f"[ERROR] Icon file not found: {icon_path}")
                print(f"   Please ensure icon file exists at: {icon_path}")
                return
            
            # Load icon
            icon = QIcon(str(icon_path))
            
            # Verify icon was loaded successfully
            if icon.isNull():
                print(f"[ERROR] Failed to load icon from: {icon_path}")
                print(f"   Icon file may be corrupted or invalid format")
                return
            
            # Set window icon FIRST
            self.setWindowIcon(icon)
            
            # Then set application-wide icon
            QApplication.setWindowIcon(icon)
            
            # Get available icon sizes for debugging
            sizes = icon.availableSizes()
            size_info = ", ".join([f"{s.width()}x{s.height()}" for s in sizes]) if sizes else "none"
            
            print(f"[OK] Application icon loaded successfully")
            print(f"   Path: {icon_path}")
            print(f"   File size: {icon_path.stat().st_size} bytes")
            print(f"   Available sizes: {size_info}")
            print(f"   Icon null check: {icon.isNull()}")
            
        except Exception as e:
            print(f"[ERROR] Error loading icon: {e}")
            import traceback
            traceback.print_exc()
    
    def setup_ui(self):
        """Setup the main user interface"""
        # Central widget with splitter layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Create main splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(main_splitter)
        
        # Left panel (File browser and settings)
        left_panel = self.create_left_panel()
        main_splitter.addWidget(left_panel)
        
        # Center panel (Image preview and processing)
        center_panel = self.create_center_panel()
        main_splitter.addWidget(center_panel)
        
        # Right panel (Progress and logs)
        right_panel = self.create_right_panel()
        main_splitter.addWidget(right_panel)
        
        # Set splitter proportions
        main_splitter.setSizes([350, 800, 450])
        main_splitter.setCollapsible(0, False)
        main_splitter.setCollapsible(1, False)
        main_splitter.setCollapsible(2, False)
    
    def create_left_panel(self):
        """Create left panel with unified file selection and output location tabs"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        
        # Create tab widget for organized sections
        tab_widget = QTabWidget()
        
        # File Selection Tab
        file_tab = QWidget()
        file_layout = QVBoxLayout(file_tab)
        
        # File type selection
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Process:"))
        
        self.file_type_combo = QComboBox()
        self.file_type_combo.addItems(["Single Image", "Video File", "Image Sequence"])
        self.file_type_combo.currentTextChanged.connect(self.on_file_type_changed)
        type_layout.addWidget(self.file_type_combo)
        type_layout.addStretch()
        
        file_layout.addLayout(type_layout)
        
        # Unified browse button
        browse_layout = QHBoxLayout()
        self.browse_btn = QPushButton("📂 Select Files")
        self.browse_btn.clicked.connect(self.browse_files)
        browse_layout.addWidget(self.browse_btn)
        browse_layout.addStretch()
        
        file_layout.addLayout(browse_layout)
        
        # Selected files preview
        self.files_list = QListWidget()
        self.files_list.setMaximumHeight(200)
        self.files_list.setAlternatingRowColors(True)
        file_layout.addWidget(QLabel("Selected Files:"))
        file_layout.addWidget(self.files_list)
        
        # Clear files button
        clear_layout = QHBoxLayout()
        self.clear_files_btn = QPushButton("🗑️ Clear Files")
        self.clear_files_btn.clicked.connect(self.clear_files)
        self.clear_files_btn.setEnabled(False)
        clear_layout.addWidget(self.clear_files_btn)
        clear_layout.addStretch()
        file_layout.addLayout(clear_layout)
        
        file_layout.addStretch()
        tab_widget.addTab(file_tab, "📁 Files")
        
        # Output Location Tab  
        output_tab = QWidget()
        output_layout = QVBoxLayout(output_tab)
        
        output_select_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output Directory:"))
        
        self.output_path_label = QLabel("No location selected")
        self.output_path_label.setStyleSheet("color: gray; font-style: italic; border: 1px solid #555; padding: 8px; border-radius: 4px; background-color: #3c3c3c;")
        self.output_path_label.setWordWrap(True)
        output_layout.addWidget(self.output_path_label)
        
        output_button_layout = QHBoxLayout()
        self.output_browse_btn = QPushButton("📁 Browse Folder")
        self.output_browse_btn.clicked.connect(self.select_output_folder)
        output_button_layout.addWidget(self.output_browse_btn)
        
        self.output_clear_btn = QPushButton("🗑️ Clear")
        self.output_clear_btn.clicked.connect(self.clear_output_folder)
        self.output_clear_btn.setEnabled(False)
        output_button_layout.addWidget(self.output_clear_btn)
        
        # Add refresh button to rescan output directory
        self.output_refresh_btn = QPushButton("Refresh")
        self.output_refresh_btn.clicked.connect(self.scan_output_results)
        self.output_refresh_btn.setEnabled(False)
        self.output_refresh_btn.setToolTip("Rescan output directory for processed results")
        output_button_layout.addWidget(self.output_refresh_btn)
        
        output_layout.addLayout(output_button_layout)
        output_layout.addStretch()
        
        tab_widget.addTab(output_tab, "💾 Output")
        
        layout.addWidget(tab_widget)
        
        return panel
    
    def create_center_panel(self):
        """Create center panel with image preview"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        
        # Header with title and device info
        header_layout = QHBoxLayout()
        
        # Title text (without emoji)
        title_label = QLabel("<b>Creative Relight</b>")
        title_label.setObjectName("titleLabel")
        header_layout.addWidget(title_label)
        
        # Lightbulb icon
        lightbulb_path = get_asset_path("assets/Lightbulb.png")
        if lightbulb_path.exists():
            lightbulb_label = QLabel()
            lightbulb_pixmap = QPixmap(str(lightbulb_path))
            # Scale to appropriate size (24x24 for subtle branding)
            scaled_pixmap = lightbulb_pixmap.scaled(24, 24, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            lightbulb_label.setPixmap(scaled_pixmap)
            lightbulb_label.setToolTip("Creative Relight")
            header_layout.addWidget(lightbulb_label)
        else:
            print(f"[WARNING] Lightbulb icon not found: {lightbulb_path}")
        
        # VFX Passes subtitle text
        vfx_passes_label = QLabel("VFX Passes")
        vfx_passes_label.setObjectName("vfxPassesLabel")
        header_layout.addWidget(vfx_passes_label)
        
        header_layout.addStretch()
        
        self.device_label = QLabel(f"Device: {self.device.upper()}")
        self.device_label.setObjectName("deviceLabel")
        header_layout.addWidget(self.device_label)
        
        layout.addLayout(header_layout)
        
        # Image Preview with Pass Selection
        preview_group = QGroupBox("Image Preview & Results")
        preview_layout = QVBoxLayout(preview_group)
        
        # Pass selection buttons
        pass_button_layout = QHBoxLayout()
        self.pass_buttons = {}
        
        # Create buttons for each pass type (Normal moved near Albedo)
        for pass_name in ['Original', 'Albedo', 'Normal', 'Specular', 'Depth']:
            btn = QPushButton(pass_name)
            btn.setCheckable(True)
            btn.setObjectName("passButton")
            btn.clicked.connect(lambda checked, name=pass_name: self.on_pass_selected(name))
            self.pass_buttons[pass_name.lower()] = btn
            pass_button_layout.addWidget(btn)
        
        # Set Original as default
        self.pass_buttons['original'].setChecked(True)
        self.current_pass = 'original'
        
        pass_button_layout.addStretch()
        preview_layout.addLayout(pass_button_layout)
        
        # Image/Video preview widget
        self.media_preview = MediaPreviewWidget()
        # Connect media preview frame changes to update timeline
        self.media_preview.frame_changed.connect(self.on_media_frame_changed)
        # Connect files dropped signal to handle drag-and-drop with auto-detection
        self.media_preview.files_dropped.connect(self.on_files_drag_dropped)
        preview_layout.addWidget(self.media_preview)
        
        # Timeline for image sequences (initially hidden)
        self.timeline_group = QGroupBox("Sequence Timeline")
        timeline_layout = QVBoxLayout(self.timeline_group)
        
        # Frame slider
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.valueChanged.connect(self.on_frame_changed)
        timeline_layout.addWidget(self.frame_slider)
        
        # Frame info
        frame_info_layout = QHBoxLayout()
        self.frame_label = QLabel("Frame: 0 / 0")
        frame_info_layout.addWidget(self.frame_label)
        frame_info_layout.addStretch()
        
        # Play controls
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.clicked.connect(self.toggle_playback)
        frame_info_layout.addWidget(self.play_btn)
        
        timeline_layout.addLayout(frame_info_layout)
        preview_layout.addWidget(self.timeline_group)
        
        # Hide timeline initially
        self.timeline_group.setVisible(False)
        self.is_playing = False
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self.next_frame)
        
        layout.addWidget(preview_group)
        
        # Processing Controls
        controls_group = QGroupBox("Processing Controls")
        controls_layout = QVBoxLayout(controls_group)
        
        # Process button
        button_layout = QHBoxLayout()
        
        self.process_btn = QPushButton("Start Processing")
        self.process_btn.setObjectName("processButton")
        self.process_btn.setMinimumHeight(50)
        self.process_btn.clicked.connect(self.start_processing)
        button_layout.addWidget(self.process_btn)
        
        self.stop_btn = QPushButton("🚫 Stop")
        self.stop_btn.setObjectName("stopButton")
        self.stop_btn.setMinimumHeight(50)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_processing)
        button_layout.addWidget(self.stop_btn)
        
        controls_layout.addLayout(button_layout)
        
        layout.addWidget(controls_group)
        
        return panel
    
    def create_right_panel(self):
        """Create right panel with processing settings and progress"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        
        # Processing Settings Section
        settings_group = QGroupBox("⚙️ Processing Settings")
        settings_layout = QVBoxLayout(settings_group)
        
        # Create tab widget for settings
        settings_tabs = QTabWidget()
        
        # General tab with Export Options
        general_tab = QWidget()
        general_layout = QVBoxLayout(general_tab)
        
        # Export Options (moved from left panel)
        general_layout.addWidget(QLabel("Export Components:"))
        
        # Export checkboxes
        self.export_checkboxes = {}
        for component in ['Albedo', 'Specular', 'Depth', 'Normal']:
            checkbox = QCheckBox(f"Export {component}")
            checkbox.setChecked(True)
            self.export_checkboxes[component.lower()] = checkbox
            general_layout.addWidget(checkbox)
        
        # Add spacing before Temporal Consistency section
        general_layout.addSpacing(20)
        
        # Temporal Consistency Section
        temporal_label = QLabel("Temporal Consistency")
        temporal_label.setStyleSheet("font-weight: bold; font-size: 11pt;")
        general_layout.addWidget(temporal_label)
        
        # Enable temporal smoothing checkbox
        self.enable_temporal = QCheckBox("Enable Temporal Smoothing")
        self.enable_temporal.setChecked(False)  # Unchecked by default - user enables if needed
        self.enable_temporal.setToolTip("Reduces flickering between frames in videos and image sequences")
        general_layout.addWidget(self.enable_temporal)
        
        # Smoothing strength with help button
        strength_header_layout = QHBoxLayout()
        strength_label = QLabel("Smoothing Strength:")
        
        help_button = QPushButton("?")
        help_button.setMaximumSize(20, 20)
        help_button.setStyleSheet("""
            QPushButton {
                background-color: #4a90e2;
                color: white;
                border-radius: 10px;
                font-weight: bold;
                font-size: 11px;
                border: none;
            }
            QPushButton:hover {
                background-color: #357abd;
            }
        """)
        help_button.setToolTip(
            "Click for more information about Smoothing Strength"
        )
        help_button.setCursor(Qt.CursorShape.WhatsThisCursor)
        help_button.clicked.connect(self.show_temporal_help)
        
        strength_header_layout.addWidget(strength_label)
        strength_header_layout.addWidget(help_button)
        strength_header_layout.addStretch()
        general_layout.addLayout(strength_header_layout)
        
        # Strength slider with value label
        strength_slider_layout = QHBoxLayout()
        self.temporal_strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.temporal_strength_slider.setRange(1, 10)
        self.temporal_strength_slider.setValue(3)
        self.temporal_strength_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.temporal_strength_slider.setTickInterval(1)
        
        self.temporal_strength_label = QLabel("3")
        self.temporal_strength_label.setMinimumWidth(30)
        self.temporal_strength_label.setStyleSheet("font-weight: bold; color: #4a90e2;")
        
        # Connect slider to update label
        self.temporal_strength_slider.valueChanged.connect(
            lambda v: self.temporal_strength_label.setText(str(v))
        )
        
        # Connect enable checkbox to toggle slider
        self.enable_temporal.toggled.connect(
            lambda enabled: self.temporal_strength_slider.setEnabled(enabled)
        )
        self.enable_temporal.toggled.connect(
            lambda enabled: self.temporal_strength_label.setEnabled(enabled)
        )
        
        strength_slider_layout.addWidget(self.temporal_strength_slider)
        strength_slider_layout.addWidget(self.temporal_strength_label)
        general_layout.addLayout(strength_slider_layout)
        
        general_layout.addStretch()
        settings_tabs.addTab(general_tab, "General")
        
        settings_layout.addWidget(settings_tabs)
        layout.addWidget(settings_group)
        
        # Progress Section
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_panel = ProgressPanel()
        progress_layout.addWidget(self.progress_panel)
        
        layout.addWidget(progress_group)
        
        return panel
    
    def setup_menu_bar(self):
        """Setup application menu bar"""
        menubar = self.menuBar()
        
        # File Menu
        file_menu = menubar.addMenu('&File')
        
        # Open action
        open_action = QAction('&Open Files...', self)
        open_action.setShortcut('Ctrl+O')
        open_action.setStatusTip('Open image files for processing')
        open_action.triggered.connect(self.browse_files)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction('E&xit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.setStatusTip('Exit application')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Help Menu
        help_menu = menubar.addMenu('&Help')
        
        about_action = QAction('&About', self)
        about_action.setStatusTip('About Creative Relight')
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def setup_toolbar(self):
        """Setup application toolbar with links on the right"""
        toolbar = self.addToolBar('Links')
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        
        # Add spacer to push buttons to the right
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)
        
        # Patreon with logo icon
        patreon_icon_path = get_asset_path("assets/patreon_icon_24x24.png")
        if patreon_icon_path.exists():
            patreon_icon = QIcon(str(patreon_icon_path))
        else:
            patreon_icon = QIcon()
            print(f"[WARNING] Patreon icon not found: {patreon_icon_path}")
        patreon_action = QAction(patreon_icon, 'Patreon', self)
        patreon_action.setStatusTip('Support us on Patreon')
        patreon_action.triggered.connect(lambda: self.open_url('https://www.patreon.com/c/CreativeTwins'))
        toolbar.addAction(patreon_action)
        
        # Website
        website_action = QAction(QIcon(), '🌐 Website', self)
        website_action.setStatusTip('Visit our website')
        website_action.triggered.connect(lambda: self.open_url('https://www.creative-twins.com/'))
        toolbar.addAction(website_action)
    
    def setup_status_bar(self):
        """Setup status bar"""
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")
        
        # Add permanent widgets
        self.file_count_label = QLabel("Files: 0")
        self.status_bar.addPermanentWidget(self.file_count_label)
        
        self.memory_label = QLabel("Memory: --")
        self.status_bar.addPermanentWidget(self.memory_label)
    
    def apply_custom_styles(self):
        """Apply custom CSS styles"""
        style = """
            QMainWindow {
                background-color: #2b2b2b;
            }
            
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555;
                border-radius: 8px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            
            #titleLabel {
                font-size: 24px;
                font-weight: bold;
                color: #ffffff;
                margin: 10px 0;
            }
            
            #vfxPassesLabel {
                font-size: 16px;
                color: #999999;
                margin: 10px 0;
                margin-left: 5px;
            }
            
            #deviceLabel {
                font-size: 14px;
                color: #4caf50;
                font-weight: bold;
            }
            
            #processButton {
                background-color: #42a5f5;
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 16px;
                font-weight: bold;
            }
            
            #processButton:hover {
                background-color: #1976d2;
            }
            
            #processButton:pressed {
                background-color: #0d47a1;
            }
            
            #stopButton {
                background-color: #ffffff;
                color: #000000;
                border: none;
                border-radius: 6px;
                font-size: 16px;
                font-weight: bold;
            }
            
            #stopButton:hover {
                background-color: #cccccc;
                color: #000000;
            }
            
            #stopButton:pressed {
                background-color: #999999;
                color: #000000;
            }
            
            #stopButton:disabled {
                background-color: #f0f0f0;
                color: #999999;
                border: 1px solid #ddd;
            }
            
            QPushButton#passButton {
                background-color: #4a4a4a;
                color: white;
                border: 2px solid #666;
                border-radius: 6px;
                font-size: 12px;
                font-weight: bold;
                padding: 8px 16px;
                margin: 2px;
            }
            
            QPushButton#passButton:hover {
                background-color: #5a5a5a;
                border-color: #42a5f5;
                color: #42a5f5;
            }
            
            QPushButton#passButton:checked {
                background-color: #42a5f5;
                border-color: #1976d2;
                color: white;
            }
            
            QPushButton#passButton:checked:hover {
                background-color: #1976d2;
                border-color: #0d47a1;
            }
            
            QProgressBar {
                border: 2px solid #555;
                border-radius: 5px;
                text-align: center;
                background-color: #3c3c3c;
                color: #FFFFFF;
                font-weight: bold;
            }
            
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
        """
        self.setStyleSheet(style)
    
    def initialize_models(self):
        """Initialize AI models in background with loading dialog (Cloud Edition - checks/downloads models first)"""
        # CLOUD EDITION: Check if models exist, download if missing
        from pathlib import Path
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        
        try:
            # Configure PyTorch Hub cache location FIRST (before any model loading)
            from src.utils.paths import configure_pytorch_hub_cache
            configure_pytorch_hub_cache()
            
            from src.utils.model_checker import ModelChecker
            
            # Check model status
            checker = ModelChecker()
            status = checker.get_models_status()
            
            if not status['all_exist']:
                # Models missing - show download dialog
                print(f"[CLOUD] Missing {status['missing_files']} model files, starting download...")
                self.show_model_download_dialog(checker, status)
            else:
                # Models exist - proceed to initialize
                print(f"[CLOUD] All models present ({status['existing_files']} files)")
                self.load_models_into_memory()
                
        except Exception as e:
            print(f"[CLOUD] Error checking models: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to regular initialization
            self.load_models_into_memory()
    
    def show_model_download_dialog(self, checker, status):
        """Show dialog to download missing models"""
        from PyQt6.QtCore import QThread, pyqtSignal
        from .download_dialog import ModelDownloadDialog
        
        # Show download dialog
        download_dialog = ModelDownloadDialog(self)
        
        # Ensure dialog is visible and on top
        download_dialog.show()
        download_dialog.activateWindow()
        download_dialog.raise_()
        QApplication.processEvents()  # Force UI update
        
        # Download in background thread
        class DownloadWorker(QThread):
            progress_update = pyqtSignal(str, int, int)
            finished = pyqtSignal(bool, str)
            
            def __init__(self, checker):
                super().__init__()
                self.checker = checker
                
            def run(self):
                def progress_callback(file_name, downloaded, total):
                    self.progress_update.emit(file_name, downloaded, total)
                
                try:
                    success, message = self.checker.download_missing_models(progress_callback)
                    self.finished.emit(success, message)
                except Exception as e:
                    self.finished.emit(False, str(e))
        
        self.download_worker = DownloadWorker(checker)
        
        def on_download_progress(file_name, downloaded, total):
            download_dialog.update_progress(file_name, downloaded, total)
        
        def on_download_finished(success, message):
            if success:
                download_dialog.set_completed()
                # Small delay to show completion
                QTimer.singleShot(1000, lambda: [
                    download_dialog.close(),
                    self.load_models_into_memory()
                ])
            else:
                download_dialog.close()
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.critical(self, "Download Failed", 
                    "Failed to download AI models.\n\nPlease check your internet connection and try again.")
                sys.exit(1)
        
        self.download_worker.progress_update.connect(on_download_progress)
        self.download_worker.finished.connect(on_download_finished)
        self.download_worker.start()
        
        # Show dialog modally
        download_dialog.exec()
    
    def load_models_into_memory(self):
        """Load models into memory (after ensuring they exist)"""
        from .splash_screen import SplashScreen
        
        # Show splash screen
        self.splash = SplashScreen()
        self.splash.show()
        
        # Create worker thread for model initialization
        self.model_init_worker = ModelInitWorker(self)
        
        # Connect signals
        self.model_init_worker.progress.connect(self.on_model_init_progress_splash)
        self.model_init_worker.finished.connect(self.on_model_init_finished)
        
        # Start worker
        self.model_init_worker.start()
    
    def on_model_init_progress_splash(self, message, percentage=None):
        """Handle model initialization progress updates for splash screen"""
        if hasattr(self, 'splash'):
            # Use the percentage if provided, otherwise keep existing logic
            if percentage is not None:
                self.splash.update_status(message, percentage)
            else:
                # Fallback to old behavior for backward compatibility
                if "Loading Albedo" in message or "lighting model" in message.lower():
                    self.splash.update_status("Loading Lighting Models...", 30)
                elif "Loading Video" in message or "video processor" in message.lower():
                    self.splash.update_status("Initializing Video Processor...", 50)
                elif "Loading Sequence" in message or "sequence processor" in message.lower():
                    self.splash.update_status("Initializing Sequence Processor...", 70)
                elif "geometry" in message.lower():
                    self.splash.update_status("Loading Geometry Models...", 85)
                elif "Initializing models on" in message or "Accelerating" in message:
                    self.splash.update_status("Accelerating Hardware...", 95)
                else:
                    self.splash.update_status(message, None)
        
        print(f"[INIT] {message}")
    
    def on_model_init_progress(self, message):
        """Handle model initialization progress updates"""
        if hasattr(self, 'loading_dialog'):
            self.loading_dialog.update_message(message)
        print(f"[INIT] {message}")
    
    def on_model_init_finished(self, success, error_message):
        """Handle model initialization completion"""
        # Close splash screen with fade effect
        if hasattr(self, 'splash'):
            self.splash.update_status("Ready!", 100)
            # Small delay before closing
            QTimer.singleShot(500, lambda: self.splash.close() if hasattr(self, 'splash') else None)
        
        if success:
            self.progress_panel.add_status_message("[OK] Models initialized successfully!")
            print("[OK] All AI models loaded and ready")
        else:
            # Close splash immediately on error
            if hasattr(self, 'splash'):
                self.splash.close()
            
            self.progress_panel.add_status_message(f"[ERROR] {error_message}")
            print(f"[ERROR] Model initialization failed: {error_message}")
            
            # Show error dialog to user
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(
                self,
                "Initialization Failed",
                "Failed to initialize AI models.\n\nThe application may not function correctly."
            )
    
    def on_files_selected(self, files):
        """Handle file selection from browser"""
        self.current_files = files
        self.file_count_label.setText(f"Files: {len(files)}")
        
        if files:
            # Show first image in preview
            self.media_preview.load_media(files[0])
            self.progress_panel.add_status_message(f"Selected {len(files)} file(s)")
    
    def on_files_drag_dropped(self, files, file_type):
        """Handle files/folder drag-and-drop with auto-detection and mode switching"""
        if not files:
            return
        
        # Auto-switch processing mode based on detected file type
        if file_type == 'video':
            self.file_type_combo.setCurrentText("Video File")
            self.current_files = files  # Only one video file
            message = f"Video file loaded: {Path(files[0]).name}"
        elif file_type == 'sequence':
            self.file_type_combo.setCurrentText("Image Sequence")
            self.current_files = files
            # Store folder path if available
            if files:
                self.sequence_folder = str(Path(files[0]).parent)
            message = f"Image sequence loaded: {len(files)} images"
            # Update timeline for sequence
            self.update_timeline_for_sequence()
        elif file_type == 'image':
            self.file_type_combo.setCurrentText("Single Image")
            self.current_files = files
            message = f"Image loaded: {Path(files[0]).name}"
        else:
            return
        
        # Update UI
        self.file_count_label.setText(f"Files: {len(files)}")
        self.update_files_list()
        self.clear_files_btn.setEnabled(True)
        
        # Show success message
        self.progress_panel.add_status_message(f"[OK] {message}")
        print(f"[DROP] Drag-and-drop: {message}, mode auto-switched to {self.file_type_combo.currentText()}")
    
    def start_processing(self):
        """Start processing selected files"""
        if not self.current_files:
            QMessageBox.warning(self, "Warning", "Please select files to process first.")
            return
        
        # Get export config from export checkboxes
        export_config = self.get_export_settings()
        
        if not any(export_config.values()):
            QMessageBox.warning(self, "Warning", "Please select at least one export option.")
            return
        
        # Check if output directory is set
        if not self.output_dir:
            QMessageBox.warning(self, "Warning", "Please select an output directory first.")
            return
        
        # Start processing
        self.processing = True
        self.process_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        # Initialize progress tracking
        total_files = len(self.current_files)
        self.progress_panel.start_progress(total_files)
        self.progress_panel.add_status_message(f"Starting processing of {total_files} files...")
        
        print(f"[VIDEO] Starting processing: {total_files} files, output: {self.output_dir}")
        print(f"[PREVIEW] Export config: {export_config}")
        
        # Start actual processing in a separate thread to prevent UI freezing
        self.start_background_processing(export_config, self.output_dir)
    
    def stop_processing(self):
        """Stop current processing"""
        self.processing = False
        
        # Stop the worker if it exists
        if hasattr(self, 'worker'):
            self.worker.stop()
        
        # Stop all processors - set their stop flags
        if self.image_processor:
            self.image_processor.should_stop = True
        if self.video_processor:
            self.video_processor.should_stop = True
        if self.sequence_processor:
            self.sequence_processor.stop()
        
        # Try to terminate the worker thread gracefully
        if hasattr(self, 'worker_thread') and self.worker_thread.isRunning():
            # First try to quit gracefully
            self.worker_thread.quit()
            
            # Wait for thread to finish (with timeout)
            if not self.worker_thread.wait(2000):  # 2 second timeout
                # If it doesn't finish, terminate it forcefully
                self.progress_panel.add_status_message("Forcefully terminating processing thread...")
                self.worker_thread.terminate()
                self.worker_thread.wait()  # Wait for termination to complete
        
        # Update UI immediately
        self.process_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        self.progress_panel.stop_progress()
        self.progress_panel.add_status_message("Processing stopped by user")
        
        # Stop any periodic timers
        self.stop_periodic_preview_refresh()
        
        # DON'T reset the timeline - keep the frame count visible
        # The timeline will show the last processed frame count
    def process_files(self, export_config, output_dir):
        """Process selected files with actual AI models"""
        if not self.image_processor:
            self.progress_panel.add_status_message("Models not initialized yet. Please wait...")
            return
        
        try:
            total_files = len(self.current_files)
            processed_files = 0
            
            # Determine processing type based on file type selection
            file_type = self.file_type_combo.currentText()
            
            for i, file_path in enumerate(self.current_files):
                if not self.processing:
                    break
                
                # Update progress
                from pathlib import Path
                file_name = Path(file_path).name
                self.progress_panel.set_task_progress(0, f"Processing {file_name}")
                
                # Route to correct processor based on file type selection and file extension
                processor = self.get_processor_for_file(file_path, file_type)
                if not processor:
                    self.progress_panel.add_status_message(f"No suitable processor for {file_name}")
                    continue
                
                # Process the file with the correct processor
                result = processor.process(
                    file_path,
                    output_dir,
                    export_config=export_config
                )
                
                processed_files += 1
                overall_progress = int((processed_files / total_files) * 100)
                self.progress_panel.set_overall_progress(overall_progress)
                
                # Update preview with results
                if result and 'previews' in result:
                    self.media_preview.update_previews(result['previews'])
                
                self.progress_panel.add_status_message(f"Processed {file_name}")
            
            if self.processing:
                self.progress_panel.complete_progress()
                self.progress_panel.add_status_message("All files processed successfully!")
            
        except Exception as e:
            self.progress_panel.add_status_message(f"Error processing files: {str(e)}")
            print(f"Processing error: {e}")
        
        finally:
            self.processing = False
            self.process_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
    
    def get_processor_for_file(self, file_path, file_type):
        """Get the appropriate processor for a file based on type and extension"""
        from pathlib import Path
        file_ext = Path(file_path).suffix.lower()
        
        # Video extensions
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
        # Image extensions (incl. HDR formats)
        image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif', '.webp', '.exr', '.hdr', '.pic']
        
        # If file type is explicitly set, respect that choice
        if file_type == "Video File":
            if file_ext in video_extensions:
                return self.video_processor
            else:
                self.progress_panel.add_status_message(f"Warning: {Path(file_path).name} is not a video file but Video File mode is selected")
                return None
                
        elif file_type == "Image Sequence":
            # For image sequences, use sequence processor regardless of individual file type
            return self.sequence_processor
            
        elif file_type == "Single Image":
            if file_ext in image_extensions:
                return self.image_processor
            elif file_ext in ('.exr', '.hdr', '.pic'):
                return self.image_processor
            else:
                self.progress_panel.add_status_message(f"Warning: {Path(file_path).name} is not an image file but Single Image mode is selected")
                return None
        
        # Auto-detect based on file extension if type is unclear
        if file_ext in video_extensions:
            return self.video_processor
        elif file_ext in image_extensions:
            return self.image_processor
        else:
            return None
    
    def open_files(self):
        """Open file dialog to select files"""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Images",
            "",
            "Image Files (*.png *.jpg *.jpeg *.tiff *.tif *.bmp *.exr *.hdr);;HDR Files (*.exr *.hdr);;All Files (*)"
        )
        
        if files:
            self.current_files = files
            self.progress_panel.add_status_message(f"Selected {len(files)} file(s)")
            self.file_count_label.setText(f"Files: {len(files)}")
    
    def open_folder(self):
        """Open folder dialog and set as image sequence"""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            # Set file type to Image Sequence
            self.file_type_combo.setCurrentText("Image Sequence")
            
            # Store the folder path for image sequences
            self.sequence_folder = folder
            
            # Get all image files from the folder
            from pathlib import Path
            extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif', '.exr', '.hdr']
            files = []
            for ext in extensions:
                files.extend([str(p) for p in Path(folder).glob(f'*{ext}')])
                files.extend([str(p) for p in Path(folder).glob(f'*{ext.upper()}')])
            files.sort()  # Sort for proper sequence order
            
            if files:
                self.current_files = files
                self.update_files_list()
                self.file_count_label.setText(f"Files: {len(files)}")
                self.clear_files_btn.setEnabled(True)
                
                # Show first image in preview
                self.media_preview.load_media(files[0])
                self.progress_panel.add_status_message(f"Loaded {len(files)} files from folder")
    
    def open_settings(self):
        """Open settings dialog"""
        QMessageBox.information(self, "Settings", "Settings dialog would open here.")
    
    def show_temporal_help(self):
        """Show help dialog for temporal smoothing strength"""
        msg = QMessageBox(self)
        msg.setWindowTitle("Smoothing Strength Help")
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setText("Smoothing Strength")
        msg.setInformativeText(
            "Controls the number of adjacent frames blended for temporal consistency.\n\n"
            "Lower values (1-3): Preserve more frame detail but may show slight flickering.\n\n"
            "Medium values (3-5): Balanced approach, recommended for most videos.\n\n"
            "Higher values (6-10): Create smoother video but may reduce sharpness in fast motion or cause ghosting.\n\n"
            "Tip: Start with 3 and adjust based on your footage."
        )
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()
    
    def show_about(self):
        """Show about dialog with clickable links"""
        dialog = QDialog(self)
        dialog.setWindowTitle("About Creative Relight")
        dialog.setFixedSize(450, 250)
        
        layout = QVBoxLayout()
        layout.setSpacing(15)
        
        # App name and version
        title_label = QLabel("Creative Relight v1.0.0")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # Description
        desc_label = QLabel(
            "Advanced AI-powered image processing for albedo,\n"
            "specular, depth, and normal extraction."
        )
        desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc_label.setStyleSheet("margin: 10px 0;")
        layout.addWidget(desc_label)
        
        # Copyright
        copyright_label = QLabel("© 2025 Creative Twins")
        copyright_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(copyright_label)
        
        # Licensing Agreement link
        license_label = QLabel('<a href="https://www.creative-twins.com/licenses" style="color: #2a82da;">Licensing Agreement</a>')
        license_label.setOpenExternalLinks(True)
        license_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        license_label.setStyleSheet("margin: 10px 0;")
        layout.addWidget(license_label)
        
        # OK button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        ok_button = QPushButton("OK")
        ok_button.setFixedWidth(80)
        ok_button.clicked.connect(dialog.accept)
        button_layout.addWidget(ok_button)
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        dialog.setLayout(layout)
        dialog.exec()
    
    def open_url(self, url):
        """Open URL in default browser"""
        import webbrowser
        try:
            webbrowser.open(url)
            self.progress_panel.add_status_message(f"🌐 Opening: {url}")
        except Exception as e:
            self.progress_panel.add_status_message(f"Error opening URL: {e}")
    
    def browse_files(self):
        """Browse and select files based on the current file type"""
        file_type = self.file_type_combo.currentText()
        
        if file_type == "Single Image":
            files, _ = QFileDialog.getOpenFileNames(
                self,
                "Select Image Files",
                "",
                "Image Files (*.png *.jpg *.jpeg *.tiff *.tif *.bmp *.gif *.webp *.exr *.hdr);;HDR Files (*.exr *.hdr *.pic);;All Files (*)"
            )
        elif file_type == "Video File":
            files, _ = QFileDialog.getOpenFileNames(
                self,
                "Select Video Files", 
                "",
                "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm);;All Files (*)"
            )
        elif file_type == "Image Sequence":
            folder = QFileDialog.getExistingDirectory(
                self,
                "Select Folder with Image Sequence"
            )
            if folder:
                # Store the folder path for image sequences
                self.sequence_folder = folder
                
                # Get all image files from the folder
                from pathlib import Path
                extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif', '.webp', '.exr', '.hdr', '.pic']
                files = []
                for ext in extensions:
                    files.extend([str(p) for p in Path(folder).glob(f'*{ext}')])
                    # Don't search for uppercase - modern filesystems are case-sensitive enough
                    # and this was causing duplicates
                    # files.extend([str(p) for p in Path(folder).glob(f'*{ext.upper()}')])
                
                # Remove duplicates and sort
                files = sorted(list(set(files)))
                
                # Debug output to see what files are found
                print(f"DEBUG: Found {len(files)} files in folder {folder}")
                if files:
                    print(f"DEBUG: First file: {Path(files[0]).name}")
                    print(f"DEBUG: Last file: {Path(files[-1]).name}")
                    print(f"DEBUG: Sample of files: {[Path(f).name for f in files[:5]]} ... {[Path(f).name for f in files[-5:]]}")
            else:
                files = []
        else:
            files = []
        
        if files:
            # Validate file types match selection
            valid_files = []
            invalid_files = []
            
            for file_path in files:
                detected_type = self.detect_file_type(file_path)
                if file_type == "Single Image" and detected_type == 'image':
                    valid_files.append(file_path)
                elif file_type == "Video File" and detected_type == 'video':
                    valid_files.append(file_path)
                elif file_type == "Image Sequence" and detected_type == 'image':
                    valid_files.append(file_path)
                else:
                    invalid_files.append(file_path)
            
            if invalid_files:
                from pathlib import Path
                invalid_names = [Path(f).name for f in invalid_files]
                QMessageBox.warning(
                    self, 
                    "Invalid File Types", 
                    f"The following files don't match the selected type '{file_type}':\n\n" + 
                    "\n".join(invalid_names) + 
                    f"\n\nOnly {len(valid_files)} valid files will be loaded."
                )
            
            if valid_files:
                self.current_files = valid_files
                self.update_files_list()
                self.file_count_label.setText(f"Files: {len(valid_files)}")
                self.clear_files_btn.setEnabled(True)
                
                # Update timeline for sequences
                self.update_timeline_for_sequence()
                
                # Show first file in preview
                self.media_preview.load_media(valid_files[0])
                self.progress_panel.add_status_message(f"Selected {len(valid_files)} valid file(s) for {file_type}")
            else:
                QMessageBox.warning(self, "No Valid Files", f"No files match the selected type '{file_type}'.")
        else:
            self.progress_panel.add_status_message("No files selected")

    def update_files_list(self):
        """Update the files list widget with current files"""
        self.files_list.clear()
        
        for file_path in self.current_files:
            from pathlib import Path
            file_name = Path(file_path).name
            item = QListWidgetItem(file_name)
            item.setToolTip(file_path)  # Full path on hover
            self.files_list.addItem(item)

    def clear_files(self):
        """Clear all selected files"""
        self.current_files = []
        self.files_list.clear()
        self.file_count_label.setText("Files: 0")
        self.clear_files_btn.setEnabled(False)
        self.media_preview.clear_preview()
        self.progress_panel.add_status_message("Cleared all files")

    def on_file_type_changed(self, file_type):
        """Handle file type selection change"""
        # Clear current files when changing type
        if self.current_files:
            self.clear_files()
        
        # Update browse button text
        if file_type == "Single Image":
            self.browse_btn.setText("📂 Select Images")
        elif file_type == "Video File":
            self.browse_btn.setText("📂 Select Videos") 
        elif file_type == "Image Sequence":
            self.browse_btn.setText("📂 Select Sequence Folder")

    def select_output_folder(self):
        """Select output folder for processed files"""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory", 
            "./output"
        )
        
        if folder:
            self.output_dir = folder
            # Truncate long paths for display
            display_path = folder
            if len(display_path) > 50:
                display_path = "..." + display_path[-47:]
            
            self.output_path_label.setText(display_path)
            self.output_path_label.setStyleSheet("color: #4caf50; font-weight: bold; border: 1px solid #4caf50; padding: 8px; border-radius: 4px; background-color: #2e2e2e;")
            self.output_path_label.setToolTip(folder)  # Full path on hover
            self.output_clear_btn.setEnabled(True)
            self.output_refresh_btn.setEnabled(True)
            self.progress_panel.add_status_message(f"Output folder set: {folder}")
            
            # Create output subfolders for each pass type (if they don't exist)
            self.create_output_folders()
            
            # Scan for existing processed results
            self.scan_output_results()

    def clear_output_folder(self):
        """Clear the selected output folder"""
        self.output_dir = None
        self.output_path_label.setText("No location selected")
        self.output_path_label.setStyleSheet("color: gray; font-style: italic; border: 1px solid #555; padding: 8px; border-radius: 4px; background-color: #3c3c3c;")
        self.output_path_label.setToolTip("")
        self.output_clear_btn.setEnabled(False)
        self.output_refresh_btn.setEnabled(False)
        self.progress_panel.add_status_message("Cleared output folder")
        
        # Clear cached results
        self.output_results = {}

    def create_output_folders(self):
        """Create output subfolders for each pass type"""
        if not self.output_dir:
            return
            
        try:
            output_path = Path(self.output_dir)
            
            # Create folders for each pass type
            for pass_type in ['albedo', 'normal', 'specular', 'depth']:
                folder_path = output_path / pass_type
                folder_path.mkdir(exist_ok=True)
                print(f"📁 Ensured output folder exists: {folder_path}")
            
            self.progress_panel.add_status_message("Output folders ready")
            
        except Exception as e:
            self.progress_panel.add_status_message(f"Error creating output folders: {str(e)}")
            print(f"Error creating output folders: {e}")

    def scan_output_results(self):
        """Scan output directory for existing processed results"""
        if not self.output_dir:
            return
            
        try:
            output_path = Path(self.output_dir)
            self.output_results = {}
            
            # Image extensions to look for
            image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp']
            
            # Scan each pass folder
            for pass_type in ['albedo', 'normal', 'specular', 'depth']:
                pass_folder = output_path / pass_type
                if pass_folder.exists():
                    # Find all image files in this pass folder
                    files = []
                    for ext in image_extensions:
                        # Only search for lowercase extensions (glob is case-insensitive on Windows)
                        found_files = list(pass_folder.glob(f'*{ext}'))
                        files.extend(found_files)
                    
                    # Remove duplicates using set() and sort
                    files = sorted(list(set([str(f) for f in files])))
                    
                    if files:
                        self.output_results[pass_type] = files
                        print(f"📁 Found {len(files)} {pass_type} results in output folder")
            
            # Update preview if we found results
            if self.output_results:
                total_results = sum(len(files) for files in self.output_results.values())
                self.progress_panel.add_status_message(f"Found {total_results} results across {len(self.output_results)} pass types")
                # Update current pass preview if applicable
                self.update_preview_from_output()
            else:
                self.progress_panel.add_status_message("No existing results found in output folder")
                
        except Exception as e:
            self.progress_panel.add_status_message(f"Error scanning output results: {str(e)}")
            print(f"Error scanning output results: {e}")

    def update_preview_from_output(self):
        """Update preview to show results from output directory"""
        if not hasattr(self, 'output_results') or not self.output_results:
            return
            
        current_pass = self.current_pass
        
        # Skip 'original' pass - show actual file instead
        if current_pass == 'original':
            if self.current_files:
                if self.file_type_combo.currentText() == "Image Sequence" and hasattr(self, 'current_frame_index'):
                    if self.current_frame_index < len(self.current_files):
                        self.media_preview.load_media(self.current_files[self.current_frame_index])
                else:
                    self.media_preview.load_media(self.current_files[0])
            return
        
        # Check if we have results for the current pass
        if current_pass in self.output_results:
            result_files = self.output_results[current_pass]
            
            if result_files:
                file_type = self.file_type_combo.currentText()
                
                if file_type == "Video File":
                    # For videos, show the latest processed frame or specific frame if available
                    if hasattr(self, 'current_video_frame_index') and self.current_video_frame_index is not None:
                        # Try to find the frame that matches the current video position
                        target_frame = self.current_video_frame_index
                        frame_file = None
                        
                        # Look for specific frame pattern: {pass}_{frame:06d}.png
                        for result_file in result_files:
                            result_path = Path(result_file)
                            if f"{current_pass}_{target_frame:06d}" in result_path.stem:
                                frame_file = result_file
                                break
                        
                        if frame_file:
                            self.media_preview.load_media(frame_file)
                            print(f"[PREVIEW] Loaded {current_pass} result for video frame {target_frame}: {Path(frame_file).name}")
                        else:
                            # Fallback to latest frame if specific frame not found
                            latest_frame = result_files[-1]
                            self.media_preview.load_media(latest_frame)
                            print(f"[PREVIEW] Loaded latest {current_pass} result: {Path(latest_frame).name}")
                    else:
                        # Show latest processed frame
                        latest_frame = result_files[-1]
                        self.media_preview.load_media(latest_frame)
                        print(f"[PREVIEW] Loaded latest {current_pass} result: {Path(latest_frame).name}")
                        
                elif file_type == "Image Sequence" and hasattr(self, 'current_frame_index'):
                    # For image sequences, try to match the current frame
                    if self.current_files and self.current_frame_index < len(self.current_files):
                        current_file = Path(self.current_files[self.current_frame_index])
                        current_stem = current_file.stem
                        
                        # Look for matching result file
                        matching_result = None
                        for result_file in result_files:
                            result_path = Path(result_file)
                            if current_stem in result_path.stem:
                                matching_result = result_file
                                break
                        
                        if matching_result:
                            self.media_preview.load_media(matching_result)
                            print(f"[PREVIEW] Loaded {current_pass} result for frame {self.current_frame_index + 1}: {Path(matching_result).name}")
                        else:
                            # Fallback to indexed result if no exact match
                            if self.current_frame_index < len(result_files):
                                self.media_preview.load_media(result_files[self.current_frame_index])
                            else:
                                self.media_preview.load_media(result_files[0])
                else:
                    # For single images, show first result
                    self.media_preview.load_media(result_files[0])
                    print(f"[PREVIEW] Loaded {current_pass} result: {Path(result_files[0]).name}")
        else:
            print(f"No {current_pass} results found in output directory")
            self.progress_panel.add_status_message(f"No {current_pass} results found in output directory")

    def on_pass_selected(self, pass_name):
        """Handle pass selection for preview - supports playback/scrubbing for all passes"""
        try:
            print(f"\n[PREVIEW] === on_pass_selected called for: {pass_name} ===")
            print(f"[PREVIEW] File type: {self.file_type_combo.currentText()}")
            print(f"[PREVIEW] Current files count: {len(self.current_files) if self.current_files else 0}")
            
            # Uncheck all other buttons
            for name, btn in self.pass_buttons.items():
                btn.setChecked(name == pass_name.lower())
            
            print(f"[PREVIEW] Setting current_pass to: {pass_name.lower()}")
            self.current_pass = pass_name.lower()
            
            # Update media preview to use this pass
            print(f"[PREVIEW] Calling media_preview.set_current_pass...")
            self.media_preview.set_current_pass(pass_name)
            print(f"[PREVIEW] media_preview.set_current_pass completed")
            
            # For image sequences, update to show current frame in the new pass
            if self.file_type_combo.currentText() == "Image Sequence" and hasattr(self, 'current_frame_index'):
                print(f"[PREVIEW] Image sequence detected, calling on_frame_changed({self.current_frame_index})")
                # Trigger frame change to load the correct pass result
                self.on_frame_changed(self.current_frame_index)
                print(f"[PREVIEW] on_frame_changed completed")
            else:
                # For single images/videos, just update the preview
                print(f"[PREVIEW] Not an image sequence, calling update_preview_for_current_pass...")
                self.update_preview_for_current_pass()
                print(f"[PREVIEW] update_preview_for_current_pass completed")
            
            self.progress_panel.add_status_message(f"Viewing {pass_name} pass")
            print(f"[PREVIEW] === on_pass_selected completed successfully ===\n")
            
        except Exception as e:
            print(f"\n[PREVIEW] ❌ EXCEPTION in on_pass_selected: {e}")
            import traceback
            traceback.print_exc()
            self.progress_panel.add_status_message(f"❌ Error switching to {pass_name} pass: {e}")
            print(f"[PREVIEW] === on_pass_selected failed ===\n")

    def update_preview_for_current_pass(self):
        """Update preview to show the current selected pass"""
        # First, try to load from output directory if available
        if hasattr(self, 'output_results') and self.output_results:
            self.update_preview_from_output()
            return
        
        # Fallback to in-memory results (during processing)
        if hasattr(self, 'current_results') and self.current_results:
            if self.current_pass in self.current_results:
                # If we have results for this pass, show them
                if self.file_type_combo.currentText() == "Image Sequence" and hasattr(self, 'current_frame_index'):
                    # For sequences, show the current frame
                    frame_results = self.current_results[self.current_pass]
                    if isinstance(frame_results, list) and self.current_frame_index < len(frame_results):
                        self.media_preview.load_media(frame_results[self.current_frame_index])
                else:
                    # For single images/videos, show the result
                    self.media_preview.load_media(self.current_results[self.current_pass])
            elif self.current_pass == 'original' and self.current_files:
                # Show original file
                if self.file_type_combo.currentText() == "Image Sequence" and hasattr(self, 'current_frame_index'):
                    if self.current_frame_index < len(self.current_files):
                        self.media_preview.load_media(self.current_files[self.current_frame_index])
                else:
                    self.media_preview.load_media(self.current_files[0])
        elif self.current_pass == 'original' and self.current_files:
            # Show original file if no processed results available
            if self.file_type_combo.currentText() == "Image Sequence" and hasattr(self, 'current_frame_index'):
                if self.current_frame_index < len(self.current_files):
                    self.media_preview.load_media(self.current_files[self.current_frame_index])
            else:
                self.media_preview.load_media(self.current_files[0])

    def on_frame_changed(self, frame_index):
        """Handle frame slider change for image sequences - works for ALL passes"""
        if not self.current_files:
            print("DEBUG: on_frame_changed called but no current_files")
            return
            
        # Ensure frame_index is within valid range
        if frame_index < 0 or frame_index >= len(self.current_files):
            print(f"WARNING: Frame index {frame_index} out of range (0 to {len(self.current_files) - 1})")
            return
            
        self.current_frame_index = frame_index
        max_frames = len(self.current_files)
        
        # Update frame label (1-based for user display)
        user_frame_number = frame_index + 1
        self.frame_label.setText(f"Frame: {user_frame_number} / {max_frames}")
        
        # Load the appropriate file based on current pass
        from pathlib import Path
        
        if self.current_pass == 'original':
            # Load original source file
            current_file = self.current_files[frame_index]
            self.media_preview.load_media(current_file)
            print(f"[VIDEO] Frame {user_frame_number}: {Path(current_file).name} (Original)")
        else:
            # Load processed result for current pass
            if hasattr(self, 'output_results') and self.current_pass in self.output_results:
                result_files = self.output_results[self.current_pass]
                
                if result_files and frame_index < len(result_files):
                    # Load the corresponding processed frame
                    result_file = result_files[frame_index]
                    self.media_preview.load_media(result_file)
                    print(f"[PREVIEW] Frame {user_frame_number}: {Path(result_file).name} ({self.current_pass.title()})")
                else:
                    # Fallback to original if no processed result available
                    if frame_index < len(self.current_files):
                        current_file = self.current_files[frame_index]
                        self.media_preview.load_media(current_file)
                        print(f"[WARNING] Frame {user_frame_number}: No {self.current_pass} result, showing original")
            else:
                # No processed results yet, show original
                if frame_index < len(self.current_files):
                    current_file = self.current_files[frame_index]
                    self.media_preview.load_media(current_file)
                    print(f"[WARNING] Frame {user_frame_number}: No {self.current_pass} results available, showing original")

    def toggle_playback(self):
        """Toggle playback for image sequences and videos"""
        file_type = self.file_type_combo.currentText()
        
        # Allow playback for both Image Sequence and Video File
        if not self.current_files or file_type not in ["Image Sequence", "Video File"]:
            return
            
        if self.is_playing:
            self.play_timer.stop()
            self.play_btn.setText("▶ Play")
            self.is_playing = False
        else:
            self.play_timer.start(42)  # 24 FPS (1000ms / 24 = ~42ms per frame)
            self.play_btn.setText("⏸ Pause")
            self.is_playing = True

    def next_frame(self):
        """Go to next frame during playback"""
        if not self.current_files:
            return
            
        current_frame = self.frame_slider.value()
        max_frame = self.frame_slider.maximum()
        
        if current_frame < max_frame:
            self.frame_slider.setValue(current_frame + 1)
        else:
            # Loop back to beginning
            self.frame_slider.setValue(0)

    def update_timeline_for_sequence(self):
        """Update timeline controls for image sequence"""
        from pathlib import Path  # Import Path at the top of the method
        file_type = self.file_type_combo.currentText()
        
        if file_type == "Image Sequence" and self.current_files:
            self.timeline_group.setVisible(True)
            num_files = len(self.current_files)
            
            # Debug output to diagnose the doubling issue
            print(f"DEBUG: update_timeline_for_sequence called")
            print(f"DEBUG: self.current_files has {num_files} items")
            print(f"DEBUG: First 5 files: {[Path(f).name for f in self.current_files[:5]]}")
            print(f"DEBUG: Last 5 files: {[Path(f).name for f in self.current_files[-5:]]}")
            
            # Set slider range: 0 to (num_files - 1)
            self.frame_slider.setMaximum(num_files - 1)
            self.frame_slider.setValue(0)
            
            # Initialize frame tracking
            self.current_frame_index = 0
            
            # Update frame label: Frame 1 of N (user-friendly 1-based counting)
            self.frame_label.setText(f"Frame: 1 / {num_files}")
            
            # Load and display first frame
            if self.current_files:
                first_file = self.current_files[0]
                self.media_preview.load_media(first_file)
                print(f"Timeline updated: {num_files} frames in sequence")
                print(f"First frame loaded: {Path(first_file).name}")
                print(f"Slider range: 0 to {num_files - 1}")
        else:
            self.timeline_group.setVisible(False)

    def get_processing_settings(self):
        """Get processing settings from the settings panel"""
        return {
            'export_config': self.get_export_settings()
        }
    
    def start_background_processing(self, export_config, output_dir):
        """Start processing in a background thread"""
        self.worker_thread = QThread()
        self.worker = ProcessingWorker(self, export_config, output_dir)
        self.worker.moveToThread(self.worker_thread)
        
        # Connect thread lifecycle
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        
        # Connect progress signals to progress panel and main window
        self.worker.progress.connect(self.on_worker_progress)
        self.worker.files_progress.connect(self.on_worker_files_progress)
        self.worker.speed.connect(self.on_worker_speed)
        self.worker.preview.connect(self.on_worker_preview)
        self.worker.status.connect(self.on_worker_status)
        self.worker.frame_progress.connect(self.on_worker_frame_progress)
        self.worker.time_remaining.connect(self.on_worker_time_remaining)
        
        # Connect worker finished to cleanup
        self.worker.finished.connect(self.on_processing_finished)
        
        # Start periodic preview refresh for video processing
        self.start_periodic_preview_refresh()
        
        self.worker_thread.start()

    def on_processing_finished(self):
        """Handle processing completion or cancellation"""
        self.processing = False
        self.process_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        # Stop the elapsed time timer (complete_progress stops the timer)
        self.progress_panel.complete_progress()
        
        # Stop periodic refresh
        self.stop_periodic_preview_refresh()
        
        # Scan for results after a short delay, then setup timeline scrubbing
        QTimer.singleShot(1000, self.scan_output_results)
        QTimer.singleShot(1500, self.setup_timeline_scrubbing_after_processing)
    
    def on_worker_time_remaining(self, seconds):
        """Handle time remaining updates from worker"""
        self.progress_panel.update_time_remaining(seconds)
        
        # Format time for display
        hrs = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        formatted = f"{hrs}:{mins:02d}:{secs:02d}" if hrs > 0 else f"{mins}:{secs:02d}"
        print(f"⏰ Time remaining: {formatted}")
    
    def on_worker_frame_progress(self, current_frame, total_frames, filename):
        """Handle frame-specific progress updates for videos"""
        # Update the files processed label to show frame progress
        self.progress_panel.files_label.setText(f"{current_frame} / {total_frames} frames")
        
        # Update timeline slider and label for video processing (like image sequences)
        if self.file_type_combo.currentText() == "Video File":
            # Make timeline visible for videos during processing
            self.timeline_group.setVisible(True)
            
            # Set slider range to match video frames
            self.frame_slider.setMaximum(total_frames - 1)
            self.frame_slider.setValue(current_frame - 1)  # current_frame is 1-based
            
            # Update frame label to show current progress
            self.frame_label.setText(f"Frame: {current_frame} / {total_frames}")
        
        print(f"🎬 Frame progress: {current_frame}/{total_frames} for {filename}")

    def on_worker_progress(self, percent, message):
        """Handle overall progress updates"""
        self.progress_panel.set_overall_progress(percent)
        if message:
            # Update current task description
            self.progress_panel.set_task_progress(percent, message)

    def on_worker_files_progress(self, processed_items, total_items):
        """Handle file/frame count progress updates"""
        self.progress_panel.update_file_progress(processed_items, total_items)
        
        # Determine unit name based on file type
        file_type = self.file_type_combo.currentText()
        if file_type == "Video File":
            unit_name = "frames"
        else:
            unit_name = "files"
        
        # Update status bar and progress panel
        self.progress_panel.files_label.setText(f"{processed_items} / {total_items} {unit_name}")
        self.file_count_label.setText(f"{unit_name.title()}: {processed_items} / {total_items}")
        
        print(f"[STATS] Progress: {processed_items}/{total_items} {unit_name} processed")

    def on_worker_speed(self, items_per_second):
        """Handle processing speed updates"""
        # Determine unit name based on file type
        file_type = self.file_type_combo.currentText()
        if file_type == "Video File":
            unit_name = "frames/sec"
        else:
            unit_name = "files/sec"
        
        # Update speed display
        self.progress_panel.speed_label.setText(f"{items_per_second:.2f} {unit_name}")
        print(f"[SPEED] Processing speed: {items_per_second:.2f} {unit_name}")

    def on_media_frame_changed(self, frame_index):
        """Handle media frame changes for video preview"""
        self.current_video_frame_index = frame_index
        print(f"[VIDEO] Video frame changed to: {frame_index}")

    def get_export_settings(self):
        """Get export settings from checkboxes"""
        return {
            component: checkbox.isChecked() 
            for component, checkbox in self.export_checkboxes.items()
        }

    def detect_file_type(self, file_path):
        """Detect if file is image or video based on extension"""
        from pathlib import Path
        file_ext = Path(file_path).suffix.lower()
        
        video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"]
        image_extensions = [".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif", ".webp"]
        
        if file_ext in video_extensions:
            return "video"
        elif file_ext in image_extensions:
            return "image"
        else:
            return "unknown"
    
    def on_worker_preview(self, previews):
        """Handle preview updates from worker"""
        if previews and isinstance(previews, dict):
            # Update media preview with new results
            self.media_preview.update_previews(previews)
            
            # Store previews for current pass switching
            if not hasattr(self, 'current_results'):
                self.current_results = {}
            self.current_results.update(previews)
            
            # Update current pass preview if needed
            if self.current_pass in previews:
                self.media_preview.load_media(previews[self.current_pass])
                print(f"[PREVIEW] Updated preview for {self.current_pass} pass during processing")

    def on_worker_status(self, msg):
        """Handle status updates from worker"""
        self.progress_panel.add_status_message(msg)
        
        # Update status bar
        self.status_bar.showMessage(msg)
        
        # If processing completed successfully, refresh output results
        if "[OK]... All" in msg and "processed successfully" in msg:
            QTimer.singleShot(1000, self.scan_output_results)  # Delay to ensure files are written
        
        # For video processing, refresh preview more frequently to show latest frames
        elif "Video File" in self.file_type_combo.currentText() and self.processing:
            # Check if this is a frame completion message
            if "[OK]... Processed" in msg:
                # Periodically refresh output results during video processing
                QTimer.singleShot(500, self.refresh_video_preview)

    def refresh_video_preview(self):
        """Refresh preview during video processing to show latest processed frames"""
        if not self.processing or not self.output_dir:
            return
        
        try:
            # Quickly rescan just the current pass folder
            if self.current_pass != 'original':
                output_path = Path(self.output_dir)
                pass_folder = output_path / self.current_pass
                
                if pass_folder.exists():
                    # Find latest files in current pass folder
                    image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp']
                    files = []
                    for ext in image_extensions:
                        files.extend(list(pass_folder.glob(f'*{ext}')))
                    
                    # Sort by modification time to get latest
                    files = sorted(files, key=lambda x: x.stat().st_mtime)
                    
                    if files:
                        # Update the cached results for this pass
                        if not hasattr(self, 'output_results'):
                            self.output_results = {}
                        self.output_results[self.current_pass] = [str(f) for f in files]
                        
                        # Update preview with latest frame
                        latest_file = str(files[-1])
                        self.media_preview.load_media(latest_file)
                        
                        frame_name = Path(latest_file).stem
                        print(f"[VIDEO] Updated video preview with latest {self.current_pass} frame: {frame_name}")
                        
        except Exception as e:
            print(f"Error refreshing video preview: {e}")

    def start_periodic_preview_refresh(self):
        """Start periodic preview refresh for video processing"""
        if not hasattr(self, 'preview_refresh_timer'):
            self.preview_refresh_timer = QTimer()
            self.preview_refresh_timer.timeout.connect(self.refresh_video_preview)
        
        # Refresh every 2 seconds during processing to show latest frames
        if self.processing:
            self.preview_refresh_timer.start(2000)
            print("[REFRESH] Started periodic preview refresh for real-time updates")

    def stop_periodic_preview_refresh(self):
        """Stop periodic preview refresh"""
        if hasattr(self, 'preview_refresh_timer'):
            self.preview_refresh_timer.stop()
            print("[STOP] Stopped periodic preview refresh")

    def on_media_frame_changed(self, frame_index):
        """Handle media frame changes for video preview"""
        self.current_video_frame_index = frame_index
        print(f"[VIDEO] Video frame changed to: {frame_index}")

    def get_export_settings(self):
        """Get export settings from checkboxes"""
        return {
            component: checkbox.isChecked() 
            for component, checkbox in self.export_checkboxes.items()
        }

    def detect_file_type(self, file_path):
        """Detect if file is image or video based on extension"""
        from pathlib import Path
        file_ext = Path(file_path).suffix.lower()
        
        video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"]
        image_extensions = [".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif", ".webp"]
        
        if file_ext in video_extensions:
            return "video"
        elif file_ext in image_extensions:
            return "image"
        else:
            return "unknown"
    
    def setup_timeline_scrubbing_after_processing(self):
        """Setup timeline scrubbing for processed results after processing completes"""
        file_type = self.file_type_combo.currentText()
        
        # For videos and sequences, enable timeline scrubbing through processed frames
        if file_type in ["Video File", "Image Sequence"] and hasattr(self, 'output_results'):
            # Get the processed frames from any pass (they should all have the same count)
            processed_frames = None
            for pass_type in ['albedo', 'normal', 'specular', 'depth']:
                if pass_type in self.output_results and self.output_results[pass_type]:
                    processed_frames = self.output_results[pass_type]
                    break
            
            if processed_frames:
                num_frames = len(processed_frames)
                
                # Make timeline visible and configure it
                self.timeline_group.setVisible(True)
                self.frame_slider.setMaximum(num_frames - 1)
                self.frame_slider.setValue(0)
                
                # Update current_files to use processed frames for scrubbing
                self.current_files = processed_frames
                self.current_frame_index = 0
                
                # Update frame label
                self.frame_label.setText(f"Frame: 1 / {num_frames}")
                
                # Load first frame of current pass
                self.on_frame_changed(0)
                
                self.progress_panel.add_status_message(f"Timeline ready: {num_frames} frames available for scrubbing")
                print(f"[TIMELINE] Scrubbing enabled for {num_frames} processed frames")
