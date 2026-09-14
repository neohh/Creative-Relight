"""
File Browser Widget for Creative Relight
Advanced file selection and management interface
"""

import os
from pathlib import Path
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *


class FileBrowserWidget(QWidget):
    """Advanced file browser with preview and filtering"""
    
    file_selected = pyqtSignal(str)
    folder_changed = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_directory = Path.home()
        self.supported_formats = {
            'image': ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif', '.webp', '.exr', '.hdr', '.pic'],
            'video': ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
        }
        self.setup_ui()
        self.setup_connections()
        
    def setup_ui(self):
        """Setup the file browser interface"""
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        
        # Current directory display
        dir_layout = QHBoxLayout()
        self.dir_label = QLabel("Current Directory:")
        self.dir_path = QLineEdit()
        self.dir_path.setReadOnly(True)
        self.browse_dir_btn = QPushButton("Browse")
        self.browse_dir_btn.setMaximumWidth(80)
        
        dir_layout.addWidget(self.dir_label)
        dir_layout.addWidget(self.dir_path)
        dir_layout.addWidget(self.browse_dir_btn)
        layout.addLayout(dir_layout)
        
        # File filter
        filter_layout = QHBoxLayout()
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All Supported", "Images Only", "Videos Only"])
        self.filter_combo.setCurrentText("All Supported")
        
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search files...")
        
        filter_layout.addWidget(QLabel("Filter:"))
        filter_layout.addWidget(self.filter_combo)
        filter_layout.addWidget(QLabel("Search:"))
        filter_layout.addWidget(self.search_edit)
        layout.addLayout(filter_layout)
        
        # File list
        self.file_list = QListWidget()
        self.file_list.setAlternatingRowColors(True)
        self.file_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        layout.addWidget(self.file_list)
        
        # Quick action buttons
        button_layout = QHBoxLayout()
        self.select_file_btn = QPushButton("Select File")
        self.select_folder_btn = QPushButton("Select Folder")
        self.refresh_btn = QPushButton("Refresh")
        
        button_layout.addWidget(self.select_file_btn)
        button_layout.addWidget(self.select_folder_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.refresh_btn)
        layout.addLayout(button_layout)
        
        # Set initial directory
        self.set_directory(str(self.current_directory))
        
    def setup_connections(self):
        """Setup signal connections"""
        self.browse_dir_btn.clicked.connect(self.browse_directory)
        self.select_file_btn.clicked.connect(self.select_single_file)
        self.select_folder_btn.clicked.connect(self.select_folder)
        self.refresh_btn.clicked.connect(self.refresh_file_list)
        self.filter_combo.currentTextChanged.connect(self.apply_filter)
        self.search_edit.textChanged.connect(self.apply_search)
        self.file_list.itemDoubleClicked.connect(self.on_file_double_clicked)
        self.file_list.itemSelectionChanged.connect(self.on_selection_changed)
        
    def set_directory(self, directory):
        """Set the current directory and refresh file list"""
        try:
            path = Path(directory)
            if path.exists() and path.is_dir():
                self.current_directory = path
                self.dir_path.setText(str(path))
                self.refresh_file_list()
                self.folder_changed.emit(str(path))
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not access directory: {e}")
            
    def browse_directory(self):
        """Open directory browser dialog"""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Directory", str(self.current_directory)
        )
        if directory:
            self.set_directory(directory)
            
    def select_single_file(self):
        """Open file selection dialog"""
        file_filters = "All Supported (*.png *.jpg *.jpeg *.tiff *.bmp *.gif *.exr *.hdr *.mp4 *.avi *.mov);;"
        file_filters += "Image Files (*.png *.jpg *.jpeg *.tiff *.bmp *.gif);;"
        file_filters += "HDR Files (*.exr *.hdr *.pic);;"
        file_filters += "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv);;"
        file_filters += "All Files (*)"
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select File", str(self.current_directory), file_filters
        )
        if file_path:
            self.file_selected.emit(file_path)
            # Update directory to file's parent
            self.set_directory(str(Path(file_path).parent))
            
    def select_folder(self):
        """Select folder for batch processing"""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Folder for Batch Processing", str(self.current_directory)
        )
        if directory:
            self.set_directory(directory)
            
    def refresh_file_list(self):
        """Refresh the file list based on current directory and filters"""
        self.file_list.clear()
        
        try:
            if not self.current_directory.exists():
                return
                
            # Get all files in directory
            files = []
            for item in self.current_directory.iterdir():
                if item.is_file():
                    files.append(item)
                    
            # Apply filter
            filtered_files = self.filter_files(files)
            
            # Apply search
            if self.search_edit.text():
                search_term = self.search_edit.text().lower()
                filtered_files = [f for f in filtered_files if search_term in f.name.lower()]
                
            # Sort files
            filtered_files.sort(key=lambda x: x.name.lower())
            
            # Add files to list
            for file_path in filtered_files:
                item = QListWidgetItem()
                item.setText(file_path.name)
                item.setData(Qt.ItemDataRole.UserRole, str(file_path))
                
                # Set icon based on file type
                if self.is_image_file(file_path):
                    item.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogDetailedView))
                elif self.is_video_file(file_path):
                    item.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon))
                else:
                    item.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon))
                    
                # Set tooltip with file info
                stat = file_path.stat()
                size_mb = stat.st_size / (1024 * 1024)
                tooltip = f"Size: {size_mb:.2f} MB\\nPath: {file_path}"
                item.setToolTip(tooltip)
                
                self.file_list.addItem(item)
                
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not refresh file list: {e}")
            
    def filter_files(self, files):
        """Filter files based on selected filter"""
        filter_type = self.filter_combo.currentText()
        
        if filter_type == "Images Only":
            return [f for f in files if self.is_image_file(f)]
        elif filter_type == "Videos Only":
            return [f for f in files if self.is_video_file(f)]
        elif filter_type == "All Supported":
            return [f for f in files if self.is_supported_file(f)]
        else:  # All Files
            return files
            
    def is_image_file(self, file_path):
        """Check if file is a supported image format"""
        return file_path.suffix.lower() in self.supported_formats['image']
        
    def is_video_file(self, file_path):
        """Check if file is a supported video format"""
        return file_path.suffix.lower() in self.supported_formats['video']
        
    def is_supported_file(self, file_path):
        """Check if file is in any supported format"""
        return self.is_image_file(file_path) or self.is_video_file(file_path)
        
    def apply_filter(self):
        """Apply selected filter to file list"""
        self.refresh_file_list()
        
    def apply_search(self):
        """Apply search filter to file list"""
        self.refresh_file_list()
        
    def on_file_double_clicked(self, item):
        """Handle file double-click"""
        file_path = item.data(Qt.ItemDataRole.UserRole)
        if file_path:
            self.file_selected.emit(file_path)
            
    def on_selection_changed(self):
        """Handle selection change"""
        current_item = self.file_list.currentItem()
        if current_item:
            file_path = current_item.data(Qt.ItemDataRole.UserRole)
            # Could emit preview signal here for immediate preview
            
    def get_selected_file(self):
        """Get currently selected file path"""
        current_item = self.file_list.currentItem()
        if current_item:
            return current_item.data(Qt.ItemDataRole.UserRole)
        return None
        
    def get_all_files(self):
        """Get all files in current directory matching current filter"""
        files = []
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            file_path = item.data(Qt.ItemDataRole.UserRole)
            if file_path:
                files.append(file_path)
        return files


# Alias for backward compatibility
FileBrowser = FileBrowserWidget
