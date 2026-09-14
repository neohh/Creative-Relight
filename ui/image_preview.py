"""
Image Preview Widget for Creative Relight
Advanced image display with zoom, pan, and comparison features
"""

import os
from pathlib import Path
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *


class ImagePreviewWidget(QLabel):
    """Advanced image preview with zoom, pan, and comparison features"""
    
    image_loaded = pyqtSignal(str)
    zoom_changed = pyqtSignal(float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.setStyleSheet("border: 2px solid #555555; background-color: #333333;")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText("No Image Loaded\\nDrag and drop an image here")
        
        # Image data
        self.original_pixmap = None
        self.scaled_pixmap = None
        self.image_path = None
        self.zoom_factor = 1.0
        self.pan_offset = QPoint(0, 0)
        
        # Mouse interaction
        self.last_pan_point = QPoint()
        self.dragging = False
        
        # Enable drag and drop
        self.setAcceptDrops(True)
        
    def load_image(self, image_path):
        """Load image from file path"""
        try:
            if not os.path.exists(image_path):
                self.show_error("File not found")
                return False
                
            # Load image
            self.original_pixmap = QPixmap(image_path)
            
            if self.original_pixmap.isNull():
                self.show_error("Invalid image format")
                return False
                
            self.image_path = image_path
            self.zoom_factor = 1.0
            self.pan_offset = QPoint(0, 0)
            
            # Fit image to widget initially
            self.fit_to_window()
            
            self.image_loaded.emit(image_path)
            return True
            
        except Exception as e:
            self.show_error(f"Error loading image: {str(e)}")
            return False
            
    def set_pixmap(self, pixmap):
        """Set pixmap directly"""
        if pixmap and not pixmap.isNull():
            self.original_pixmap = pixmap
            self.zoom_factor = 1.0
            self.pan_offset = QPoint(0, 0)
            self.fit_to_window()
            
    def fit_to_window(self):
        """Fit image to widget size"""
        if not self.original_pixmap:
            return
            
        widget_size = self.size()
        pixmap_size = self.original_pixmap.size()
        
        # Calculate zoom factor to fit image in widget
        scale_x = widget_size.width() / pixmap_size.width()
        scale_y = widget_size.height() / pixmap_size.height()
        self.zoom_factor = min(scale_x, scale_y, 1.0)  # Don't zoom in beyond 100%
        
        self.pan_offset = QPoint(0, 0)
        self.update_display()
        
    def actual_size(self):
        """Show image at actual size (100%)"""
        if not self.original_pixmap:
            return
            
        self.zoom_factor = 1.0
        self.pan_offset = QPoint(0, 0)
        self.update_display()
        
    def set_zoom(self, zoom):
        """Set zoom factor"""
        if not self.original_pixmap:
            return
            
        # Clamp zoom between 0.1x and 10x
        self.zoom_factor = max(0.1, min(10.0, zoom))
        self.update_display()
        self.zoom_changed.emit(self.zoom_factor)
        
    def zoom_in(self):
        """Zoom in by 25%"""
        self.set_zoom(self.zoom_factor * 1.25)
        
    def zoom_out(self):
        """Zoom out by 25%"""
        self.set_zoom(self.zoom_factor / 1.25)
        
    def update_display(self):
        """Update the displayed image"""
        if not self.original_pixmap:
            return
            
        # Scale image
        scaled_size = self.original_pixmap.size() * self.zoom_factor
        self.scaled_pixmap = self.original_pixmap.scaled(
            scaled_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        
        # Create display pixmap with pan offset
        widget_size = self.size()
        display_pixmap = QPixmap(widget_size)
        display_pixmap.fill(QColor(51, 51, 51))  # Dark gray background
        
        painter = QPainter(display_pixmap)
        
        # Calculate position to center image
        x = (widget_size.width() - self.scaled_pixmap.width()) // 2 + self.pan_offset.x()
        y = (widget_size.height() - self.scaled_pixmap.height()) // 2 + self.pan_offset.y()
        
        painter.drawPixmap(x, y, self.scaled_pixmap)
        painter.end()
        
        self.setPixmap(display_pixmap)
        
    def show_error(self, message):
        """Show error message"""
        self.original_pixmap = None
        self.scaled_pixmap = None
        self.image_path = None
        self.setText(f"Error: {message}")
        
    def has_image(self):
        """Check if an image is loaded"""
        return self.original_pixmap is not None
        
    def get_image_info(self):
        """Get information about the loaded image"""
        if not self.original_pixmap:
            return None
            
        return {
            'path': self.image_path,
            'width': self.original_pixmap.width(),
            'height': self.original_pixmap.height(),
            'zoom': self.zoom_factor
        }
        
    def clear_image(self):
        """Clear the current image"""
        self.original_pixmap = None
        self.scaled_pixmap = None
        self.image_path = None
        self.zoom_factor = 1.0
        self.pan_offset = QPoint(0, 0)
        self.setText("No Image Loaded\nDrag and drop an image here")
        self.update()

    # Event handlers for mouse interaction
    def mousePressEvent(self, event):
        """Handle mouse press for panning"""
        if event.button() == Qt.MouseButton.LeftButton and self.original_pixmap:
            self.last_pan_point = event.pos()
            self.dragging = True
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            
    def mouseMoveEvent(self, event):
        """Handle mouse move for panning"""
        if self.dragging and self.original_pixmap:
            delta = event.pos() - self.last_pan_point
            self.pan_offset += delta
            self.last_pan_point = event.pos()
            self.update_display()
            
    def mouseReleaseEvent(self, event):
        """Handle mouse release"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            
    def wheelEvent(self, event):
        """Handle mouse wheel for zooming"""
        if self.original_pixmap:
            # Get zoom delta
            delta = event.angleDelta().y()
            zoom_in = delta > 0
            
            # Calculate new zoom
            if zoom_in:
                new_zoom = self.zoom_factor * 1.1
            else:
                new_zoom = self.zoom_factor / 1.1
                
            self.set_zoom(new_zoom)
            
    def resizeEvent(self, event):
        """Handle widget resize"""
        super().resizeEvent(event)
        if self.original_pixmap:
            self.update_display()
            
    # Drag and drop support
    def dragEnterEvent(self, event):
        """Handle drag enter"""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and self.is_image_file(urls[0].toLocalFile()):
                event.acceptProposedAction()
                
    def dropEvent(self, event):
        """Handle file drop"""
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if self.is_image_file(file_path):
                self.load_image(file_path)
                event.acceptProposedAction()
                
    def is_image_file(self, file_path):
        """Check if file is a supported image format"""
        supported_formats = ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif', '.webp']
        return Path(file_path).suffix.lower() in supported_formats

    def update_previews(self, previews):
        """Update preview with processing results"""
        if not previews:
            return
            
        # If previews is a dict with different result types, show the first available
        if isinstance(previews, dict):
            # Look for common result types in order of preference
            for key in ['hr_alb', 'gry_alb', 'albedo', 'result', 'output']:
                if key in previews:
                    preview_image = previews[key]
                    break
            else:
                # If none of the expected keys found, use the first available
                preview_image = list(previews.values())[0]
        else:
            preview_image = previews
            
        # Load the preview image
        if isinstance(preview_image, str):
            # If it's a file path
            self.load_image(preview_image)
        else:
            # If it's a PIL Image or numpy array, convert and display
            try:
                from PIL import Image
                import numpy as np
                
                if isinstance(preview_image, np.ndarray):
                    # Convert numpy array to PIL Image
                    if preview_image.dtype != np.uint8:
                        preview_image = (preview_image * 255).astype(np.uint8)
                    preview_image = Image.fromarray(preview_image)
                
                if isinstance(preview_image, Image.Image):
                    # Convert PIL Image to QPixmap
                    from PyQt6.QtGui import QPixmap, QImage
                    
                    # Convert PIL to QImage
                    preview_image = preview_image.convert('RGB')
                    w, h = preview_image.size
                    qimage = QImage(preview_image.tobytes(), w, h, QImage.Format.Format_RGB888)
                    
                    # Convert to QPixmap and display
                    pixmap = QPixmap.fromImage(qimage)
                    self.set_pixmap(pixmap)
                    
            except Exception as e:
                print(f"Error updating preview: {e}")
                self.show_error(f"Could not display preview: {e}")

class ComparisonImagePreview(QWidget):
    """Side-by-side image comparison widget"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        """Setup comparison interface"""
        layout = QHBoxLayout(self)
        
        # Before image
        self.before_group = QGroupBox("Original")
        before_layout = QVBoxLayout(self.before_group)
        self.before_preview = ImagePreviewWidget()
        before_layout.addWidget(self.before_preview)
        
        # After image
        self.after_group = QGroupBox("Processed")
        after_layout = QVBoxLayout(self.after_group)
        self.after_preview = ImagePreviewWidget()
        after_layout.addWidget(self.after_preview)
        
        layout.addWidget(self.before_group)
        layout.addWidget(self.after_group)
        
    def set_before_image(self, image_path):
        """Set the before image"""
        self.before_preview.load_image(image_path)
        
    def set_after_image(self, image_path):
        """Set the after image"""
        self.after_preview.load_image(image_path)
        
    def set_before_pixmap(self, pixmap):
        """Set before image from pixmap"""
        self.before_preview.set_pixmap(pixmap)
        
    def set_after_pixmap(self, pixmap):
        """Set after image from pixmap"""
        self.after_preview.set_pixmap(pixmap)


# Alias for backward compatibility
ImagePreview = ImagePreviewWidget
