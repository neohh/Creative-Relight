"""
Settings Panel Widget for Creative Relight
Simple export options and temporal consistency controls
"""

from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *


class SettingsPanel(QGroupBox):
    """Settings panel for export options and temporal consistency"""
    
    settings_changed = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__("Processing Settings", parent)
        self.setup_ui()
        self.setup_connections()
        
    def setup_ui(self):
        """Setup the settings panel interface"""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 20, 15, 15)
        
        # Export Components section
        export_label = QLabel("Export Components")
        export_label.setStyleSheet("font-weight: bold; font-size: 11pt; color: #333;")
        layout.addWidget(export_label)
        
        self.export_albedo = QCheckBox("Export Albedo")
        self.export_specular = QCheckBox("Export Specular")
        self.export_depth = QCheckBox("Export Depth")
        self.export_normal = QCheckBox("Export Normal")
        
        # Set default values
        self.export_albedo.setChecked(True)
        self.export_specular.setChecked(True)
        self.export_depth.setChecked(True)
        self.export_normal.setChecked(True)
        
        layout.addWidget(self.export_albedo)
        layout.addWidget(self.export_specular)
        layout.addWidget(self.export_depth)
        layout.addWidget(self.export_normal)
        
        # Spacing between sections
        layout.addSpacing(20)
        
        # Temporal Consistency section
        temporal_label = QLabel("Temporal Consistency")
        temporal_label.setStyleSheet("font-weight: bold; font-size: 11pt; color: #333;")
        layout.addWidget(temporal_label)
        
        # Enable checkbox with tooltip
        self.enable_temporal = QCheckBox("Enable Temporal Smoothing")
        self.enable_temporal.setChecked(True)
        self.enable_temporal.setToolTip("Reduces flickering between frames in video sequences")
        layout.addWidget(self.enable_temporal)
        
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
            "Strength is the number of adjacent frames\n"
            "blended for temporal consistency.\n\n"
            "• Higher value enhances consistency but\n"
            "  may cause ghosting.\n"
            "• Lower value reduces artifacts but may\n"
            "  lead to inconsistency."
        )
        help_button.setCursor(Qt.CursorShape.WhatsThisCursor)
        
        strength_header_layout.addWidget(strength_label)
        strength_header_layout.addWidget(help_button)
        strength_header_layout.addStretch()
        layout.addLayout(strength_header_layout)
        
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
        
        strength_slider_layout.addWidget(self.temporal_strength_slider)
        strength_slider_layout.addWidget(self.temporal_strength_label)
        layout.addLayout(strength_slider_layout)
        
        # Push everything to the top
        layout.addStretch()
        
    def setup_connections(self):
        """Setup signal connections"""
        # Export checkboxes
        self.export_albedo.toggled.connect(self.emit_settings_changed)
        self.export_specular.toggled.connect(self.emit_settings_changed)
        self.export_depth.toggled.connect(self.emit_settings_changed)
        self.export_normal.toggled.connect(self.emit_settings_changed)
        
        # Temporal consistency controls
        self.enable_temporal.toggled.connect(self.on_temporal_toggled)
        self.temporal_strength_slider.valueChanged.connect(self.update_temporal_strength_label)
    
    def on_temporal_toggled(self, enabled):
        """Handle temporal consistency toggle"""
        self.temporal_strength_slider.setEnabled(enabled)
        self.temporal_strength_label.setEnabled(enabled)
        self.emit_settings_changed()
    
    def update_temporal_strength_label(self, value):
        """Update temporal strength label and emit settings change"""
        self.temporal_strength_label.setText(str(value))
        self.emit_settings_changed()
    
    
    def emit_settings_changed(self):
        """Emit settings changed signal with current values"""
        settings = self.get_settings()
        self.settings_changed.emit(settings)
    
    def get_settings(self):
        """Get current settings as dictionary"""
        return {
            'export_albedo': self.export_albedo.isChecked(),
            'export_specular': self.export_specular.isChecked(),
            'export_depth': self.export_depth.isChecked(),
            'export_normal': self.export_normal.isChecked(),
            'enable_temporal': self.enable_temporal.isChecked(),
            'temporal_strength': self.temporal_strength_slider.value()
        }
    
    def set_settings(self, settings):
        """Apply settings from dictionary"""
        if 'export_albedo' in settings:
            self.export_albedo.setChecked(settings['export_albedo'])
        if 'export_specular' in settings:
            self.export_specular.setChecked(settings['export_specular'])
        if 'export_depth' in settings:
            self.export_depth.setChecked(settings['export_depth'])
        if 'export_normal' in settings:
            self.export_normal.setChecked(settings['export_normal'])
        if 'enable_temporal' in settings:
            self.enable_temporal.setChecked(settings['enable_temporal'])
        if 'temporal_strength' in settings:
            self.temporal_strength_slider.setValue(settings['temporal_strength'])

