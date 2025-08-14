#!/usr/bin/env python3
"""
Clean Simple GUI module for Force Control Experiment
Enhanced functionality with the exact same clean styling as the original
"""

import sys
import numpy as np
from scipy import signal
from collections import deque
import queue as queue_module
import threading
import time 

# GUI imports
try:
    import matplotlib
    matplotlib.use('Qt5Agg')
    from PyQt5 import QtWidgets, QtCore, QtGui
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                QHBoxLayout, QLabel, QCheckBox, QSlider, QGroupBox,
                                QSpinBox, QComboBox)
    from PyQt5.QtCore import QTimer, Qt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    print("GUI libraries not available")


class ForceDataFilter:
    """Real-time force data filtering with multiple filter options"""
    
    def __init__(self, buffer_size=100):
        self.buffer_size = buffer_size
        self.raw_buffer = deque(maxlen=buffer_size)
        self.filtered_buffer = deque(maxlen=buffer_size)
        self.time_buffer = deque(maxlen=buffer_size)
        
        # Filter parameters
        self.filter_enabled = True
        self.filter_type = "butterworth"
        self.cutoff_freq = 3.0
        self.filter_order = 2
        self.window_size = 5
        self.savgol_order = 2
        
        # Butterworth filter state
        self.sampling_freq = 60.0
        self.reset_filter()
    
    def reset_filter(self):
        """Reset filter state"""
        if self.filter_type == "butterworth":
            nyquist = 0.5 * self.sampling_freq
            normal_cutoff = self.cutoff_freq / nyquist
            normal_cutoff = min(normal_cutoff, 0.99)
            self.b, self.a = signal.butter(self.filter_order, normal_cutoff, btype='low')
            self.zi = signal.lfilter_zi(self.b, self.a)
            self.filter_initialized = False
    
    def update_filter_params(self, filter_type=None, cutoff_freq=None, 
                           filter_order=None, window_size=None, savgol_order=None):
        """Update filter parameters"""
        if filter_type is not None:
            self.filter_type = filter_type
        if cutoff_freq is not None:
            self.cutoff_freq = cutoff_freq
        if filter_order is not None:
            self.filter_order = filter_order
        if window_size is not None:
            self.window_size = window_size
        if savgol_order is not None:
            self.savgol_order = savgol_order
        
        self.reset_filter()
    
    def add_data_point(self, timestamp, raw_force):
        """Add new data point and return filtered value"""
        self.raw_buffer.append(raw_force)
        self.time_buffer.append(timestamp)
        
        if not self.filter_enabled:
            filtered_force = raw_force
        else:
            filtered_force = self._apply_filter(raw_force)
        
        self.filtered_buffer.append(filtered_force)
        return filtered_force
    
    def _apply_filter(self, new_value):
        """Apply the selected filter"""
        if self.filter_type == "moving_average":
            if len(self.raw_buffer) >= self.window_size:
                return np.mean(list(self.raw_buffer)[-self.window_size:])
            else:
                return np.mean(list(self.raw_buffer))
        
        elif self.filter_type == "butterworth":
            if not self.filter_initialized:
                self.zi = self.zi * new_value
                self.filter_initialized = True
                return new_value
            else:
                filtered_val, self.zi = signal.lfilter(self.b, self.a, [new_value], zi=self.zi)
                return filtered_val[0]
        
        elif self.filter_type == "savgol":
            if len(self.raw_buffer) >= self.window_size:
                recent_data = np.array(list(self.raw_buffer)[-self.window_size:])
                if len(recent_data) > self.savgol_order:
                    filtered_data = signal.savgol_filter(recent_data, 
                                                       min(self.window_size, len(recent_data)), 
                                                       self.savgol_order)
                    return filtered_data[-1]
            return new_value
        
        else:
            return new_value
    
    def get_filtered_data(self):
        """Get current filtered data arrays"""
        return np.array(list(self.time_buffer)), np.array(list(self.filtered_buffer))
    
    def get_raw_data(self):
        """Get current raw data arrays"""
        return np.array(list(self.time_buffer)), np.array(list(self.raw_buffer))


class SimplePlotCanvas(FigureCanvas):
    """Exactly same plot styling as original, but with filtering capability"""
    
    def __init__(self, parent=None, grip_config=None, width=12, height=8, dpi=100, sampling_frequency=60.0):
        if not GUI_AVAILABLE:
            return
            
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        
        self.grip_config = grip_config
        
        # Initialize data filter
        self.data_filter = ForceDataFilter(buffer_size=200)
        self.data_filter.sampling_freq = sampling_frequency
        self.data_filter.reset_filter() # reinitialize filter with new sampling frequency
        # Create single subplot for force - EXACT same as original
        self.ax_force = self.fig.add_subplot(1, 1, 1)
        
        # Configure force plot - EXACT same styling as original
        self.ax_force.set_title(f'Force Monitor - {grip_config["description"]}', fontsize=14, weight='bold')
        self.ax_force.set_xlabel('Time (s)', fontsize=12)
        self.ax_force.set_ylabel('Force (N)', fontsize=12)
        self.ax_force.grid(True, alpha=0.3)
        
        # Get force limits from grip config
        self.min_force = grip_config["min_force"]
        self.max_force = grip_config["max_force"]
        
        # Initialize plot elements - same colors as original but with both raw and filtered
        self.raw_line, = self.ax_force.plot([], [], 'lightblue', linewidth=1, 
                                          label='Raw Force', alpha=0.7, zorder=2)
        self.filtered_line, = self.ax_force.plot([], [], 'b-', linewidth=3, 
                                                label='Filtered Force', zorder=3)
        
        # Force bands will be initialized later
        self.bands_initialized = False
        
        # Add legend and formatting - EXACT same as original
        self.ax_force.legend(loc='upper right')
        self.fig.tight_layout()
        
        # Track plot visibility
        self.show_raw = True
        self.show_filtered = True
        
    def update_filter_settings(self, **kwargs):
        """Update filter settings"""
        self.data_filter.update_filter_params(**kwargs)
    
    def set_filter_enabled(self, enabled):
        """Enable/disable filtering"""
        self.data_filter.filter_enabled = enabled
    
    def set_plot_visibility(self, show_raw=True, show_filtered=True):
        """Set which plots to show"""
        self.show_raw = show_raw
        self.show_filtered = show_filtered
        
        self.raw_line.set_visible(show_raw)
        self.filtered_line.set_visible(show_filtered)
        
        # Update legend
        if show_raw and show_filtered:
            self.raw_line.set_label('Raw Force')
            self.filtered_line.set_label('Filtered Force')
        elif show_filtered:
            self.filtered_line.set_label('Force')
            self.raw_line.set_label('')
        elif show_raw:
            self.raw_line.set_label('Force')
            self.filtered_line.set_label('')
        
        self.ax_force.legend(loc='upper right')
    
    def initialize_force_bands(self, time_range):
        """Initialize force bands - corrected logic but same visual style"""
        if self.bands_initialized:
            return
            
        # Define band boundaries
        band_upper_limit = self.max_force + 2.0
        
        # CORRECTED force zone logic:
        # Transition zone (0 to min_force): Gray - necessary for contact detection
        # Optimal zone (min_force to max_force): Green - target range
        # Overload zone (max_force+): Red - avoid
        
        self.transition_band = self.ax_force.axhspan(0, self.min_force, 
                                                   alpha=0.2, color='gray', 
                                                   label=f'Transition Zone (0-{self.min_force:.1f}N)', zorder=1)
        
        self.optimal_band = self.ax_force.axhspan(self.min_force, self.max_force, 
                                                alpha=0.2, color='green', 
                                                label=f'Optimal Zone ({self.min_force:.1f}-{self.max_force:.1f}N)', zorder=1)
        
        self.overload_band = self.ax_force.axhspan(self.max_force, band_upper_limit, 
                                                 alpha=0.2, color='red', 
                                                 label=f'Overload Zone (>{self.max_force:.1f}N)', zorder=1)
        
        # Add horizontal lines - EXACT same styling as original
        self.ax_force.axhline(y=self.min_force, color='darkgreen', linestyle='--', 
                            alpha=0.7, linewidth=2, label=f'Min Force ({self.min_force:.1f}N)')
        self.ax_force.axhline(y=self.max_force, color='darkgreen', linestyle='--', 
                            alpha=0.7, linewidth=2, label=f'Max Force ({self.max_force:.1f}N)')
        
        # Update legend
        self.ax_force.legend(loc='upper right', fontsize=10)
        
        self.bands_initialized = True
        
    def update_plots(self, timestamps, measured_forces, window_seconds=15):
        """Update plots with new data and filtering"""
        if not timestamps or len(timestamps) == 0:
            return
        
        try:
            # Add latest data point to filter
            latest_time = timestamps[-1]
            latest_force = measured_forces[-1]
            filtered_force = self.data_filter.add_data_point(latest_time, latest_force)
            
            # Get filtered data arrays
            filter_times, filter_forces = self.data_filter.get_filtered_data()
            
            # Time window
            current_time = latest_time
            start_time = max(0, current_time - window_seconds)
            
            # Filter data for time window
            times = np.array(timestamps)
            measured = np.array(measured_forces)
            
            time_mask = times >= start_time
            window_times = times[time_mask]
            window_measured = measured[time_mask]
            
            # Filter the filtered data for time window
            if len(filter_times) > 0:
                filter_mask = filter_times >= start_time
                window_filter_times = filter_times[filter_mask]
                window_filter_forces = filter_forces[filter_mask]
            else:
                window_filter_times = np.array([])
                window_filter_forces = np.array([])
            
            if len(window_times) == 0:
                return
            
            # Initialize bands if not done yet
            if not self.bands_initialized:
                self.initialize_force_bands([start_time, current_time])
            
            # Update plot lines
            if self.show_raw and len(window_times) > 0:
                self.raw_line.set_data(window_times, window_measured)
            
            if self.show_filtered and len(window_filter_times) > 0:
                self.filtered_line.set_data(window_filter_times, window_filter_forces)
            
            # Auto-scale plot - EXACT same as original
            self.ax_force.set_xlim(start_time, current_time)
            
            # Set Y-axis - EXACT same as original
            y_min = -0.5
            y_max = self.max_force + 2.5
            self.ax_force.set_ylim(y_min, y_max)
            
            self.draw()
            
        except Exception as e:
            print(f"Plot update error: {e}")


class FilterControlPanel(QGroupBox):
    """Clean filter control panel with better visibility for settings"""
    
    def __init__(self, canvas, parent=None):
        super().__init__("Filter Settings", parent)
        self.canvas = canvas
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(12)  # More space between controls
        
        # Filter enable/disable
        self.filter_checkbox = QCheckBox("Enable Filtering")
        self.filter_checkbox.setChecked(True)
        self.filter_checkbox.stateChanged.connect(self.on_filter_toggle)
        layout.addWidget(self.filter_checkbox)
        
        # Filter type selection - WIDER layout
        filter_layout = QVBoxLayout()  # Changed to vertical for better visibility
        filter_layout.addWidget(QLabel("Filter Type:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["butterworth", "moving_average", "savgol"])
        self.filter_combo.setCurrentText("butterworth")
        self.filter_combo.currentTextChanged.connect(self.on_filter_type_changed)
        self.filter_combo.setMinimumWidth(200)  # Ensure dropdown is wide enough
        filter_layout.addWidget(self.filter_combo)
        layout.addLayout(filter_layout)
        
        # Cutoff frequency (for Butterworth) - BETTER layout
        self.cutoff_layout = QVBoxLayout()
        cutoff_label = QLabel("Cutoff Frequency (Hz):")
        self.cutoff_layout.addWidget(cutoff_label)
        
        cutoff_container = QHBoxLayout()
        self.cutoff_slider = QSlider(Qt.Horizontal)
        self.cutoff_slider.setRange(1, 20)  # 0.1 to 2.0 Hz
        self.cutoff_slider.setValue(20)  # 2.0 Hz
        self.cutoff_slider.valueChanged.connect(self.on_cutoff_changed)
        self.cutoff_slider.setMinimumWidth(150)  # Make slider wider
        cutoff_container.addWidget(self.cutoff_slider)
        
        self.cutoff_label = QLabel("2.0")
        self.cutoff_label.setMinimumWidth(30)
        self.cutoff_label.setStyleSheet("font-weight: bold; color: #0066cc;")
        cutoff_container.addWidget(self.cutoff_label)
        
        self.cutoff_layout.addLayout(cutoff_container)
        layout.addLayout(self.cutoff_layout)
        
        # Filter order - BETTER layout
        order_layout = QVBoxLayout()
        order_layout.addWidget(QLabel("Filter Order:"))
        self.order_spin = QSpinBox()
        self.order_spin.setRange(1, 6)
        self.order_spin.setValue(2)
        self.order_spin.valueChanged.connect(self.on_order_changed)
        self.order_spin.setMinimumWidth(100)
        order_layout.addWidget(self.order_spin)
        layout.addLayout(order_layout)
        
        # Window size (for moving average and Savgol) - BETTER layout
        self.window_layout = QVBoxLayout()
        self.window_label = QLabel("Window Size:")
        self.window_layout.addWidget(self.window_label)
        self.window_spin = QSpinBox()
        self.window_spin.setRange(3, 21)
        self.window_spin.setValue(5)
        self.window_spin.valueChanged.connect(self.on_window_changed)
        self.window_spin.setMinimumWidth(100)
        self.window_layout.addWidget(self.window_spin)
        layout.addLayout(self.window_layout)
        
        # Initially hide window controls
        self.window_label.setVisible(False)
        self.window_spin.setVisible(False)
        
        # Add separator for better organization
        separator = QLabel()
        separator.setFixedHeight(1)
        separator.setStyleSheet("background-color: #cccccc; margin: 10px 0;")
        layout.addWidget(separator)
        
        # Plot visibility controls - BETTER layout
        vis_layout = QVBoxLayout()
        vis_layout.addWidget(QLabel("Display Options:"))
        
        self.show_raw_cb = QCheckBox("Show Raw Data")
        self.show_raw_cb.setChecked(True)
        self.show_raw_cb.stateChanged.connect(self.on_visibility_changed)
        vis_layout.addWidget(self.show_raw_cb)
        
        self.show_filtered_cb = QCheckBox("Show Filtered Data")
        self.show_filtered_cb.setChecked(True)
        self.show_filtered_cb.stateChanged.connect(self.on_visibility_changed)
        vis_layout.addWidget(self.show_filtered_cb)
        
        layout.addLayout(vis_layout)
        
        # Add stretch to push everything to top
        layout.addStretch()
        
        self.setLayout(layout)
    
    def on_filter_toggle(self):
        enabled = self.filter_checkbox.isChecked()
        self.canvas.set_filter_enabled(enabled)
    
    def on_filter_type_changed(self):
        filter_type = self.filter_combo.currentText()
        self.canvas.update_filter_settings(filter_type=filter_type)
        
        # Show/hide relevant controls
        is_butterworth = filter_type == "butterworth"
        is_windowed = filter_type in ["moving_average", "savgol"]
        
        # Cutoff frequency only for Butterworth
        for i in range(self.cutoff_layout.count()):
            item = self.cutoff_layout.itemAt(i)
            if item and item.widget():
                item.widget().setVisible(is_butterworth)
            elif item and item.layout():
                for j in range(item.layout().count()):
                    widget = item.layout().itemAt(j).widget()
                    if widget:
                        widget.setVisible(is_butterworth)
        
        # Window size for moving average and Savgol
        self.window_label.setVisible(is_windowed)
        self.window_spin.setVisible(is_windowed)
    
    def on_cutoff_changed(self):
        cutoff = self.cutoff_slider.value() / 10.0
        self.cutoff_label.setText(f"{cutoff:.1f}")
        self.canvas.update_filter_settings(cutoff_freq=cutoff)
    
    def on_order_changed(self):
        order = self.order_spin.value()
        self.canvas.update_filter_settings(filter_order=order, savgol_order=order)
    
    def on_window_changed(self):
        window = self.window_spin.value()
        self.canvas.update_filter_settings(window_size=window)
    
    def on_visibility_changed(self):
        show_raw = self.show_raw_cb.isChecked()
        show_filtered = self.show_filtered_cb.isChecked()
        self.canvas.set_plot_visibility(show_raw, show_filtered)


class SimpleForceGUI(QMainWindow):
    """EXACT same styling as original GUI but with enhanced functionality"""
    
    def __init__(self, grip_name, duration, max_force, pattern, grip_config, sampling_frequency=60.0):
        if not GUI_AVAILABLE:
            return
            
        super().__init__()
        
        # Store grip configuration
        self.grip_config = grip_config

        # Store sampling frequency
        self.sampling_frequency = sampling_frequency    
        
        # EXACT same window title and size as original
        self.setWindowTitle(f"Force Control Monitor - {grip_name} ({self.grip_config['min_force']:.1f}-{self.grip_config['max_force']:.1f}N)")
        self.setGeometry(100, 100, 1200, 800)
        
        # Experiment parameters
        self.grip_name = grip_name
        self.duration = duration
        self.max_force = max_force
        self.pattern = pattern
        
        # Data storage for GUI
        self.gui_data = {
            'timestamps': [],
            'measured_forces': []
        }
        
        # Initialize current_phase
        self.current_phase = "NEUTRAL"
        
        self.initUI()
        
        # Update timer - EXACT same as original
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.setInterval(33)  # ~30 Hz
        
    def initUI(self):
        """Initialize UI with compact text boxes and larger plot area"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout - horizontal to accommodate control panel
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)
        
        # Left side - plots and info
        left_layout = QVBoxLayout()
        left_layout.setSpacing(8)  # Reduced spacing between elements
        
        # Info panel - COMPACT boxes that fit text tightly
        info_layout = QHBoxLayout()
        info_layout.setSpacing(10)
        
        self.status_label = QLabel("Status: Starting...")
        self.status_label.setStyleSheet("font-weight: bold; padding: 4px 8px; background-color: #e8f5e8; border-radius: 4px;")
        self.status_label.setFixedHeight(28)  # Fixed compact height
        info_layout.addWidget(self.status_label)
        
        self.phase_label = QLabel("Phase: NEUTRAL")
        self.phase_label.setStyleSheet("font-weight: bold; padding: 4px 8px; background-color: #fff3cd; border-radius: 4px;")
        self.phase_label.setFixedHeight(28)  # Fixed compact height
        info_layout.addWidget(self.phase_label)
        
        # Force range info - corrected text but compact styling
        min_f = self.grip_config["min_force"]
        max_f = self.grip_config["max_force"]
        self.range_label = QLabel(f"Optimal Range: {min_f:.1f}N - {max_f:.1f}N")
        self.range_label.setStyleSheet("font-weight: bold; padding: 4px 8px; background-color: #d4edda; border-radius: 4px;")
        self.range_label.setFixedHeight(28)  # Fixed compact height
        info_layout.addWidget(self.range_label)
        
        left_layout.addLayout(info_layout)
        
        # Current values panel - COMPACT styling
        values_layout = QHBoxLayout()
        values_layout.setSpacing(10)
        
        self.current_force_label = QLabel("Current Force: 0.0N")
        self.current_force_label.setStyleSheet("font-family: monospace; font-size: 14px; font-weight: bold; padding: 4px 8px; background-color: #f8f9fa; border-radius: 4px;")
        self.current_force_label.setFixedHeight(30)  # Fixed compact height
        values_layout.addWidget(self.current_force_label)
        
        self.force_status_label = QLabel("Status: Within Range")
        self.force_status_label.setStyleSheet("font-family: monospace; font-size: 14px; font-weight: bold; padding: 4px 8px; background-color: #d4edda; border-radius: 4px;")
        self.force_status_label.setFixedHeight(30)  # Fixed compact height
        values_layout.addWidget(self.force_status_label)
        
        left_layout.addLayout(values_layout)
        
        # Plot canvas - NOW TAKES UP MOST OF THE SPACE
        self.canvas = SimplePlotCanvas(self, grip_config=self.grip_config, width=12, height=8, sampling_frequency=self.sampling_frequency)
        left_layout.addWidget(self.canvas, 1)  # Give it stretch factor so it expands
        
        # Stats panel - COMPACT styling  
        self.stats_label = QLabel("Samples: 0 | Avg Force: -- | Time in Range: --")
        self.stats_label.setStyleSheet("font-family: monospace; padding: 4px 8px; background-color: #e9ecef; border-radius: 4px;")
        self.stats_label.setFixedHeight(26)  # Fixed compact height
        left_layout.addWidget(self.stats_label)
        
        # Add left layout to main (3/4 width)
        main_layout.addLayout(left_layout, 3)
        
        # Right side - WIDER filter controls so you can see settings
        self.filter_panel = FilterControlPanel(self.canvas)
        self.filter_panel.setMinimumWidth(320)  # Much wider so you can see settings
        self.filter_panel.setMaximumWidth(400)  # But not too wide
        main_layout.addWidget(self.filter_panel, 1)
    
    def update_data(self, timestamp, measured_force, phase=""):
        """Update data for GUI - EXACT same as original"""
        self.gui_data['timestamps'].append(timestamp)
        self.gui_data['measured_forces'].append(measured_force)
        
        # Keep only recent data to prevent memory issues
        max_samples = 2000
        if len(self.gui_data['timestamps']) > max_samples:
            for key in self.gui_data:
                self.gui_data[key] = self.gui_data[key][-max_samples:]
        
        # Update phase
        self.current_phase = phase
    
    def get_force_status(self, force):
        """Determine force status - corrected logic but same colors"""
        min_f = self.grip_config["min_force"]
        max_f = self.grip_config["max_force"]
        
        if force < min_f:
            return "TRANSITION", "#f8f9fa"  # Light gray (neutral transition)
        elif force > max_f:
            return "OVERLOAD", "#ffebee"    # Light red (warning)
        else:
            return "OPTIMAL", "#e8f5e8"     # Light green (good)
    
    def update_display(self):
        """Update GUI display - enhanced but same interface"""
        if not self.gui_data['timestamps']:
            return
        
        try:
            # Update plots with filtering
            # window_seconds = self.duration * 4 if self.pattern == "all" else self.duration + 5
            window_seconds = 15  # Fixed window size for simplicity
            self.canvas.update_plots(
                self.gui_data['timestamps'],
                self.gui_data['measured_forces'],
                window_seconds=window_seconds
            )
            
            # Update current values
            if self.gui_data['timestamps']:
                latest_time = self.gui_data['timestamps'][-1]
                latest_measured = self.gui_data['measured_forces'][-1]
                
                # Get filtered value for display
                if hasattr(self.canvas.data_filter, 'filtered_buffer') and len(self.canvas.data_filter.filtered_buffer) > 0:
                    latest_filtered = list(self.canvas.data_filter.filtered_buffer)[-1]
                    display_force = latest_filtered if self.canvas.data_filter.filter_enabled else latest_measured
                else:
                    display_force = latest_measured
                
                # Update current force display
                self.current_force_label.setText(f"Current Force: {display_force:.2f}N")
                
                # Update force status with corrected logic
                status_text, bg_color = self.get_force_status(display_force)
                self.force_status_label.setText(f"Status: {status_text}")
                self.force_status_label.setStyleSheet(f"font-family: monospace; font-size: 14px; font-weight: bold; padding: 8px; background-color: {bg_color}; border-radius: 4px;")
                
                # Update phase - EXACT same styling as original
                self.phase_label.setText(f"Phase: {self.current_phase}")
                
                if self.current_phase == "APPROACH":
                    self.phase_label.setStyleSheet("font-weight: bold; padding: 8px; background-color: #fff3cd; border-radius: 4px;")
                elif self.current_phase == "FORCE_CONTROL":
                    self.phase_label.setStyleSheet("font-weight: bold; padding: 8px; background-color: #d4edda; border-radius: 4px;")
                else:
                    self.phase_label.setStyleSheet("font-weight: bold; padding: 8px; background-color: #e2e3e5; border-radius: 4px;")
            
            # Update statistics - enhanced info but same styling
            if len(self.gui_data['timestamps']) > 10:
                recent_count = min(100, len(self.gui_data['timestamps']))
                recent_measured = np.array(self.gui_data['measured_forces'][-recent_count:])
                
                avg_force = np.mean(recent_measured)
                
                # Calculate time in optimal range
                min_f = self.grip_config["min_force"]
                max_f = self.grip_config["max_force"]
                in_optimal = np.logical_and(recent_measured >= min_f, recent_measured <= max_f)
                optimal_percent = np.mean(in_optimal) * 100
                
                filter_status = "ON" if self.canvas.data_filter.filter_enabled else "OFF"
                
                self.stats_label.setText(
                    f"Samples: {len(self.gui_data['timestamps'])} | "
                    f"Avg Force: {avg_force:.2f}N | "
                    f"Time in Optimal: {optimal_percent:.1f}% | "
                    f"Filter: {filter_status}"
                )
            
        except Exception as e:
            print(f"GUI update error: {e}")
    
    def start_updates(self):
        """Start GUI updates - EXACT same as original"""
        self.update_timer.start()
        self.status_label.setText("Status: Experiment Running")
    
    def stop_updates(self):
        """Stop GUI updates - EXACT same as original"""
        self.update_timer.stop()
        self.status_label.setText("Status: Experiment Completed")
    
    def closeEvent(self, event):
        """Handle window close - EXACT same as original"""
        try:
            if hasattr(self, 'update_timer') and self.update_timer.isActive():
                self.update_timer.stop()
        except:
            pass
        event.accept()

class ThreadedForceGUI(SimpleForceGUI):
    """Modified GUI that consumes data from control thread via queue"""
    
    def __init__(self, grip_name, duration, max_force, pattern, grip_config, 
                 data_queue, control_thread=None, sampling_frequency=60.0):
        super().__init__(grip_name, duration, max_force, pattern, grip_config, sampling_frequency)
        
        self.data_queue = data_queue
        self.control_thread = control_thread
        self.experiment_completed = False
        
        # GUI update timer (replaces the direct update calls)
        from PyQt5.QtCore import QTimer
        self.gui_timer = QTimer()
        self.gui_timer.timeout.connect(self.consume_control_data)
        self.gui_timer.setInterval(33)  # 30Hz GUI updates
        
    def start_updates(self):
        """Start GUI updates via timer"""
        self.gui_timer.start()
        self.status_label.setText("Status: Experiment Running")
        print("GUI updates started at 30Hz")
    
    def stop_updates(self):
        """Stop GUI updates"""
        self.gui_timer.stop()
        self.status_label.setText("Status: Experiment Completed")
        print("GUI updates stopped")
    
    def consume_control_data(self):
        """Consume data from control thread (runs at 30Hz)"""
        latest_sample = None
        
        # Drain the queue - only keep latest for GUI
        while True:
            try:
                sample = self.data_queue.get_nowait()
                
                # Check for completion
                if isinstance(sample, dict) and sample.get("done", False):
                    self.experiment_completed = True
                    reason = sample.get("reason", "unknown")
                    print(f"Experiment completed: {reason}")
                    self.stop_updates()
                    return
                
                latest_sample = sample
            except queue_module.Empty:
                break
        
        if latest_sample is None:
            return
        
        # Update GUI data (same as before)
        self.gui_data['timestamps'].append(latest_sample['timestamp'])
        self.gui_data['measured_forces'].append(latest_sample['measured_force'])
        
        # Keep only recent data
        max_samples = 2000
        if len(self.gui_data['timestamps']) > max_samples:
            for key in self.gui_data:
                self.gui_data[key] = self.gui_data[key][-max_samples:]
        
        self.current_phase = latest_sample['phase']
        
        # Update display (same as before)
        self.update_display()
    
    def get_experiment_data(self):
        """Get full experiment data from control thread"""
        if self.control_thread:
            return self.control_thread.get_experiment_data()
        return None

# Convenience function for backward compatibility
def create_force_gui(grip_name, duration, max_force, pattern, grip_config):
    """Create and return a SimpleForceGUI instance"""
    return SimpleForceGUI(grip_name, duration, max_force, pattern, grip_config)