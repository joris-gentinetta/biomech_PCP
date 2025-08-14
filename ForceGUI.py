#!/usr/bin/env python3
import sys
import time
import numpy as np
import threading
from collections import deque
import argparse

# PyQt5 imports for GUI
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QLineEdit, QComboBox, QCheckBox, QGroupBox, QTextEdit, QTabWidget,
    QSlider, QSpinBox, QGridLayout, QProgressBar
)
from PyQt5.QtCore import QTimer, pyqtSignal, QObject
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Import your prosthetic hand class
sys.path.append('C:/Users/Emanuel Wicki/Documents/MIT/biomech_PCP/helpers')
from psyonicHand import psyonicArm

class AutomaticBaselineCalibrator(QObject):
    """
    Automatic baseline calibration system for force sensors
    """
    
    # Signals for GUI updates
    calibration_progress = pyqtSignal(int)  # Progress percentage
    calibration_complete = pyqtSignal(dict)  # Results dictionary
    calibration_status = pyqtSignal(str)    # Status messages
    
    def __init__(self, num_sensors=30, sampling_rate=100):
        super().__init__()
        self.num_sensors = num_sensors
        self.sampling_rate = sampling_rate
        
        # Calibration parameters
        self.calibration_duration = 10.0  # seconds
        self.required_samples = int(self.calibration_duration * self.sampling_rate)
        
        # Calibration data storage
        self.calibration_data = []
        self.is_calibrating = False
        self.calibration_complete_flag = False
        
        # Calibration results
        self.baseline_offsets = np.zeros(num_sensors)
        self.noise_thresholds = np.zeros(num_sensors)  # Max values after offset removal
        
        # Post-calibration filtering
        self.filtering_enabled = False
        
        # Statistics for validation
        self.calibration_stats = {}
    
    def start_calibration(self):
        """Start the 10-second calibration process"""
        if self.is_calibrating:
            return False
        
        print("=== STARTING AUTOMATIC BASELINE CALIBRATION ===")
        print("IMPORTANT: Do NOT touch any force sensors for the next 10 seconds!")
        print("Keep the prosthetic hand completely relaxed and untouched.")
        print()
        
        # Reset calibration data
        self.calibration_data = []
        self.is_calibrating = True
        self.calibration_complete_flag = False
        
        # Emit status signals
        self.calibration_status.emit("Starting calibration - Do NOT touch sensors!")
        self.calibration_progress.emit(0)
        
        return True
    
    def add_calibration_sample(self, force_readings):
        """
        Add a sample during calibration period
        force_readings: numpy array of shape (num_sensors,)
        """
        if not self.is_calibrating:
            return False
        
        # Store the sample
        self.calibration_data.append(force_readings.copy())
        
        # Calculate progress
        progress = int((len(self.calibration_data) / self.required_samples) * 100)
        self.calibration_progress.emit(min(progress, 100))
        
        # Update status
        elapsed_time = len(self.calibration_data) / self.sampling_rate
        remaining_time = self.calibration_duration - elapsed_time
        
        if remaining_time > 0:
            self.calibration_status.emit(
                f"Calibrating... {remaining_time:.1f}s remaining - Keep sensors untouched!"
            )
        
        # Check if calibration is complete
        if len(self.calibration_data) >= self.required_samples:
            self.complete_calibration()
            return True
        
        return False
    
    def complete_calibration(self):
        """Complete the calibration process and calculate baselines"""
        if not self.is_calibrating or len(self.calibration_data) < self.required_samples:
            return
        
        print("\n=== PROCESSING CALIBRATION DATA ===")
        self.calibration_status.emit("Processing calibration data...")
        
        # Convert to numpy array for analysis
        calibration_array = np.array(self.calibration_data)  # Shape: (samples, sensors)
        
        print(f"Collected {len(self.calibration_data)} samples ({self.calibration_duration}s)")
        print(f"Data shape: {calibration_array.shape}")
        
        # Step 1: Calculate baseline offsets (mean values)
        self.baseline_offsets = np.mean(calibration_array, axis=0)
        
        # Step 2: Remove offsets from calibration data
        offset_corrected_data = calibration_array - self.baseline_offsets
        
        # Step 3: Find maximum absolute values after offset removal (noise threshold)
        # Use a more robust approach - take 100th percentile instead of absolute max
        # This helps avoid outliers while still capturing the noise level
        self.noise_thresholds = np.percentile(np.abs(offset_corrected_data), 100, axis=0)
        
        # Ensure minimum threshold to avoid division by zero or too-sensitive filtering
        min_threshold = 0.01  # Minimum 0.01N threshold
        self.noise_thresholds = np.maximum(self.noise_thresholds, min_threshold)
        
        # Store comprehensive statistics
        self.calibration_stats = {
            'baseline_offsets': self.baseline_offsets,
            'noise_thresholds': self.noise_thresholds,
            'samples_collected': len(self.calibration_data),
            'duration': self.calibration_duration,
            'raw_data_mean': np.mean(calibration_array, axis=0),
            'raw_data_std': np.std(calibration_array, axis=0),
            'raw_data_min': np.min(calibration_array, axis=0),
            'raw_data_max': np.max(calibration_array, axis=0),
            'offset_corrected_max': np.max(np.abs(offset_corrected_data), axis=0),
            'offset_corrected_95th': self.noise_thresholds
        }
        
        # Print detailed results
        self.print_calibration_results()
        
        # Validate calibration quality
        if self.validate_calibration():
            self.is_calibrating = False
            self.calibration_complete_flag = True
            self.filtering_enabled = True
            
            # Emit completion signal
            self.calibration_complete.emit(self.calibration_stats)
            self.calibration_status.emit("Calibration complete! Zero baseline established.")
            self.calibration_progress.emit(100)
            
            print("=== CALIBRATION SUCCESSFUL ===")
            print("You can now use the force sensors normally.")
            print("All readings will be automatically corrected for offset and noise.")
            
        else:
            self.calibration_status.emit("Calibration failed! Please retry - ensure no sensor contact.")
            self.is_calibrating = False
            print("=== CALIBRATION FAILED ===")
            print("Please retry calibration - ensure sensors are completely untouched.")
    
    def print_calibration_results(self):
        """Print detailed calibration results"""
        print("\n=== CALIBRATION RESULTS ===")
        
        # Overall statistics
        print(f"Samples collected: {len(self.calibration_data)}")
        print(f"Duration: {self.calibration_duration}s")
        print(f"Sampling rate: {self.sampling_rate} Hz")
        
        # Per-sensor results (summarized)
        print(f"\nSensor Statistics Summary:")
        print(f"Baseline offset range: {np.min(self.baseline_offsets):.4f} to {np.max(self.baseline_offsets):.4f}")
        print(f"Noise threshold range: {np.min(self.noise_thresholds):.4f} to {np.max(self.noise_thresholds):.4f}")
        
        # Show some raw statistics for debugging
        if 'offset_corrected_max' in self.calibration_stats:
            max_abs_values = self.calibration_stats['offset_corrected_max']
            print(f"Max absolute values (after offset removal): {np.min(max_abs_values):.4f} to {np.max(max_abs_values):.4f}")
            print(f"95th percentile values (used as thresholds): {np.min(self.noise_thresholds):.4f} to {np.max(self.noise_thresholds):.4f}")
        
        # Finger-by-finger summary (assuming 5 fingers × 6 sensors each)
        finger_names = ["Index", "Middle", "Ring", "Pinky", "Thumb"]
        print(f"\nPer-Finger Summary:")
        print(f"{'Finger':<8} {'Avg Offset':<12} {'Avg Noise Thr':<12} {'Max Variation':<12}")
        print("-" * 50)
        
        calibration_array = np.array(self.calibration_data)
        
        for i, finger in enumerate(finger_names):
            start_idx = i * 6
            end_idx = start_idx + 6
            
            finger_offsets = self.baseline_offsets[start_idx:end_idx]
            finger_thresholds = self.noise_thresholds[start_idx:end_idx]
            
            # Calculate max variation during calibration for this finger
            finger_data = calibration_array[:, start_idx:end_idx]
            finger_variation = np.max([np.max(finger_data[:, j]) - np.min(finger_data[:, j]) for j in range(6)])
            
            avg_offset = np.mean(finger_offsets)
            avg_threshold = np.mean(finger_thresholds)
            
            print(f"{finger:<8} {avg_offset:<12.4f} {avg_threshold:<12.4f} {finger_variation:<12.4f}")
        
        # Additional diagnostic info
        print(f"\nDiagnostic Info:")
        print(f"Total sensors: {self.num_sensors}")
        print(f"Sensors with offset > 0.1N: {np.sum(np.abs(self.baseline_offsets) > 0.1)}")
        print(f"Sensors with noise threshold > 0.1N: {np.sum(self.noise_thresholds > 0.1)}")
        print(f"Minimum noise threshold: {np.min(self.noise_thresholds):.6f}")
        print(f"Maximum noise threshold: {np.max(self.noise_thresholds):.6f}")
    
    def validate_calibration(self):
        """Validate calibration quality with more realistic thresholds"""
        # Check for reasonable offset values
        if np.any(np.abs(self.baseline_offsets) > 100):  # Very large offsets
            print("WARNING: Very large baseline offsets detected!")
            print(f"Max offset: {np.max(np.abs(self.baseline_offsets)):.3f}")
            return False
        
        # Check for extremely high noise (suggests movement during calibration)
        if np.any(self.noise_thresholds > 50):  # Very high noise
            print("WARNING: Very high noise thresholds detected - sensors may have been touched!")
            print(f"Max noise threshold: {np.max(self.noise_thresholds):.3f}")
            return False
        
        # Relax the minimum noise threshold - very stable sensors are actually good!
        if np.any(self.noise_thresholds < 0.0001):  # Extremely low (possibly broken sensors)
            print("WARNING: Extremely low noise thresholds - possible sensor malfunction!")
            print(f"Min noise threshold: {np.min(self.noise_thresholds):.6f}")
            return False
        
        # Check for reasonable data collection
        if len(self.calibration_data) < self.required_samples * 0.9:  # Less than 90% of expected samples
            print(f"WARNING: Insufficient calibration data! Got {len(self.calibration_data)}, expected {self.required_samples}")
            return False
        
        # Additional validation: check for obvious sensor movement during calibration
        calibration_array = np.array(self.calibration_data)
        for sensor_idx in range(self.num_sensors):
            sensor_data = calibration_array[:, sensor_idx]
            sensor_range = np.max(sensor_data) - np.min(sensor_data)
            
            # If any sensor shows large variation, it might indicate movement
            if sensor_range > 5.0:  # More than 5N variation during "no-touch" period
                print(f"WARNING: Large variation detected on sensor {sensor_idx} (range: {sensor_range:.3f}N)")
                print("This suggests sensors were touched or moved during calibration!")
                return False
        
        print("Calibration validation PASSED!")
        print(f"Baseline offsets: {np.min(self.baseline_offsets):.3f} to {np.max(self.baseline_offsets):.3f}")
        print(f"Noise thresholds: {np.min(self.noise_thresholds):.3f} to {np.max(self.noise_thresholds):.3f}")
        return True
    
    def apply_filtering(self, raw_forces):
        """
        Apply calibrated filtering to raw force data
        Returns corrected forces with zero baseline and noise removal
        """
        if not self.calibration_complete_flag or not self.filtering_enabled:
            return raw_forces
        
        # Step 1: Remove baseline offset
        offset_corrected = raw_forces - self.baseline_offsets
        
        # Step 2: Apply noise threshold (subtract max noise value)
        noise_corrected = offset_corrected - self.noise_thresholds
        
        # Step 3: Ensure no negative forces (clamp to zero)
        filtered_forces = np.maximum(noise_corrected, 0.0)
        
        return filtered_forces
    
    def get_calibration_summary(self):
        """Get a summary of calibration results for display"""
        if not self.calibration_complete_flag:
            return "Calibration not completed"
        
        summary = []
        summary.append(f"Calibration Status: Complete")
        summary.append(f"Duration: {self.calibration_duration}s")
        summary.append(f"Samples: {len(self.calibration_data)}")
        summary.append(f"")
        summary.append(f"Offset Range: {np.min(self.baseline_offsets):.3f} to {np.max(self.baseline_offsets):.3f}")
        summary.append(f"Noise Threshold Range: {np.min(self.noise_thresholds):.3f} to {np.max(self.noise_thresholds):.3f}")
        
        return "\n".join(summary)
    
    def reset_calibration(self):
        """Reset all calibration data"""
        self.calibration_data = []
        self.is_calibrating = False
        self.calibration_complete_flag = False
        self.filtering_enabled = False
        
        self.baseline_offsets = np.zeros(self.num_sensors)
        self.noise_thresholds = np.zeros(self.num_sensors)
        self.calibration_stats = {}
        
        self.calibration_status.emit("Calibration reset")
        self.calibration_progress.emit(0)

class ForceGuiCanvas(FigureCanvas):
    """Custom matplotlib canvas for force plotting"""
    
    def __init__(self, parent=None, width=12, height=8, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        
        # Create subplots for each finger + total
        self.axes = {}
        
        # Create 2x3 grid
        finger_names = ["index", "middle", "ring", "pinky", "thumb"]
        for i, finger in enumerate(finger_names):
            ax = self.fig.add_subplot(2, 3, i + 1)
            ax.set_title(f'{finger.capitalize()} Force', fontsize=10)
            ax.set_xlabel('Time (s)', fontsize=8)
            ax.set_ylabel('Force (N)', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
            self.axes[finger] = ax
        
        # Total grip force plot
        total_ax = self.fig.add_subplot(2, 3, 6)
        total_ax.set_title('Total Grip Force', fontsize=10, weight='bold')
        total_ax.set_xlabel('Time (s)', fontsize=8)
        total_ax.set_ylabel('Force (N)', fontsize=8)
        total_ax.grid(True, alpha=0.3)
        total_ax.tick_params(labelsize=8)
        self.axes['total'] = total_ax
        
        self.fig.tight_layout()
        
        # Data lines
        self.lines = {}
        for finger in finger_names:
            line, = self.axes[finger].plot([], [], 'b-', linewidth=2)
            self.lines[finger] = line
        
        # Total grip force line
        total_line, = self.axes['total'].plot([], [], 'r-', linewidth=3)
        self.lines['total'] = total_line
    
    def update_plots(self, timestamps, force_data, window_seconds=15):
        """Update all plots with new data - thread-safe version"""
        if not timestamps or len(timestamps) == 0:
            return
        
        try:
            # Convert to numpy arrays for easier manipulation
            times = np.array(timestamps)
            current_time = times[-1] if len(times) > 0 else 0
            start_time = max(0, current_time - window_seconds)
            
            # Create time mask for window
            time_mask = times >= start_time
            window_times = times[time_mask]
            
            if len(window_times) == 0:
                return
            
            # Update individual finger plots
            finger_names = ["index", "middle", "ring", "pinky", "thumb"]
            for finger in finger_names:
                if finger in force_data and len(force_data[finger]) > 0:
                    # Ensure arrays are same length before masking
                    finger_data = np.array(force_data[finger])
                    
                    # Handle length mismatch by truncating to minimum length
                    min_length = min(len(times), len(finger_data))
                    if min_length == 0:
                        continue
                    
                    times_truncated = times[:min_length]
                    finger_data_truncated = finger_data[:min_length]
                    
                    # Recalculate time mask for truncated data
                    time_mask_truncated = times_truncated >= start_time
                    window_times_truncated = times_truncated[time_mask_truncated]
                    finger_forces = finger_data_truncated[time_mask_truncated]
                    
                    if len(finger_forces) > 0 and len(window_times_truncated) > 0:
                        self.lines[finger].set_data(window_times_truncated, finger_forces)
                        
                        # Auto-scale
                        ax = self.axes[finger]
                        ax.set_xlim(start_time, current_time)
                        max_force = max(finger_forces) if len(finger_forces) > 0 else 2
                        ax.set_ylim(0, max(max_force * 1.1, 2))
            
            # Update total grip force
            if 'total_grip' in force_data and len(force_data['total_grip']) > 0:
                total_data = np.array(force_data['total_grip'])
                
                # Handle length mismatch
                min_length = min(len(times), len(total_data))
                if min_length > 0:
                    times_truncated = times[:min_length]
                    total_data_truncated = total_data[:min_length]
                    
                    # Recalculate time mask
                    time_mask_truncated = times_truncated >= start_time
                    window_times_truncated = times_truncated[time_mask_truncated]
                    total_forces = total_data_truncated[time_mask_truncated]
                    
                    if len(total_forces) > 0 and len(window_times_truncated) > 0:
                        self.lines['total'].set_data(window_times_truncated, total_forces)
                        
                        # Auto-scale
                        ax = self.axes['total']
                        ax.set_xlim(start_time, current_time)
                        max_total = max(total_forces) if len(total_forces) > 0 else 5
                        ax.set_ylim(0, max(max_total * 1.1, 5))
            
            self.draw()
            
        except Exception as e:
            # Silently handle plotting errors to prevent crashes
            print(f"Plot update error (non-critical): {e}")
            pass

class ForceMonitorGUI(QMainWindow):
    """Main GUI window for force monitoring with automatic baseline calibration"""
    
    def __init__(self, connected_arm):
        super().__init__()
        self.setWindowTitle("Prosthetic Hand Force Monitor with Auto-Calibration")
        self.setGeometry(100, 100, 1600, 900)
        
        # Use the pre-connected arm
        self.arm = connected_arm
        self.connected = True
        self.collecting_data = False
        
        # Data storage
        self.finger_names = ["index", "middle", "ring", "pinky", "thumb"]
        self.force_history = {
            'timestamps': [],
            'index': [], 'middle': [], 'ring': [], 'pinky': [], 'thumb': [],
            'total_grip': []
        }
        
        # Current force data
        self.current_forces = {}
        self.start_time = None
        
        # Add automatic baseline calibrator
        self.baseline_calibrator = AutomaticBaselineCalibrator(
            num_sensors=30,  # 5 fingers × 6 sensors
            sampling_rate=100
        )
        
        # Connect calibrator signals
        self.baseline_calibrator.calibration_progress.connect(self.update_calibration_progress)
        self.baseline_calibrator.calibration_complete.connect(self.on_calibration_complete)
        self.baseline_calibrator.calibration_status.connect(self.update_calibration_status)
        
        self.initUI()
        
        # Data collection thread and timer
        self.data_thread = None
        self.stop_event = None
        
        # Update timer for GUI
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.setInterval(100)  # 10 Hz GUI updates
        
        # Show connected status
        self.update_status("Hand connected - CALIBRATION REQUIRED before use")
    
    def initUI(self):
        """Initialize the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Left side: Controls
        controls_widget = QWidget()
        controls_widget.setMaximumWidth(380)
        controls_layout = QVBoxLayout(controls_widget)
        
        # Connection status group
        conn_group = QGroupBox("Hardware Status")
        conn_layout = QVBoxLayout(conn_group)
        
        # Status label
        self.status_label = QLabel("Status: Connected")
        self.status_label.setStyleSheet("padding: 5px; border: 1px solid green; background-color: #e8f5e8;")
        conn_layout.addWidget(self.status_label)
        
        # Disconnect button
        self.disconnect_btn = QPushButton("Disconnect Hand")
        self.disconnect_btn.clicked.connect(self.disconnect_hand)
        conn_layout.addWidget(self.disconnect_btn)
        
        controls_layout.addWidget(conn_group)
        
        # CALIBRATION GROUP - Added first and prominently
        calibration_group = QGroupBox(" AUTOMATIC BASELINE CALIBRATION")
        calibration_group.setStyleSheet("QGroupBox { font-weight: bold; color: red; }")
        calibration_layout = QVBoxLayout(calibration_group)
        
        # Instructions
        instructions = QLabel(
            "REQUIRED BEFORE FIRST USE:\n"
            "1. Ensure NO force sensors are touched\n"
            "2. Keep prosthetic hand completely relaxed\n"
            "3. Click 'Start Calibration' for 10-second baseline capture\n"
            "4. This removes offset, drift, and noise automatically"
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("padding: 8px; background-color: #fff3cd; border: 2px solid #ffc107; border-radius: 5px;")
        calibration_layout.addWidget(instructions)
        
        # Calibration button
        self.calibrate_auto_btn = QPushButton("🚀 START 10-SECOND CALIBRATION")
        self.calibrate_auto_btn.clicked.connect(self.start_automatic_calibration)
        self.calibrate_auto_btn.setStyleSheet(
            "font-weight: bold; padding: 12px; background-color: #007bff; color: white; border-radius: 5px;"
        )
        calibration_layout.addWidget(self.calibrate_auto_btn)
        
        # Progress bar
        self.calibration_progress_bar = QProgressBar()
        self.calibration_progress_bar.setVisible(False)
        self.calibration_progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #28a745; }")
        calibration_layout.addWidget(self.calibration_progress_bar)
        
        # Calibration status label
        self.calibration_status_label = QLabel("Status: Not calibrated - Data collection disabled")
        self.calibration_status_label.setStyleSheet("padding: 5px; border: 1px solid #dc3545; background-color: #f8d7da; color: #721c24;")
        calibration_layout.addWidget(self.calibration_status_label)
        
        # Reset button
        self.reset_calibration_btn = QPushButton("Reset Calibration")
        self.reset_calibration_btn.clicked.connect(self.reset_calibration)
        calibration_layout.addWidget(self.reset_calibration_btn)
        
        # Calibration summary
        self.calibration_summary = QTextEdit()
        self.calibration_summary.setMaximumHeight(100)
        self.calibration_summary.setReadOnly(True)
        self.calibration_summary.setFont(QtGui.QFont("Courier", 8))
        self.calibration_summary.setPlainText("No calibration data available")
        calibration_layout.addWidget(self.calibration_summary)
        
        controls_layout.addWidget(calibration_group)
        
        # Data collection group
        data_group = QGroupBox("Data Collection")
        data_layout = QVBoxLayout(data_group)
        
        self.start_btn = QPushButton("Start Data Collection")
        self.start_btn.clicked.connect(self.start_data_collection)
        self.start_btn.setEnabled(False)  # Disabled until calibration
        data_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Stop Data Collection")
        self.stop_btn.clicked.connect(self.stop_data_collection)
        self.stop_btn.setEnabled(False)
        data_layout.addWidget(self.stop_btn)
        
        self.clear_btn = QPushButton("Clear Data")
        self.clear_btn.clicked.connect(self.clear_data)
        data_layout.addWidget(self.clear_btn)
        
        # Data count label
        self.data_count_label = QLabel("Samples: 0")
        data_layout.addWidget(self.data_count_label)
        
        controls_layout.addWidget(data_group)
        
        # Display settings group
        display_group = QGroupBox("Display Settings")
        display_layout = QVBoxLayout(display_group)
        
        # Time window
        window_layout = QHBoxLayout()
        window_layout.addWidget(QLabel("Time Window (s):"))
        self.window_spin = QSpinBox()
        self.window_spin.setRange(5, 60)
        self.window_spin.setValue(15)
        window_layout.addWidget(self.window_spin)
        display_layout.addLayout(window_layout)
        
        # Update rate
        rate_layout = QHBoxLayout()
        rate_layout.addWidget(QLabel("Update Rate (Hz):"))
        self.rate_spin = QSpinBox()
        self.rate_spin.setRange(1, 30)
        self.rate_spin.setValue(10)
        self.rate_spin.valueChanged.connect(self.update_rate_changed)
        rate_layout.addWidget(self.rate_spin)
        display_layout.addLayout(rate_layout)
        
        controls_layout.addWidget(display_group)
        
        # Current values display
        values_group = QGroupBox("Current Force Values (N)")
        values_layout = QVBoxLayout(values_group)
        
        self.values_text = QTextEdit()
        self.values_text.setMaximumHeight(180)
        self.values_text.setFont(QtGui.QFont("Courier", 9))
        self.values_text.setReadOnly(True)
        values_layout.addWidget(self.values_text)
        
        controls_layout.addWidget(values_group)
        
        # Statistics display
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_text = QTextEdit()
        self.stats_text.setMaximumHeight(150)
        self.stats_text.setFont(QtGui.QFont("Courier", 8))
        self.stats_text.setReadOnly(True)
        stats_layout.addWidget(self.stats_text)
        
        controls_layout.addWidget(stats_group)
        
        # Hand control group (optional)
        control_group = QGroupBox("Hand Control")
        control_layout = QVBoxLayout(control_group)
        
        # Quick position buttons
        pos_layout = QHBoxLayout()
        
        self.open_btn = QPushButton("Open Hand")
        self.open_btn.clicked.connect(self.open_hand)
        pos_layout.addWidget(self.open_btn)
        
        self.close_btn = QPushButton("Close Hand")
        self.close_btn.clicked.connect(self.close_hand)
        pos_layout.addWidget(self.close_btn)
        
        control_layout.addLayout(pos_layout)
        controls_layout.addWidget(control_group)
        
        controls_layout.addStretch()
        
        # Right side: Plots
        self.canvas = ForceGuiCanvas(self, width=10, height=8)
        
        # Add to main layout
        main_layout.addWidget(controls_widget)
        main_layout.addWidget(self.canvas, stretch=1)
    
    def start_automatic_calibration(self):
        """Start the automatic calibration process"""
        if self.baseline_calibrator.start_calibration():
            self.calibrate_auto_btn.setEnabled(False)
            self.calibrate_auto_btn.setText("🔄 CALIBRATING...")
            self.calibration_progress_bar.setVisible(True)
            self.calibration_progress_bar.setValue(0)
            
            # Disable data collection during calibration if running
            if self.collecting_data:
                self.stop_data_collection()
            
            # Start data collection specifically for calibration
            self.start_calibration_data_collection()
    
    def start_calibration_data_collection(self):
        """Start data collection specifically for calibration"""
        if not self.connected or not self.arm:
            return
        
        self.collecting_data = True
        self.start_time = time.time()
        self.stop_event = threading.Event()
        
        # Start calibration data collection thread
        self.data_thread = threading.Thread(target=self.calibration_data_worker, daemon=True)
        self.data_thread.start()
    
    def calibration_data_worker(self):
        """Data collection worker specifically for calibration"""
        while not self.stop_event.is_set() and self.baseline_calibrator.is_calibrating:
            try:
                if not self.arm or not self.connected:
                    break
                
                # Collect raw force data from all sensors
                raw_forces = []
                
                for finger in self.finger_names:
                    for sensor_idx in range(6):
                        sensor_name = f"{finger}{sensor_idx}_Force"
                        force_value = self.arm.sensors.get(sensor_name, 0)
                        raw_forces.append(force_value)
                
                # Send to calibrator
                calibration_complete = self.baseline_calibrator.add_calibration_sample(
                    np.array(raw_forces)
                )
                
                if calibration_complete:
                    break
                
                # Sleep for sampling rate
                time.sleep(0.01)  # 100 Hz
                
            except Exception as e:
                print(f"Calibration data collection error: {e}")
                time.sleep(0.1)
        
        # Stop data collection
        self.collecting_data = False
    
    def update_calibration_progress(self, progress):
        """Update calibration progress bar"""
        self.calibration_progress_bar.setValue(progress)
    
    def update_calibration_status(self, status):
        """Update calibration status"""
        self.calibration_status_label.setText(status)
    
    def on_calibration_complete(self, stats):
        """Handle calibration completion"""
        self.calibrate_auto_btn.setEnabled(True)
        self.calibrate_auto_btn.setText("🚀 START 10-SECOND CALIBRATION")
        self.calibration_progress_bar.setVisible(False)
        
        # Update status
        self.calibration_status_label.setText("Status: Calibrated - Ready for data collection")
        self.calibration_status_label.setStyleSheet("padding: 5px; border: 1px solid #28a745; background-color: #d4edda; color: #155724;")
        
        # Update summary
        summary = self.baseline_calibrator.get_calibration_summary()
        self.calibration_summary.setPlainText(summary)
        
        # Enable data collection
        self.start_btn.setEnabled(True)
        
        # Update main status
        self.update_status("Hand calibrated and ready for data collection")
    
    def reset_calibration(self):
        """Reset calibration"""
        self.baseline_calibrator.reset_calibration()
        
        # Update UI
        self.calibration_status_label.setText("Status: Not calibrated - Data collection disabled")
        self.calibration_status_label.setStyleSheet("padding: 5px; border: 1px solid #dc3545; background-color: #f8d7da; color: #721c24;")
        self.calibration_summary.setPlainText("No calibration data available")
        
        # Disable data collection until recalibrated
        self.start_btn.setEnabled(False)
        self.update_status("Hand connected - CALIBRATION REQUIRED before use")
    
    def update_status(self, status):
        """Update status label"""
        self.status_label.setText(f"Status: {status}")
    
    def disconnect_hand(self):
        """Disconnect from prosthetic hand"""
        try:
            # Stop data collection first
            if self.collecting_data:
                self.stop_data_collection()
            
            # Close arm connection
            if self.arm:
                self.arm.close()
                self.arm = None
            
            self.connected = False
            self.update_status("Hand disconnected")
            
        except Exception as e:
            self.update_status(f"Disconnect error: {str(e)}")
        
        finally:
            # Update UI
            self.disconnect_btn.setEnabled(False)
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
            self.open_btn.setEnabled(False)
            self.close_btn.setEnabled(False)
    
    def data_collection_worker(self):
        """Enhanced data collection worker with automatic filtering"""
        while not self.stop_event.is_set():
            try:
                if not self.arm or not self.connected:
                    break
                
                # Get current timestamp
                current_time = time.time() - self.start_time
                
                # Collect RAW force data from all sensors
                raw_forces = []
                force_data = {}
                
                for finger in self.finger_names:
                    finger_forces = []
                    for sensor_idx in range(6):
                        sensor_name = f"{finger}{sensor_idx}_Force"
                        force_value = self.arm.sensors.get(sensor_name, 0)
                        finger_forces.append(force_value)
                        raw_forces.append(force_value)
                    
                    force_data[finger] = {'individual': finger_forces}
                
                # APPLY AUTOMATIC BASELINE FILTERING
                raw_forces_array = np.array(raw_forces)
                filtered_forces = self.baseline_calibrator.apply_filtering(raw_forces_array)
                
                # Reconstruct force_data with filtered values
                idx = 0
                total_grip_force = 0
                
                for finger in self.finger_names:
                    finger_filtered = filtered_forces[idx:idx+6]
                    force_data[finger]['individual'] = finger_filtered.tolist()
                    force_data[finger]['total'] = np.sum(finger_filtered)
                    total_grip_force += force_data[finger]['total']
                    idx += 6
                
                # Store current forces for GUI access (thread-safe single assignment)
                current_forces_snapshot = force_data.copy()
                current_forces_snapshot['total_grip'] = total_grip_force
                current_forces_snapshot['timestamp'] = current_time
                self.current_forces = current_forces_snapshot
                
                # Thread-safe update of history - add all data atomically
                new_timestamp = current_time
                new_finger_data = {}
                for finger in self.finger_names:
                    new_finger_data[finger] = force_data[finger]['total']
                new_total_grip = total_grip_force
                
                # Atomic append to all arrays
                self.force_history['timestamps'].append(new_timestamp)
                for finger in self.finger_names:
                    self.force_history[finger].append(new_finger_data[finger])
                self.force_history['total_grip'].append(new_total_grip)
                
                # Limit history size to prevent memory issues - maintain synchronization
                max_samples = 10000
                if len(self.force_history['timestamps']) > max_samples:
                    # Remove oldest samples from all arrays simultaneously
                    samples_to_remove = len(self.force_history['timestamps']) - max_samples
                    self.force_history['timestamps'] = self.force_history['timestamps'][samples_to_remove:]
                    for finger in self.finger_names:
                        self.force_history[finger] = self.force_history[finger][samples_to_remove:]
                    self.force_history['total_grip'] = self.force_history['total_grip'][samples_to_remove:]
                
                # Sleep for sampling rate (approximately 100 Hz)
                time.sleep(0.01)
                
            except Exception as e:
                print(f"Data collection error: {e}")
                time.sleep(0.1)
    
    def start_data_collection(self):
        """Start data collection - only works after calibration"""
        if not self.baseline_calibrator.calibration_complete_flag:
            self.update_status("ERROR: Calibration required before data collection!")
            return
        
        if not self.connected or not self.arm:
            self.update_status("Hand not connected!")
            return
        
        if self.collecting_data:
            return
        
        # Clear previous data
        self.clear_data()
        
        # Set up data collection
        self.collecting_data = True
        self.start_time = time.time()
        self.stop_event = threading.Event()
        
        # Start data collection thread
        self.data_thread = threading.Thread(target=self.data_collection_worker, daemon=True)
        self.data_thread.start()
        
        # Start GUI updates
        self.update_timer.start()
        
        # Update UI
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.update_status("Data collection started (filtered data)")
    
    def stop_data_collection(self):
        """Stop data collection"""
        if not self.collecting_data:
            return
        
        # Stop data collection
        self.collecting_data = False
        if self.stop_event:
            self.stop_event.set()
        
        # Wait for thread to finish
        if self.data_thread and self.data_thread.is_alive():
            self.data_thread.join(timeout=2)
        
        # Stop GUI updates
        self.update_timer.stop()
        
        # Update UI
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.update_status("Data collection stopped")
    
    def clear_data(self):
        """Clear all recorded data"""
        for key in self.force_history:
            self.force_history[key] = []
        self.current_forces = {}
        self.update_status("Data cleared")
    
    def open_hand(self):
        """Open the prosthetic hand"""
        try:
            if self.arm and self.connected:
                open_pos = [0, 0, 0, 0, 0, 0]  # All joints to 0 degrees
                self.arm.mainControlLoop(posDes=np.array([open_pos]), period=2)
                self.update_status("Hand opened")
        except Exception as e:
            self.update_status(f"Open hand error: {e}")
    
    def close_hand(self):
        """Close the prosthetic hand"""
        try:
            if self.arm and self.connected:
                close_pos = [80, 80, 80, 80, 80, 0]  # Close fingers, keep thumb rotation
                self.arm.mainControlLoop(posDes=np.array([close_pos]), period=2)
                self.update_status("Hand closed")
        except Exception as e:
            self.update_status(f"Close hand error: {e}")
    
    def update_rate_changed(self, rate):
        """Update the GUI refresh rate"""
        if self.update_timer.isActive():
            self.update_timer.setInterval(1000 // rate)
    
    def update_display(self):
        """Update the GUI display - with thread safety"""
        if not self.current_forces:
            return
        
        try:
            # Create thread-safe snapshots of the data
            timestamps_snapshot = self.force_history['timestamps'].copy()
            force_data_snapshot = {}
            
            for key in self.force_history:
                if key != 'timestamps':
                    force_data_snapshot[key] = self.force_history[key].copy()
            
            # Update plots with snapshots
            if timestamps_snapshot:
                window_seconds = self.window_spin.value()
                self.canvas.update_plots(
                    timestamps_snapshot,
                    force_data_snapshot,
                    window_seconds
                )
            
            # Update current values display
            self.update_values_display()
            
            # Update statistics with snapshot
            self.update_statistics(force_data_snapshot, timestamps_snapshot)
            
            # Update data count
            count = len(timestamps_snapshot)
            self.data_count_label.setText(f"Samples: {count}")
            
        except Exception as e:
            # Handle any display errors gracefully
            print(f"Display update error (non-critical): {e}")
            pass
    
    def update_values_display(self):
        """Update current force values display"""
        if not self.current_forces:
            self.values_text.setPlainText("No data available")
            return
        
        text = f"Time: {self.current_forces.get('timestamp', 0):.2f}s\n\n"
        
        # Add calibration status indicator
        if self.baseline_calibrator.calibration_complete_flag:
            text += "📍 FILTERED DATA (Offset & Noise Removed)\n"
        else:
            text += "⚠️ RAW DATA (No Calibration Applied)\n"
        
        text += "Total Force per Finger:\n"
        text += "-" * 35 + "\n"
        
        for finger in self.finger_names:
            if finger in self.current_forces:
                total = self.current_forces[finger]['total']
                text += f"{finger.capitalize():>8}: {total:6.3f} N\n"
        
        text += "-" * 35 + "\n"
        text += f"{'TOTAL GRIP':>8}: {self.current_forces.get('total_grip', 0):6.3f} N"
        
        self.values_text.setPlainText(text)
    
    def update_statistics(self, force_data_snapshot=None, timestamps_snapshot=None):
        """Update statistics display with optional snapshots"""
        # Use snapshots if provided, otherwise use current data
        if force_data_snapshot is None:
            force_data_snapshot = self.force_history
        if timestamps_snapshot is None:
            timestamps_snapshot = self.force_history['timestamps']
            
        if not timestamps_snapshot or len(timestamps_snapshot) < 10:
            self.stats_text.setPlainText("Not enough data for statistics")
            return
        
        try:
            text = "Statistics (last 100 samples):\n"
            if self.baseline_calibrator.calibration_complete_flag:
                text += "📍 FILTERED DATA\n"
            else:
                text += "⚠️ RAW DATA\n"
            text += "-" * 30 + "\n"
            
            # Get recent data (last 100 samples)
            recent_count = min(100, len(timestamps_snapshot))
            
            for finger in self.finger_names + ["total_grip"]:
                if finger in force_data_snapshot and force_data_snapshot[finger]:
                    recent_data = force_data_snapshot[finger][-recent_count:]
                    
                    if recent_data and len(recent_data) > 0:
                        mean_val = np.mean(recent_data)
                        max_val = np.max(recent_data)
                        std_val = np.std(recent_data)
                        
                        display_name = "TOTAL" if finger == "total_grip" else finger.capitalize()
                        text += f"{display_name:>8}:\n"
                        text += f"  Mean: {mean_val:5.3f} N\n"
                        text += f"  Max:  {max_val:5.3f} N\n"
                        text += f"  Std:  {std_val:5.3f} N\n\n"
            
            self.stats_text.setPlainText(text)
            
        except Exception as e:
            self.stats_text.setPlainText(f"Statistics error: {e}")
    
    def closeEvent(self, event):
        """Handle application close"""
        try:
            # Stop data collection
            if self.collecting_data:
                self.stop_data_collection()
            
            # Disconnect hand
            if self.connected:
                self.disconnect_hand()
                
        except Exception as e:
            print(f"Error during close: {e}")
        
        event.accept()

def connect_prosthetic_hand(hand_side="left"):
    """Connect to prosthetic hand using the same method as psyonicHand.py"""
    print(f"Connecting to {hand_side} prosthetic hand...")
    
    try:
        # Create arm instance (same as psyonicHand.py)
        print("Creating psyonicArm instance...")
        arm = psyonicArm(hand=hand_side)
        
        # Initialize sensors (includes zeroing)
        print("Initializing sensors...")
        arm.initSensors()
        print("Sensors initialized.")
        
        # Start communications
        print("Starting communications...")
        arm.startComms()
        print("Communications started.")
        
        print("Prosthetic hand connected successfully!")
        return arm
        
    except Exception as e:
        print(f"✗ Failed to connect to prosthetic hand: {e}")
        return None

def main():
    """Main function - connect hand first, then start GUI"""
    parser = argparse.ArgumentParser(description="Prosthetic Hand Force Monitor GUI with Auto-Calibration")
    parser.add_argument("--hand", choices=["left", "right"], default="left",
                        help="Hand side (default: left)")
    args = parser.parse_args()
    
    print("=== Prosthetic Hand Force Monitor with Auto-Calibration ===")
    print(f"Hand side: {args.hand}")
    print()
    
    # Step 1: Connect to prosthetic hand (same as psyonicHand.py)
    arm = connect_prosthetic_hand(args.hand)
    if arm is None:
        print("Failed to connect to prosthetic hand. Exiting.")
        return
    
    print()
    print("Starting GUI...")
    print("⚠️ IMPORTANT: You MUST run calibration before collecting data!")
    print("   The calibration removes offset, drift, and noise automatically.")
    
    # Step 2: Start GUI with connected arm
    app = QApplication(sys.argv)
    
    # Create and show the main window with pre-connected arm
    window = ForceMonitorGUI(arm)
    window.show()
    
    print("GUI started. Please run the 10-second calibration first!")
    
    try:
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        # Clean up
        try:
            if arm:
                arm.close()
                print("Prosthetic hand disconnected.")
        except:
            pass

if __name__ == "__main__":
    main()