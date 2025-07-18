#!/usr/bin/env python3
import sys
import time
import numpy as np
import threading
from collections import deque
import argparse
import os
import pandas as pd
import yaml

# PyQt5 imports for GUI
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QLineEdit, QComboBox, QCheckBox, QGroupBox, QTextEdit, QTabWidget,
    QSlider, QSpinBox, QGridLayout, QProgressBar, QFileDialog
)
from PyQt5.QtCore import QTimer, pyqtSignal, QThread
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Import your prosthetic hand class
sys.path.append('C:/Users/Emanuel Wicki/Documents/MIT/biomech_PCP/helpers')
from psyonicHand import psyonicArm
from helpers.EMGClass import EMG

# Grip configurations
GRIP_CONFIGURATIONS = {
    "pinch": {
        "description": "Thumb-index pinch grip",
        "active_fingers": ["index", "thumb"],
        "neutral_position": [0, 0, 0, 0, 0, 0],
        "target_position": [60, 0, 0, 0, 60, -60],
        "contact_threshold": 0.8,
        "force_modulation": [1.0, 0.0, 0.0, 0.0, 1.0],
        "max_approach_speed": 20.0,
    },
    "power_grip": {
        "description": "Full hand power grip",
        "active_fingers": ["index", "middle", "ring", "pinky", "thumb"],
        "neutral_position": [0, 0, 0, 0, 0, 0],
        "target_position": [90, 90, 90, 90, 40, -90],
        "contact_threshold": 0.8,
        "force_modulation": [1.0, 1.0, 1.0, 1.0, 1.0],
        "max_approach_speed": 15.0,
    },
    "tripod": {
        "description": "Tripod grip (thumb, index, middle)",
        "active_fingers": ["index", "middle", "thumb"],
        "neutral_position": [0, 0, 0, 0, 0, 0],
        "target_position": [60, 60, 0, 0, 65, -80],
        "contact_threshold": 0.8,
        "force_modulation": [1.0, 1.0, 0.0, 0.0, 1.0],
        "max_approach_speed": 18.0,
    },
    "hook": {
        "description": "Hook grip (fingers only, no thumb)",
        "active_fingers": ["index", "middle", "ring", "pinky"],
        "neutral_position": [0, 0, 0, 0, 0, 0],
        "target_position": [110, 110, 110, 110, 0, 0],
        "contact_threshold": 0.8,
        "force_modulation": [1.0, 1.0, 1.0, 1.0, 0.0],
        "max_approach_speed": 15.0,
    }
}

def make_timestamps_unique(timestamps):
    """Make timestamps unique by adding small increments to duplicates"""
    timestamps = np.array(timestamps)
    for i in range(1, len(timestamps)):
        if timestamps[i] <= timestamps[i - 1]:
            timestamps[i] = timestamps[i - 1] + 1e-6  # add 1 microsecond
    return timestamps

class AdaptiveGripController:
    """Controller for adaptive grip force control"""
    
    def __init__(self, arm, grip_config):
        self.arm = arm
        self.grip_config = grip_config
        self.joint_names = ["index", "middle", "ring", "pinky", "thumbFlex", "thumbRot"]
        self.finger_names = ["index", "middle", "ring", "pinky", "thumb"]
        
        # State variables
        self.phase = "NEUTRAL"
        self.contact_detected = False
        self.contact_position = None
        self.current_position = np.array(grip_config["neutral_position"], dtype=np.float64)
        
        # PID parameters - More aggressive for higher forces
        self.kp = 3.0   # Increased from 1.5
        self.ki = 0.1   # Increased from 0.05
        self.kd = 0.5   # Increased from 0.3
        self.integral_error = 0.0
        self.previous_error = 0.0
        
        # Control limits - Smaller changes for smoother high-frequency control
        self.max_position_change = 0.5  # Much smaller for high-frequency updates
        self.control_rate = 200.0  # Target 200 Hz control rate
        
        # Force control parameters
        self.force_buildup_mode = False
        self.force_buildup_threshold = 3.0
        
        # Data filtering for smooth display (longer buffer for high-freq data)
        self.force_filter_buffer = deque(maxlen=20)  # More samples for smoothing
        self.filtered_force = 0.0
        
        # High-frequency control parameters
        self.last_control_time = 0.0
        self.control_dt_target = 1.0 / self.control_rate  # 5ms target
        
    def get_finger_forces(self):
        """Get force for each of the 5 fingers"""
        finger_forces = {}
        for finger in self.finger_names:
            finger_total = 0.0
            for sensor_idx in range(6):
                sensor_name = f"{finger}{sensor_idx}_Force"
                force_value = self.arm.sensors.get(sensor_name, 0.0)
                finger_total += force_value
            finger_forces[finger] = finger_total
        return finger_forces
    
    def get_grip_force(self, filtered=True):
        """Get current total grip force from active fingers with optional filtering"""
        finger_forces = self.get_finger_forces()
        total_force = 0.0
        for finger in self.grip_config["active_fingers"]:
            total_force += finger_forces.get(finger, 0.0)
        
        if filtered:
            # Apply rolling average filter for smoother display
            self.force_filter_buffer.append(total_force)
            self.filtered_force = np.mean(list(self.force_filter_buffer))
            return self.filtered_force
        else:
            # Return raw force for control
            return total_force
    
    def detect_contact(self):
        """Detect contact with object"""
        current_force = self.get_grip_force(filtered=False)  # Use raw force for detection
        if not self.contact_detected:
            if current_force > self.grip_config["contact_threshold"]:
                self.contact_detected = True
                self.contact_position = self.current_position.astype(np.float64).copy()
                self.phase = "CONTACT"
                return True
        return self.contact_detected
    
    def map_finger_to_joint_modulation(self):
        """Map 5-finger modulation to 6-joint modulation"""
        finger_modulation = np.array(self.grip_config["force_modulation"], dtype=np.float64)
        joint_modulation = np.zeros(6, dtype=np.float64)
        joint_modulation[0] = finger_modulation[0]  # index
        joint_modulation[1] = finger_modulation[1]  # middle
        joint_modulation[2] = finger_modulation[2]  # ring
        joint_modulation[3] = finger_modulation[3]  # pinky
        joint_modulation[4] = finger_modulation[4]  # thumbFlex
        joint_modulation[5] = finger_modulation[4]  # thumbRot
        return joint_modulation
    
    def approach_phase(self, dt):
        """Move towards target until contact"""
        if self.detect_contact():
            return True
        
        if self.current_position.dtype != np.float64:
            self.current_position = self.current_position.astype(np.float64)
        
        neutral_pos = np.array(self.grip_config["neutral_position"], dtype=np.float64)
        target_pos = np.array(self.grip_config["target_position"], dtype=np.float64)
        joint_force_modulation = self.map_finger_to_joint_modulation()
        
        direction = (target_pos - self.current_position) * joint_force_modulation
        max_movement = self.grip_config["max_approach_speed"] * dt
        movement_magnitude = np.linalg.norm(direction)
        
        if movement_magnitude > 0:
            normalized_direction = direction / movement_magnitude
            actual_movement = normalized_direction * min(max_movement, movement_magnitude)
            actual_movement = actual_movement.astype(np.float64)
            
            self.current_position = self.current_position + actual_movement
            self.current_position = np.clip(self.current_position, neutral_pos, target_pos)
        
        return False
    
    def force_control_phase(self, target_force, dt):
        """High-frequency smooth force control with smaller incremental changes"""
        current_force = self.get_grip_force(filtered=False)  # Use raw force for control
        force_error = target_force - current_force
        
        # Scale PID gains for high-frequency control (smaller gains, more frequent updates)
        if target_force > self.force_buildup_threshold:
            self.force_buildup_mode = True
            # Reduce gains for high-frequency control to prevent oscillation
            kp_active = self.kp * 0.8  # Lower gains for stability
            ki_active = self.ki * 0.5
            kd_active = self.kd * 0.6
            max_change_active = self.max_position_change * 1.2  # Slightly larger for force buildup
            fine_control_scale = 0.4
            deadband = 0.3
        else:
            self.force_buildup_mode = False
            # Standard gains scaled for high frequency
            kp_active = self.kp * 0.6  # Lower gains for smooth high-freq control
            ki_active = self.ki * 0.3
            kd_active = self.kd * 0.4
            max_change_active = self.max_position_change
            fine_control_scale = 0.2
            deadband = 0.15
        
        # PID control with high-frequency tuning
        self.integral_error += force_error * dt
        max_integral = 8.0 if self.force_buildup_mode else 5.0  # Smaller integral limits
        self.integral_error = np.clip(self.integral_error, -max_integral, max_integral)
        
        derivative_error = (force_error - self.previous_error) / dt if dt > 0 else 0
        self.previous_error = force_error
        
        pid_output = (kp_active * force_error + 
                     ki_active * self.integral_error + 
                     kd_active * derivative_error)
        
        # Adaptive deadband
        if abs(force_error) < deadband:
            pid_output = 0.0
        
        # Convert to position change with smooth scaling
        joint_force_modulation = self.map_finger_to_joint_modulation()
        position_change = pid_output * joint_force_modulation * fine_control_scale
        
        # Very small position change limits for smooth motion
        position_change = np.clip(position_change, -max_change_active, max_change_active)
        
        # Even smaller minimum change threshold for high-frequency updates
        min_change_threshold = 0.02  # Very small movements
        for i in range(len(position_change)):
            if abs(position_change[i]) < min_change_threshold:
                position_change[i] = 0.0
        
        # Smooth position updates
        self.current_position = self.current_position + position_change
        
        # Apply limits with extended range for force buildup
        neutral_pos = np.array(self.grip_config["neutral_position"], dtype=np.float64)
        target_pos = np.array(self.grip_config["target_position"], dtype=np.float64)
        contact_pos = self.contact_position if self.contact_position is not None else neutral_pos
        
        if self.force_buildup_mode:
            # Allow 30% extension beyond target for high forces
            for i in range(len(self.current_position)):
                if joint_force_modulation[i] > 0:
                    range_extension = (target_pos[i] - contact_pos[i]) * 0.3
                    min_pos = min(contact_pos[i], target_pos[i])
                    max_pos = max(contact_pos[i], target_pos[i] + range_extension)
                    self.current_position[i] = np.clip(self.current_position[i], min_pos, max_pos)
        else:
            # Standard limits
            for i in range(len(self.current_position)):
                if joint_force_modulation[i] > 0:
                    min_pos = min(contact_pos[i], target_pos[i])
                    max_pos = max(contact_pos[i], target_pos[i])
                    self.current_position[i] = np.clip(self.current_position[i], min_pos, max_pos)
        
        return current_force, force_error, pid_output
    
    def reset_controller(self):
        """Reset to initial state"""
        self.phase = "NEUTRAL"
        self.contact_detected = False
        self.contact_position = None
        self.current_position = np.array(self.grip_config["neutral_position"], dtype=np.float64)
        self.integral_error = 0.0
        self.previous_error = 0.0

class ForceControlCanvas(FigureCanvas):
    """Canvas for real-time force control visualization"""
    
    def __init__(self, parent=None, width=10, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        
        # Create subplots
        self.ax_main = self.fig.add_subplot(2, 1, 1)  # Main force plot
        self.ax_error = self.fig.add_subplot(2, 1, 2)  # Error plot
        
        # Main force plot
        self.ax_main.set_title('Real-Time Force Control', fontsize=14, weight='bold')
        self.ax_main.set_ylabel('Force (N)', fontsize=12)
        self.ax_main.grid(True, alpha=0.3)
        
        # Error plot
        self.ax_error.set_title('Force Error', fontsize=12)
        self.ax_error.set_xlabel('Time (s)', fontsize=12)
        self.ax_error.set_ylabel('Error (N)', fontsize=12)
        self.ax_error.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        
        # Initialize lines
        self.target_line, = self.ax_main.plot([], [], 'r-', linewidth=3, label='Target Force')
        self.actual_line, = self.ax_main.plot([], [], 'b-', linewidth=2, label='Actual Force')
        self.error_line, = self.ax_error.plot([], [], 'g-', linewidth=2, label='Force Error')
        
        self.ax_main.legend(fontsize=10)
        self.ax_error.legend(fontsize=10)
    
    def update_plots(self, timestamps, target_forces, actual_forces, window_seconds=15):
        """Update real-time plots"""
        if not timestamps or len(timestamps) == 0:
            return
        
        try:
            times = np.array(timestamps)
            targets = np.array(target_forces)
            actuals = np.array(actual_forces)
            errors = targets - actuals
            
            # Apply time window
            current_time = times[-1]
            start_time = max(0, current_time - window_seconds)
            mask = times >= start_time
            
            window_times = times[mask]
            window_targets = targets[mask]
            window_actuals = actuals[mask]
            window_errors = errors[mask]
            
            if len(window_times) > 0:
                # Update main plot
                self.target_line.set_data(window_times, window_targets)
                self.actual_line.set_data(window_times, window_actuals)
                
                self.ax_main.set_xlim(start_time, current_time)
                if len(window_targets) > 0 and len(window_actuals) > 0:
                    all_forces = np.concatenate([window_targets, window_actuals])
                    force_min, force_max = np.min(all_forces), np.max(all_forces)
                    margin = (force_max - force_min) * 0.1
                    self.ax_main.set_ylim(max(0, force_min - margin), force_max + margin)
                
                # Update error plot
                self.error_line.set_data(window_times, window_errors)
                self.ax_error.set_xlim(start_time, current_time)
                if len(window_errors) > 0:
                    error_max = max(abs(np.min(window_errors)), abs(np.max(window_errors)))
                    self.ax_error.set_ylim(-error_max * 1.1, error_max * 1.1)
                
                # Add reference lines
                self.ax_main.axhline(y=0, color='k', linestyle='--', alpha=0.3)
                self.ax_error.axhline(y=0, color='k', linestyle='-', alpha=0.5)
            
            self.draw()
            
        except Exception as e:
            print(f"Plot update error: {e}")

class AdaptiveGripGUI(QMainWindow):
    """Main GUI for adaptive grip control with real-time monitoring"""
    
    def __init__(self, connected_arm, person_id, out_root="data"):
        super().__init__()
        self.setWindowTitle("Adaptive Grip Force Control with Real-Time Monitoring")
        self.setGeometry(100, 100, 1600, 1000)
        
        # Hardware
        self.arm = connected_arm
        self.controller = None
        
        # Data storage setup
        self.person_id = person_id
        self.out_root = out_root
        
        # EMG recording setup
        self.emg = None
        self.emg_thread = None
        self.recording_emg = False
        self.raw_emg_data = []
        self.raw_emg_timestamps = []
        
        # Experiment state
        self.experiment_running = False
        self.current_grip_config = None
        self.force_trajectory = None
        self.force_timestamps = None
        self.experiment_start_time = None
        
        # Data storage with filtering support
        self.experiment_data = {
            'timestamps': [],
            'target_forces': [],
            'actual_forces': [],
            'actual_forces_filtered': [],  # For smooth plotting
            'grip_positions': [],
            'phases': [],
            'force_errors': []
        }
        
        # Data filtering for smooth display
        self.display_force_buffer = deque(maxlen=5)  # For GUI display smoothing
        
        self.initUI()
        
        # Update timer - Optimized for smooth display
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.setInterval(50)  # 20 Hz display updates (enough for smooth GUI)
        self.last_control_time = 0.0
    
    def initUI(self):
        """Initialize user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel: Controls
        controls_widget = QWidget()
        controls_widget.setMaximumWidth(400)
        controls_layout = QVBoxLayout(controls_widget)
        
        # Person ID display
        person_group = QGroupBox("Session Info")
        person_layout = QVBoxLayout(person_group)
        self.person_label = QLabel(f"Person ID: {self.person_id}")
        self.person_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        person_layout.addWidget(self.person_label)
        controls_layout.addWidget(person_group)
        
        # Experiment setup
        setup_group = QGroupBox("Experiment Setup")
        setup_layout = QVBoxLayout(setup_group)
        
        # Grip selection
        grip_layout = QHBoxLayout()
        grip_layout.addWidget(QLabel("Grip Type:"))
        self.grip_combo = QComboBox()
        self.grip_combo.addItems(list(GRIP_CONFIGURATIONS.keys()))
        self.grip_combo.currentTextChanged.connect(self.grip_changed)
        grip_layout.addWidget(self.grip_combo)
        setup_layout.addLayout(grip_layout)
        
        # Grip description
        self.grip_description = QLabel("Select a grip type")
        self.grip_description.setWordWrap(True)
        self.grip_description.setStyleSheet("font-style: italic; color: #666;")
        setup_layout.addWidget(self.grip_description)
        
        # Force trajectory
        traj_layout = QHBoxLayout()
        traj_layout.addWidget(QLabel("Force Pattern:"))
        self.pattern_combo = QComboBox()
        self.pattern_combo.addItems(["sine", "step", "ramp", "constant"])
        traj_layout.addWidget(self.pattern_combo)
        setup_layout.addLayout(traj_layout)
        
        # Load custom trajectory
        self.load_traj_btn = QPushButton("Load Force Trajectory CSV")
        self.load_traj_btn.clicked.connect(self.load_force_trajectory)
        setup_layout.addWidget(self.load_traj_btn)
        
        # Force parameters
        param_layout = QGridLayout()
        param_layout.addWidget(QLabel("Max Force (N):"), 0, 0)
        self.max_force_spin = QSpinBox()
        self.max_force_spin.setRange(1, 20)
        self.max_force_spin.setValue(5)
        param_layout.addWidget(self.max_force_spin, 0, 1)
        
        param_layout.addWidget(QLabel("Duration (s):"), 1, 0)
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(5, 60)
        self.duration_spin.setValue(15)
        param_layout.addWidget(self.duration_spin, 1, 1)
        
        setup_layout.addLayout(param_layout)
        controls_layout.addWidget(setup_group)
        
        # Experiment control
        control_group = QGroupBox("Experiment Control")
        control_layout = QVBoxLayout(control_group)
        
        self.prepare_btn = QPushButton("Prepare Experiment")
        self.prepare_btn.clicked.connect(self.prepare_experiment)
        control_layout.addWidget(self.prepare_btn)
        
        self.start_btn = QPushButton("Start Experiment")
        self.start_btn.clicked.connect(self.start_experiment)
        self.start_btn.setEnabled(False)
        control_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Stop Experiment")
        self.stop_btn.clicked.connect(self.stop_experiment)
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.stop_btn)
        
        self.reset_btn = QPushButton("Reset to Neutral")
        self.reset_btn.clicked.connect(self.reset_to_neutral)
        control_layout.addWidget(self.reset_btn)
        
        controls_layout.addWidget(control_group)
        
        # Status and info
        status_group = QGroupBox("Status & Information")
        status_layout = QVBoxLayout(status_group)
        
        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet("font-weight: bold;")
        status_layout.addWidget(self.status_label)
        
        self.phase_label = QLabel("Phase: NEUTRAL")
        status_layout.addWidget(self.phase_label)
        
        self.force_info = QTextEdit()
        self.force_info.setMaximumHeight(120)
        self.force_info.setReadOnly(True)
        self.force_info.setFont(QtGui.QFont("Courier", 9))
        status_layout.addWidget(self.force_info)
        
        controls_layout.addWidget(status_group)
        
        # Auto-save info
        save_group = QGroupBox("Auto-Save")
        save_layout = QVBoxLayout(save_group)
        
        self.save_path_label = QLabel("Save Path: Not set")
        self.save_path_label.setWordWrap(True)
        self.save_path_label.setStyleSheet("font-size: 10px;")
        save_layout.addWidget(self.save_path_label)
        
        controls_layout.addWidget(save_group)
        controls_layout.addStretch()
        
        # Right panel: Real-time plots
        self.canvas = ForceControlCanvas(self, width=12, height=8)
        
        # Add to main layout
        main_layout.addWidget(controls_widget)
        main_layout.addWidget(self.canvas, stretch=1)
        
        # Initialize
        self.grip_changed()
    
    def grip_changed(self):
        """Handle grip type selection change"""
        grip_name = self.grip_combo.currentText()
        if grip_name in GRIP_CONFIGURATIONS:
            config = GRIP_CONFIGURATIONS[grip_name]
            self.grip_description.setText(
                f"{config['description']}\n"
                f"Active fingers: {', '.join(config['active_fingers'])}\n"
                f"Contact threshold: {config['contact_threshold']} N"
            )
            
            # Update save path
            self.update_save_path()
    
    def update_save_path(self):
        """Update the save path display"""
        grip_name = self.grip_combo.currentText()
        
        # Create directory structure like s1.5 script
        exp_parent = os.path.join(
            self.out_root,
            self.person_id,
            "recordings",
            f"{grip_name}_interaction",
            "experiments"
        )
        
        # Find next experiment number
        exp_idx = 1
        while os.path.exists(os.path.join(exp_parent, str(exp_idx))):
            exp_idx += 1
        
        save_path = os.path.join(exp_parent, str(exp_idx))
        self.save_path_label.setText(f"Save Path: {save_path}")
    
    def start_emg_recording(self):
        """Start EMG recording in background thread"""
        try:
            if self.emg is None:
                print("Initializing EMG...")
                self.emg = EMG()
            
            self.raw_emg_data = []
            self.raw_emg_timestamps = []
            self.recording_emg = True
            
            def emg_capture_loop():
                try:
                    print("Starting EMG communication...")
                    # Use the same method as s1.5 script
                    self.emg.startCommunication()
                    
                    # Wait for first data
                    print("Waiting for first EMG data...")
                    timeout_counter = 0
                    while getattr(self.emg, 'OS_time', None) is None and timeout_counter < 100:
                        time.sleep(0.01)
                        timeout_counter += 1
                    
                    if getattr(self.emg, 'OS_time', None) is None:
                        print("WARNING: EMG not responding, continuing without EMG")
                        return
                    
                    print("EMG data detected, starting recording...")
                    first_emg_time = self.emg.OS_time
                    last_time = self.emg.OS_time
                    
                    while self.recording_emg:
                        try:
                            time_sample = self.emg.OS_time
                            
                            # Add timing validation (same as s1.5)
                            if (time_sample - last_time)/1e6 > 0.1:
                                print(f'EMG Read time: {time_sample}, expected time: {last_time}')
                                print('EMG alignment lost. Please restart the EMG board.')
                                break
                            
                            # Only collect data when we have new timestamps
                            elif time_sample > last_time:
                                emg_sample = np.asarray(self.emg.rawEMG)
                                self.raw_emg_data.append(list(emg_sample))
                                self.raw_emg_timestamps.append((time_sample - first_emg_time) / 1e6)
                            
                            last_time = time_sample
                            time.sleep(0.001)  # Small delay to prevent busy waiting
                        except Exception as e:
                            print(f"EMG read error: {e}")
                            break
                        
                except Exception as e:
                    print(f"EMG capture setup error: {e}")
                finally:
                    try:
                        if hasattr(self.emg, 'exitEvent'):
                            self.emg.exitEvent.set()
                        time.sleep(0.1)
                        print("EMG recording thread finished")
                    except:
                        pass
            
            self.emg_thread = threading.Thread(target=emg_capture_loop, daemon=True)
            self.emg_thread.start()
            print("EMG thread started")
            
        except Exception as e:
            print(f"Failed to start EMG recording: {e}")
            self.emg = None
    
    def stop_emg_recording(self):
        """Stop EMG recording"""
        print("Stopping EMG recording...")
        self.recording_emg = False
        if self.emg_thread:
            self.emg_thread.join(timeout=3.0)
        print(f"EMG recording stopped. Captured {len(self.raw_emg_data)} samples.")
    
    def load_force_trajectory(self):
        """Load custom force trajectory from CSV"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Force Trajectory", "", "CSV Files (*.csv);;All Files (*)"
        )
        if file_path:
            try:
                data = pd.read_csv(file_path)
                if 'timestamp' not in data.columns or 'force' not in data.columns:
                    QtWidgets.QMessageBox.warning(
                        self, "Error", "CSV must contain 'timestamp' and 'force' columns"
                    )
                    return
                
                self.force_timestamps = data['timestamp'].values
                self.force_trajectory = data['force'].values
                
                QtWidgets.QMessageBox.information(
                    self, "Success", 
                    f"Loaded trajectory with {len(self.force_trajectory)} points\n"
                    f"Duration: {self.force_timestamps[-1]:.1f} seconds"
                )
                
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Error", f"Failed to load file: {e}")
    
    def generate_force_trajectory(self):
        """Generate force trajectory based on current settings"""
        duration = self.duration_spin.value()
        max_force = self.max_force_spin.value()
        pattern = self.pattern_combo.currentText()
        
        dt = 1.0 / 60.0  # 60 Hz
        timestamps = np.arange(0, duration, dt)
        
        if pattern == "sine":
            base_force = max_force * 0.4
            amplitude = max_force * 0.3
            frequency = 0.1
            forces = base_force + amplitude * np.sin(2 * np.pi * frequency * timestamps)
        elif pattern == "step":
            forces = np.ones_like(timestamps) * max_force * 0.3
            forces[len(forces)//4:len(forces)//2] = max_force * 0.6
            forces[3*len(forces)//4:] = max_force * 0.9
        elif pattern == "ramp":
            forces = np.ones_like(timestamps) * max_force * 0.2
            ramp_up = np.linspace(max_force * 0.2, max_force, len(forces)//3)
            ramp_down = np.linspace(max_force, max_force * 0.2, len(forces)//3)
            forces[len(forces)//3:2*len(forces)//3] = ramp_up
            forces[2*len(forces)//3:] = ramp_down[:len(forces) - 2*len(forces)//3]
        else:  # constant
            forces = np.ones_like(timestamps) * max_force * 0.5
        
        return timestamps, forces
    
    def prepare_experiment(self):
        """Prepare experiment - move to neutral and set up controller"""
        try:
            # Get current grip configuration
            grip_name = self.grip_combo.currentText()
            self.current_grip_config = GRIP_CONFIGURATIONS[grip_name]
            
            # Generate or use loaded trajectory
            if self.force_trajectory is None:
                self.force_timestamps, self.force_trajectory = self.generate_force_trajectory()
            
            # Initialize controller
            self.controller = AdaptiveGripController(self.arm, self.current_grip_config)
            
            # Move to neutral position
            self.status_label.setText("Status: Moving to neutral position...")
            neutral_pos = np.array(self.current_grip_config["neutral_position"], dtype=np.float64)
            self.arm.mainControlLoop(posDes=neutral_pos.reshape(1, -1), period=2, emg=None)
            self.controller.current_position = neutral_pos.copy()
            
            self.status_label.setText("Status: Ready for object placement")
            self.start_btn.setEnabled(True)
            self.prepare_btn.setEnabled(False)
            
            QtWidgets.QMessageBox.information(
                self, "Ready", 
                f"Experiment prepared!\n\n"
                f"Grip: {self.current_grip_config['description']}\n"
                f"Please place the object and click 'Start Experiment'"
            )
            
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to prepare experiment: {e}")
    
    def start_experiment(self):
        """Start the adaptive grip experiment"""
        if not self.controller:
            return
        
        # Clear previous data
        self.clear_data()
        
        # Start EMG recording FIRST
        try:
            print("Starting EMG recording...")
            self.start_emg_recording()
            # Give EMG time to initialize
            time.sleep(0.5)
        except Exception as e:
            print(f"Warning: Could not start EMG recording: {e}")
        
        # Start experiment
        self.experiment_running = True
        self.experiment_start_time = time.time()
        self.force_start_time = None
        
        # Reset controller
        self.controller.reset_controller()
        self.controller.phase = "APPROACH"
        
        # CRITICAL: Start hand recording using the SAME method as s1.5
        print("Starting hand recording...")
        self.arm.resetRecording()
        self.arm.recording = True
        
        # Start control loop at maximum frequency
        self.control_timer = QTimer()
        self.control_timer.timeout.connect(self.control_loop)
        self.control_timer.setInterval(5)  # 5ms = 200 Hz target
        self.control_timer.start()
        
        # Start GUI updates at lower frequency
        self.update_timer.start()
        
        # Update UI
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Status: Experiment running - APPROACH phase")
        
        print("Experiment started - recording EMG and hand data")
    
    def control_loop(self):
        """High-frequency control loop - Use mainControlLoop like s1.5 for proper recording"""
        if not self.experiment_running or not self.controller:
            return
        
        try:
            current_time = time.time() - self.experiment_start_time
            
            # Calculate actual dt for this control cycle
            if hasattr(self, 'last_control_time') and self.last_control_time > 0:
                actual_dt = current_time - self.last_control_time
            else:
                actual_dt = self.controller.control_dt_target
            self.last_control_time = current_time
            
            # Use actual measured dt for more accurate control
            dt = actual_dt
            
            # Get current force
            current_force_raw = self.controller.get_grip_force(filtered=False)
            
            target_force = 0.0
            force_error = 0.0
            
            if self.controller.phase == "APPROACH":
                # Approach phase
                contact_detected = self.controller.approach_phase(dt)
                
                if contact_detected:
                    self.controller.phase = "FORCE_CONTROL"
                    self.force_start_time = current_time
                    self.status_label.setText("Status: CONTACT detected - FORCE CONTROL active")
                    self.phase_label.setText("Phase: FORCE_CONTROL")
            
            elif self.controller.phase == "FORCE_CONTROL":
                # Force control phase
                trajectory_time = current_time - self.force_start_time
                
                if trajectory_time <= self.force_timestamps[-1]:
                    target_force = np.interp(trajectory_time, self.force_timestamps, self.force_trajectory)
                    current_force_raw, force_error, pid_output = self.controller.force_control_phase(target_force, dt)
                else:
                    # Trajectory complete
                    self.stop_experiment()
                    return
            
            # CRITICAL FIX: Use mainControlLoop like s1.5 for proper recording
            # Create a single-row trajectory for this timestep
            position_command = self.controller.current_position.reshape(1, -1)
            
            # Use mainControlLoop with very short period (like s1.5 does)
            # This ensures proper recording of all sensor data
            self.arm.mainControlLoop(posDes=position_command, period=0.001, emg=None)
            
            # Store data less frequently to avoid memory issues (every 10th sample)
            if len(self.experiment_data['timestamps']) == 0 or current_time - self.experiment_data['timestamps'][-1] >= 0.033:  # ~30 Hz data storage
                current_force_display = self.controller.get_grip_force(filtered=True)
                
                self.experiment_data['timestamps'].append(current_time)
                self.experiment_data['target_forces'].append(target_force)
                self.experiment_data['actual_forces'].append(current_force_raw)
                self.experiment_data['actual_forces_filtered'].append(current_force_display)
                self.experiment_data['grip_positions'].append(self.controller.current_position.copy())
                self.experiment_data['phases'].append(self.controller.phase)
                self.experiment_data['force_errors'].append(force_error)
            
        except Exception as e:
            print(f"Control loop error: {e}")
            self.stop_experiment()
    
    def stop_experiment(self):
        """Stop the experiment and save all data"""
        print("Stopping experiment...")
        self.experiment_running = False
        
        if hasattr(self, 'control_timer'):
            self.control_timer.stop()
        
        self.update_timer.stop()
        
        # Stop hand recording FIRST
        print("Stopping hand recording...")
        if hasattr(self.arm, 'recording'):
            self.arm.recording = False
        
        # Stop EMG recording
        self.stop_emg_recording()
        
        # Return to neutral
        if self.controller:
            try:
                neutral_pos = np.array(self.current_grip_config["neutral_position"], dtype=np.float64)
                self.arm.mainControlLoop(posDes=neutral_pos.reshape(1, -1), period=2, emg=None)
            except Exception as e:
                print(f"Warning: Could not return to neutral: {e}")
        
        # Update UI
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.prepare_btn.setEnabled(True)
        self.status_label.setText("Status: Experiment completed - Saving data...")
        
        # Auto-save all data
        self.auto_save_experiment_data()
        
        # Analyze results
        if self.experiment_data['timestamps']:
            self.analyze_results()
    
    def auto_save_experiment_data(self):
        """Automatically save experiment data in s1.5 format"""
        try:
            grip_name = self.grip_combo.currentText()
            
            # Create directory structure exactly like s1.5 script
            exp_parent = os.path.join(
                self.out_root,
                self.person_id,
                "recordings",
                f"{grip_name}_interaction",
                "experiments"
            )
            
            # Find next experiment number
            exp_idx = 1
            while os.path.exists(os.path.join(exp_parent, str(exp_idx))):
                exp_idx += 1
            
            base_dir = os.path.join(exp_parent, str(exp_idx))
            os.makedirs(base_dir, exist_ok=True)
            
            print(f"Saving experiment data to: {base_dir}")
            
            files_saved = []
            
            # 1. Save raw EMG data (like s1.5)
            if self.raw_emg_data and len(self.raw_emg_data) > 0:
                try:
                    raw_emg_array = np.vstack(self.raw_emg_data)
                    np.save(os.path.join(base_dir, "raw_emg.npy"), raw_emg_array)
                    files_saved.append("raw_emg.npy")
                    
                    raw_timestamps_unique = make_timestamps_unique(self.raw_emg_timestamps)
                    np.save(os.path.join(base_dir, "raw_timestamps.npy"), np.array(raw_timestamps_unique))
                    files_saved.append("raw_timestamps.npy")
                    
                    print(f"✓ Saved raw_emg.npy with {len(self.raw_emg_data)} samples")
                    print(f"✓ EMG duration: {self.raw_emg_timestamps[-1] - self.raw_emg_timestamps[0]:.2f}s")
                except Exception as e:
                    print(f"✗ Failed to save EMG data: {e}")
            else:
                print("✗ No EMG data to save")
            
            # 2. Save hand angle data (like s1.5)
            if hasattr(self.arm, 'recordedData') and self.arm.recordedData and len(self.arm.recordedData) > 1:
                try:
                    raw_data = self.arm.recordedData
                    headers = raw_data[0]  # first row is header names
                    data_rows = raw_data[1:]
                    
                    if data_rows:
                        rec = np.array(data_rows, dtype=float)
                        ts = rec[:, 0]
                        ts -= ts[0]  # normalize timestamps
                        
                        np.save(os.path.join(base_dir, "angles.npy"), rec)
                        files_saved.append("angles.npy")
                        
                        angle_timestamps_unique = make_timestamps_unique(ts)
                        np.save(os.path.join(base_dir, "angle_timestamps.npy"), angle_timestamps_unique)
                        files_saved.append("angle_timestamps.npy")
                        
                        with open(os.path.join(base_dir, "angles_header.txt"), "w") as f:
                            f.write(",".join(headers))
                        files_saved.append("angles_header.txt")
                        
                        print(f"✓ Saved angles.npy with {len(rec)} frames")
                        print(f"✓ Angles duration: {ts[-1]:.2f}s")
                    else:
                        print("✗ No angle data rows found")
                except Exception as e:
                    print(f"✗ Failed to save angle data: {e}")
            else:
                print("✗ No hand angle data to save")
                print(f"   recordedData exists: {hasattr(self.arm, 'recordedData')}")
                if hasattr(self.arm, 'recordedData'):
                    print(f"   recordedData length: {len(self.arm.recordedData) if self.arm.recordedData else 0}")
            
            # 3. Save adaptive grip data (target vs actual forces)
            if self.experiment_data['timestamps']:
                try:
                    adaptive_grip_data = {
                        'timestamps': np.array(self.experiment_data['timestamps']),
                        'target_forces': np.array(self.experiment_data['target_forces']),
                        'actual_forces': np.array(self.experiment_data['actual_forces']),
                        'force_errors': np.array(self.experiment_data['force_errors']),
                        'phases': self.experiment_data['phases'],
                        'grip_positions': np.array(self.experiment_data['grip_positions'])
                    }
                    
                    # Save as numpy file
                    np.save(os.path.join(base_dir, f"adaptive_grip_{grip_name}.npy"), adaptive_grip_data)
                    files_saved.append(f"adaptive_grip_{grip_name}.npy")
                    
                    # Also save as CSV for easy analysis
                    df = pd.DataFrame({
                        'timestamp': self.experiment_data['timestamps'],
                        'target_force': self.experiment_data['target_forces'],
                        'actual_force': self.experiment_data['actual_forces'],
                        'force_error': self.experiment_data['force_errors'],
                        'phase': self.experiment_data['phases']
                    })
                    df.to_csv(os.path.join(base_dir, f"adaptive_grip_{grip_name}.csv"), index=False)
                    files_saved.append(f"adaptive_grip_{grip_name}.csv")
                    
                    print(f"✓ Saved adaptive_grip_{grip_name}.npy and .csv")
                except Exception as e:
                    print(f"✗ Failed to save adaptive grip data: {e}")
            
            # 4. Save experiment configuration
            try:
                config_data = {
                    'person_id': self.person_id,
                    'grip_type': grip_name,
                    'grip_config': self.current_grip_config,
                    'force_pattern': self.pattern_combo.currentText(),
                    'max_force': self.max_force_spin.value(),
                    'duration': self.duration_spin.value(),
                    'experiment_summary': {
                        'total_samples': len(self.experiment_data['timestamps']),
                        'total_duration': self.experiment_data['timestamps'][-1] if self.experiment_data['timestamps'] else 0,
                        'emg_samples': len(self.raw_emg_data),
                        'angle_samples': len(self.arm.recordedData) - 1 if hasattr(self.arm, 'recordedData') and self.arm.recordedData else 0
                    }
                }
                
                with open(os.path.join(base_dir, 'experiment_config.yaml'), 'w') as f:
                    yaml.safe_dump(config_data, f)
                files_saved.append('experiment_config.yaml')
                
                print(f"✓ Saved experiment_config.yaml")
            except Exception as e:
                print(f"✗ Failed to save config: {e}")
            
            # Update UI
            self.status_label.setText(f"Status: Data saved to experiment {exp_idx}")
            self.update_save_path()  # Update for next experiment
            
            # Show results
            files_list = "\n".join([f"- {f}" for f in files_saved])
            missing_files = []
            expected_files = ["raw_emg.npy", "raw_timestamps.npy", "angles.npy", "angle_timestamps.npy", "angles_header.txt"]
            for expected in expected_files:
                if expected not in files_saved:
                    missing_files.append(expected)
            
            message = f"Experiment data saved to:\n{base_dir}\n\nFiles saved:\n{files_list}"
            if missing_files:
                message += f"\n\nMissing files:\n" + "\n".join([f"- {f}" for f in missing_files])
                message += "\n\nCheck console output for error details."
            
            QtWidgets.QMessageBox.information(self, "Data Saved", message)
            
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Save Error", f"Failed to save data: {e}")
            print(f"Save error: {e}")
            import traceback
            traceback.print_exc()
    
    def reset_to_neutral(self):
        """Reset hand to neutral position"""
        try:
            if self.current_grip_config:
                neutral_pos = np.array(self.current_grip_config["neutral_position"], dtype=np.float64)
            else:
                neutral_pos = np.array([0, 0, 0, 0, 0, 0], dtype=np.float64)
            
            self.arm.mainControlLoop(posDes=neutral_pos.reshape(1, -1), period=2, emg=None)
            self.status_label.setText("Status: Reset to neutral position")
            
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to reset: {e}")
    
    def update_display(self):
        """Update GUI display"""
        if not self.experiment_running:
            return
        
        try:
            # Update plots with filtered data for smooth visualization
            if self.experiment_data['timestamps']:
                self.canvas.update_plots(
                    self.experiment_data['timestamps'],
                    self.experiment_data['target_forces'],
                    self.experiment_data['actual_forces_filtered']  # Use filtered data for plotting
                )
            
            # Update force info with enhanced details
            if self.controller:
                current_force = self.controller.get_grip_force(filtered=True)  # Use filtered for display
                finger_forces = self.controller.get_finger_forces()
                
                info_text = f"Current Total Force: {current_force:.2f} N\n"
                info_text += f"Control Mode: {'FORCE_BUILDUP' if self.controller.force_buildup_mode else 'STANDARD'}\n\n"
                
                info_text += "Finger Forces:\n"
                for finger in self.controller.finger_names:
                    if finger in finger_forces:
                        active_marker = " *" if finger in self.controller.grip_config["active_fingers"] else ""
                        info_text += f"  {finger.capitalize()}: {finger_forces[finger]:.1f} N{active_marker}\n"
                
                if self.experiment_data['target_forces']:
                    target = self.experiment_data['target_forces'][-1]
                    error = abs(current_force - target)
                    error_pct = (error / target * 100) if target > 0 else 0
                    info_text += f"\nTarget Force: {target:.2f} N\n"
                    info_text += f"Force Error: {error:.2f} N ({error_pct:.1f}%)\n"
                    
                    # Show position info
                    if hasattr(self.controller, 'contact_position') and self.controller.contact_position is not None:
                        pos_range = np.array(self.controller.grip_config["target_position"]) - self.controller.contact_position
                        pos_used = self.controller.current_position - self.controller.contact_position
                        pos_pct = np.mean(pos_used / pos_range * 100) if np.mean(pos_range) > 0 else 0
                        info_text += f"Position Used: {pos_pct:.0f}% of available range"
                
                self.force_info.setPlainText(info_text)
                
                # Update phase
                self.phase_label.setText(f"Phase: {self.controller.phase}")
        
        except Exception as e:
            print(f"Display update error: {e}")
    
    def analyze_results(self):
        """Analyze experiment results"""
        timestamps = np.array(self.experiment_data['timestamps'])
        target_forces = np.array(self.experiment_data['target_forces'])
        actual_forces = np.array(self.experiment_data['actual_forces'])
        phases = self.experiment_data['phases']
        
        # Find contact point
        contact_idx = next((i for i, phase in enumerate(phases) if phase == "FORCE_CONTROL"), None)
        
        if contact_idx is not None:
            # Analyze force control performance
            control_targets = target_forces[contact_idx:]
            control_actuals = actual_forces[contact_idx:]
            
            if len(control_actuals) > 0:
                force_errors = control_targets - control_actuals
                rmse = np.sqrt(np.mean(force_errors**2))
                mae = np.mean(np.abs(force_errors))
                max_error = np.max(np.abs(force_errors))
                
                contact_time = timestamps[contact_idx]
                duration = timestamps[-1] - contact_time
                
                # Show results
                QtWidgets.QMessageBox.information(
                    self, "Experiment Results",
                    f"Experiment Analysis:\n\n"
                    f"Contact detected at: {contact_time:.1f}s\n"
                    f"Force control duration: {duration:.1f}s\n\n"
                    f"Force Control Performance:\n"
                    f"  RMSE: {rmse:.2f} N\n"
                    f"  MAE: {mae:.2f} N\n"
                    f"  Max Error: {max_error:.2f} N\n\n"
                    f"Total samples: {len(timestamps)}"
                )
    
    def clear_data(self):
        """Clear experiment data"""
        self.experiment_data = {
            'timestamps': [],
            'target_forces': [],
            'actual_forces': [],
            'actual_forces_filtered': [],
            'grip_positions': [],
            'phases': [],
            'force_errors': []
        }
        self.force_info.clear()
        if hasattr(self, 'display_force_buffer'):
            self.display_force_buffer.clear()
    
    def closeEvent(self, event):
        """Handle application close"""
        if self.experiment_running:
            self.stop_experiment()
        
        # Stop EMG recording
        self.stop_emg_recording()
        
        try:
            if self.arm:
                self.arm.close()
        except:
            pass
        
        event.accept()

def connect_prosthetic_hand(hand_side="left"):
    """Connect to prosthetic hand using proven method"""
    print(f"Connecting to {hand_side} prosthetic hand...")
    
    try:
        print("Creating psyonicArm instance...")
        arm = psyonicArm(hand=hand_side)
        
        print("Initializing sensors...")
        arm.initSensors()
        print("Sensors initialized.")
        
        print("Starting communications...")
        arm.startComms()
        print("Communications started.")
        
        print("✓ Prosthetic hand connected successfully!")
        return arm
        
    except Exception as e:
        print(f"✗ Failed to connect to prosthetic hand: {e}")
        return None

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Adaptive Grip Control with Real-Time GUI and Auto-Save")
    parser.add_argument("--person_id", "-p", required=True,
                        help="Person ID (folder under data/)")
    parser.add_argument("--hand", choices=["left", "right"], default="left",
                        help="Hand side (default: left)")
    parser.add_argument("--out_root", "-o", default="data",
                        help="Root data directory (default: ./data)")
    
    args = parser.parse_args()
    
    print("=== Adaptive Grip Control with Real-Time GUI and Auto-Save ===")
    print(f"Person ID: {args.person_id}")
    print(f"Hand side: {args.hand}")
    print(f"Output root: {args.out_root}")
    print()
    
    # Step 1: Connect to prosthetic hand
    arm = connect_prosthetic_hand(args.hand)
    if arm is None:
        print("Failed to connect to prosthetic hand. Exiting.")
        return
    
    print()
    print("Starting GUI...")
    
    # Step 2: Start GUI with connected arm and person ID
    app = QApplication(sys.argv)
    
    # Create and show the main window with person ID for auto-save
    window = AdaptiveGripGUI(arm, args.person_id, args.out_root)
    window.show()
    
    print("GUI started successfully!")
    print()
    print("Instructions:")
    print("1. Select grip type and force pattern")
    print("2. Click 'Prepare Experiment' to move to neutral position")
    print("3. Place object in hand")
    print("4. Click 'Start Experiment' to begin adaptive grip control")
    print("5. Watch real-time force control in the plots")
    print("6. Data will be automatically saved when experiment completes")
    print(f"7. Files will be saved in: data/{args.person_id}/recordings/<grip>_interaction/experiments/")
    print()
    
    try:
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        try:
            if arm:
                arm.close()
                print("Prosthetic hand disconnected.")
        except:
            pass

if __name__ == "__main__":
    main()