#!/usr/bin/env python3
import sys
import time
import numpy as np
import threading
import argparse
import os
import yaml

# GUI imports (only imported if --gui flag is used)
try:
    import matplotlib
    matplotlib.use('Qt5Agg')  # Set backend before importing pyplot
    from PyQt5 import QtWidgets, QtCore, QtGui
    from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel
    from PyQt5.QtCore import QTimer
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False


# Import your prosthetic hand class
sys.path.append('C:/Users/Emanuel Wicki/Documents/MIT/biomech_PCP/helpers')
from psyonicHand import psyonicArm
from helpers.EMGClass import EMG
try: 
    from helpers.EMGClass import EMG
    EMG_AVAILABLE = True
except ImportError:
    EMG_AVAILABLE = False
    print("EMG class not available. EMG recording will be skipped.")    


# Enhanced grip configurations with min/max force limits
GRIP_CONFIGURATIONS = {
    "pinch": {
        "description": "Thumb-index pinch grip",
        "active_fingers": ["index", "thumb"],
        "neutral_position": [0, 0, 0, 0, 0, 0],
        "target_position": [67, 0, 0, 0, 67, -67],
        "contact_threshold": 0.3,
        "force_modulation": [1.0, 0.0, 0.0, 0.0, 1.0],
        "max_approach_speed": 30.0,
        "min_force": 1.0,  # Minimum force in Newtons
        "max_force": 8.0,  # Maximum force in Newtons
    },
    "power_grip": {
        "description": "Full hand power grip",
        "active_fingers": ["index", "middle", "ring", "pinky", "thumb"],
        "neutral_position": [0, 0, 0, 0, 0, 0],
        "target_position": [90, 90, 90, 90, 40, -90],
        "contact_threshold": 0.8,
        "force_modulation": [1.0, 1.0, 1.0, 1.0, 1.0],
        "max_approach_speed": 25.0,
        "min_force": 2.0,  # Minimum force in Newtons
        "max_force": 15.0,  # Maximum force in Newtons
    },
    "tripod": {
        "description": "Tripod grip (thumb, index, middle)",
        "active_fingers": ["index", "middle", "thumb"],
        "neutral_position": [0, 0, 0, 0, 0, 0],
        "target_position": [60, 60, 0, 0, 65, -80],
        "contact_threshold": 0.8,
        "force_modulation": [1.0, 1.0, 0.0, 0.0, 1.0],
        "max_approach_speed": 28.0,
        "min_force": 1.5,  # Minimum force in Newtons
        "max_force": 10.0,  # Maximum force in Newtons
    },
    "hook": {
        "description": "Hook grip (fingers only, no thumb)",
        "active_fingers": ["index", "middle", "ring", "pinky"],
        "neutral_position": [0, 0, 0, 0, 0, 0],
        "target_position": [110, 110, 110, 110, 0, 0],
        "contact_threshold": 0.8,
        "force_modulation": [1.0, 1.0, 1.0, 1.0, 0.0],
        "max_approach_speed": 25.0,
        "min_force": 2.5,  # Minimum force in Newtons
        "max_force": 12.0,  # Maximum force in Newtons
    }
}

def make_timestamps_unique(timestamps):
    """Make timestamps unique by adding small increments to duplicates"""
    timestamps = np.array(timestamps)
    for i in range(1, len(timestamps)):
        if timestamps[i] <= timestamps[i - 1]:
            timestamps[i] = timestamps[i - 1] + 1e-6  # add 1 microsecond
    return timestamps

class SimpleAdaptiveGripController:
    """Simplified controller for adaptive grip force control without GUI overhead"""
    
    def __init__(self, arm, grip_config, control_frequency=60.0):
        self.arm = arm
        self.grip_config = grip_config
        self.joint_names = ["index", "middle", "ring", "pinky", "thumbFlex", "thumbRot"]
        self.finger_names = ["index", "middle", "ring", "pinky", "thumb"]
        
        # Control frequency
        self.control_frequency = control_frequency
        self.control_dt = 1.0 / control_frequency
        
        # State variables
        self.phase = "NEUTRAL"
        self.contact_detected = False
        self.contact_position = None
        self.current_position = np.array(grip_config["neutral_position"], dtype=np.float64)
        
        # PID parameters - Tuned for the specified frequency
        self.kp = 2.5
        self.ki = 0.05
        self.kd = 0.3
        self.integral_error = 0.0
        self.previous_error = 0.0
        
        # Control limits - Adjusted for frequency
        frequency_scale = 60.0 / control_frequency  # Scale for different frequencies
        self.max_position_change = 0.8 * frequency_scale
        
        # Force control parameters
        self.force_buildup_mode = False
        self.force_buildup_threshold = 3.0
        
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
    
    def get_grip_force(self):
        """Get current total grip force from active fingers"""
        finger_forces = self.get_finger_forces()
        total_force = 0.0
        for finger in self.grip_config["active_fingers"]:
            total_force += finger_forces.get(finger, 0.0)
        return total_force
    
    def detect_contact(self):
        """Detect contact with object"""
        current_force = self.get_grip_force()
        if not self.contact_detected:
            if current_force > self.grip_config["contact_threshold"]:
                self.contact_detected = True
                self.contact_position = self.current_position.copy()
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
        
        neutral_pos = np.array(self.grip_config["neutral_position"], dtype=np.float64)
        target_pos = np.array(self.grip_config["target_position"], dtype=np.float64)
        joint_force_modulation = self.map_finger_to_joint_modulation()
        
        direction = (target_pos - self.current_position) * joint_force_modulation
        max_movement = self.grip_config["max_approach_speed"] * dt
        movement_magnitude = np.linalg.norm(direction)
        
        if movement_magnitude > 0:
            normalized_direction = direction / movement_magnitude
            actual_movement = normalized_direction * min(max_movement, movement_magnitude)
            
            self.current_position = self.current_position + actual_movement
            self.current_position = np.clip(self.current_position, neutral_pos, target_pos)
        
        return False
    
    def force_control_phase(self, target_force, dt):
        """Simple force control without filtering overhead"""
        current_force = self.get_grip_force()
        force_error = target_force - current_force
        
        # Determine control mode
        if target_force > self.force_buildup_threshold:
            self.force_buildup_mode = True
            kp_active = self.kp * 0.9
            ki_active = self.ki * 0.5
            kd_active = self.kd * 0.6
            max_change_active = self.max_position_change * 1.2
            fine_control_scale = 0.6
            deadband = 0.5
        else:
            self.force_buildup_mode = False
            kp_active = self.kp
            ki_active = self.ki
            kd_active = self.kd
            max_change_active = self.max_position_change
            fine_control_scale = 0.4
            deadband = 0.3
        
        # PID control
        self.integral_error += force_error * dt
        max_integral = 8.0 if self.force_buildup_mode else 5.0
        self.integral_error = np.clip(self.integral_error, -max_integral, max_integral)
        
        derivative_error = (force_error - self.previous_error) / dt if dt > 0 else 0
        self.previous_error = force_error
        
        pid_output = (kp_active * force_error + 
                     ki_active * self.integral_error + 
                     kd_active * derivative_error)
        
        # Apply deadband
        if abs(force_error) < deadband:
            pid_output = 0.0
        
        # Convert to position change
        joint_force_modulation = self.map_finger_to_joint_modulation()
        position_change = pid_output * joint_force_modulation * fine_control_scale
        position_change = np.clip(position_change, -max_change_active, max_change_active)
        
        # Apply minimum change threshold
        min_change_threshold = 0.05
        for i in range(len(position_change)):
            if abs(position_change[i]) < min_change_threshold:
                position_change[i] = 0.0
        
        # Update position
        self.current_position = self.current_position + position_change
        
        # Apply limits
        neutral_pos = np.array(self.grip_config["neutral_position"], dtype=np.float64)
        target_pos = np.array(self.grip_config["target_position"], dtype=np.float64)
        contact_pos = self.contact_position if self.contact_position is not None else neutral_pos
        
        if self.force_buildup_mode:
            # Allow extension for high forces
            for i in range(len(self.current_position)):
                if joint_force_modulation[i] > 0:
                    range_extension = (target_pos[i] - contact_pos[i]) * 0.6
                    min_pos = min(contact_pos[i], target_pos[i])
                    max_pos = max(contact_pos[i], target_pos[i] + range_extension)
                    self.current_position[i] = np.clip(self.current_position[i], min_pos, max_pos)
        else:
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

def generate_force_trajectory(duration, max_force, pattern, min_force=0.0):
    """Generate force trajectory based on settings, respecting min/max force limits"""
    dt = 1.0 / 60.0  # 60 Hz
    
    # Ensure max_force doesn't exceed grip limits
    force_range = max_force - min_force
    
    if pattern == "all":
        # Stitch all patterns together - each gets equal duration
        patterns = ["sine", "ramp", "step", "constant"]
        pattern_duration = duration / len(patterns)
        
        all_timestamps = []
        all_forces = []
        current_time_offset = 0
        
        print(f"Generating 'all' pattern: {len(patterns)} patterns × {pattern_duration:.1f}s each = {duration}s total")
        print(f"Force range: {min_force:.1f}N - {max_force:.1f}N")
        
        for i, sub_pattern in enumerate(patterns):
            print(f"  Pattern {i+1}/{len(patterns)}: {sub_pattern} ({pattern_duration:.1f}s)")
            
            # Generate timestamps for this pattern
            pattern_timestamps = np.arange(0, pattern_duration, dt)
            
            # Generate forces for this specific pattern
            if sub_pattern == "sine":
                # Sine wave oscillating between min_force + 10% range and max_force - 10% range
                base_force = min_force + force_range * 0.5
                amplitude = force_range * 0.3
                frequency = 0.1
                pattern_forces = base_force + amplitude * np.sin(2 * np.pi * frequency * pattern_timestamps)
            elif sub_pattern == "step":
                # Step pattern within force range
                pattern_forces = np.ones_like(pattern_timestamps) * (min_force + force_range * 0.3)
                pattern_forces[len(pattern_forces)//4:len(pattern_forces)//2] = min_force + force_range * 0.6
                pattern_forces[3*len(pattern_forces)//4:] = min_force + force_range * 0.9
            elif sub_pattern == "ramp":
                # Ramp pattern within force range
                pattern_forces = np.ones_like(pattern_timestamps) * min_force
                ramp_up = np.linspace(min_force, max_force, len(pattern_forces)//3)
                ramp_down = np.linspace(max_force, min_force, len(pattern_forces)//3)
                pattern_forces[len(pattern_forces)//3:2*len(pattern_forces)//3] = ramp_up
                pattern_forces[2*len(pattern_forces)//3:] = ramp_down[:len(pattern_forces) - 2*len(pattern_forces)//3]
            else:  # constant
                pattern_forces = np.ones_like(pattern_timestamps) * (min_force + force_range * 0.5)
            
            # Ensure forces stay within limits
            pattern_forces = np.clip(pattern_forces, min_force, max_force)
            
            # Adjust timestamps to be continuous
            adjusted_timestamps = pattern_timestamps + current_time_offset
            
            # Append to overall trajectory
            all_timestamps.extend(adjusted_timestamps)
            all_forces.extend(pattern_forces)
            
            # Update time offset for next pattern
            current_time_offset += pattern_duration
        
        print(f"Generated complete trajectory with {len(all_forces)} points over {all_timestamps[-1]:.1f}s")
        return np.array(all_timestamps), np.array(all_forces)
    
    else:
        # Single pattern mode (original behavior, but respecting force limits)
        timestamps = np.arange(0, duration, dt)
        
        if pattern == "sine":
            base_force = min_force + force_range * 0.5
            amplitude = force_range * 0.3
            frequency = 0.1
            forces = base_force + amplitude * np.sin(2 * np.pi * frequency * timestamps)
        elif pattern == "step":
            forces = np.ones_like(timestamps) * (min_force + force_range * 0.3)
            forces[len(forces)//4:len(forces)//2] = min_force + force_range * 0.6
            forces[3*len(forces)//4:] = min_force + force_range * 0.9
        elif pattern == "ramp":
            forces = np.ones_like(timestamps) * min_force
            ramp_up = np.linspace(min_force, max_force, len(forces)//3)
            ramp_down = np.linspace(max_force, min_force, len(forces)//3)
            forces[len(forces)//3:2*len(forces)//3] = ramp_up
            forces[2*len(forces)//3:] = ramp_down[:len(forces) - 2*len(forces)//3]
        else:  # constant
            forces = np.ones_like(timestamps) * (min_force + force_range * 0.5)
        
        # Ensure forces stay within limits
        forces = np.clip(forces, min_force, max_force)
        
        return timestamps, forces

def start_emg_recording(enable_emg=True):
    """Start EMG recording in background thread if enabled"""
    if not enable_emg:
        print("EMG recording disabled via --no_emg flag")
        def dummy_stop_recording():
            return [], []  # Return empty EMG data
        return dummy_stop_recording
    
    if not EMG_AVAILABLE:
        print("EMG class not available. Skipping EMG recording.")
        def dummy_stop_recording():
            return [], []
        return dummy_stop_recording
    
    emg = EMG()
    raw_emg_data = []
    raw_emg_timestamps = []
    recording_emg = True
    
    def emg_capture_loop():
        nonlocal recording_emg
        try:
            print("Starting EMG communication...")
            emg.startCommunication()
            
            # Wait for first data
            print("Waiting for first EMG data...")
            timeout_counter = 0
            while getattr(emg, 'OS_time', None) is None and timeout_counter < 100:
                time.sleep(0.01)
                timeout_counter += 1
            
            if getattr(emg, 'OS_time', None) is None:
                print("WARNING: EMG not responding, continuing without EMG")
                return
            
            print("EMG data detected, starting recording...")
            first_emg_time = emg.OS_time
            last_time = emg.OS_time
            
            while recording_emg:
                try:
                    time_sample = emg.OS_time
                    
                    if (time_sample - last_time)/1e6 > 0.1:
                        print(f'EMG Read time: {time_sample}, expected time: {last_time}')
                        print('EMG alignment lost. Please restart the EMG board.')
                        break
                    
                    elif time_sample > last_time:
                        emg_sample = np.asarray(emg.rawEMG)
                        raw_emg_data.append(list(emg_sample))
                        raw_emg_timestamps.append((time_sample - first_emg_time) / 1e6)
                    
                    last_time = time_sample
                    time.sleep(0.001)
                except Exception as e:
                    print(f"EMG read error: {e}")
                    break
                    
        except Exception as e:
            print(f"EMG capture setup error: {e}")
        finally:
            try:
                if hasattr(emg, 'exitEvent'):
                    emg.exitEvent.set()
                time.sleep(0.1)
                print("EMG recording thread finished")
            except:
                pass
    
    emg_thread = threading.Thread(target=emg_capture_loop, daemon=True)
    emg_thread.start()
    
    def stop_recording():
        nonlocal recording_emg
        recording_emg = False
        emg_thread.join(timeout=3.0)
        return raw_emg_data, raw_emg_timestamps
    
    return stop_recording

class SimplePlotCanvas(FigureCanvas):
    """Simple matplotlib canvas for force visualization with force bands"""
    
    def __init__(self, parent=None, grip_config=None, width=12, height=8, dpi=100):
        if not GUI_AVAILABLE:
            return
            
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        
        self.grip_config = grip_config
        
        # Create single subplot for force with bands
        self.ax_force = self.fig.add_subplot(1, 1, 1)
        
        # Configure force plot
        self.ax_force.set_title(f'Force Monitor - {grip_config["description"]}', fontsize=14, weight='bold')
        self.ax_force.set_xlabel('Time (s)', fontsize=12)
        self.ax_force.set_ylabel('Force (N)', fontsize=12)
        self.ax_force.grid(True, alpha=0.3)
        
        # Get force limits from grip config
        self.min_force = grip_config["min_force"]
        self.max_force = grip_config["max_force"]
        
        # Initialize plot elements
        self.measured_line, = self.ax_force.plot([], [], 'b-', linewidth=3, label='Measured Force', zorder=3)
        
        # Create force bands
        # Band 1: 0N to min_force (light red - too low)
        self.low_band = None
        # Band 2: min_force to max_force (light green - optimal range)
        self.optimal_band = None
        # Band 3: max_force to max_force+2 (light red - too high)
        self.high_band = None
        
        # Add legend and formatting
        self.ax_force.legend(loc='upper right')
        self.fig.tight_layout()
        
        # Track if bands have been initialized
        self.bands_initialized = False
        
    def initialize_force_bands(self, time_range):
        """Initialize the force bands once we have a time range"""
        if self.bands_initialized:
            return
            
        # Define band boundaries
        band_upper_limit = self.max_force + 2.0  # Max + 2N as requested
        
        # Create the bands as filled areas
        self.low_band = self.ax_force.axhspan(0, self.min_force, 
                                            alpha=0.2, color='red', 
                                            label=f'Too Low (0-{self.min_force:.1f}N)', zorder=1)
        
        self.optimal_band = self.ax_force.axhspan(self.min_force, self.max_force, 
                                                alpha=0.2, color='green', 
                                                label=f'Optimal ({self.min_force:.1f}-{self.max_force:.1f}N)', zorder=1)
        
        self.high_band = self.ax_force.axhspan(self.max_force, band_upper_limit, 
                                             alpha=0.2, color='red', 
                                             label=f'Too High ({self.max_force:.1f}-{band_upper_limit:.1f}N)', zorder=1)
        
        # Add horizontal lines for clear boundaries
        self.ax_force.axhline(y=self.min_force, color='darkgreen', linestyle='--', 
                            alpha=0.7, linewidth=2, label=f'Min Force ({self.min_force:.1f}N)')
        self.ax_force.axhline(y=self.max_force, color='darkgreen', linestyle='--', 
                            alpha=0.7, linewidth=2, label=f'Max Force ({self.max_force:.1f}N)')
        
        # Update legend
        self.ax_force.legend(loc='upper right', fontsize=10)
        
        self.bands_initialized = True
        
    def update_plots(self, timestamps, measured_forces, window_seconds=30):
        """Update plots with new data and force bands"""
        if not timestamps or len(timestamps) == 0:
            return
        
        try:
            times = np.array(timestamps)
            measured = np.array(measured_forces)
            
            # Time window - for "all" pattern, show more of the trajectory
            current_time = times[-1] if len(times) > 0 else 0
            start_time = max(0, current_time - window_seconds)
            
            time_mask = times >= start_time
            window_times = times[time_mask]
            window_measured = measured[time_mask]
            
            if len(window_times) == 0:
                return
            
            # Initialize bands if not done yet
            if not self.bands_initialized:
                self.initialize_force_bands([start_time, current_time])
            
            # Update measured force line
            self.measured_line.set_data(window_times, window_measured)
            
            # Auto-scale plot
            self.ax_force.set_xlim(start_time, current_time)
            
            # Set Y-axis to show the full force range with bands
            y_min = -0.5  # Show slightly below 0
            y_max = self.max_force + 2.5  # Show slightly above max+2
            self.ax_force.set_ylim(y_min, y_max)
            
            self.draw()
            
        except Exception as e:
            print(f"Plot update error: {e}")

class SimpleForceGUI(QMainWindow):
    """Enhanced GUI for force control visualization with force bands"""
    
    def __init__(self, grip_name, duration, max_force, pattern):
        if not GUI_AVAILABLE:
            return
            
        super().__init__()
        
        # Get grip configuration for this grip
        self.grip_config = GRIP_CONFIGURATIONS[grip_name]
        
        self.setWindowTitle(f"Force Control Monitor - {grip_name} ({self.grip_config['min_force']:.1f}-{self.grip_config['max_force']:.1f}N)")
        self.setGeometry(100, 100, 1000, 700)
        
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
        
        # Update timer - 30Hz as requested
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.setInterval(33)  # ~30 Hz (1000/30 = 33ms)
        
    def initUI(self):
        """Initialize the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        
        # Info panel - enhanced with force range info
        info_layout = QHBoxLayout()
        
        self.status_label = QLabel("Status: Starting...")
        self.status_label.setStyleSheet("font-weight: bold; padding: 8px; background-color: #e8f5e8; border-radius: 4px;")
        info_layout.addWidget(self.status_label)
        
        self.phase_label = QLabel("Phase: NEUTRAL")
        self.phase_label.setStyleSheet("font-weight: bold; padding: 8px; background-color: #fff3cd; border-radius: 4px;")
        info_layout.addWidget(self.phase_label)
        
        # Force range info
        min_f = self.grip_config["min_force"]
        max_f = self.grip_config["max_force"]
        self.range_label = QLabel(f"Force Range: {min_f:.1f}N - {max_f:.1f}N")
        self.range_label.setStyleSheet("font-weight: bold; padding: 8px; background-color: #d4edda; border-radius: 4px;")
        info_layout.addWidget(self.range_label)
        
        layout.addLayout(info_layout)
        
        # Current values panel
        values_layout = QHBoxLayout()
        
        self.current_force_label = QLabel("Current Force: 0.0N")
        self.current_force_label.setStyleSheet("font-family: monospace; font-size: 14px; font-weight: bold; padding: 8px; background-color: #f8f9fa; border-radius: 4px;")
        values_layout.addWidget(self.current_force_label)
        
        self.force_status_label = QLabel("Status: Within Range")
        self.force_status_label.setStyleSheet("font-family: monospace; font-size: 14px; font-weight: bold; padding: 8px; background-color: #d4edda; border-radius: 4px;")
        values_layout.addWidget(self.force_status_label)
        
        layout.addLayout(values_layout)
        
        # Plot canvas - pass grip config for force bands
        self.canvas = SimplePlotCanvas(self, grip_config=self.grip_config, width=12, height=8)
        layout.addWidget(self.canvas)
        
        # Stats panel
        self.stats_label = QLabel("Samples: 0 | Avg Force: -- | Time in Range: --")
        self.stats_label.setStyleSheet("font-family: monospace; padding: 8px; background-color: #e9ecef; border-radius: 4px;")
        layout.addWidget(self.stats_label)
    
    def update_data(self, timestamp, measured_force, phase=""):
        """Update data for GUI (called from experiment thread)"""
        # Thread-safe data update
        self.gui_data['timestamps'].append(timestamp)
        self.gui_data['measured_forces'].append(measured_force)
        
        # Keep only recent data to prevent memory issues
        max_samples = 2000
        if len(self.gui_data['timestamps']) > max_samples:
            for key in self.gui_data:
                self.gui_data[key] = self.gui_data[key][-max_samples:]
        
        # Update phase (thread-safe single assignment)
        self.current_phase = phase
    
    def get_force_status(self, force):
        """Determine if force is within acceptable range"""
        min_f = self.grip_config["min_force"]
        max_f = self.grip_config["max_force"]
        
        if force < min_f:
            return "TOO LOW", "#ffebee"  # Light red background
        elif force > max_f:
            return "TOO HIGH", "#ffebee"  # Light red background
        else:
            return "OPTIMAL", "#e8f5e8"  # Light green background
    
    def update_display(self):
        """Update GUI display at 30Hz"""
        if not self.gui_data['timestamps']:
            return
        
        try:
            # Update plots - for "all" pattern, show longer window to see pattern transitions
            window_seconds = self.duration * 4 if self.pattern == "all" else self.duration + 5
            self.canvas.update_plots(
                self.gui_data['timestamps'],
                self.gui_data['measured_forces'],
                window_seconds=window_seconds
            )
            
            # Update current values
            if self.gui_data['timestamps']:
                latest_time = self.gui_data['timestamps'][-1]
                latest_measured = self.gui_data['measured_forces'][-1]
                
                # Update current force display
                self.current_force_label.setText(f"Current Force: {latest_measured:.2f}N")
                
                # Update force status with color coding
                status_text, bg_color = self.get_force_status(latest_measured)
                self.force_status_label.setText(f"Status: {status_text}")
                self.force_status_label.setStyleSheet(f"font-family: monospace; font-size: 14px; font-weight: bold; padding: 8px; background-color: {bg_color}; border-radius: 4px;")
                
                # Update phase
                self.phase_label.setText(f"Phase: {self.current_phase}")
                
                if self.current_phase == "APPROACH":
                    self.phase_label.setStyleSheet("font-weight: bold; padding: 8px; background-color: #fff3cd; border-radius: 4px;")
                elif self.current_phase == "FORCE_CONTROL":
                    self.phase_label.setStyleSheet("font-weight: bold; padding: 8px; background-color: #d4edda; border-radius: 4px;")
                else:
                    self.phase_label.setStyleSheet("font-weight: bold; padding: 8px; background-color: #e2e3e5; border-radius: 4px;")
            
            # Update statistics (last 100 samples)
            if len(self.gui_data['timestamps']) > 10:
                recent_count = min(100, len(self.gui_data['timestamps']))
                recent_measured = np.array(self.gui_data['measured_forces'][-recent_count:])
                
                avg_force = np.mean(recent_measured)
                
                # Calculate time in optimal range
                min_f = self.grip_config["min_force"]
                max_f = self.grip_config["max_force"]
                in_range = np.logical_and(recent_measured >= min_f, recent_measured <= max_f)
                time_in_range_percent = np.mean(in_range) * 100
                
                self.stats_label.setText(
                    f"Samples: {len(self.gui_data['timestamps'])} | "
                    f"Avg Force: {avg_force:.2f}N | "
                    f"Time in Range: {time_in_range_percent:.1f}% | "
                    f"Range: {min_f:.1f}-{max_f:.1f}N"
                )
            
        except Exception as e:
            print(f"GUI update error: {e}")
    
    def start_updates(self):
        """Start the GUI update timer"""
        self.update_timer.start()
        self.status_label.setText("Status: Experiment Running")
    
    def stop_updates(self):
        """Stop the GUI update timer"""
        self.update_timer.stop()
        self.status_label.setText("Status: Experiment Completed")
    
    def closeEvent(self, event):
        """Handle window close event"""
        try:
            if hasattr(self, 'update_timer') and self.update_timer.isActive():
                self.update_timer.stop()
        except:
            pass
        event.accept()

def run_force_control_with_gui(arm, grip_name, duration, max_force, pattern, 
                              control_frequency, person_id, out_root, enable_emg=True):
    """Run force control experiment with GUI visualization"""
    
    if not GUI_AVAILABLE:
        print("Error: PyQt5 not available. Install PyQt5 to use GUI mode.")
        print("Falling back to command-line mode...")
        return run_force_control_experiment(arm, grip_name, duration, max_force, pattern, 
                                          control_frequency, person_id, out_root, enable_emg)
    
    # Get grip configuration
    grip_config = GRIP_CONFIGURATIONS[grip_name]
    
    print(f"\n=== Starting Force Control Experiment with GUI ===")
    print(f"Grip: {grip_name}")
    print(f"Description: {grip_config['description']}")
    print(f"Force Range: {grip_config['min_force']:.1f}N - {grip_config['max_force']:.1f}N")
    print(f"Duration: {duration}s")
    print(f"Trajectory Max Force: {max_force}N")
    print(f"Pattern: {pattern}")
    if pattern == "all":
        print(f"Will run ALL patterns: sine, ramp, step, constant ({duration/4:.1f}s each)")
    print(f"Control Frequency: {control_frequency}Hz")
    print(f"EMG Recording: {'Enabled' if enable_emg else 'Disabled'}")
    
    # Validate that trajectory max_force doesn't exceed grip max_force
    if max_force > grip_config["max_force"]:
        print(f"WARNING: Trajectory max force ({max_force}N) exceeds grip max force ({grip_config['max_force']}N)")
        print(f"Clamping trajectory to grip maximum: {grip_config['max_force']}N")
        max_force = grip_config["max_force"]
    
    # Initialize Qt Application
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    # Create GUI
    gui = SimpleForceGUI(grip_name, duration, max_force, pattern)
    gui.show()
    
    # Process events to ensure GUI is shown
    app.processEvents()
    
    # Generate force trajectory using grip's min/max force
    force_timestamps, force_trajectory = generate_force_trajectory(
        duration, max_force, pattern, min_force=grip_config["min_force"]
    )
    
    # Initialize controller
    controller = SimpleAdaptiveGripController(arm, grip_config, control_frequency)
    
    # Move to neutral position
    neutral_pos = np.array(grip_config["neutral_position"], dtype=np.float64)
    arm.mainControlLoop(posDes=neutral_pos.reshape(1, -1), period=2, emg=None)
    controller.current_position = neutral_pos.copy()
    
    input("\nPlace object in hand and press Enter to start experiment...")
    
    # Start EMG recording (or skip if disabled)
    if enable_emg:
        print("Starting EMG recording...")
    else:
        print("Skipping EMG recording (disabled via --no_emg flag)")
    stop_emg_recording = start_emg_recording(enable_emg)
    time.sleep(0.5)
    
    # Start experiment
    print("Starting experiment with GUI...")
    experiment_start_time = time.time()
    force_start_time = None
    
    # Reset controller
    controller.reset_controller()
    controller.phase = "APPROACH"
    
    # Start hand recording
    arm.resetRecording()
    arm.recording = True
    
    # Data storage (same as original)
    experiment_data = {
        'timestamps': [],
        'target_forces': [],
        'actual_forces': [],
        'grip_positions': [],
        'phases': [],
        'force_errors': []
    }
    
    # Start GUI updates
    gui.start_updates()
    
    # Control loop (same frequency as original, but GUI updates at 30Hz)
    control_dt = 1.0 / control_frequency
    next_control_time = time.time()
    
    print(f"Running control loop at {control_frequency}Hz with GUI at 30Hz...")
    print("Phase: APPROACH - waiting for contact...")
    
    def process_qt_events():
        """Process Qt events to keep GUI responsive"""
        app.processEvents()
    
    while True:
        current_time = time.time()
        
        # Process GUI events to keep it responsive
        process_qt_events()
        
        # Wait for next control cycle
        if current_time < next_control_time:
            sleep_time = next_control_time - current_time
            if sleep_time > 0.001:
                time.sleep(sleep_time)
            continue
        
        experiment_time = current_time - experiment_start_time
        
        # Control logic (same as original)
        current_force_raw = controller.get_grip_force()
        target_force = 0.0
        force_error = 0.0
        
        if controller.phase == "APPROACH":
            contact_detected = controller.approach_phase(control_dt)
            
            if contact_detected:
                controller.phase = "FORCE_CONTROL"
                force_start_time = experiment_time
                print(f"CONTACT detected at {experiment_time:.2f}s - starting FORCE CONTROL")
        
        elif controller.phase == "FORCE_CONTROL":
            trajectory_time = experiment_time - force_start_time
            
            if trajectory_time <= force_timestamps[-1]:
                target_force = np.interp(trajectory_time, force_timestamps, force_trajectory)
                current_force_raw, force_error, pid_output = controller.force_control_phase(target_force, control_dt)
            else:
                print(f"Trajectory completed at {experiment_time:.2f}s")
                break
        
        # Send position command
        position_command = controller.current_position.reshape(1, -1)
        arm.mainControlLoop(posDes=position_command, period=0.001, emg=None)
        
        # Store data (same as original)
        experiment_data['timestamps'].append(experiment_time)
        experiment_data['target_forces'].append(target_force)
        experiment_data['actual_forces'].append(current_force_raw)
        experiment_data['grip_positions'].append(controller.current_position.copy())
        experiment_data['phases'].append(controller.phase)
        experiment_data['force_errors'].append(force_error)
        
        # Update GUI data (this is called at control frequency but GUI updates at 30Hz)
        gui.update_data(experiment_time, current_force_raw, controller.phase)
        
        # Progress reporting (reduced frequency)
        if len(experiment_data['timestamps']) % (control_frequency * 2) == 0:
            if controller.phase == "FORCE_CONTROL":
                # Enhanced progress reporting with force status
                force_status = "OPTIMAL"
                if current_force_raw < grip_config["min_force"]:
                    force_status = "TOO LOW"
                elif current_force_raw > grip_config["max_force"]:
                    force_status = "TOO HIGH"
                
                print(f"Time: {experiment_time:.1f}s | Target: {target_force:.2f}N | Actual: {current_force_raw:.2f}N | Status: {force_status}")
        
        # Schedule next control cycle
        next_control_time += control_dt
        
        # Safety timeout
        if experiment_time > duration + 10:
            print("Safety timeout reached")
            break
    
    # Stop GUI updates
    gui.stop_updates()
    
    # Stop recording
    print("Stopping recordings...")
    arm.recording = False
    raw_emg_data, raw_emg_timestamps = stop_emg_recording()
    
    # Return to neutral
    print("Returning to neutral position...")
    neutral_pos = np.array(grip_config["neutral_position"], dtype=np.float64)
    arm.mainControlLoop(posDes=neutral_pos.reshape(1, -1), period=2, emg=None)
    
    # Save data (same as original)
    print("Saving data...")
    save_experiment_data(person_id, out_root, grip_name, experiment_data, 
                        raw_emg_data, raw_emg_timestamps, arm, grip_config, 
                        max_force, duration, pattern, control_frequency, enable_emg)
    
    # Print enhanced results with force range analysis
    analyze_experiment_results(experiment_data, grip_config)
    
    print("\nExperiment completed! Close the GUI window when ready.")
    
    # Keep GUI open until user closes it - but don't block forever
    try:
        import signal
        def signal_handler(sig, frame):
            print("\nForcing GUI close...")
            app.quit()
        
        signal.signal(signal.SIGINT, signal_handler)
        app.exec_()
    except KeyboardInterrupt:
        print("\nGUI closed by user")
    
    return True

def run_force_control_experiment(arm, grip_name, duration, max_force, pattern, 
                                control_frequency, person_id, out_root, enable_emg=True):
    """Run the force control experiment without GUI"""
    
    # Get grip configuration
    grip_config = GRIP_CONFIGURATIONS[grip_name]
    
    print(f"\n=== Starting Force Control Experiment ===")
    print(f"Grip: {grip_name}")
    print(f"Description: {grip_config['description']}")
    print(f"Force Range: {grip_config['min_force']:.1f}N - {grip_config['max_force']:.1f}N")
    print(f"Duration: {duration}s")
    print(f"Trajectory Max Force: {max_force}N")
    print(f"Pattern: {pattern}")
    if pattern == "all":
        print(f"Will run ALL patterns: sine, ramp, step, constant ({duration/4:.1f}s each)")
    print(f"Control Frequency: {control_frequency}Hz")
    print(f"EMG Recording: {'Enabled' if enable_emg else 'Disabled'}")
    
    # Validate grip name
    if grip_name not in GRIP_CONFIGURATIONS:
        print(f"Error: Unknown grip '{grip_name}'. Available: {list(GRIP_CONFIGURATIONS.keys())}")
        return False
    
    print(f"Active fingers: {', '.join(grip_config['active_fingers'])}")
    
    # Validate that trajectory max_force doesn't exceed grip max_force
    if max_force > grip_config["max_force"]:
        print(f"WARNING: Trajectory max force ({max_force}N) exceeds grip max force ({grip_config['max_force']}N)")
        print(f"Clamping trajectory to grip maximum: {grip_config['max_force']}N")
        max_force = grip_config["max_force"]
    
    # Generate force trajectory using grip's min/max force
    force_timestamps, force_trajectory = generate_force_trajectory(
        duration, max_force, pattern, min_force=grip_config["min_force"]
    )
    print(f"Generated force trajectory with {len(force_trajectory)} points")
    
    # Initialize controller
    controller = SimpleAdaptiveGripController(arm, grip_config, control_frequency)
    
    # Move to neutral position
    print("\nMoving to neutral position...")
    neutral_pos = np.array(grip_config["neutral_position"], dtype=np.float64)
    arm.mainControlLoop(posDes=neutral_pos.reshape(1, -1), period=2, emg=None)
    controller.current_position = neutral_pos.copy()
    
    input("\nPlace object in hand and press Enter to start experiment...")
    
    # Start EMG recording (or skip if disabled)
    if enable_emg:
        print("Starting EMG recording...")
    else:
        print("Skipping EMG recording (disabled via --no_emg flag)")
    stop_emg_recording = start_emg_recording(enable_emg)
    time.sleep(0.5)  # Give EMG time to initialize (or just a brief pause)
    
    # Start experiment
    print("Starting experiment...")
    experiment_start_time = time.time()
    force_start_time = None
    
    # Reset controller
    controller.reset_controller()
    controller.phase = "APPROACH"
    
    # Start hand recording
    arm.resetRecording()
    arm.recording = True
    
    # Data storage
    experiment_data = {
        'timestamps': [],
        'target_forces': [],
        'actual_forces': [],
        'grip_positions': [],
        'phases': [],
        'force_errors': []
    }
    
    # Control loop
    control_dt = 1.0 / control_frequency
    next_control_time = time.time()
    
    print(f"Running control loop at {control_frequency}Hz...")
    print("Phase: APPROACH - waiting for contact...")
    
    while True:
        current_time = time.time()
        
        # Wait for next control cycle
        if current_time < next_control_time:
            sleep_time = next_control_time - current_time
            if sleep_time > 0.001:  # Only sleep if meaningful time
                time.sleep(sleep_time)
            continue
        
        experiment_time = current_time - experiment_start_time
        
        # Get current force and execute control
        current_force_raw = controller.get_grip_force()
        target_force = 0.0
        force_error = 0.0
        
        if controller.phase == "APPROACH":
            contact_detected = controller.approach_phase(control_dt)
            
            if contact_detected:
                controller.phase = "FORCE_CONTROL"
                force_start_time = experiment_time
                print(f"CONTACT detected at {experiment_time:.2f}s - starting FORCE CONTROL")
        
        elif controller.phase == "FORCE_CONTROL":
            trajectory_time = experiment_time - force_start_time
            
            if trajectory_time <= force_timestamps[-1]:
                target_force = np.interp(trajectory_time, force_timestamps, force_trajectory)
                current_force_raw, force_error, pid_output = controller.force_control_phase(target_force, control_dt)
            else:
                print(f"Trajectory completed at {experiment_time:.2f}s")
                break
        
        # Send position command
        position_command = controller.current_position.reshape(1, -1)
        arm.mainControlLoop(posDes=position_command, period=0.001, emg=None)
        
        # Store data
        experiment_data['timestamps'].append(experiment_time)
        experiment_data['target_forces'].append(target_force)
        experiment_data['actual_forces'].append(current_force_raw)
        experiment_data['grip_positions'].append(controller.current_position.copy())
        experiment_data['phases'].append(controller.phase)
        experiment_data['force_errors'].append(force_error)
        
        # Enhanced progress reporting with force status
        if len(experiment_data['timestamps']) % (control_frequency * 2) == 0:  # Every 2 seconds
            if controller.phase == "FORCE_CONTROL":
                force_status = "OPTIMAL"
                if current_force_raw < grip_config["min_force"]:
                    force_status = "TOO LOW"
                elif current_force_raw > grip_config["max_force"]:
                    force_status = "TOO HIGH"
                
                print(f"Time: {experiment_time:.1f}s | Target: {target_force:.2f}N | Actual: {current_force_raw:.2f}N | Status: {force_status}")
        
        # Schedule next control cycle
        next_control_time += control_dt
        
        # Safety timeout
        if experiment_time > duration + 10:  # 10 seconds extra
            print("Safety timeout reached")
            break
    
    # Stop recording
    print("Stopping recordings...")
    arm.recording = False
    raw_emg_data, raw_emg_timestamps = stop_emg_recording()
    
    # Return to neutral
    print("Returning to neutral position...")
    neutral_pos = np.array(grip_config["neutral_position"], dtype=np.float64)
    arm.mainControlLoop(posDes=neutral_pos.reshape(1, -1), period=2, emg=None)
    
    # Save data
    print("Saving data...")
    save_experiment_data(person_id, out_root, grip_name, experiment_data, 
                        raw_emg_data, raw_emg_timestamps, arm, grip_config, 
                        max_force, duration, pattern, control_frequency, enable_emg)
    
    # Print enhanced results
    analyze_experiment_results(experiment_data, grip_config)
    
    return True

def save_experiment_data(person_id, out_root, grip_name, experiment_data, 
                        raw_emg_data, raw_emg_timestamps, arm, grip_config, 
                        max_force, duration, pattern, control_frequency, enable_emg=True):
    """Save experiment data in s1.5 format"""
    
    # Create directory structure
    exp_parent = os.path.join(
        out_root, person_id, "recordings", f"{grip_name}_interaction", "experiments"
    )
    
    exp_idx = 1
    while os.path.exists(os.path.join(exp_parent, str(exp_idx))):
        exp_idx += 1
    
    base_dir = os.path.join(exp_parent, str(exp_idx))
    os.makedirs(base_dir, exist_ok=True)
    
    print(f"Saving to: {base_dir}")
    
    files_saved = []
    
    # Save EMG data (only if EMG was enabled and data exists)
    if enable_emg and raw_emg_data and len(raw_emg_data) > 0:
        try:
            raw_emg_array = np.vstack(raw_emg_data)
            np.save(os.path.join(base_dir, "raw_emg.npy"), raw_emg_array)
            files_saved.append("raw_emg.npy")
            
            raw_timestamps_unique = make_timestamps_unique(raw_emg_timestamps)
            np.save(os.path.join(base_dir, "raw_timestamps.npy"), np.array(raw_timestamps_unique))
            files_saved.append("raw_timestamps.npy")
            
            print(f"Saved EMG data: {len(raw_emg_data)} samples, {raw_emg_timestamps[-1] - raw_emg_timestamps[0]:.2f}s")
        except Exception as e:
            print(f"Failed to save EMG data: {e}")
    elif not enable_emg:
        print("EMG data not saved (disabled via --no_emg flag)")
    else:
        print("No EMG data to save")
    
    # Save angle data
    if hasattr(arm, 'recordedData') and arm.recordedData and len(arm.recordedData) > 1:
        try:
            raw_data = arm.recordedData
            headers = raw_data[0]
            data_rows = raw_data[1:]
            
            if data_rows:
                rec = np.array(data_rows, dtype=float)
                ts = rec[:, 0]
                ts -= ts[0]
                
                np.save(os.path.join(base_dir, "angles.npy"), rec)
                files_saved.append("angles.npy")
                
                angle_timestamps_unique = make_timestamps_unique(ts)
                np.save(os.path.join(base_dir, "angle_timestamps.npy"), angle_timestamps_unique)
                files_saved.append("angle_timestamps.npy")
                
                with open(os.path.join(base_dir, "angles_header.txt"), "w") as f:
                    f.write(",".join(headers))
                files_saved.append("angles_header.txt")
                
                print(f"Saved angle data: {len(rec)} frames, {ts[-1]:.2f}s")
        except Exception as e:
            print(f"Failed to save angle data: {e}")
    
    # Save experiment configuration
    try:
        config_data = {
            'person_id': person_id,
            'grip_type': grip_name,
            'grip_config': grip_config,
            'force_pattern': pattern,
            'max_force': max_force,
            'duration': duration,
            'control_frequency': control_frequency,
            'emg_enabled': enable_emg,
            'experiment_summary': {
                'total_samples': len(experiment_data['timestamps']),
                'total_duration': experiment_data['timestamps'][-1] if experiment_data['timestamps'] else 0,
                'emg_samples': len(raw_emg_data) if enable_emg else 0,
                'angle_samples': len(arm.recordedData) - 1 if hasattr(arm, 'recordedData') and arm.recordedData else 0
            }
        }
        
        with open(os.path.join(base_dir, 'experiment_config.yaml'), 'w') as f:
            yaml.safe_dump(config_data, f)
        files_saved.append('experiment_config.yaml')
        
        print(f"Saved experiment config")
    except Exception as e:
        print(f"Failed to save config: {e}")
    
    print(f"Saved {len(files_saved)} files to experiment {exp_idx}")

def analyze_experiment_results(experiment_data, grip_config):
    """Analyze and print experiment results with force range analysis"""
    timestamps = np.array(experiment_data['timestamps'])
    target_forces = np.array(experiment_data['target_forces'])
    actual_forces = np.array(experiment_data['actual_forces'])
    phases = experiment_data['phases']
    
    # Find contact point
    contact_idx = next((i for i, phase in enumerate(phases) if phase == "FORCE_CONTROL"), None)
    
    if contact_idx is not None:
        control_targets = target_forces[contact_idx:]
        control_actuals = actual_forces[contact_idx:]
        
        if len(control_actuals) > 0:
            force_errors = control_targets - control_actuals
            rmse = np.sqrt(np.mean(force_errors**2))
            mae = np.mean(np.abs(force_errors))
            max_error = np.max(np.abs(force_errors))
            
            contact_time = timestamps[contact_idx]
            duration = timestamps[-1] - contact_time
            
            # Force range analysis
            min_f = grip_config["min_force"]
            max_f = grip_config["max_force"]
            
            # Calculate time spent in different force ranges
            too_low = np.sum(control_actuals < min_f)
            optimal = np.sum((control_actuals >= min_f) & (control_actuals <= max_f))
            too_high = np.sum(control_actuals > max_f)
            total_samples = len(control_actuals)
            
            too_low_percent = (too_low / total_samples) * 100
            optimal_percent = (optimal / total_samples) * 100
            too_high_percent = (too_high / total_samples) * 100
            
            avg_force = np.mean(control_actuals)
            
            print(f"\n=== Experiment Results ===")
            print(f"Contact detected at: {contact_time:.1f}s")
            print(f"Force control duration: {duration:.1f}s")
            print(f"Total samples: {len(timestamps)}")
            print(f"\n=== Force Range Analysis ===")
            print(f"Grip Force Range: {min_f:.1f}N - {max_f:.1f}N")
            print(f"Average Force: {avg_force:.2f}N")
            print(f"Time in Optimal Range: {optimal_percent:.1f}% ({optimal}/{total_samples} samples)")
            print(f"Time Too Low (<{min_f:.1f}N): {too_low_percent:.1f}% ({too_low}/{total_samples} samples)")
            print(f"Time Too High (>{max_f:.1f}N): {too_high_percent:.1f}% ({too_high}/{total_samples} samples)")
            print(f"\n=== Control Performance ===")
            print(f"RMSE: {rmse:.2f}N")
            print(f"MAE: {mae:.2f}N")
            print(f"Max Error: {max_error:.2f}N")

def connect_prosthetic_hand(hand_side="left"):
    """Connect to prosthetic hand"""
    print(f"Connecting to {hand_side} prosthetic hand...")
    
    try:
        arm = psyonicArm(hand=hand_side)
        arm.initSensors()
        arm.startComms()
        print("Prosthetic hand connected successfully!")
        return arm
    except Exception as e:
        print(f"Failed to connect to prosthetic hand: {e}")
        return None

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Command-Line Force Control Experiment with Force Bands")
    parser.add_argument("--person_id", "-p", required=True,
                        help="Person ID (folder under data/)")
    parser.add_argument("--grip", "-g", required=True, 
                        choices=list(GRIP_CONFIGURATIONS.keys()),
                        help="Grip type")
    parser.add_argument("--duration", "-d", type=float, default=20.0,
                        help="Experiment duration in seconds (default: 60)")
    parser.add_argument("--max_force", "-f", type=float, default=6.0,
                        help="Maximum trajectory force in Newtons (default: 6.0, will be clamped to grip max)")
    parser.add_argument("--pattern", choices=["sine", "step", "ramp", "constant", "all"], 
                        default="sine", help="Force pattern (default: sine). 'all' stitches all patterns together.")
    parser.add_argument("--frequency", type=float, default=100.0,
                        help="Control frequency in Hz (default: 200)")
    parser.add_argument("--hand", choices=["left", "right"], default="left",
                        help="Hand side (default: left)")
    parser.add_argument("--out_root", "-o", default="data",
                        help="Root data directory (default: ./data)")
    parser.add_argument("--gui", action="store_true",
                        help="Show GUI for real-time force visualization with force bands")
    parser.add_argument("--no_emg", action="store_true",
                        help="Disable EMG recording (useful for testing without EMG hardware)")
    
    args = parser.parse_args()
    
    # Check for conflicting EMG settings
    enable_emg = not args.no_emg
    
    print("=== Command-Line Force Control Experiment ===")
    print(f"Person ID: {args.person_id}")
    print(f"Grip: {args.grip}")
    
    # Show grip-specific force limits
    if args.grip in GRIP_CONFIGURATIONS:
        grip_config = GRIP_CONFIGURATIONS[args.grip]
        print(f"Grip Force Range: {grip_config['min_force']:.1f}N - {grip_config['max_force']:.1f}N")
    
    print(f"Duration: {args.duration}s")
    print(f"Max Trajectory Force: {args.max_force}N")
    print(f"Pattern: {args.pattern}")
    print(f"Control Frequency: {args.frequency}Hz")
    print(f"Hand side: {args.hand}")
    print(f"Output root: {args.out_root}")
    print(f"GUI Mode: {'Enabled' if args.gui else 'Disabled'}")
    print(f"EMG Recording: {'Enabled' if enable_emg else 'Disabled (--no_emg flag)'}")
    
    if args.no_emg:
        print("EMG recording disabled - useful for testing without EMG hardware")
    
    # Connect to prosthetic hand
    arm = connect_prosthetic_hand(args.hand)
    if arm is None:
        print("Failed to connect to prosthetic hand. Exiting.")
        return
    
    try:
        if args.gui:
            # Run with GUI
            success = run_force_control_with_gui(
                arm=arm,
                grip_name=args.grip,
                duration=args.duration,
                max_force=args.max_force,
                pattern=args.pattern,
                control_frequency=args.frequency,
                person_id=args.person_id,
                out_root=args.out_root,
                enable_emg=enable_emg
            )
        else:
            # Run the experiment without GUI
            success = run_force_control_experiment(
                arm=arm,
                grip_name=args.grip,
                duration=args.duration,
                max_force=args.max_force,
                pattern=args.pattern,
                control_frequency=args.frequency,
                person_id=args.person_id,
                out_root=args.out_root,
                enable_emg=enable_emg
            )
        
        if success:
            print("\n Experiment completed successfully!")
            print(f"Data saved to: data/{args.person_id}/recordings/{args.grip}_interaction/experiments/")
            if not enable_emg:
                print(" Note: No EMG data was recorded (--no_emg flag was used)")
        else:
            print("\n Experiment failed!")
            
    except KeyboardInterrupt:
        print("\n\n Experiment interrupted by user")
    except Exception as e:
        print(f"\n\n Experiment failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            if arm:
                arm.close()
                print("Prosthetic hand disconnected.")
        except:
            pass

if __name__ == "__main__":
    main()