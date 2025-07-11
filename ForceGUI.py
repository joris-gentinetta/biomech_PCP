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
from PyQt5.QtCore import QTimer, pyqtSignal, QThread
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Import your prosthetic hand class
sys.path.append('C:/Users/Emanuel Wicki/Documents/MIT/biomech_PCP/helpers')
from psyonicHand import psyonicArm

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
    """Main GUI window for force monitoring - takes pre-connected arm"""
    
    def __init__(self, connected_arm):
        super().__init__()
        self.setWindowTitle("Prosthetic Hand Force Monitor")
        self.setGeometry(100, 100, 1400, 900)
        
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
        
        self.initUI()
        
        # Data collection thread and timer
        self.data_thread = None
        self.stop_event = None
        
        # Update timer for GUI
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.setInterval(100)  # 10 Hz GUI updates
        
        # Show connected status
        self.update_status("Hand connected and ready")
    
    def initUI(self):
        """Initialize the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Left side: Controls
        controls_widget = QWidget()
        controls_widget.setMaximumWidth(320)
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
        
        # Data collection group
        data_group = QGroupBox("Data Collection")
        data_layout = QVBoxLayout(data_group)
        
        self.start_btn = QPushButton("Start Data Collection")
        self.start_btn.clicked.connect(self.start_data_collection)
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
        """Worker function for data collection thread - with thread safety"""
        while not self.stop_event.is_set():
            try:
                if not self.arm or not self.connected:
                    break
                
                # Get current timestamp
                current_time = time.time() - self.start_time
                
                # Collect force data from all fingers
                force_data = {}
                total_grip_force = 0
                
                for finger in self.finger_names:
                    finger_total = 0
                    finger_forces = []
                    
                    # Read all 6 sensors for this finger
                    for sensor_idx in range(6):
                        sensor_name = f"{finger}{sensor_idx}_Force"
                        force_value = self.arm.sensors.get(sensor_name, 0)
                        finger_forces.append(force_value)
                        finger_total += force_value
                    
                    force_data[finger] = {
                        'individual': finger_forces,
                        'total': finger_total
                    }
                    total_grip_force += finger_total
                
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
        """Start data collection"""
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
        self.update_status("Data collection started")
    
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
        text += "Total Force per Finger:\n"
        text += "-" * 25 + "\n"
        
        for finger in self.finger_names:
            if finger in self.current_forces:
                total = self.current_forces[finger]['total']
                text += f"{finger.capitalize():>8}: {total:6.2f} N\n"
        
        text += "-" * 25 + "\n"
        text += f"{'TOTAL GRIP':>8}: {self.current_forces.get('total_grip', 0):6.2f} N"
        
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
                        text += f"  Mean: {mean_val:5.2f} N\n"
                        text += f"  Max:  {max_val:5.2f} N\n"
                        text += f"  Std:  {std_val:5.2f} N\n\n"
            
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
    parser = argparse.ArgumentParser(description="Prosthetic Hand Force Monitor GUI")
    parser.add_argument("--hand", choices=["left", "right"], default="left",
                        help="Hand side (default: left)")
    args = parser.parse_args()
    
    print("=== Prosthetic Hand Force Monitor ===")
    print(f"Hand side: {args.hand}")
    print()
    
    # Step 1: Connect to prosthetic hand (same as psyonicHand.py)
    arm = connect_prosthetic_hand(args.hand)
    if arm is None:
        print("Failed to connect to prosthetic hand. Exiting.")
        return
    
    print()
    print("Starting GUI...")
    
    # Step 2: Start GUI with connected arm
    app = QApplication(sys.argv)
    
    # Create and show the main window with pre-connected arm
    window = ForceMonitorGUI(arm)
    window.show()
    
    print("GUI started. You can now start data collection.")
    
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