#!/usr/bin/env python3
import argparse
import os
import time
import threading
import numpy as np
import cv2
import yaml
import matplotlib.pyplot as plt
from os.path import join
from helpers.hand_poses import hand_poses
from psyonicHand import psyonicArm
from helpers.EMGClass import EMG
from helpers.BesselFilter import BesselFilterArr
import pandas as pd

# Define grip configurations with neutral and target positions
GRIP_CONFIGURATIONS = {
    "pinch": {
        "description": "Thumb-index pinch grip",
        "active_fingers": ["index", "thumb"],  # Changed to match finger naming
        "neutral_position": [0, 0, 0, 0, 0, 0],  # Open hand
        "target_position": [60, 0, 0, 0, 60, -60],  # Full pinch closure
        "contact_threshold": 0.8,  # Force threshold to detect contact (N)
        "force_modulation": [1.0, 0.0, 0.0, 0.0, 1.0],  # Which FINGERS to modulate (5 fingers)
        "max_approach_speed": 20.0,  # degrees per second during approach
    },
    "power_grip": {
        "description": "Full hand power grip",
        "active_fingers": ["index", "middle", "ring", "pinky", "thumb"],
        "neutral_position": [0, 0, 0, 0, 0, 0],
        "target_position": [90, 90, 90, 90, 40, -90],  # Full closure
        "contact_threshold": 0.8,
        "force_modulation": [1.0, 1.0, 1.0, 1.0, 1.0],  # All 5 fingers
        "max_approach_speed": 15.0,
    },
    "tripod": {
        "description": "Tripod grip (thumb, index, middle)",
        "active_fingers": ["index", "middle", "thumb"],
        "neutral_position": [0, 0, 0, 0, 0, 0],
        "target_position": [60, 60, 0, 0, 65, -80],
        "contact_threshold": 0.8,
        "force_modulation": [1.0, 1.0, 0.0, 0.0, 1.0],  # Index, middle, thumb only
        "max_approach_speed": 18.0,
    },
    "hook": {
        "description": "Hook grip (fingers only, no thumb)",
        "active_fingers": ["index", "middle", "ring", "pinky"],
        "neutral_position": [0, 0, 0, 0, 0, 0],
        "target_position": [110, 110, 110, 110, 0, 0],
        "contact_threshold": 0.8,
        "force_modulation": [1.0, 1.0, 1.0, 1.0, 0.0],  # Fingers only, no thumb
        "max_approach_speed": 15.0,
    }
}

class AdaptiveGripController:
    """
    Controller for adaptive grip force control
    """
    
    def __init__(self, arm, grip_config):
        self.arm = arm
        self.grip_config = grip_config
        self.joint_names = ["index", "middle", "ring", "pinky", "thumbFlex", "thumbRot"]  # 6 DoF
        self.finger_names = ["index", "middle", "ring", "pinky", "thumb"]  # 5 fingers with force sensors
        
        # State variables
        self.phase = "NEUTRAL"  # NEUTRAL -> APPROACH -> CONTACT -> FORCE_CONTROL
        self.contact_detected = False
        self.contact_position = None
        # Ensure current_position is float64 array
        self.current_position = np.array(grip_config["neutral_position"], dtype=np.float64)
        
        # Force control parameters
        self.force_history = []
        self.position_history = []
        self.timestamps = []
        
        # PID parameters for force control - Much more conservative
        self.kp = 1.5   # Proportional gain (reduced from 8.0)
        self.ki = 0.05  # Integral gain (reduced from 0.5)
        self.kd = 0.3   # Derivative gain (reduced from 1.0)
        self.integral_error = 0.0
        self.previous_error = 0.0
        
        # Safety and control limits - Much smaller adjustments
        self.max_position_change = 1.0  # max degrees change per control cycle (reduced from 5.0)
        self.control_rate = 60.0  # Hz
        
        print(f"Adaptive Grip Controller initialized:")
        print(f"  Grip: {grip_config['description']}")
        print(f"  Active fingers: {grip_config['active_fingers']}")
        print(f"  Contact threshold: {grip_config['contact_threshold']} N")
        print(f"  Target position: {grip_config['target_position']}")
        print(f"  Force modulation (5 fingers): {grip_config['force_modulation']}")
    
    def get_finger_forces(self):
        """Get force for each of the 5 fingers (sum of 6 sensors per finger)"""
        finger_forces = {}
        
        for finger in self.finger_names:
            finger_total = 0.0
            # Sum all 6 sensors for this finger
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
        
        # Sum only active fingers
        for finger in self.grip_config["active_fingers"]:
            total_force += finger_forces.get(finger, 0.0)
        
        return total_force
    
    def detect_contact(self):
        """Detect when contact is made with object"""
        current_force = self.get_grip_force()
        
        if not self.contact_detected:
            if current_force > self.grip_config["contact_threshold"]:
                self.contact_detected = True
                # Ensure contact position is float64
                self.contact_position = self.current_position.astype(np.float64).copy()
                self.phase = "CONTACT"
                print(f"CONTACT DETECTED! Force: {current_force:.2f} N")
                print(f"Contact position: {self.contact_position}")
                return True
        
        return self.contact_detected
    
    def approach_phase(self, dt):
        """
        Gradually move towards target position until contact is detected
        """
        if self.detect_contact():
            return True  # Contact detected, move to next phase
        
        # Ensure current_position stays float64
        if self.current_position.dtype != np.float64:
            self.current_position = self.current_position.astype(np.float64)
        
        # Calculate movement towards target - ensure all arrays are float64
        neutral_pos = np.array(self.grip_config["neutral_position"], dtype=np.float64)
        target_pos = np.array(self.grip_config["target_position"], dtype=np.float64)
        
        # Get joint-level force modulation by mapping finger modulation to joints
        joint_force_modulation = self.map_finger_to_joint_modulation()
        
        # Direction vector from current to target (only for active joints)
        direction = (target_pos - self.current_position) * joint_force_modulation
        
        # Limit movement speed
        max_movement = self.grip_config["max_approach_speed"] * dt  # degrees per timestep
        movement_magnitude = np.linalg.norm(direction)
        
        if movement_magnitude > 0:
            # Normalize and scale movement
            normalized_direction = direction / movement_magnitude
            actual_movement = normalized_direction * min(max_movement, movement_magnitude)
            
            # Ensure actual_movement is float64
            actual_movement = actual_movement.astype(np.float64)
            
            # Update position (both arrays are now float64)
            self.current_position = self.current_position + actual_movement
            
            # Apply joint limits
            self.current_position = np.clip(
                self.current_position, 
                neutral_pos, 
                target_pos
            )
        
        return False  # Still approaching
    
    def map_finger_to_joint_modulation(self):
        """
        Map finger force modulation (5 elements) to joint modulation (6 elements)
        finger_modulation: [index, middle, ring, pinky, thumb]
        joint_modulation:  [index, middle, ring, pinky, thumbFlex, thumbRot]
        """
        finger_modulation = np.array(self.grip_config["force_modulation"], dtype=np.float64)
        joint_modulation = np.zeros(6, dtype=np.float64)
        
        # Map fingers to joints
        joint_modulation[0] = finger_modulation[0]  # index -> index
        joint_modulation[1] = finger_modulation[1]  # middle -> middle  
        joint_modulation[2] = finger_modulation[2]  # ring -> ring
        joint_modulation[3] = finger_modulation[3]  # pinky -> pinky
        joint_modulation[4] = finger_modulation[4]  # thumb -> thumbFlex
        joint_modulation[5] = finger_modulation[4]  # thumb -> thumbRot (same as thumbFlex)
        
        return joint_modulation
    
    def force_control_phase(self, target_force, dt):
        """
        Control grip force by modulating position between contact and target
        Uses conservative adjustments for smooth force control
        """
        current_force = self.get_grip_force()
        force_error = target_force - current_force
        
        # PID control with conservative gains
        self.integral_error += force_error * dt
        
        # More restrictive anti-windup for smoother control
        max_integral = 10.0  # Reduced from 50.0
        self.integral_error = np.clip(self.integral_error, -max_integral, max_integral)
        
        derivative_error = (force_error - self.previous_error) / dt if dt > 0 else 0
        self.previous_error = force_error
        
        # PID output (position adjustment) - now much smaller
        pid_output = (self.kp * force_error + 
                     self.ki * self.integral_error + 
                     self.kd * derivative_error)
        
        # Add deadband to prevent tiny oscillations
        deadband = 0.2  # Don't adjust for errors smaller than 0.2N
        if abs(force_error) < deadband:
            pid_output = 0.0
        
        # Convert PID output to position change with additional scaling
        joint_force_modulation = self.map_finger_to_joint_modulation()
        
        # Additional scaling factor for even smaller movements
        fine_control_scale = 0.3  # Scale down movements by another 70%
        position_change = pid_output * joint_force_modulation * fine_control_scale
        
        # Very restrictive position change limits
        max_change = self.max_position_change  # Already reduced to 1.0 degrees
        position_change = np.clip(position_change, -max_change, max_change)
        
        # Only apply changes if they're significant enough
        min_change_threshold = 0.1  # degrees
        for i in range(len(position_change)):
            if abs(position_change[i]) < min_change_threshold:
                position_change[i] = 0.0
        
        # Update position
        self.current_position = self.current_position + position_change
        
        # Apply limits: between contact position and target position
        neutral_pos = np.array(self.grip_config["neutral_position"], dtype=np.float64)
        target_pos = np.array(self.grip_config["target_position"], dtype=np.float64)
        contact_pos = self.contact_position if self.contact_position is not None else neutral_pos
        
        # For each joint, limit between contact and target
        for i in range(len(self.current_position)):
            if joint_force_modulation[i] > 0:  # Only limit active joints
                min_pos = min(contact_pos[i], target_pos[i])
                max_pos = max(contact_pos[i], target_pos[i])
                self.current_position[i] = np.clip(self.current_position[i], min_pos, max_pos)
        
        return current_force, force_error, pid_output
    
    def reset_controller(self):
        """Reset controller to initial state"""
        self.phase = "NEUTRAL"
        self.contact_detected = False
        self.contact_position = None
        # Ensure reset position is float64
        self.current_position = np.array(self.grip_config["neutral_position"], dtype=np.float64)
        self.integral_error = 0.0
        self.previous_error = 0.0
        self.force_history = []
        self.position_history = []
        self.timestamps = []
        print("Controller reset to neutral position")

def load_force_trajectory(filepath):
    """
    Load target force trajectory from CSV file
    Expected format: columns 'timestamp' and 'force'
    """
    try:
        data = pd.read_csv(filepath)
        if 'timestamp' not in data.columns or 'force' not in data.columns:
            raise ValueError("CSV must contain 'timestamp' and 'force' columns")
        
        timestamps = data['timestamp'].values
        forces = data['force'].values
        
        print(f"Loaded force trajectory: {len(timestamps)} points")
        print(f"Duration: {timestamps[-1] - timestamps[0]:.1f} seconds")
        print(f"Force range: {forces.min():.1f} - {forces.max():.1f} N")
        
        return timestamps, forces
    
    except Exception as e:
        print(f"Error loading force trajectory: {e}")
        return None, None

def generate_test_force_trajectory(duration=20.0, pattern="sine"):
    """
    Generate test force trajectory patterns
    """
    dt = 1.0 / 60.0  # 60 Hz
    timestamps = np.arange(0, duration, dt)
    
    if pattern == "sine":
        # Sinusoidal force pattern
        base_force = 5.0
        amplitude = 3.0
        frequency = 0.1  # Hz
        forces = base_force + amplitude * np.sin(2 * np.pi * frequency * timestamps)
        
    elif pattern == "step":
        # Step pattern
        forces = np.ones_like(timestamps) * 2.0
        forces[len(forces)//4:len(forces)//2] = 6.0
        forces[3*len(forces)//4:] = 8.0
        
    elif pattern == "ramp":
        # Ramp up and down
        forces = np.ones_like(timestamps) * 2.0
        ramp_up = np.linspace(2.0, 8.0, len(forces)//3)
        ramp_down = np.linspace(8.0, 2.0, len(forces)//3)
        forces[len(forces)//3:2*len(forces)//3] = ramp_up
        forces[2*len(forces)//3:] = ramp_down[:len(forces) - 2*len(forces)//3]
        
    else:
        # Constant force
        forces = np.ones_like(timestamps) * 5.0
    
    return timestamps, forces

def start_raw_emg_recorder(base_dir, enable_video=False, sync_event=None):
    """Start EMG and video recording threads"""
    emg = EMG()
    
    raw_history = []
    raw_timestamps = []
    video_timestamps = []
    stop_event = threading.Event()
    
    # Video thread
    video_thread = None
    if enable_video:
        def video_loop():
            cap = cv2.VideoCapture(0)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_path = join(base_dir, 'webcam.mp4')
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = 30.0
            writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
            
            if sync_event:
                sync_event.wait()
            
            video_first_time = None
            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break
                now = time.time()
                if video_first_time is None:
                    video_first_time = now
                video_timestamps.append(now - video_first_time)
                writer.write(frame)
                time.sleep(1.0 / fps)
            
            cap.release()
            writer.release()
        
        video_thread = threading.Thread(target=video_loop, daemon=True)
        video_thread.start()
    
    # EMG thread
    def emg_loop():
        try:
            emg.startCommunication()
            
            while getattr(emg, 'OS_time', None) is None:
                time.sleep(0.001)
            
            if sync_event:
                sync_event.wait()
            
            first_emg_time = emg.OS_time
            last_time = emg.OS_time
            
            while not stop_event.is_set():
                time_sample = emg.OS_time
                
                if (time_sample - last_time) / 1e6 > 0.1:
                    raise ValueError('EMG alignment lost')
                elif time_sample > last_time:
                    emg_sample = np.asarray(emg.rawEMG)
                    raw_history.append(list(emg_sample))
                    raw_timestamps.append((time_sample - first_emg_time) / 1e6)
                
                last_time = time_sample
                time.sleep(0.001)
        
        except Exception as e:
            print(f"EMG error: {e}")
        finally:
            emg.exitEvent.set()
    
    emg_thread = threading.Thread(target=emg_loop, daemon=True)
    emg_thread.start()
    
    return stop_event, emg_thread, video_thread, raw_history, raw_timestamps, video_timestamps

def run_adaptive_grip_experiment(arm, grip_name, grip_config, force_trajectory_data, 
                                base_dir, enable_emg=False, enable_video=False):
    """
    Run complete adaptive grip experiment
    """
    print(f"\n{'='*60}")
    print(f"ADAPTIVE GRIP EXPERIMENT: {grip_name}")
    print(f"{'='*60}")
    
    target_timestamps, target_forces = force_trajectory_data
    
    # Initialize controller
    controller = AdaptiveGripController(arm, grip_config)
    
    # Start EMG/video recording if enabled
    sync_event = None
    emg_data = None
    if enable_emg or enable_video:
        sync_event = threading.Event()
        stop_event, emg_thread, video_thread, raw_history, raw_timestamps, video_timestamps = \
            start_raw_emg_recorder(base_dir, enable_video, sync_event)
    
    # Experiment data collection
    experiment_data = {
        'timestamps': [],
        'target_forces': [],
        'actual_forces': [],
        'grip_positions': [],
        'phases': [],
        'force_errors': [],
        'contact_position': None,
        'pid_outputs': []
    }
    
    # Start recording
    arm.resetRecording()
    arm.recording = True
    
    print("\n1. MOVING TO NEUTRAL POSITION...")
    # Move to neutral position - ensure float64 for arm.mainControlLoop
    neutral_pos = np.array(grip_config["neutral_position"], dtype=np.float64)
    arm.mainControlLoop(posDes=neutral_pos.reshape(1, -1), period=2, emg=None)
    # Ensure controller position stays float64
    controller.current_position = neutral_pos.copy().astype(np.float64)
    
    print("\n2. READY FOR OBJECT PLACEMENT")
    print(f"   Place object for {grip_config['description']}")
    input("   Press Enter when object is positioned and ready...")
    
    # Start synchronized recording
    if sync_event:
        sync_event.set()
        time.sleep(0.1)
    
    print("\n3. STARTING ADAPTIVE GRIP SEQUENCE...")
    print("   Phase 1: APPROACH - Moving towards target until contact")
    
    # Control loop
    start_time = time.time()
    dt = 1.0 / controller.control_rate
    force_start_time = None
    
    controller.phase = "APPROACH"
    
    try:
        while True:
            loop_start = time.time()
            current_time = time.time() - start_time
            
            # Get current force
            current_force = controller.get_grip_force()
            
            if controller.phase == "APPROACH":
                # Approach phase - move towards target until contact
                contact_detected = controller.approach_phase(dt)
                target_force = 0.0
                force_error = 0.0
                pid_output = 0.0
                
                if contact_detected:
                    controller.phase = "FORCE_CONTROL"
                    force_start_time = current_time
                    print(f"\n   Phase 2: FORCE CONTROL - Following target force trajectory")
                    print(f"   Contact detected at {current_time:.1f}s, force: {current_force:.2f}N")
            
            elif controller.phase == "FORCE_CONTROL":
                # Force control phase - follow target trajectory
                trajectory_time = current_time - force_start_time
                
                # Interpolate target force from trajectory
                if trajectory_time <= target_timestamps[-1]:
                    target_force = np.interp(trajectory_time, target_timestamps, target_forces)
                    current_force, force_error, pid_output = controller.force_control_phase(target_force, dt)
                else:
                    # Trajectory complete
                    print(f"\n   TRAJECTORY COMPLETE at {current_time:.1f}s")
                    break
            
            # Send position command to hand
            # Use the correct method for psyonicArm - convert to list first
            position_command = controller.current_position.tolist()
            arm.handCom = position_command
            
            # Record data
            experiment_data['timestamps'].append(current_time)
            experiment_data['target_forces'].append(target_force)
            experiment_data['actual_forces'].append(current_force)
            experiment_data['grip_positions'].append(controller.current_position.copy())
            experiment_data['phases'].append(controller.phase)
            experiment_data['force_errors'].append(force_error)
            experiment_data['pid_outputs'].append(pid_output)
            
            # Store contact position when detected
            if controller.contact_detected and experiment_data['contact_position'] is None:
                experiment_data['contact_position'] = controller.contact_position.copy()
            
            # Print progress
            if len(experiment_data['timestamps']) % 120 == 0:  # Every 2 seconds
                print(f"   Time: {current_time:5.1f}s | Phase: {controller.phase:12s} | "
                      f"Force: {current_force:5.1f}N | Target: {target_force:5.1f}N")
            
            # Control loop timing
            elapsed = time.time() - loop_start
            sleep_time = max(0, dt - elapsed)
            time.sleep(sleep_time)
    
    except KeyboardInterrupt:
        print("\n   EXPERIMENT INTERRUPTED")
    
    # Cleanup
    print("\n4. RETURNING TO NEUTRAL POSITION...")
    arm.mainControlLoop(posDes=neutral_pos.reshape(1, -1), period=3, emg=None)
    arm.recording = False
    
    # Stop EMG/video recording
    if enable_emg or enable_video:
        stop_event.set()
        emg_thread.join()
        if enable_video:
            video_thread.join()
    
    print("\n5. SAVING EXPERIMENT DATA...")
    
    # Save experiment data
    experiment_file = join(base_dir, f"adaptive_grip_{grip_name}.npy")
    np.save(experiment_file, experiment_data)
    
    # Save prosthetic data
    if hasattr(arm, 'recordedData') and arm.recordedData:
        prosthetic_data = np.array(arm.recordedData[1:], dtype=float)
        np.save(join(base_dir, f"prosthetic_data_{grip_name}.npy"), prosthetic_data)
        
        with open(join(base_dir, f"prosthetic_header_{grip_name}.txt"), "w") as f:
            f.write(",".join(arm.recordedData[0]))
    
    # Save EMG data
    if enable_emg and raw_history:
        np.save(join(base_dir, f"raw_emg_{grip_name}.npy"), np.vstack(raw_history))
        np.save(join(base_dir, f"raw_timestamps_{grip_name}.npy"), np.array(raw_timestamps))
    
    if enable_video and video_timestamps:
        np.save(join(base_dir, f"video_timestamps_{grip_name}.npy"), np.array(video_timestamps))
    
    print(f"   Experiment data saved to: {base_dir}")
    
    # Analyze results
    analyze_experiment_results(experiment_data, grip_name)
    
    return experiment_data

def analyze_experiment_results(experiment_data, grip_name):
    """Analyze and display experiment results"""
    timestamps = np.array(experiment_data['timestamps'])
    target_forces = np.array(experiment_data['target_forces'])
    actual_forces = np.array(experiment_data['actual_forces'])
    phases = experiment_data['phases']
    
    # Find phase transitions
    contact_idx = next((i for i, phase in enumerate(phases) if phase == "FORCE_CONTROL"), None)
    
    print(f"\n{'='*40}")
    print(f"EXPERIMENT ANALYSIS: {grip_name}")
    print(f"{'='*40}")
    
    if contact_idx is not None:
        contact_time = timestamps[contact_idx]
        print(f"Contact detected at: {contact_time:.2f}s")
        print(f"Contact position: {experiment_data['contact_position']}")
        
        # Analyze force control phase
        control_timestamps = timestamps[contact_idx:]
        control_target = target_forces[contact_idx:]
        control_actual = actual_forces[contact_idx:]
        
        if len(control_actual) > 0:
            force_errors = control_target - control_actual
            rmse = np.sqrt(np.mean(force_errors**2))
            mae = np.mean(np.abs(force_errors))
            max_error = np.max(np.abs(force_errors))
            
            print(f"\nForce Control Performance:")
            print(f"  RMSE: {rmse:.2f} N")
            print(f"  MAE:  {mae:.2f} N")
            print(f"  Max Error: {max_error:.2f} N")
            print(f"  Control Duration: {control_timestamps[-1] - control_timestamps[0]:.1f}s")
    else:
        print("No contact detected during experiment")
    
    print(f"Total Experiment Duration: {timestamps[-1]:.1f}s")

def main():
    parser = argparse.ArgumentParser(description="Adaptive Grip Force Control Training")
    parser.add_argument("--person_id", "-p", required=True, help="Person ID")
    parser.add_argument("--grip_type", "-g", required=True, 
                       choices=list(GRIP_CONFIGURATIONS.keys()),
                       help="Grip type for experiment")
    parser.add_argument("--out_root", "-o", default="data", help="Output directory")
    parser.add_argument("--hand_side", "-s", choices=["left", "right"], default="left")
    parser.add_argument("--force_file", "-f", help="CSV file with target force trajectory")
    parser.add_argument("--force_pattern", choices=["sine", "step", "ramp", "constant"], 
                       default="sine", help="Generated force pattern if no file provided")
    parser.add_argument("--duration", type=float, default=15.0, 
                       help="Force trajectory duration (seconds)")
    parser.add_argument("--max_force", type=float, default=8.0, help="Maximum force (N)")
    parser.add_argument("--emg", action="store_true", help="Enable EMG recording")
    parser.add_argument("--video", action="store_true", help="Enable video recording")
    parser.add_argument("--iterations", type=int, default=1, help="Number of experiment iterations")
    
    args = parser.parse_args()
    
    # Create output directory
    exp_parent = join(args.out_root, args.person_id, "adaptive_grip_experiments", args.grip_type)
    exp_idx = 1
    while os.path.exists(join(exp_parent, str(exp_idx))):
        exp_idx += 1
    base_dir = join(exp_parent, str(exp_idx))
    os.makedirs(base_dir, exist_ok=True)
    
    # Get grip configuration
    grip_config = GRIP_CONFIGURATIONS[args.grip_type]
    
    # Load or generate force trajectory
    if args.force_file:
        target_timestamps, target_forces = load_force_trajectory(args.force_file)
        if target_timestamps is None:
            return
    else:
        target_timestamps, target_forces = generate_test_force_trajectory(
            duration=args.duration, pattern=args.force_pattern
        )
        # Scale forces to max_force
        target_forces = target_forces * (args.max_force / np.max(target_forces))
    
    print(f"Adaptive Grip Force Control Training")
    print(f"Grip Type: {args.grip_type} ({grip_config['description']})")
    print(f"Output Directory: {base_dir}")
    print(f"Force Trajectory: {len(target_timestamps)} points, {target_timestamps[-1]:.1f}s duration")
    
    # Initialize prosthetic arm
    arm = psyonicArm(hand=args.hand_side)
    arm.initSensors()
    arm.startComms()
    
    try:
        # Run experiments
        for iteration in range(args.iterations):
            print(f"\n{'='*60}")
            print(f"ITERATION {iteration + 1} / {args.iterations}")
            print(f"{'='*60}")
            
            # Create iteration subdirectory
            iter_dir = join(base_dir, f"iteration_{iteration + 1}")
            os.makedirs(iter_dir, exist_ok=True)
            
            # Run experiment
            experiment_data = run_adaptive_grip_experiment(
                arm, args.grip_type, grip_config, 
                (target_timestamps, target_forces),
                iter_dir, args.emg, args.video
            )
            
            # Rest between iterations
            if iteration < args.iterations - 1:
                print(f"\nResting 10 seconds before next iteration...")
                time.sleep(10.0)
        
        # Save experiment summary
        summary = {
            'grip_type': args.grip_type,
            'grip_config': grip_config,
            'force_pattern': args.force_pattern,
            'duration': args.duration,
            'max_force': args.max_force,
            'iterations': args.iterations,
            'person_id': args.person_id
        }
        
        with open(join(base_dir, "experiment_summary.yaml"), 'w') as f:
            yaml.safe_dump(summary, f)
        
        print(f"\n{'='*60}")
        print("ALL EXPERIMENTS COMPLETE!")
        print(f"{'='*60}")
        print(f"Results saved to: {base_dir}")
        
    finally:
        arm.close()
        print("Prosthetic arm disconnected.")

if __name__ == "__main__":
    main()