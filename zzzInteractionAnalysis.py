#!/usr/bin/env python3
"""
Force-only processing script for interaction experiments.
Processes raw force data without EMG alignment - just force filtering and normalization.
"""

import os
import numpy as np
import yaml
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

SKIP_MOVEMENTS = {"calibration", "Calibration", "calib", "Calib"}

def is_interaction_experiment(movement_name):
    """Check if this is a force interaction experiment."""
    return "_interaction" in movement_name.lower()

def extract_finger_forces_by_indices(angles_data):
    """
    Extract and sum force sensor data for each finger using known column indices.
    Based on the header: force sensors start at column 48 and follow pattern:
    index0_Force through index5_Force (cols 48-53)
    middle0_Force through middle5_Force (cols 54-59) 
    ring0_Force through ring5_Force (cols 60-65)
    pinky0_Force through pinky5_Force (cols 66-71)
    thumb0_Force through thumb5_Force (cols 72-77)
    
    Args:
        angles_data: numpy array of shape (n_samples, n_features)
    
    Returns:
        force_data: numpy array of shape (n_samples, 5) for [index, middle, ring, pinky, thumb]
        finger_names: list of finger names
    """
    finger_names = ['index', 'middle', 'ring', 'pinky', 'thumb']
    force_data = np.zeros((angles_data.shape[0], len(finger_names)))
    
    # Define the starting column index for each finger's force sensors
    force_start_indices = {
        'index': 48,   # index0_Force through index5_Force
        'middle': 54,  # middle0_Force through middle5_Force  
        'ring': 60,    # ring0_Force through ring5_Force
        'pinky': 66,   # pinky0_Force through pinky5_Force
        'thumb': 72    # thumb0_Force through thumb5_Force
    }
    
    print(f"Extracting force data using known column indices")
    print(f"Angles data shape: {angles_data.shape}")
    
    for finger_idx, finger in enumerate(finger_names):
        start_col = force_start_indices[finger]
        end_col = start_col + 6  # 6 sensors per finger
        
        if end_col <= angles_data.shape[1]:
            # Sum the 6 force sensors for this finger
            finger_force_sum = np.sum(angles_data[:, start_col:end_col], axis=1)
            force_data[:, finger_idx] = finger_force_sum
            
            non_zero_count = np.sum(finger_force_sum > 0.001)
            print(f"  {finger} (cols {start_col}-{end_col-1}): mean={np.mean(finger_force_sum):.3f}, max={np.max(finger_force_sum):.3f}, non-zero={non_zero_count}/{len(finger_force_sum)}")
        else:
            print(f"  {finger}: columns {start_col}-{end_col-1} exceed data width ({angles_data.shape[1]})")
    
    return force_data, finger_names

def extract_finger_forces(angles_data, headers):
    """
    Extract and sum force sensor data for each finger from the angles array.
    Try header-based extraction first, fall back to index-based if needed.
    """
    # First try index-based extraction (more reliable)
    if angles_data.shape[1] >= 78:  # Need at least 78 columns for force data
        print("Using index-based force extraction (more reliable)")
        return extract_finger_forces_by_indices(angles_data)
    
    # Fall back to header-based extraction
    print("Using header-based force extraction")
    finger_names = ['index', 'middle', 'ring', 'pinky', 'thumb']
    force_data = np.zeros((angles_data.shape[0], len(finger_names)))
    
    print(f"Extracting force data from angles array with shape {angles_data.shape}")
    print(f"Headers available: {len(headers)} columns")
    
    for finger_idx, finger in enumerate(finger_names):
        finger_force_sum = np.zeros(angles_data.shape[0])
        sensors_found = 0
        
        # Sum all 6 force sensors for this finger
        for sensor_idx in range(6):
            force_sensor_name = f"{finger}{sensor_idx}_Force"
            try:
                col_idx = headers.index(force_sensor_name)
                finger_force_sum += angles_data[:, col_idx]
                sensors_found += 1
            except ValueError:
                print(f"Warning: Force sensor {force_sensor_name} not found in headers")
        
        if sensors_found > 0:
            force_data[:, finger_idx] = finger_force_sum
            print(f"  {finger}: found {sensors_found}/6 sensors, sum range [{np.min(finger_force_sum):.3f}, {np.max(finger_force_sum):.3f}]")
        else:
            print(f"  {finger}: no force sensors found!")
    
    return force_data, finger_names

def process_force_realtime_style(force_data, sampling_freq, static_offsets=None):
    """
    Process force data to match the real-time filtering pipeline in s5_inference.py
    BUT using zero-phase filtering for superior offline processing.
    
    This matches the real-time filter characteristics (Butterworth, 3Hz, 2nd order)
    while providing zero-lag filtering for training data.
    
    Args:
        force_data: numpy array of shape (n_samples, 5) for per-finger forces
        sampling_freq: sampling frequency in Hz
        static_offsets: numpy array of shape (5,) with static baseline offsets
                       If None, uses mean of first 10% of data as baseline
    
    Returns:
        filtered_force: processed force data with same characteristics as real-time
                       but with zero-phase filtering for better training data
    """
    from scipy import signal
    
    # 1. Static baseline correction (like zeroJoints calibration in real-time)
    if static_offsets is None:
        # Estimate static baseline from first 10% of data (assuming start is baseline)
        baseline_samples = int(0.1 * len(force_data))
        static_offsets = np.mean(force_data[:baseline_samples, :], axis=0)
        print(f"Estimated static offsets: {static_offsets}")
    
    static_corrected = np.maximum(force_data - static_offsets[None, :], 0)
    
    # 2. Butterworth filter - SAME CHARACTERISTICS as real-time but zero-phase
    # Real-time uses: 3Hz cutoff, 2nd order Butterworth
    # We use: 3Hz cutoff, 2nd order Butterworth + filtfilt (zero-phase)
    nyquist = 0.5 * sampling_freq
    normal_cutoff = min(3.0 / nyquist, 0.99)
    b, a = signal.butter(2, normal_cutoff, btype='low')
    
    # Apply ZERO-PHASE filtering to each finger (this is the key advantage for training)
    hf_filtered = np.zeros_like(static_corrected)
    for finger_idx in range(static_corrected.shape[1]):
        hf_filtered[:, finger_idx] = signal.filtfilt(b, a, static_corrected[:, finger_idx])
    
    # 3. Since all your data is "during contact", skip adaptive baseline correction
    # Just ensure non-negative values (matching real-time clipping)
    final_filtered = np.maximum(hf_filtered, 0)
    
    print(f"Applied zero-phase Butterworth filter (3Hz, 2nd order) - matches real-time characteristics")
    return final_filtered

def normalize_force_data(force_data, finger_names, max_forces_per_finger=None):
    """
    Normalize force data to [0, 1] range based on maximum achievable force per finger
    
    Args:
        force_data: numpy array of shape (n_samples, 5) 
        finger_names: list of finger names ['index', 'middle', 'ring', 'pinky', 'thumb']
        max_forces_per_finger: dict of max forces per finger (in Newtons)
                              If None, uses reasonable defaults
    
    Returns:
        normalized_force: force data normalized to [0, 1]
    """
    if max_forces_per_finger is None:
        # Default maximum forces based on prosthetic hand capabilities
        max_forces_per_finger = {
            'index': 12.0,   
            'middle': 12.0,    
            'ring': 12.0,    
            'pinky': 12.0,   
            'thumb': 12.0    
        }
    
    normalized_force = np.copy(force_data)
    
    for finger_idx, finger_name in enumerate(finger_names):
        max_force = max_forces_per_finger[finger_name]
        normalized_force[:, finger_idx] = np.clip(force_data[:, finger_idx] / max_force, 0, 1)
        
        print(f"  {finger_name}: max_force={max_force}N, "
              f"raw_range=[{np.min(force_data[:, finger_idx]):.2f}, {np.max(force_data[:, finger_idx]):.2f}], "
              f"normalized_range=[{np.min(normalized_force[:, finger_idx]):.3f}, {np.max(normalized_force[:, finger_idx]):.3f}]")
    
    return normalized_force

def plot_processed_force_data(data_dir, force_data, finger_names, timestamps, force_units="N", save_plot=True):
    """Plot the processed force data"""
    import matplotlib.pyplot as plt
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Define colors for each finger
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    # Plot 1: All fingers together
    ax1 = axes[0]
    for i, finger in enumerate(finger_names):
        ax1.plot(timestamps, force_data[:, i], label=finger, color=colors[i], linewidth=1.5)
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel(f'Force ({force_units})')
    ax1.set_title('Processed Force Data - All Fingers')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add force statistics to the plot
    total_force = np.sum(force_data, axis=1)
    max_total = np.max(total_force)
    mean_total = np.mean(total_force)
    if force_units == "N":
        ax1.text(0.02, 0.98, f'Max Total Force: {max_total:.3f}N\nMean Total Force: {mean_total:.3f}N', 
                 transform=ax1.transAxes, verticalalignment='top', 
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax1.text(0.02, 0.98, f'Max Total Force: {max_total:.3f}\nMean Total Force: {mean_total:.3f}', 
                 transform=ax1.transAxes, verticalalignment='top', 
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 2: Stacked view
    ax2 = axes[1]
    
    # Stack the forces to show contribution
    bottom = np.zeros(len(timestamps))
    for i, finger in enumerate(finger_names):
        ax2.fill_between(timestamps, bottom, bottom + force_data[:, i], 
                        label=finger, color=colors[i], alpha=0.7)
        bottom += force_data[:, i]
    
    ax2.plot(timestamps, total_force, 'k-', linewidth=2, label='Total Force')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel(f'Force ({force_units})')
    ax2.set_title('Processed Force Data - Stacked View')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add experiment info
    exp_name = os.path.basename(os.path.dirname(os.path.dirname(data_dir)))
    exp_num = os.path.basename(data_dir)
    fig.suptitle(f'Processed Force Analysis: {exp_name} - Experiment {exp_num}\nDirectory: {data_dir}', fontsize=14)
    
    plt.tight_layout()
    
    # Save plot if requested
    if save_plot:
        plot_file = os.path.join(data_dir, "processed_force_analysis.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {plot_file}")
    
    plt.show()
    
    return fig

def process_force_data_only(data_dir, hand_side='left', normalize=False):
    """
    Process only force data from an interaction experiment.
    No EMG processing or alignment - just force filtering.
    
    Args:
        data_dir: Directory containing angles.npy, angle_timestamps.npy, angles_header.txt
        hand_side: Hand side for column naming
        normalize: Whether to normalize forces to [0,1] range
    """
    
    print(f"\n=== Processing Force Data Only ===")
    print(f"Data directory: {data_dir}")
    print(f"Normalization: {'Enabled' if normalize else 'Disabled'}")
    
    # Load angle data (which contains force sensors)
    angles_file = os.path.join(data_dir, 'angles.npy')
    timestamps_file = os.path.join(data_dir, 'angle_timestamps.npy') 
    headers_file = os.path.join(data_dir, 'angles_header.txt')
    
    if not os.path.exists(angles_file):
        raise FileNotFoundError(f"angles.npy not found in {data_dir}")
    
    # Load data
    angles = np.load(angles_file)
    timestamps = np.load(timestamps_file) if os.path.exists(timestamps_file) else None
    
    print(f"Loaded angles data: {angles.shape}")
    if timestamps is not None:
        print(f"Loaded timestamps: {len(timestamps)} samples, {timestamps[-1]:.2f}s duration")
    
    # Load headers if available
    headers = None
    if os.path.exists(headers_file):
        with open(headers_file, 'r') as f:
            header_content = f.read().strip()
            headers = [h.strip() for h in header_content.split(',')]
        print(f"Loaded {len(headers)} headers")
    else:
        print("Warning: No angles_header.txt found - using index-based extraction")
    
    # Load experiment config to trim pre-contact data if available
    config_path = os.path.join(data_dir, 'experiment_config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        duration = config.get('duration', 0)
        total_duration = config.get('total_duration', 0)
        
        if duration > 0 and total_duration > 0:
            contact_offset = total_duration - duration
            print(f"Trimming pre-contact data: Removing first {contact_offset:.3f}s")
            
            if timestamps is not None:
                # Trim data to only include data after contact detection
                mask = timestamps >= (timestamps[0] + contact_offset)
                angles = angles[mask, :]
                timestamps = timestamps[mask]
                print(f"After trimming: {angles.shape[0]} samples, {timestamps[-1]:.2f}s duration")
        else:
            print("No contact offset found in config - processing all data")
    else:
        print("Warning: experiment_config.yaml not found - processing all data")
    
    # Extract force data
    print(f"\n=== Extracting Force Data ===")
    force_data, finger_names = extract_finger_forces(angles, headers)
    
    if force_data is None or len(finger_names) == 0:
        raise ValueError("Failed to extract force data")
    
    # Calculate sampling frequency
    if timestamps is not None:
        sampling_freq = (len(timestamps) - 1) / (timestamps[-1] - timestamps[0])
        print(f"Force sampling frequency: {sampling_freq:.1f} Hz")
    else:
        sampling_freq = 60.0  # Default assumption
        timestamps = np.arange(len(force_data)) / sampling_freq
        print(f"No timestamps found - assuming {sampling_freq} Hz, created synthetic timestamps")
    
    # Apply real-time style filtering
    print(f"\n=== Filtering Force Data (Real-time Style) ===")
    filtered_force = process_force_realtime_style(force_data, sampling_freq)
    
    # Optionally normalize force data  
    if normalize:
        print(f"\n=== Normalizing Force Data ===")
        final_force = normalize_force_data(filtered_force, finger_names)
        force_units = "Normalized"
    else:
        print(f"Skipping normalization - keeping forces in Newtons")
        final_force = filtered_force
        force_units = "N"
    
    # Downsample to 60Hz if needed
    target_freq = 60.0
    if sampling_freq > target_freq * 1.1:  # Only downsample if significantly higher
        downsample_step = int(round(sampling_freq / target_freq))
        print(f"\n=== Downsampling from {sampling_freq:.1f}Hz to ~{target_freq}Hz ===")
        print(f"Downsample step: {downsample_step}")
        
        final_force_60hz = final_force[::downsample_step, :]
        timestamps_60hz = timestamps[::downsample_step]
        
        # Reset timestamps to start at 0
        timestamps_60hz = timestamps_60hz - timestamps_60hz[0]
        
        final_freq = (len(timestamps_60hz) - 1) / (timestamps_60hz[-1] - timestamps_60hz[0]) if len(timestamps_60hz) > 1 else target_freq
        print(f"Final sampling frequency: {final_freq:.1f} Hz")
        print(f"Final data shape: {final_force_60hz.shape}")
        
        final_force = final_force_60hz
        final_timestamps = timestamps_60hz
    else:
        print(f"No downsampling needed (current: {sampling_freq:.1f}Hz)")
        final_timestamps = timestamps - timestamps[0]  # Reset to start at 0
    
    # Save processed data
    print(f"\n=== Saving Processed Force Data ===")
    
    # Save as numpy arrays
    np.save(os.path.join(data_dir, 'processed_force_data.npy'), final_force)
    np.save(os.path.join(data_dir, 'processed_force_timestamps.npy'), final_timestamps)
    
    print(f"Processing complete!")
    print(f"Final force data shape: {final_force.shape}")
    print(f"Final duration: {final_timestamps[-1]:.2f}s")
    
    # Print summary statistics
    print(f"\n=== Final Force Data Summary ===")
    for i, finger in enumerate(finger_names):
        finger_data = final_force[:, i]
        if normalize:
            print(f"  {finger}: mean={np.mean(finger_data):.3f}, max={np.max(finger_data):.3f}, "
                  f"std={np.std(finger_data):.3f}, non-zero={np.sum(finger_data > 0.001)}/{len(finger_data)}")
        else:
            print(f"  {finger}: mean={np.mean(finger_data):.3f}N, max={np.max(finger_data):.3f}N, "
                  f"std={np.std(finger_data):.3f}N, non-zero={np.sum(finger_data > 0.001)}/{len(finger_data)}")
    
    # Plot the processed force data
    print(f"\n=== Creating Force Plots ===")
    plot_processed_force_data(data_dir, final_force, finger_names, final_timestamps, force_units)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process force data from a specific interaction experiment')
    parser.add_argument('--person_id', required=True, help='Person ID (e.g. Emanuel_FirstTries)')
    parser.add_argument('--movement', required=True, help='Interaction movement name (e.g. pinch_interaction)')
    parser.add_argument('--experiment', required=True, help='Experiment number (e.g. 1, 2, 3)')
    parser.add_argument('--out_root', default='data', help='Root directory (default: data)')
    parser.add_argument("--hand_side", "-s", choices=["left", "right"], default="left", 
                       help="Side of the prosthetic hand")
    parser.add_argument('--normalize', action='store_true', help='Normalize forces to [0,1] range')
    
    args = parser.parse_args()
    
    # Build the specific experiment directory path
    exp_dir = os.path.join(args.out_root, args.person_id, "recordings", 
                          args.movement, "experiments", args.experiment)
    
    if not os.path.exists(exp_dir):
        print(f"Error: Experiment directory not found: {exp_dir}")
        print(f"Available experiments in {os.path.dirname(exp_dir)}:")
        parent_dir = os.path.dirname(exp_dir)
        if os.path.exists(parent_dir):
            experiments = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
            experiments.sort()
            for exp in experiments:
                print(f"  - {exp}")
        exit(1)
    
    # Check if this is actually an interaction experiment
    if not is_interaction_experiment(args.movement):
        print(f"Warning: '{args.movement}' doesn't appear to be an interaction experiment")
        print("Continuing anyway...")
    
    print(f"Processing experiment: {args.person_id}/{args.movement}/experiments/{args.experiment}")
    print(f"Full path: {exp_dir}")
    
    try:
        process_force_data_only(exp_dir, hand_side=args.hand_side, normalize=args.normalize)
    except Exception as e:
        print(f"Failed to process experiment: {e}")
        import traceback
        traceback.print_exc()