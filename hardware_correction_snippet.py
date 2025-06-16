
# Hardware-calibrated EMG timestamp correction
# Based on hardware sync test results showing 34.049 ms/s drift
# Add this to your s2.5_preprocess_data.py script:

def hardware_corrected_end_align_data(emg_data, emg_timestamps, angles_data, angles_timestamps):
    """
    Hardware-calibrated alignment that corrects for your specific EMG clock drift.
    Measured drift: 34.049 ms/s (EMG internal: 985.8 Hz vs system: 1115.9 Hz)
    """
    import numpy as np
    
    print("Applying hardware-calibrated drift correction...")
    
    # Your specific hardware correction factor
    DRIFT_CORRECTION_FACTOR = 1.13205094
    
    # Correct EMG timestamps for hardware clock drift
    corrected_emg_timestamps = emg_timestamps * DRIFT_CORRECTION_FACTOR
    
    print(f"Applied drift correction factor: {DRIFT_CORRECTION_FACTOR}")
    print(f"This eliminates ~34ms/s drift ({34*len(emg_timestamps)/1000:.1f}ms total)")
    
    # Standard end-alignment with corrected timestamps
    shift = corrected_emg_timestamps[-1] - angles_timestamps[-1]
    emg_timestamps_shifted = corrected_emg_timestamps - shift
    
    # Find overlapping region
    mask = (emg_timestamps_shifted >= 0) & (emg_timestamps_shifted <= angles_timestamps[-1])
    
    # Apply mask
    emg_aligned = emg_data[:, mask] if emg_data.ndim == 2 else emg_data[mask]
    emg_timestamps_aligned = emg_timestamps_shifted[mask]
    
    print(f"Hardware drift correction: kept {np.sum(mask)}/{len(mask)} EMG samples")
    
    return emg_aligned, emg_timestamps_aligned, angles_data, angles_timestamps

# USAGE: In your process_emg_and_angles() function, replace:
#   emg, emg_timestamps, angles, angles_timestamps = end_align_data(...)
# with:
#   emg, emg_timestamps, angles, angles_timestamps = hardware_corrected_end_align_data(...)
