# Modified s2.5 script with hardware drift correction
# Generated automatically from hardware synchronization test
import numpy as np


def hardware_corrected_end_align_data(emg_data, emg_timestamps, angles_data, angles_timestamps):
    """
    Hardware-calibrated alignment that corrects for EMG clock drift.
    Based on hardware synchronization test results.
    """
    print("Applying hardware-calibrated drift correction...")
    
    # Hardware-measured drift correction factor
    DRIFT_CORRECTION_FACTOR = 1.13205094
    
    # Correct EMG timestamps for hardware clock drift
    corrected_emg_timestamps = emg_timestamps * DRIFT_CORRECTION_FACTOR
    
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


# Replace the call to end_align_data() with:
# emg, emg_timestamps, angles, angles_timestamps = hardware_corrected_end_align_data(
#     emg, emg_timestamps, angles, angles_timestamps
# )
