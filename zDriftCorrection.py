import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
import os
import argparse

def detect_peaks_in_signal(signal_data, timestamps, min_prominence=None, min_distance_sec=0.5):
    """
    Detect peaks in a signal with minimum prominence and distance constraints.
    
    Args:
        signal_data: 1D signal array
        timestamps: corresponding timestamps
        min_prominence: minimum peak prominence (auto if None)
        min_distance_sec: minimum distance between peaks in seconds
    
    Returns:
        peak_indices: indices of peaks in signal
        peak_times: timestamps of peaks
        peak_values: values at peaks
    """
    # Calculate sampling frequency
    fs = len(timestamps) / (timestamps[-1] - timestamps[0])
    min_distance_samples = int(min_distance_sec * fs)
    
    # Auto-detect prominence if not provided
    if min_prominence is None:
        min_prominence = np.std(signal_data) * 0.5
    
    # Find peaks
    peaks, properties = signal.find_peaks(
        signal_data,
        prominence=min_prominence,
        distance=min_distance_samples
    )
    
    return peaks, timestamps[peaks], signal_data[peaks]

def detect_angle_cycles(angle_data, timestamps, channel_idx=26):
    """
    Detect angle cycles (peaks) in the angle signal.
    
    Args:
        angle_data: angle data (time x channels)
        timestamps: angle timestamps
        channel_idx: angle channel to analyze
    
    Returns:
        cycle_indices: indices of angle peaks
        cycle_times: timestamps of angle peaks
        cycle_values: angle values at peaks
    """
    angle_signal = angle_data[:, channel_idx]
    
    print(f"Detecting angle cycles in channel {channel_idx}...")
    print(f"  Angle range: {np.min(angle_signal):.1f} to {np.max(angle_signal):.1f}")
    
    # Find peaks in angle signal
    peak_indices, peak_times, peak_values = detect_peaks_in_signal(
        angle_signal, timestamps, 
        min_prominence=np.std(angle_signal) * 0.3,
        min_distance_sec=0.8  # Expect at least 0.8s between cycles
    )
    
    print(f"  Found {len(peak_indices)} angle cycles")
    print(f"  Average cycle period: {np.mean(np.diff(peak_times)):.2f}s")
    
    return peak_indices, peak_times, peak_values

def detect_emg_peaks(emg_data, timestamps, channel_idx=0):
    """
    Detect EMG activation peaks using envelope.
    
    Args:
        emg_data: EMG data (time x channels)
        timestamps: EMG timestamps
        channel_idx: EMG channel to analyze
    
    Returns:
        peak_indices: indices of EMG peaks
        peak_times: timestamps of EMG peaks
        emg_envelope: processed EMG envelope
    """
    from scipy import signal as sp_signal
    
    emg_signal = emg_data[:, channel_idx]
    fs = len(timestamps) / (timestamps[-1] - timestamps[0])
    
    print(f"Detecting EMG peaks in channel {channel_idx}...")
    print(f"  Sampling frequency: {fs:.1f} Hz")
    
    # Process EMG to get envelope
    nyquist = fs / 2
    
    # Bandpass filter (20-500 Hz)
    low_cutoff = 20 / nyquist
    high_cutoff = min(500, nyquist * 0.9) / nyquist
    
    try:
        bandpass_sos = sp_signal.butter(N=4, Wn=[low_cutoff, high_cutoff], btype='bandpass', output='sos')
        filtered_signal = sp_signal.sosfiltfilt(bandpass_sos, emg_signal)
    except:
        print("  Warning: Bandpass filter failed, using high-pass only")
        highpass_sos = sp_signal.butter(N=4, Wn=low_cutoff, btype='highpass', output='sos')
        filtered_signal = sp_signal.sosfiltfilt(highpass_sos, emg_signal)
    
    # Rectification
    rectified_signal = np.abs(filtered_signal)
    
    # Envelope extraction (3 Hz low-pass)
    envelope_cutoff = 3 / nyquist
    envelope_sos = sp_signal.butter(N=4, Wn=envelope_cutoff, btype='lowpass', output='sos')
    emg_envelope = sp_signal.sosfiltfilt(envelope_sos, rectified_signal)
    
    print(f"  EMG envelope range: {np.min(emg_envelope):.4f} to {np.max(emg_envelope):.4f}")
    
    # Find peaks in envelope
    peak_indices, peak_times, peak_values = detect_peaks_in_signal(
        emg_envelope, timestamps,
        min_prominence=np.std(emg_envelope) * 0.5,
        min_distance_sec=0.6  # Minimum distance between EMG peaks
    )
    
    print(f"  Found {len(peak_indices)} EMG peaks")
    
    return peak_indices, peak_times, emg_envelope

def calculate_phase_drift(emg_peak_times, angle_peak_times):
    """
    Calculate how EMG peaks drift relative to angle cycles over time.
    
    Args:
        emg_peak_times: timestamps of EMG peaks
        angle_peak_times: timestamps of angle cycle peaks
    
    Returns:
        drift_times: time points where drift was measured
        phase_shifts: phase shift at each time point (0-1, where 0.5 = middle of cycle)
        angle_periods: period of each angle cycle
    """
    
    print(f"Calculating phase drift...")
    print(f"  EMG peaks: {len(emg_peak_times)}")
    print(f"  Angle cycles: {len(angle_peak_times)}")
    
    drift_times = []
    phase_shifts = []
    angle_periods = []
    
    # For each EMG peak, find which angle cycle it belongs to and calculate phase
    for emg_time in emg_peak_times:
        # Find the angle cycle that contains this EMG peak
        cycle_before = angle_peak_times[angle_peak_times <= emg_time]
        cycle_after = angle_peak_times[angle_peak_times > emg_time]
        
        if len(cycle_before) > 0 and len(cycle_after) > 0:
            cycle_start = cycle_before[-1]  # Last cycle before EMG peak
            cycle_end = cycle_after[0]      # First cycle after EMG peak
            
            # Calculate phase within the cycle (0 = at peak, 0.5 = middle of cycle)
            cycle_duration = cycle_end - cycle_start
            time_since_cycle_start = emg_time - cycle_start
            phase = time_since_cycle_start / cycle_duration
            
            drift_times.append(emg_time)
            phase_shifts.append(phase)
            angle_periods.append(cycle_duration)
    
    drift_times = np.array(drift_times)
    phase_shifts = np.array(phase_shifts)
    angle_periods = np.array(angle_periods)
    
    print(f"  Calculated drift for {len(drift_times)} EMG peaks")
    print(f"  Phase shift range: {np.min(phase_shifts):.3f} to {np.max(phase_shifts):.3f}")
    print(f"  Average angle period: {np.mean(angle_periods):.2f}s")
    
    return drift_times, phase_shifts, angle_periods

def create_drift_correction_function(drift_times, phase_shifts, target_phase=0.2):
    """
    Create a function to correct the drift by warping EMG timestamps.
    
    Args:
        drift_times: time points where drift was measured
        phase_shifts: measured phase shifts
        target_phase: desired phase for EMG peaks (0.2 = 20% into cycle)
    
    Returns:
        correction_function: function to apply to EMG timestamps
        phase_corrections: corrections applied at each drift time
    """
    
    # Calculate required phase corrections
    phase_corrections = target_phase - phase_shifts
    
    print(f"Creating drift correction function...")
    print(f"  Target phase: {target_phase:.2f}")
    print(f"  Phase corrections range: {np.min(phase_corrections):.3f} to {np.max(phase_corrections):.3f}")
    
    # Smooth the corrections to avoid discontinuities
    from scipy import ndimage
    smoothed_corrections = ndimage.gaussian_filter1d(phase_corrections, sigma=2.0)
    
    # Create interpolation function
    if len(drift_times) >= 2:
        correction_interp = interp1d(
            drift_times, smoothed_corrections, 
            kind='linear', bounds_error=False, 
            fill_value=(smoothed_corrections[0], smoothed_corrections[-1])
        )
    else:
        # Fallback for insufficient data
        avg_correction = np.mean(smoothed_corrections) if len(smoothed_corrections) > 0 else 0
        correction_interp = lambda t: np.full_like(t, avg_correction)
    
    def correction_function(emg_timestamps, angle_periods_interp_func):
        """
        Apply drift correction to EMG timestamps.
        
        Args:
            emg_timestamps: original EMG timestamps
            angle_periods_interp_func: function to get angle period at any time
        
        Returns:
            corrected_timestamps: drift-corrected EMG timestamps
        """
        # Get phase corrections for all EMG timestamps
        phase_corrections_all = correction_interp(emg_timestamps)
        
        # Convert phase corrections to time corrections using local angle periods
        angle_periods_all = angle_periods_interp_func(emg_timestamps)
        time_corrections = phase_corrections_all * angle_periods_all
        
        # Apply corrections
        corrected_timestamps = emg_timestamps + time_corrections
        
        return corrected_timestamps
    
    return correction_function, smoothed_corrections

def apply_peak_based_drift_correction(emg_data, emg_timestamps, angles_data, angle_timestamps,
                                    emg_channel=0, angle_channel=26, target_phase=0.2):
    """
    Apply peak-based drift correction to align EMG peaks with angle cycles.
    
    Args:
        emg_data: EMG data (time x channels)
        emg_timestamps: EMG timestamps
        angles_data: angle data (time x channels)
        angle_timestamps: angle timestamps
        emg_channel: EMG channel for analysis
        angle_channel: angle channel for analysis
        target_phase: target phase for EMG peaks in angle cycles
    
    Returns:
        corrected_emg_timestamps: drift-corrected EMG timestamps
        correction_info: dictionary with correction details
    """
    
    print(f"\n=== PEAK-BASED DRIFT CORRECTION ===")
    
    # Step 1: Detect angle cycles
    angle_peak_indices, angle_peak_times, angle_peak_values = detect_angle_cycles(
        angles_data, angle_timestamps, angle_channel
    )
    
    # Step 2: Detect EMG peaks
    emg_peak_indices, emg_peak_times, emg_envelope = detect_emg_peaks(
        emg_data, emg_timestamps, emg_channel
    )
    
    # Step 3: Calculate phase drift
    if len(emg_peak_times) < 3 or len(angle_peak_times) < 3:
        print("Warning: Not enough peaks detected for drift correction")
        return emg_timestamps, {
            'method': 'insufficient_peaks',
            'emg_peaks': len(emg_peak_times),
            'angle_peaks': len(angle_peak_times)
        }
    
    drift_times, phase_shifts, angle_periods = calculate_phase_drift(
        emg_peak_times, angle_peak_times
    )
    
    if len(drift_times) < 3:
        print("Warning: Not enough drift measurements for correction")
        return emg_timestamps, {
            'method': 'insufficient_drift_data',
            'drift_measurements': len(drift_times)
        }
    
    # Step 4: Create drift correction function
    correction_function, phase_corrections = create_drift_correction_function(
        drift_times, phase_shifts, target_phase
    )
    
    # Step 5: Create angle period interpolation function
    angle_period_times = angle_peak_times[:-1]  # All but last peak
    angle_period_values = np.diff(angle_peak_times)  # Periods between peaks
    
    angle_period_interp = interp1d(
        angle_period_times, angle_period_values,
        kind='linear', bounds_error=False,
        fill_value=(angle_period_values[0], angle_period_values[-1])
    )
    
    # Step 6: Apply correction
    corrected_emg_timestamps = correction_function(emg_timestamps, angle_period_interp)
    
    print(f"\nDrift correction complete!")
    print(f"  Original EMG duration: {emg_timestamps[-1] - emg_timestamps[0]:.3f}s")
    print(f"  Corrected EMG duration: {corrected_emg_timestamps[-1] - corrected_emg_timestamps[0]:.3f}s")
    print(f"  Max time correction: {np.max(np.abs(corrected_emg_timestamps - emg_timestamps)):.3f}s")
    
    correction_info = {
        'method': 'peak_based_drift_correction',
        'emg_channel': emg_channel,
        'angle_channel': angle_channel,
        'target_phase': target_phase,
        'emg_peak_times': emg_peak_times,
        'angle_peak_times': angle_peak_times,
        'drift_times': drift_times,
        'phase_shifts': phase_shifts,
        'phase_corrections': phase_corrections,
        'emg_envelope': emg_envelope,
        'angle_periods': angle_periods
    }
    
    return corrected_emg_timestamps, correction_info

def plot_drift_analysis(emg_data, emg_timestamps, emg_timestamps_corrected,
                       angles_data, angle_timestamps, correction_info, save_path=None):
    """
    Create comprehensive plots showing the drift analysis and correction.
    """
    
    fig, axes = plt.subplots(4, 2, figsize=(20, 16))
    
    emg_channel = correction_info['emg_channel']
    angle_channel = correction_info['angle_channel']
    
    # Get data for plotting
    angle_signal = angles_data[:, angle_channel]
    emg_envelope = correction_info.get('emg_envelope', emg_data[:, emg_channel])
    
    # Plot 1: Phase drift over time
    if 'drift_times' in correction_info:
        axes[0, 0].plot(correction_info['drift_times'], correction_info['phase_shifts'], 'bo-', alpha=0.7)
        axes[0, 0].axhline(y=correction_info['target_phase'], color='r', linestyle='--', label=f"Target phase ({correction_info['target_phase']:.2f})")
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Phase in Angle Cycle')
        axes[0, 0].set_title('EMG Peak Phase Drift Over Time')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
    
    # Plot 2: Phase corrections applied
    if 'phase_corrections' in correction_info:
        axes[0, 1].plot(correction_info['drift_times'], correction_info['phase_corrections'], 'ro-', alpha=0.7)
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Phase Correction Applied')
        axes[0, 1].set_title('Phase Corrections Applied')
        axes[0, 1].grid(True)
    
    # Plot 3: Original alignment (first 20 seconds)
    time_mask = emg_timestamps <= 20
    angle_mask = angle_timestamps <= 20
    
    ax3_emg = axes[1, 0]
    ax3_angle = ax3_emg.twinx()
    
    step = max(1, np.sum(time_mask) // 2000)
    ax3_emg.plot(emg_timestamps[time_mask][::step], emg_envelope[time_mask][::step], 'b-', alpha=0.7, label=f'EMG envelope Ch{emg_channel}')
    ax3_angle.plot(angle_timestamps[angle_mask][::step], angle_signal[angle_mask][::step], 'k-', alpha=0.8, label=f'Angle Ch{angle_channel}')
    
    # Mark peaks
    if 'emg_peak_times' in correction_info:
        emg_peaks_in_range = correction_info['emg_peak_times'][correction_info['emg_peak_times'] <= 20]
        for peak_time in emg_peaks_in_range:
            ax3_emg.axvline(x=peak_time, color='blue', alpha=0.5, linestyle=':', linewidth=1)
    
    if 'angle_peak_times' in correction_info:
        angle_peaks_in_range = correction_info['angle_peak_times'][correction_info['angle_peak_times'] <= 20]
        for peak_time in angle_peaks_in_range:
            ax3_angle.axvline(x=peak_time, color='black', alpha=0.5, linestyle=':', linewidth=1)
    
    ax3_emg.set_xlabel('Time (s)')
    ax3_emg.set_ylabel('EMG Envelope', color='b')
    ax3_angle.set_ylabel('Angle (deg)', color='k')
    ax3_emg.set_title('Original Alignment (First 20s)')
    ax3_emg.legend(loc='upper left')
    ax3_angle.legend(loc='upper right')
    ax3_emg.grid(True)
    
    # Plot 4: Corrected alignment (first 20 seconds)
    time_mask_corr = emg_timestamps_corrected <= 20
    
    ax4_emg = axes[1, 1]
    ax4_angle = ax4_emg.twinx()
    
    step = max(1, np.sum(time_mask_corr) // 2000)
    ax4_emg.plot(emg_timestamps_corrected[time_mask_corr][::step], emg_envelope[time_mask_corr][::step], 'b-', alpha=0.7, label=f'EMG envelope Ch{emg_channel} (corrected)')
    ax4_angle.plot(angle_timestamps[angle_mask][::step], angle_signal[angle_mask][::step], 'k-', alpha=0.8, label=f'Angle Ch{angle_channel}')
    
    # Mark peaks in corrected data
    if 'emg_peak_times' in correction_info:
        for i, peak_time in enumerate(correction_info['emg_peak_times']):
            if peak_time <= 20:
                # Find corresponding corrected time
                peak_idx = np.argmin(np.abs(emg_timestamps - peak_time))
                corrected_peak_time = emg_timestamps_corrected[peak_idx]
                ax4_emg.axvline(x=corrected_peak_time, color='blue', alpha=0.5, linestyle=':', linewidth=1)
    
    if 'angle_peak_times' in correction_info:
        angle_peaks_in_range = correction_info['angle_peak_times'][correction_info['angle_peak_times'] <= 20]
        for peak_time in angle_peaks_in_range:
            ax4_angle.axvline(x=peak_time, color='black', alpha=0.5, linestyle=':', linewidth=1)
    
    ax4_emg.set_xlabel('Time (s)')
    ax4_emg.set_ylabel('EMG Envelope', color='b')
    ax4_angle.set_ylabel('Angle (deg)', color='k')
    ax4_emg.set_title('Corrected Alignment (First 20s)')
    ax4_emg.legend(loc='upper left')
    ax4_angle.legend(loc='upper right')
    ax4_emg.grid(True)
    
    # Plot 5: Original alignment (last 20 seconds)
    time_end = emg_timestamps[-1]
    time_mask = emg_timestamps >= (time_end - 20)
    angle_end = angle_timestamps[-1]
    angle_mask = angle_timestamps >= (angle_end - 20)
    
    ax5_emg = axes[2, 0]
    ax5_angle = ax5_emg.twinx()
    
    step = max(1, np.sum(time_mask) // 2000)
    ax5_emg.plot(emg_timestamps[time_mask][::step], emg_envelope[time_mask][::step], 'b-', alpha=0.7, label=f'EMG envelope Ch{emg_channel}')
    ax5_angle.plot(angle_timestamps[angle_mask][::step], angle_signal[angle_mask][::step], 'k-', alpha=0.8, label=f'Angle Ch{angle_channel}')
    
    # Mark peaks
    if 'emg_peak_times' in correction_info:
        emg_peaks_in_range = correction_info['emg_peak_times'][correction_info['emg_peak_times'] >= (time_end - 20)]
        for peak_time in emg_peaks_in_range:
            ax5_emg.axvline(x=peak_time, color='blue', alpha=0.5, linestyle=':', linewidth=1)
    
    if 'angle_peak_times' in correction_info:
        angle_peaks_in_range = correction_info['angle_peak_times'][correction_info['angle_peak_times'] >= (angle_end - 20)]
        for peak_time in angle_peaks_in_range:
            ax5_angle.axvline(x=peak_time, color='black', alpha=0.5, linestyle=':', linewidth=1)
    
    ax5_emg.set_xlabel('Time (s)')
    ax5_emg.set_ylabel('EMG Envelope', color='b')
    ax5_angle.set_ylabel('Angle (deg)', color='k')
    ax5_emg.set_title('Original Alignment (Last 20s)')
    ax5_emg.legend(loc='upper left')
    ax5_angle.legend(loc='upper right')
    ax5_emg.grid(True)
    
    # Plot 6: Corrected alignment (last 20 seconds)
    time_end_corr = emg_timestamps_corrected[-1]
    time_mask_corr = emg_timestamps_corrected >= (time_end_corr - 20)
    
    ax6_emg = axes[2, 1]
    ax6_angle = ax6_emg.twinx()
    
    step = max(1, np.sum(time_mask_corr) // 2000)
    ax6_emg.plot(emg_timestamps_corrected[time_mask_corr][::step], emg_envelope[time_mask_corr][::step], 'b-', alpha=0.7, label=f'EMG envelope Ch{emg_channel} (corrected)')
    ax6_angle.plot(angle_timestamps[angle_mask][::step], angle_signal[angle_mask][::step], 'k-', alpha=0.8, label=f'Angle Ch{angle_channel}')
    
    # Mark corrected peaks
    if 'emg_peak_times' in correction_info:
        for i, peak_time in enumerate(correction_info['emg_peak_times']):
            if peak_time >= (time_end - 20):
                # Find corresponding corrected time
                peak_idx = np.argmin(np.abs(emg_timestamps - peak_time))
                corrected_peak_time = emg_timestamps_corrected[peak_idx]
                ax6_emg.axvline(x=corrected_peak_time, color='blue', alpha=0.5, linestyle=':', linewidth=1)
    
    if 'angle_peak_times' in correction_info:
        angle_peaks_in_range = correction_info['angle_peak_times'][correction_info['angle_peak_times'] >= (angle_end - 20)]
        for peak_time in angle_peaks_in_range:
            ax6_angle.axvline(x=peak_time, color='black', alpha=0.5, linestyle=':', linewidth=1)
    
    ax6_emg.set_xlabel('Time (s)')
    ax6_emg.set_ylabel('EMG Envelope', color='b')
    ax6_angle.set_ylabel('Angle (deg)', color='k')
    ax6_emg.set_title('Corrected Alignment (Last 20s)')
    ax6_emg.legend(loc='upper left')
    ax6_angle.legend(loc='upper right')
    ax6_emg.grid(True)
    
    # Plot 7: Timestamp comparison
    axes[3, 0].plot(emg_timestamps, np.ones_like(emg_timestamps), 'b.', alpha=0.5, markersize=1, label='Original EMG')
    axes[3, 0].plot(emg_timestamps_corrected, np.ones_like(emg_timestamps_corrected) * 1.1, 'r.', alpha=0.5, markersize=1, label='Corrected EMG')
    axes[3, 0].plot(angle_timestamps, np.ones_like(angle_timestamps) * 0.9, 'g.', alpha=0.5, markersize=1, label='Angles')
    axes[3, 0].set_ylabel('Timeline')
    axes[3, 0].set_xlabel('Time (s)')
    axes[3, 0].set_title('Timestamp Comparison')
    axes[3, 0].legend()
    axes[3, 0].grid(True)
    
    # Plot 8: Time corrections applied
    time_corrections = emg_timestamps_corrected - emg_timestamps
    step = max(1, len(time_corrections) // 5000)
    axes[3, 1].plot(emg_timestamps[::step], time_corrections[::step], 'r-', alpha=0.7)
    axes[3, 1].set_xlabel('Time (s)')
    axes[3, 1].set_ylabel('Time Correction (s)')
    axes[3, 1].set_title('Time Corrections Applied to EMG')
    axes[3, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Drift correction analysis saved to: {save_path}")
    
    return fig

def main():
    parser = argparse.ArgumentParser(description='Peak-based drift correction for EMG-angle synchronization')
    # parser.add_argument('data_dir', help='Directory containing the recorded data')
    parser.add_argument('--emg-channel', type=int, default=0, help='EMG channel for analysis (default: 0)')
    parser.add_argument('--angle-channel', type=int, default=26, help='Angle channel for analysis (default: 2)')
    parser.add_argument('--target-phase', type=float, default=0.2, help='Target phase for EMG peaks (0-1, default: 0.2)')
    
    args = parser.parse_args()
    data_dir = "data/GG/recordings/handOpCl/experiments/1"
    # Load data
    print(f"Loading data from {data_dir}...")
    emg_data = np.load(os.path.join(data_dir, "raw_emg.npy"))
    emg_timestamps = np.load(os.path.join(data_dir, "raw_timestamps.npy"))
    angles_data = np.load(os.path.join(data_dir, "angles.npy"))
    angle_timestamps = np.load(os.path.join(data_dir, "angle_timestamps.npy"))
    
    print(f"Data loaded - EMG: {emg_data.shape}, Angles: {angles_data.shape}")
    
    # Apply traditional end-alignment first
    print("Applying end-alignment...")
    shift = emg_timestamps[-1] - angle_timestamps[-1]
    emg_timestamps_aligned = emg_timestamps - shift
    
    # Mask for overlapping data
    mask = (emg_timestamps_aligned >= angle_timestamps[0]) & (emg_timestamps_aligned <= angle_timestamps[-1])
    emg_data_aligned = emg_data[mask, :]
    emg_timestamps_aligned = emg_timestamps_aligned[mask]
    
    print(f"After end-alignment: kept {np.sum(mask)}/{len(mask)} EMG samples")
    
    # Apply peak-based drift correction
    emg_timestamps_corrected, correction_info = apply_peak_based_drift_correction(
        emg_data_aligned, emg_timestamps_aligned, angles_data, angle_timestamps,
        emg_channel=args.emg_channel, angle_channel=args.angle_channel, 
        target_phase=args.target_phase
    )
    
    # Create analysis plots
    plot_path = os.path.join(data_dir, 'peak_drift_correction_analysis.png')
    plot_drift_analysis(emg_data_aligned, emg_timestamps_aligned, emg_timestamps_corrected,
                       angles_data, angle_timestamps, correction_info, save_path=plot_path)
    
    # Save results
    np.save(os.path.join(data_dir, 'emg_peak_corrected.npy'), emg_data_aligned)
    np.save(os.path.join(data_dir, 'emg_timestamps_peak_corrected.npy'), emg_timestamps_corrected)
    
    print("✅ Peak-based drift correction complete!")
    return emg_timestamps_corrected, correction_info

if __name__ == "__main__":
    main()