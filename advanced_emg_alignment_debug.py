import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import minimize_scalar
import pandas as pd
from scipy.stats import pearsonr

def analyze_temporal_drift(emg_data, angle_data, timestamps, window_size=5.0, overlap=0.5):
    """
    Analyze how the phase relationship changes over time using sliding windows.
    
    Args:
        emg_data: EMG data [samples, channels]
        angle_data: Angle data [samples, angles]
        timestamps: Time stamps
        window_size: Size of analysis window in seconds
        overlap: Overlap between windows (0-1)
    
    Returns:
        Dictionary with drift analysis results
    """
    fs = 1.0 / np.mean(np.diff(timestamps))
    window_samples = int(window_size * fs)
    step_samples = int(window_samples * (1 - overlap))
    
    n_windows = (len(timestamps) - window_samples) // step_samples + 1
    
    results = {
        'window_centers': [],
        'lags_per_channel': {ch: [] for ch in range(emg_data.shape[1])},
        'correlations_per_channel': {ch: [] for ch in range(emg_data.shape[1])},
        'window_times': []
    }
    
    for w in range(n_windows):
        start_idx = w * step_samples
        end_idx = start_idx + window_samples
        
        if end_idx > len(timestamps):
            break
            
        window_center = timestamps[start_idx + window_samples // 2]
        results['window_centers'].append(window_center)
        results['window_times'].append((timestamps[start_idx], timestamps[end_idx-1]))
        
        emg_window = emg_data[start_idx:end_idx]
        angle_window = angle_data[start_idx:end_idx, 26]  # Assuming index finger
        
        for ch in range(emg_data.shape[1]):
            # Cross-correlation analysis for this window
            emg_ch = emg_window[:, ch]
            
            # Compute cross-correlation
            correlation = signal.correlate(angle_window, emg_ch, mode='full')
            lags = signal.correlation_lags(len(angle_window), len(emg_ch), mode='full')
            lag_times = lags / fs
            
            # Find optimal lag within reasonable bounds
            max_lag_samples = int(1.0 * fs)  # 1 second max
            valid_mask = np.abs(lags) <= max_lag_samples
            
            if np.any(valid_mask):
                valid_correlation = correlation[valid_mask]
                valid_lag_times = lag_times[valid_mask]
                
                max_idx = np.argmax(np.abs(valid_correlation))
                optimal_lag = valid_lag_times[max_idx]
                max_corr = valid_correlation[max_idx]
            else:
                optimal_lag = 0
                max_corr = 0
            
            results['lags_per_channel'][ch].append(optimal_lag)
            results['correlations_per_channel'][ch].append(max_corr)
    
    return results

def analyze_muscle_function_types(emg_data, angle_data, timestamps, angle_idx=26):
    """
    Analyze whether different EMG channels represent different muscle functions:
    - Agonist: peaks with angle peaks (flexors during flexion)
    - Antagonist: peaks when angle is at minimum (extensors during flexion)
    - Stabilizer: constant activation or different pattern
    """
    angle_signal = angle_data[:, angle_idx]
    
    # Find angle peaks and valleys
    peaks, _ = signal.find_peaks(angle_signal, height=np.percentile(angle_signal, 75))
    valleys, _ = signal.find_peaks(-angle_signal, height=-np.percentile(angle_signal, 25))
    
    # Calculate angle velocity and acceleration
    dt = np.mean(np.diff(timestamps))
    angle_velocity = np.gradient(angle_signal, dt)
    angle_acceleration = np.gradient(angle_velocity, dt)
    
    results = {}
    
    for ch in range(emg_data.shape[1]):
        emg_ch = emg_data[:, ch]
        
        # Correlations with different angle phases
        corr_position = pearsonr(emg_ch, angle_signal)[0]
        corr_velocity = pearsonr(emg_ch, np.abs(angle_velocity))[0]
        corr_acceleration = pearsonr(emg_ch, np.abs(angle_acceleration))[0]
        
        # EMG at peaks vs valleys
        if len(peaks) > 0 and len(valleys) > 0:
            emg_at_peaks = np.mean(emg_ch[peaks])
            emg_at_valleys = np.mean(emg_ch[valleys])
            peak_valley_ratio = emg_at_peaks / (emg_at_valleys + 1e-10)
        else:
            emg_at_peaks = emg_at_valleys = peak_valley_ratio = 0
        
        # Phase classification
        if corr_position > 0.3 and peak_valley_ratio > 1.5:
            muscle_type = "Agonist (Flexor)"
        elif corr_position < -0.3 and peak_valley_ratio < 0.67:
            muscle_type = "Antagonist (Extensor)"
        elif corr_velocity > 0.4:
            muscle_type = "Movement Facilitator"
        elif np.std(emg_ch) / np.mean(emg_ch) < 0.5:
            muscle_type = "Stabilizer"
        else:
            muscle_type = "Mixed/Unclear"
        
        results[ch] = {
            'muscle_type': muscle_type,
            'position_correlation': corr_position,
            'velocity_correlation': corr_velocity,
            'acceleration_correlation': corr_acceleration,
            'peak_valley_ratio': peak_valley_ratio,
            'emg_at_peaks': emg_at_peaks,
            'emg_at_valleys': emg_at_valleys
        }
    
    return results

def detect_synchronization_issues(emg_data, angle_data, timestamps):
    """
    Detect potential synchronization issues between EMG and angle data.
    """
    issues = []
    
    # Check for different sampling rates
    emg_fs = 1.0 / np.mean(np.diff(timestamps))
    
    # Check for temporal drift using cross-correlation
    drift_analysis = analyze_temporal_drift(emg_data, angle_data, timestamps)
    
    # Look for channels with highly variable lag over time
    for ch in range(emg_data.shape[1]):
        lags = np.array(drift_analysis['lags_per_channel'][ch])
        lag_std = np.std(lags)
        lag_range = np.max(lags) - np.min(lags)
        
        if lag_std > 0.1:  # More than 100ms standard deviation
            issues.append(f"Channel {ch}: High lag variability (std={lag_std:.3f}s, range={lag_range:.3f}s)")
        
        if lag_range > 0.3:  # More than 300ms total drift
            issues.append(f"Channel {ch}: Significant temporal drift ({lag_range:.3f}s over recording)")
    
    # Check for potential clock drift
    window_centers = np.array(drift_analysis['window_centers'])
    for ch in range(emg_data.shape[1]):
        lags = np.array(drift_analysis['lags_per_channel'][ch])
        if len(lags) > 5:
            # Fit linear trend to detect systematic drift
            slope, intercept = np.polyfit(window_centers, lags, 1)
            if abs(slope) > 0.01:  # More than 10ms drift per second
                issues.append(f"Channel {ch}: Systematic clock drift ({slope*1000:.1f}ms/s)")
    
    return issues, drift_analysis

def plot_comprehensive_analysis(emg_data, angle_data, timestamps, channels_to_plot=None):
    """
    Create comprehensive plots for debugging alignment issues.
    """
    if channels_to_plot is None:
        # Select most active channels
        emg_activity = np.std(emg_data, axis=0)
        channels_to_plot = np.argsort(emg_activity)[-8:]  # Top 6 channels
    
    # Run analyses
    muscle_analysis = analyze_muscle_function_types(emg_data, angle_data, timestamps)
    issues, drift_analysis = detect_synchronization_issues(emg_data, angle_data, timestamps)
    
    fig = plt.figure(figsize=(20, 15))
    
    # Plot 1: Original signals with muscle type annotations
    ax1 = plt.subplot(4, 2, 1)
    angle_norm = angle_data[:, 26] / np.max(angle_data[:, 26])
    plt.plot(timestamps, angle_norm, 'k-', linewidth=2, label='Index Angle (norm)')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(channels_to_plot)))
    for i, ch in enumerate(channels_to_plot):
        emg_norm = emg_data[:, ch] / np.max(emg_data[:, ch])
        muscle_type = muscle_analysis[ch]['muscle_type']
        plt.plot(timestamps, emg_norm, color=colors[i], alpha=0.7, 
                label=f'EMG {ch} ({muscle_type})')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Amplitude')
    plt.title('EMG Channels by Muscle Function Type')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    # Plot 2: Temporal drift analysis
    ax2 = plt.subplot(4, 2, 2)
    window_centers = drift_analysis['window_centers']
    for i, ch in enumerate(channels_to_plot):
        lags = drift_analysis['lags_per_channel'][ch]
        plt.plot(window_centers, lags, 'o-', color=colors[i], label=f'EMG {ch}')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Optimal Lag (s)')
    plt.title('Temporal Drift Analysis')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Correlation matrix
    ax3 = plt.subplot(4, 2, 3)
    correlations = []
    for ch in channels_to_plot:
        corr = muscle_analysis[ch]['position_correlation']
        correlations.append(corr)
    
    plt.bar(range(len(channels_to_plot)), correlations, color=colors)
    plt.xlabel('EMG Channel')
    plt.ylabel('Position Correlation')
    plt.title('EMG-Angle Position Correlation')
    plt.xticks(range(len(channels_to_plot)), [f'EMG {ch}' for ch in channels_to_plot], rotation=45)
    plt.grid(True)
    
    # Plot 4: Phase relationship (first 10 seconds)
    ax4 = plt.subplot(4, 2, 4)
    mask_10s = timestamps <= 10
    t_10s = timestamps[mask_10s]
    angle_10s = angle_data[mask_10s, 26]
    
    plt.plot(t_10s, angle_10s / np.max(angle_10s), 'k-', linewidth=2, label='Angle')
    for i, ch in enumerate(channels_to_plot):
        emg_10s = emg_data[mask_10s, ch]
        plt.plot(t_10s, emg_10s / np.max(emg_10s), color=colors[i], alpha=0.7, label=f'EMG {ch}')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Amplitude')
    plt.title('Phase Relationships (First 10s)')
    plt.legend()
    plt.grid(True)
    
    # Plot 5: Phase relationship (last 10 seconds)
    ax5 = plt.subplot(4, 2, 5)
    mask_last = timestamps >= (timestamps[-1] - 10)
    t_last = timestamps[mask_last]
    angle_last = angle_data[mask_last, 26]
    
    plt.plot(t_last, angle_last / np.max(angle_last), 'k-', linewidth=2, label='Angle')
    for i, ch in enumerate(channels_to_plot):
        emg_last = emg_data[mask_last, ch]
        plt.plot(t_last, emg_last / np.max(emg_last), color=colors[i], alpha=0.7, label=f'EMG {ch}')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Amplitude')
    plt.title('Phase Relationships (Last 10s)')
    plt.legend()
    plt.grid(True)
    
    # Plot 6: Muscle type summary
    ax6 = plt.subplot(4, 2, 6)
    muscle_types = [muscle_analysis[ch]['muscle_type'] for ch in channels_to_plot]
    type_counts = {}
    for mt in muscle_types:
        type_counts[mt] = type_counts.get(mt, 0) + 1
    
    plt.pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%')
    plt.title('Distribution of Muscle Function Types')
    
    # Plot 7: Lag distribution
    ax7 = plt.subplot(4, 2, 7)
    all_lags = []
    for ch in channels_to_plot:
        ch_lags = drift_analysis['lags_per_channel'][ch]
        all_lags.extend(ch_lags)
    
    plt.hist(all_lags, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Lag (s)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Optimal Lags')
    plt.grid(True)
    
    # Plot 8: Issues summary (text)
    ax8 = plt.subplot(4, 2, 8)
    ax8.axis('off')
    issues_text = "Detected Issues:\n\n" + "\n".join([f"• {issue}" for issue in issues])
    if not issues:
        issues_text = "No major synchronization issues detected!"
    
    plt.text(0.1, 0.9, issues_text, transform=ax8.transAxes, fontsize=10, 
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.show()
    
    return muscle_analysis, drift_analysis, issues

def recommend_alignment_strategy(muscle_analysis, drift_analysis, issues):
    """
    Provide recommendations based on the analysis results.
    """
    recommendations = []
    
    # Check for consistent muscle types
    agonist_channels = [ch for ch, data in muscle_analysis.items() 
                       if 'Agonist' in data['muscle_type']]
    antagonist_channels = [ch for ch, data in muscle_analysis.items() 
                          if 'Antagonist' in data['muscle_type']]
    
    if len(agonist_channels) > 0:
        recommendations.append(f"Use agonist channels {agonist_channels} for primary control (should peak with angle)")
    
    if len(antagonist_channels) > 0:
        recommendations.append(f"Antagonist channels {antagonist_channels} should peak in valleys - this is normal!")
    
    # Check for drift issues
    drift_issues = [issue for issue in issues if 'drift' in issue.lower()]
    if drift_issues:
        recommendations.append("CRITICAL: Clock synchronization issues detected!")
        recommendations.append("Consider re-recording with better hardware synchronization")
    
    # Check for high variability
    high_var_issues = [issue for issue in issues if 'variability' in issue.lower()]
    if high_var_issues:
        recommendations.append("High lag variability suggests filtering or processing artifacts")
        recommendations.append("Try different filter parameters or window sizes")
    
    # General recommendations
    recommendations.append("For training data:")
    recommendations.append("- Use different alignment strategies for agonist vs antagonist muscles")
    recommendations.append("- Consider separate models for different muscle groups")
    recommendations.append("- Apply small lead time (50-150ms) for agonist muscles only")
    
    return recommendations