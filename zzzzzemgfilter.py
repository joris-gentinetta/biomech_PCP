#!/usr/bin/env python3
"""
Complete EMG processing: Rest data verification + MVC maxVals calculation + Normalization
"""

import numpy as np
import matplotlib.pyplot as plt
from helpers.BesselFilter import BesselFilterArr

def process_emg_through_pipeline(emg_data, sf, data_type="unknown", noise_levels=None):
    """
    Process EMG data through the complete filtering pipeline
    
    Parameters:
    -----------
    emg_data : np.array
        Raw EMG data
    sf : float
        Sampling frequency
    data_type : str
        Type of data ("rest", "mvc", "task") for appropriate processing
    noise_levels : np.array, optional
        Pre-calculated noise levels (for non-rest data)
    
    Returns:
    --------
    dict with processed data and statistics
    """
    
    # Ensure data is [channels, samples]
    if emg_data.shape[0] > emg_data.shape[1]:
        emg_data = emg_data.T
    
    num_channels = emg_data.shape[0]
    print(f"Processing {data_type} data: {emg_data.shape} at {sf:.1f}Hz")
    
    # Step 1: Cut artifacts
    artifact_cut = 400
    data_cut = emg_data[:, artifact_cut:]
    
    # Step 2: Bandstop filter (remove powerline)
    notch_filter = BesselFilterArr(numChannels=num_channels, order=8,
                                  critFreqs=[58, 62], fs=sf, filtType='bandstop')
    data_notch = notch_filter.filter(data_cut)
    
    # Cut transients
    transient_cut_notch = int(0.5 * sf)
    data_notch = data_notch[:, transient_cut_notch:]
    
    # Step 3: Highpass filter (remove DC offset)
    hp_filter = BesselFilterArr(numChannels=num_channels, order=4,
                               critFreqs=20, fs=sf, filtType='highpass')
    data_hp = hp_filter.filter(data_notch)
    
    # Cut transients
    transient_cut_hp = int(1.0 * sf)
    data_hp = data_hp[:, transient_cut_hp:]
    
    # Step 4: Rectification
    data_rect = np.abs(data_hp)
    
    # Step 5: Noise estimation (only for rest data)
    if data_type == "rest":
        print("  Estimating noise levels from rest data...")
        noise_levels = np.zeros(num_channels)
        for ch in range(num_channels):
            # Use 98th percentile for aggressive noise removal in rest data
            noise_levels[ch] = np.percentile(data_rect[ch, :], 99.9)
    elif noise_levels is None:
        print("  Warning: No noise levels provided for non-rest data!")
        noise_levels = np.zeros(num_channels)
    else:
        print("  Using provided noise levels...")
    
    # Step 6: Noise subtraction
    data_denoised = np.clip(data_rect - noise_levels[:, None], 0, None)
    
    # Step 7: Lowpass envelope
    lp_filter = BesselFilterArr(numChannels=num_channels, order=4,
                               critFreqs=3, fs=sf, filtType='lowpass')
    data_final = lp_filter.filter(data_denoised)
    
    # Step 8: Final artifact cut
    final_cut_samples = 400
    data_final_cut = data_final[:, final_cut_samples:]
    
    # Create time axis
    time_axis = np.arange(data_final_cut.shape[1]) / sf
    
    print(f"  Final {data_type} shape: {data_final_cut.shape}")
    
    return {
        'processed_data': data_final_cut,
        'time_axis': time_axis,
        'noise_levels': noise_levels,
        'rectified_data': data_rect,  # For MVC calculation
        'sampling_frequency': sf,
        'data_type': data_type
    }

def calculate_mvc_maxvals(mvc_processed, percentile=95):
    """
    Calculate MVC maximum values using specified percentile
    
    Parameters:
    -----------
    mvc_processed : dict
        Processed MVC data from process_emg_through_pipeline
    percentile : float
        Percentile to use for max calculation (default: 95)
    
    Returns:
    --------
    maxVals : np.array
        Maximum values for each channel
    mvc_stats : dict
        Statistics about MVC data
    """
    
    processed_data = mvc_processed['processed_data']
    num_channels = processed_data.shape[0]
    
    print(f"\n=== MVC MAX VALUES CALCULATION ===")
    print(f"Using {percentile}th percentile for robust max estimation")
    
    maxVals = np.zeros(num_channels)
    mvc_stats = {
        'percentile_used': percentile,
        'channel_stats': {}
    }
    
    for ch in range(num_channels):
        channel_data = processed_data[ch, :]
        
        # Calculate different percentiles for comparison
        p95 = np.percentile(channel_data, 95)
        p99 = np.percentile(channel_data, 99)
        max_val = np.max(channel_data)
        mean_val = np.mean(channel_data)
        
        # Use specified percentile as maxVal
        maxVals[ch] = np.percentile(channel_data, percentile)
        
        # Store detailed stats
        mvc_stats['channel_stats'][ch] = {
            'max_value': max_val,
            'p99': p99,
            'p95': p95,
            'mean': mean_val,
            'chosen_maxval': maxVals[ch]
        }
        
        if ch < 8:  # Print first 8 channels
            print(f"  Ch {ch}: max={max_val:.1f}, p99={p99:.1f}, p95={p95:.1f}, "
                  f"chosen={maxVals[ch]:.1f}")
    
    # Quality checks
    print(f"\n  MVC Quality Assessment:")
    
    # Check for reasonable maxVals
    valid_channels = maxVals > 1.0  # Should have some signal
    num_valid = np.sum(valid_channels)
    print(f"    Channels with good signal (>1.0): {num_valid}/{num_channels}")
    
    # Check for outliers
    if num_valid > 0:
        valid_maxvals = maxVals[valid_channels]
        median_maxval = np.median(valid_maxvals)
        mad = np.median(np.abs(valid_maxvals - median_maxval))
        
        print(f"    Median maxVal: {median_maxval:.1f}")
        print(f"    MAD: {mad:.1f}")
        
        # Flag potential outliers (>3 MAD from median)
        outlier_threshold = median_maxval + 3 * mad * 1.4826
        outliers = np.where(maxVals > outlier_threshold)[0]
        if len(outliers) > 0:
            print(f"      Potential outlier channels: {outliers}")
        else:
            print(f"     No outlier channels detected")

    mvc_stats['summary'] = {
        'num_valid_channels': num_valid,
        'median_maxval': np.median(maxVals[maxVals > 0]) if np.any(maxVals > 0) else 0,
        'outlier_channels': outliers.tolist() if 'outliers' in locals() else []
    }
    
    return maxVals, mvc_stats

def normalize_emg_data(emg_data, maxVals, clip_max=1.5):
    """
    Normalize EMG data using MVC maxVals
    
    Parameters:
    -----------
    emg_data : np.array, shape [channels, samples]
        Processed EMG data to normalize
    maxVals : np.array
        Maximum values for each channel from MVC
    clip_max : float
        Maximum normalized value to clip at (default: 1.5 = 150% MVC)
    
    Returns:
    --------
    normalized_data : np.array
        Normalized EMG data (0-1 range, where 1 = 100% MVC)
    norm_stats : dict
        Normalization statistics
    """
    
    print(f"\n=== EMG NORMALIZATION ===")
    print(f"Normalizing to MVC maxVals, clipping at {clip_max*100:.0f}% MVC")
    
    # Avoid division by zero
    safe_maxVals = np.where(maxVals > 0, maxVals, 1.0)
    
    # Normalize: EMG / maxVal
    normalized_data = emg_data / safe_maxVals[:, None]
    
    # Clip extreme values
    normalized_data = np.clip(normalized_data, 0, clip_max)
    
    # Calculate normalization statistics
    norm_stats = {
        'clip_max': clip_max,
        'channels_clipped': [],
        'max_normalized_values': [],
        'mean_normalized_values': []
    }
    
    for ch in range(emg_data.shape[0]):
        max_norm = np.max(normalized_data[ch, :])
        mean_norm = np.mean(normalized_data[ch, :])
        
        norm_stats['max_normalized_values'].append(max_norm)
        norm_stats['mean_normalized_values'].append(mean_norm)
        
        # Check if clipping occurred
        if max_norm >= clip_max - 0.001:  # Account for floating point precision
            norm_stats['channels_clipped'].append(ch)
        
        if ch < 8:
            clipped_str = " (CLIPPED)" if ch in norm_stats['channels_clipped'] else ""
            print(f"  Ch {ch}: max_norm={max_norm:.2f}, mean_norm={mean_norm:.3f}{clipped_str}")
    
    num_clipped = len(norm_stats['channels_clipped'])
    if num_clipped > 0:
        print(f"    {num_clipped} channels clipped at {clip_max*100:.0f}% MVC")
        print(f"      Clipped channels: {norm_stats['channels_clipped']}")
    else:
        print(f"   No channels exceeded {clip_max*100:.0f}% MVC")
    
    return normalized_data, norm_stats

def plot_rest_mvc_comparison(rest_results, mvc_results, maxVals, channels_to_plot=None):
    """
    Plot comparison of rest and MVC data processing
    """
    if channels_to_plot is None:
        channels_to_plot = [0, 1, 2, 4]
    
    fig, axes = plt.subplots(len(channels_to_plot), 3, figsize=(18, 3*len(channels_to_plot)))
    if len(channels_to_plot) == 1:
        axes = axes.reshape(1, -1)
    
    for i, ch in enumerate(channels_to_plot):
        if ch >= rest_results['processed_data'].shape[0]:
            continue
        
        # Rest data
        ax = axes[i, 0]
        ax.plot(rest_results['time_axis'], rest_results['processed_data'][ch, :], 
                'b-', linewidth=0.5, label='Rest (final)')
        ax.axhline(y=rest_results['noise_levels'][ch], color='r', linestyle='--', 
                  alpha=0.7, label=f'Noise: {rest_results["noise_levels"][ch]:.2f}')
        ax.set_title(f'Ch {ch}: Rest Data (Should be ~0)')
        ax.set_ylabel('Amplitude')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Add zero percentage
        zero_pct = np.sum(rest_results['processed_data'][ch, :] == 0) / len(rest_results['processed_data'][ch, :]) * 100
        ax.text(0.01, 0.99, f'Zeros: {zero_pct:.1f}%', transform=ax.transAxes, 
               verticalalignment='top', fontsize=8,
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # MVC data
        ax = axes[i, 1]
        ax.plot(mvc_results['time_axis'], mvc_results['processed_data'][ch, :], 
                'g-', linewidth=0.5, label='MVC (final)')
        ax.axhline(y=maxVals[ch], color='r', linestyle='-', 
                  alpha=0.8, label=f'MaxVal: {maxVals[ch]:.1f}')
        ax.set_title(f'Ch {ch}: MVC Data')
        ax.set_ylabel('Amplitude')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Add MVC stats
        max_val = np.max(mvc_results['processed_data'][ch, :])
        mean_val = np.mean(mvc_results['processed_data'][ch, :])
        ax.text(0.02, 0.98, f'Max: {max_val:.1f}\nMean: {mean_val:.1f}', 
               transform=ax.transAxes, verticalalignment='top', fontsize=8,
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # Normalized MVC data
        ax = axes[i, 2]
        normalized_mvc = mvc_results['processed_data'][ch, :] / maxVals[ch]
        ax.plot(mvc_results['time_axis'], normalized_mvc, 
                'purple', linewidth=0.5, label='MVC (normalized)')
        ax.axhline(y=1.0, color='r', linestyle='-', 
                  alpha=0.8, label='100% MVC')
        ax.set_title(f'Ch {ch}: Normalized MVC')
        ax.set_ylabel('% MVC')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Add normalization stats
        max_norm = np.max(normalized_mvc)
        mean_norm = np.mean(normalized_mvc)
        ax.text(0.02, 0.98, f'Max: {max_norm:.2f}\nMean: {mean_norm:.3f}', 
               transform=ax.transAxes, verticalalignment='top', fontsize=8,
               bbox=dict(boxstyle='round', facecolor='plum', alpha=0.8))
    
    # Set x-labels for bottom row
    for j in range(3):
        axes[-1, j].set_xlabel('Time (s)')
    
    plt.suptitle('EMG Processing: Rest vs MVC Comparison', fontsize=14)
    plt.tight_layout()
    plt.show()

def process_task_emg(task_data, task_timestamps, noise_levels, maxVals, channels_to_plot=None):
    """
    Process task EMG data using pre-calculated noise levels and maxVals
    
    Parameters:
    -----------
    task_data : np.array
        Raw task EMG data
    task_timestamps : np.array
        Task timestamps
    noise_levels : np.array
        Pre-calculated noise levels from rest data
    maxVals : np.array
        Pre-calculated maxVals from MVC data
    channels_to_plot : list, optional
        Channels to visualize
    
    Returns:
    --------
    dict with processed and normalized task data
    """
    
    # Calculate sampling frequency
    sf_task = (len(task_timestamps) - 1) / (task_timestamps[-1] - task_timestamps[0])
    
    print(f"\n{'='*60}")
    print(f"PROCESSING TASK EMG DATA")
    print(f"{'='*60}")
    print(f"Task data: {task_data.shape} at {sf_task:.1f}Hz")
    
    # Process through pipeline using pre-calculated noise levels
    task_results = process_emg_through_pipeline(task_data, sf_task, data_type="task", 
                                              noise_levels=noise_levels)
    
    # Normalize using MVC maxVals
    print(f"\nNormalizing task data using MVC maxVals...")
    normalized_task, norm_stats = normalize_emg_data(task_results['processed_data'], maxVals)
    
    # Add normalized data to results
    task_results['normalized_data'] = normalized_task
    task_results['normalization_stats'] = norm_stats
    
    # Plot task EMG results
    if channels_to_plot is None:
        channels_to_plot = [0, 1, 2, 4, 12, 13, 14, 15]
    
    plot_task_emg_results(task_results, noise_levels, maxVals, channels_to_plot)
    
    return task_results

def plot_task_emg_results(task_results, noise_levels, maxVals, channels_to_plot):
    """
    Plot task EMG processing results: raw → filtered → normalized
    """
    
    processed_data = task_results['processed_data']
    normalized_data = task_results['normalized_data']
    time_axis = task_results['time_axis']
    
    # Filter channels that exist
    max_channels = processed_data.shape[0]
    channels_to_plot = [ch for ch in channels_to_plot if ch < max_channels]
    
    fig, axes = plt.subplots(len(channels_to_plot), 3, figsize=(18, 3*len(channels_to_plot)))
    if len(channels_to_plot) == 1:
        axes = axes.reshape(1, -1)
    
    for i, ch in enumerate(channels_to_plot):
        
        # Filtered EMG (absolute values, after noise subtraction)
        ax = axes[i, 0]
        ax.plot(time_axis, processed_data[ch, :], 'b-', linewidth=0.7, alpha=0.8)
        ax.axhline(y=noise_levels[ch], color='r', linestyle='--', alpha=0.7, 
                  label=f'Noise: {noise_levels[ch]:.2f}')
        ax.set_title(f'Ch {ch}: Filtered EMG')
        ax.set_ylabel('Amplitude')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_filt = np.mean(processed_data[ch, :])
        max_filt = np.max(processed_data[ch, :])
        zero_pct = np.sum(processed_data[ch, :] == 0) / len(processed_data[ch, :]) * 100
        
        stats_text = f'Mean: {mean_filt:.2f}\nMax: {max_filt:.1f}\nZeros: {zero_pct:.1f}%'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=8,
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Normalized EMG (% MVC)
        ax = axes[i, 1]
        ax.plot(time_axis, normalized_data[ch, :], 'g-', linewidth=0.7, alpha=0.8)
        ax.axhline(y=1.0, color='r', linestyle='-', alpha=0.7, label='100% MVC')
        ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='50% MVC')
        ax.set_title(f'Ch {ch}: Normalized EMG (% MVC)')
        ax.set_ylabel('% MVC')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.5)  # 0-150% MVC
        
        # Add normalization statistics
        mean_norm = np.mean(normalized_data[ch, :])
        max_norm = np.max(normalized_data[ch, :])
        p95_norm = np.percentile(normalized_data[ch, :], 95)
        
        stats_text = f'Mean: {mean_norm:.3f}\nMax: {max_norm:.2f}\n95th: {p95_norm:.2f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=8,
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # EMG Activity Levels (color-coded by intensity)
        ax = axes[i, 2]
        
        # Create color-coded plot based on activation level
        colors = np.where(normalized_data[ch, :] < 0.1, 'blue',      # Rest: <10% MVC
                 np.where(normalized_data[ch, :] < 0.3, 'green',     # Light: 10-30% MVC  
                 np.where(normalized_data[ch, :] < 0.6, 'orange',    # Moderate: 30-60% MVC
                          'red')))                                   # High: >60% MVC
        
        ax.scatter(time_axis, normalized_data[ch, :], c=colors, s=1, alpha=0.7)
        ax.axhline(y=0.1, color='blue', linestyle=':', alpha=0.5, label='Light (10%)')
        ax.axhline(y=0.3, color='green', linestyle=':', alpha=0.5, label='Moderate (30%)')
        ax.axhline(y=0.6, color='orange', linestyle=':', alpha=0.5, label='High (60%)')
        
        ax.set_title(f'Ch {ch}: Activity Levels')
        ax.set_ylabel('% MVC')
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.5)
        
        # Add activity time percentages
        rest_time = np.sum(normalized_data[ch, :] < 0.1) / len(normalized_data[ch, :]) * 100
        light_time = np.sum((normalized_data[ch, :] >= 0.1) & (normalized_data[ch, :] < 0.3)) / len(normalized_data[ch, :]) * 100
        mod_time = np.sum((normalized_data[ch, :] >= 0.3) & (normalized_data[ch, :] < 0.6)) / len(normalized_data[ch, :]) * 100
        high_time = np.sum(normalized_data[ch, :] >= 0.6) / len(normalized_data[ch, :]) * 100
        
        stats_text = f'Rest: {rest_time:.1f}%\nLight: {light_time:.1f}%\nMod: {mod_time:.1f}%\nHigh: {high_time:.1f}%'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=7,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Set x-labels for bottom row
    for j in range(3):
        axes[-1, j].set_xlabel('Time (s)')
    
    plt.suptitle('Task EMG Processing: Filtered → Normalized → Activity Analysis', fontsize=14)
    plt.tight_layout()
    plt.show()

def complete_emg_calibration_and_task():
    """
    Complete EMG processing: calibration (rest + MVC) + task processing
    """
    import os
    
    print("=== COMPLETE EMG CALIBRATION + TASK PROCESSING ===\n")
    
    # Load calibration data
    base_dir = "data/Emanuel/recordings/Calibration/experiments/1"
    
    # Rest data
    rest_data = np.load(os.path.join(base_dir, 'calib_rest_emg.npy'))
    rest_timestamps = np.load(os.path.join(base_dir, 'calib_rest_timestamps.npy'))
    sf_rest = (len(rest_timestamps) - 1) / (rest_timestamps[-1] - rest_timestamps[0])
    
    # MVC data
    mvc_data = np.load(os.path.join(base_dir, 'calib_mvc_emg.npy'))
    mvc_timestamps = np.load(os.path.join(base_dir, 'calib_mvc_timestamps.npy'))
    sf_mvc = (len(mvc_timestamps) - 1) / (mvc_timestamps[-1] - mvc_timestamps[0])
    
    # Task data - UPDATE THIS PATH TO YOUR TASK DATA
    task_base_dir = "data/Emanuel/recordings/fingersFlEx/experiments/1"  # CHANGE THIS PATH
    task_data = np.load(os.path.join(task_base_dir, 'raw_emg.npy'))
    task_timestamps = np.load(os.path.join(task_base_dir, 'raw_timestamps.npy'))
    
    print(f"Loaded data:")
    print(f"  Rest: {rest_data.shape} at {sf_rest:.1f}Hz")
    print(f"  MVC:  {mvc_data.shape} at {sf_mvc:.1f}Hz")
    print(f"  Task: {task_data.shape} at sampling freq to be calculated")
    
    # === CALIBRATION PHASE ===
    print(f"\n{'='*50}")
    print("PHASE 1: CALIBRATION")
    print(f"{'='*50}")
    
    # Process rest data (estimates noise levels)
    rest_results = process_emg_through_pipeline(rest_data, sf_rest, data_type="rest")
    
    # Process MVC data (using rest noise levels)
    mvc_results = process_emg_through_pipeline(mvc_data, sf_mvc, data_type="mvc", 
                                             noise_levels=rest_results['noise_levels'])
    
    # Calculate MVC maxVals
    maxVals, mvc_stats = calculate_mvc_maxvals(mvc_results, percentile=90)
    
    # Plot calibration comparison
    channels_to_check = [0, 1, 2, 11, 12, 13, 14, 15]
    # channels_to_check = [0,1]
    max_channels = min(rest_data.shape[1] if rest_data.shape[0] > rest_data.shape[1] else rest_data.shape[0], 16)
    channels_to_check = [ch for ch in channels_to_check if ch < max_channels]
    
    plot_rest_mvc_comparison(rest_results, mvc_results, maxVals, channels_to_check[:8])
    
    # === TASK PROCESSING PHASE ===
    print(f"\n{'='*50}")
    print("PHASE 2: TASK PROCESSING")
    print(f"{'='*50}")
    
    # Process task data using calibrated parameters
    task_results = process_task_emg(task_data, task_timestamps, 
                                   rest_results['noise_levels'], maxVals, 
                                   channels_to_plot=channels_to_check[:8])
    
    # === SAVE RESULTS ===
    calibration_results = {
        'noise_levels': rest_results['noise_levels'],
        'maxVals': maxVals,
        'mvc_stats': mvc_stats,
        'task_results': task_results,
        'sampling_frequencies': {
            'rest': sf_rest,
            'mvc': sf_mvc,
            'task': task_results['sampling_frequency']
        },
        'channels_processed': channels_to_check
    }
    
    # Save calibration to file
    import yaml
    calibration_file = os.path.join(base_dir, 'emg_calibration_with_task.yaml')
    
    yaml_data = {
        'noise_levels': rest_results['noise_levels'].tolist(),
        'maxVals': maxVals.tolist(),
        'sampling_frequencies': calibration_results['sampling_frequencies'],
        'channels_processed': channels_to_check,
        'mvc_percentile_used': 95,
        'calibration_quality': {
            'rest_zero_percentage': np.sum(rest_results['processed_data'] == 0) / rest_results['processed_data'].size * 100,
            'mvc_valid_channels': mvc_stats['summary']['num_valid_channels'],
            'task_mean_activation': float(np.mean(task_results['normalized_data'])),
            'task_max_activation': float(np.max(task_results['normalized_data']))
        },
        'task_info': {
            'duration_seconds': float(task_results['time_axis'][-1]),
            'num_samples': int(task_results['processed_data'].shape[1])
        }
    }
    
    with open(calibration_file, 'w') as f:
        yaml.safe_dump(yaml_data, f, default_flow_style=False)
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    
    print(f"\n CALIBRATION PARAMETERS:")
    print(f"  Noise levels: {len(rest_results['noise_levels'])} channels")
    print(f"  MVC maxVals: {len(maxVals)} channels") 
    print(f"  Valid MVC channels: {mvc_stats['summary']['num_valid_channels']}")
    
    print(f"\n TASK EMG ANALYSIS:")
    print(f"  Duration: {task_results['time_axis'][-1]:.1f} seconds")
    print(f"  Mean activation: {np.mean(task_results['normalized_data']):.3f} (% MVC)")
    print(f"  Peak activation: {np.max(task_results['normalized_data']):.2f} (% MVC)")
    print(f"  Channels with >50% MVC: {np.sum(np.max(task_results['normalized_data'], axis=1) > 0.5)}")
    
    print(f"\n FILES SAVED:")
    print(f"  Calibration: {calibration_file}")
    
    print(f"\nComplete EMG processing pipeline finished successfully!")
    
    return calibration_results

def complete_emg_calibration():
    """
    Legacy function - now calls the complete version with task processing
    """
    return complete_emg_calibration_and_task()

if __name__ == "__main__":
    # Run complete calibration + task processing
    calibration_results = complete_emg_calibration_and_task()
    
    print(f"\n=== USAGE EXAMPLE ===")
    print(f"# To process new task data with these calibration parameters:")
    print(f"import yaml")
    print(f"import numpy as np")
    print(f"")
    print(f"# Load calibration")
    print(f"cal = yaml.safe_load(open('emg_calibration_with_task.yaml'))")
    print(f"noise_levels = np.array(cal['noise_levels'])")
    print(f"maxVals = np.array(cal['maxVals'])")
    print(f"")
    print(f"# Process new task EMG")
    print(f"task_results = process_task_emg(new_task_data, new_timestamps, noise_levels, maxVals)")
    print(f"normalized_emg = task_results['normalized_data']  # 0-1 scale (% MVC)")
    
    print(f"\n=== KEY INSIGHTS FROM YOUR TASK DATA ===")
    task_results = calibration_results['task_results']
    normalized = task_results['normalized_data']
    
    # Calculate overall task statistics
    channels_active = np.sum(np.max(normalized, axis=1) > 0.1)  # Channels with >10% MVC
    peak_activation = np.max(normalized)
    mean_activation = np.mean(normalized)
    
    # Time in different activation levels
    total_samples = normalized.size
    rest_samples = np.sum(normalized < 0.1)
    light_samples = np.sum((normalized >= 0.1) & (normalized < 0.3))
    moderate_samples = np.sum((normalized >= 0.3) & (normalized < 0.6))
    high_samples = np.sum(normalized >= 0.6)
    
    print(f"Task Overview:")
    print(f"  Active channels (>10% MVC): {channels_active}/{normalized.shape[0]}")
    print(f"  Peak activation: {peak_activation:.2f} ({peak_activation*100:.0f}% MVC)")
    print(f"  Mean activation: {mean_activation:.3f} ({mean_activation*100:.1f}% MVC)")
    
    print(f"Time Distribution:")
    print(f"  Rest (<10% MVC): {100*rest_samples/total_samples:.1f}%")
    print(f"  Light (10-30% MVC): {100*light_samples/total_samples:.1f}%")
    print(f"  Moderate (30-60% MVC): {100*moderate_samples/total_samples:.1f}%")
    print(f"  High (>60% MVC): {100*high_samples/total_samples:.1f}%")
    
    if peak_activation > 1.0:
        print(f"Peak activation exceeded 100% MVC - check MVC calibration!")
    if mean_activation < 0.05:
        print(f"Low overall activation - this might be a light task")
    if channels_active < normalized.shape[0] / 2:
        print(f"Only {channels_active} channels active - task may be localized")