#!/usr/bin/env python3
"""
Correct EMG processing pipeline with proper filter order
The key is to remove DC offset BEFORE calculating noise levels
"""

import numpy as np
import matplotlib.pyplot as plt
from helpers.BesselFilter import BesselFilterArr
import os
from scipy.interpolate import interp1d

def process_emg_correct_order(emg_data, sf, artifact_cut=400, show_steps=True, check_downsample_zeros=True):
    """
    Process EMG data with correct filter order to handle DC offset
    
    Correct order:
    1. Cut artifacts
    2. Highpass filter (removes DC offset and low-freq drift)
    3. Bandstop filter (removes powerline interference)
    4. Rectification
    5. Noise estimation (on properly filtered data)
    6. Noise subtraction
    7. Lowpass envelope filter
    
    Parameters:
    -----------
    emg_data : np.array, shape [samples, channels] or [channels, samples]
    sf : float
        Sampling frequency
    artifact_cut : int
        Samples to cut from beginning
    show_steps : bool
        Whether to plot intermediate steps
    
    Returns:
    --------
    processed_data : dict
        Dictionary containing all processing steps
    """
    
    # Ensure data is [channels, samples]
    if emg_data.shape[0] > emg_data.shape[1]:
        emg_data = emg_data.T
    
    num_channels, num_samples = emg_data.shape
    print(f"Processing EMG data: {num_channels} channels, {num_samples} samples")
    print(f"Sampling frequency: {sf:.1f} Hz")
    print(f"Duration: {num_samples/sf:.1f} seconds\n")
    
    # Store all processing steps
    processing_steps = {}
    
    # Step 1: Cut initial artifacts
    print("Step 1: Cutting initial artifacts...")
    data_cut = emg_data[:, artifact_cut:]
    processing_steps['1_artifact_cut'] = data_cut.copy()
    print(f"  Removed {artifact_cut} samples ({artifact_cut/sf:.2f}s)")
    print(f"  New shape: {data_cut.shape}\n")

    # Step 2: BANDSTOP FILTER (removes powerline interference)
    print("Step 3: Bandstop filtering (58-62 Hz) - REMOVES POWERLINE...")
    notch_filter = BesselFilterArr(numChannels=num_channels, order=8,
                                  critFreqs=[58, 62], fs=sf, filtType='bandstop')
    data_notch = notch_filter.filter(data_hp)

    # Cut additional transients
    transient_cut_notch = int(0.5 * sf)  # 0.5 seconds for notch
    data_notch = data_notch[:, transient_cut_notch:]
    processing_steps['3_bandstop'] = data_notch.copy()
    
    # Step 3: HIGH-PASS FILTER FIRST (removes DC offset!)
    print("Step 2: High-pass filtering (20 Hz) - REMOVES DC OFFSET...")
    hp_filter = BesselFilterArr(numChannels=num_channels, order=4, 
                               critFreqs=20, fs=sf, filtType='highpass')
    data_hp = hp_filter.filter(data_cut)
    
    # Cut filter transients
    transient_cut = int(1.0 * sf)  # 1 second for highpass
    data_hp = data_hp[:, transient_cut:]
    processing_steps['2_highpass'] = data_hp.copy()
    
    print(f"  Applied 4th order Bessel highpass at 20 Hz")
    print(f"  Cut {transient_cut} transient samples ({transient_cut/sf:.1f}s)")
    print(f"  New shape: {data_hp.shape}")
    
    # Check DC removal effectiveness
    # dc_before = np.mean(data_cut, axis=1)
    # dc_after = np.mean(data_hp, axis=1)
    # print(f"  DC offset reduction:")
    # for ch in range(min(4, num_channels)):
    #     print(f"    Ch {ch}: {dc_before[ch]:.1f} → {dc_after[ch]:.3f}")
    # print()
    
    
    print(f"  Applied 8th order Bessel bandstop 58-62 Hz")
    print(f"  Cut {transient_cut_notch} additional transient samples")
    print(f"  New shape: {data_notch.shape}\n")
    
    # Step 4: RECTIFICATION
    # print("Step 4: Full-wave rectification...")
    # data_rect = np.abs(data_hp)
    # processing_steps['4_rectified'] = data_rect.copy()
    # print(f"  Applied full-wave rectification\n")
    
    # Step 5: NOISE ESTIMATION (now on properly filtered data!)
    print("Step 5: Noise level estimation (on filtered data)...")
    noise_levels = np.zeros(num_channels)
    
    for ch in range(num_channels):
        signal = data_hp[ch, :]
        
        # Now that DC is removed, use robust statistics
        median_val = np.median(signal)
        mad = np.median(np.abs(signal - median_val))
        
        # Conservative noise estimate
        # Use MAD-based approach (more robust than mean+std)
        noise_mad = median_val + 2.0 * mad * 1.4826  # 1.4826 converts MAD to std equivalent
        
        # Alternative: use percentile approach
        noise_percentile = np.percentile(signal, 88)  # 88th percentile
        
        # Take the more conservative estimate
        noise_levels[ch] = max(noise_mad, noise_percentile)
        
        if ch < 8:  # Print for first 8 channels
            print(f"  Ch {ch}: median={median_val:.2f}, MAD={mad:.2f}, "
                  f"noise={noise_levels[ch]:.2f}")
    
    processing_steps['5_noise_levels'] = noise_levels.copy()
    print()
    
    # Step 6: NOISE SUBTRACTION
    print("Step 6: Noise subtraction...")
    data_denoised = np.clip(data_hp - noise_levels[:, None], 0, None)
    processing_steps['6_denoised'] = data_denoised.copy()
    
    # Calculate zero percentages
    print("  Zero percentages after denoising:")
    for ch in range(min(8, num_channels)):
        zero_pct = np.sum(data_denoised[ch, :] == 0) / len(data_denoised[ch, :]) * 100
        print(f"    Ch {ch}: {zero_pct:.1f}%")
    print()
    
    # Step 7: LOWPASS ENVELOPE FILTER
    print("Step 7: Lowpass envelope filtering (3 Hz)...")
    lp_filter = BesselFilterArr(numChannels=num_channels, order=4,
                               critFreqs=3, fs=sf, filtType='lowpass')
    data_final = lp_filter.filter(data_denoised)
    processing_steps['7_final'] = data_final.copy()
    
    print(f"  Applied 4th order Bessel lowpass at 3 Hz")
    print(f"  Final processing complete!\n")

    # Create time axis for final data
    time_axis = np.arange(data_final.shape[1]) / sf

    # === (OPTIONAL) ZERO CHECKS AND DOWNSAMPLING ===
    if check_downsample_zeros:
        print("\n=== BEFORE DOWNSAMPLING ===")
        all_zeros_before = np.all(data_final < 0.001, axis=0)
        print(f"Samples where ALL channels < 0.001: {np.sum(all_zeros_before)}/{data_final.shape[1]} ({100*np.sum(all_zeros_before)/data_final.shape[1]:.1f}%)")

        timestamps = time_axis
        data_60hz, timestamps_60hz = downsample_to_60hz(data_final, timestamps, target_freq=60.0)

        print("\n=== AFTER DOWNSAMPLING ===")
        all_zeros_after = np.all(data_60hz < 0.001, axis=0)
        print(f"Samples where ALL channels < 0.001: {np.sum(all_zeros_after)}/{data_60hz.shape[1]} ({100*np.sum(all_zeros_after)/data_60hz.shape[1]:.1f}%)")

        for ch in range(min(4, data_60hz.shape[0])):
            before_zeros = np.sum(data_final[ch, :] < 0.001) / data_final.shape[1] * 100
            after_zeros = np.sum(data_60hz[ch, :] < 0.001) / data_60hz.shape[1] * 100
            print(f"Ch {ch}: {before_zeros:.1f}% → {after_zeros:.1f}% near-zero")
    # === END ZERO CHECKS & DOWNSAMPLE ===

    print(f"  Applied 4th order Bessel lowpass at 3 Hz")
    print(f"  Final processing complete!\n")
    
    # Create time axis for final data
    time_axis = np.arange(data_final.shape[1]) / sf
    
    if show_steps:
        plot_processing_steps(processing_steps, time_axis, sf, 
                            channels_to_show=[0, 1, 2, 4, 12, 13, 14, 15])
    
    return {
        'processed_data': data_final,
        'noise_levels': noise_levels,
        'processing_steps': processing_steps,
        'time_axis': time_axis,
        'sampling_frequency': sf
    }

def plot_processing_steps(processing_steps, time_axis, sf, channels_to_show=None):
    """
    Plot the EMG processing pipeline steps
    """
    if channels_to_show is None:
        channels_to_show = [0, 1, 2, 3]
    
    # Select key steps to visualize
    key_steps = [
        ('1_artifact_cut', 'Raw (artifacts cut)'),
        ('2_highpass', 'After Highpass (DC removed)'),
        ('3_bandstop', 'After Bandstop'),
        ('4_rectified', 'Rectified'),
        ('6_denoised', 'Denoised'),
        ('7_final', 'Final (envelope)')
    ]
    
    n_steps = len(key_steps)
    n_channels = len(channels_to_show)
    
    fig, axes = plt.subplots(n_channels, n_steps, figsize=(4*n_steps, 3*n_channels))
    if n_channels == 1:
        axes = axes.reshape(1, -1)
    if n_steps == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle('EMG Processing Pipeline (Correct Filter Order)', fontsize=16)
    
    for i, ch in enumerate(channels_to_show):
        for j, (step_key, step_name) in enumerate(key_steps):
            ax = axes[i, j]
            
            if step_key in processing_steps:
                data = processing_steps[step_key]
                
                # Adjust time axis length to match data
                if step_key == '1_artifact_cut':
                    # This is the longest signal, create its own time axis
                    step_time = np.arange(data.shape[1]) / sf
                else:
                    # Use the final time axis, but truncate if necessary
                    step_time = time_axis[:min(len(time_axis), data.shape[1])]
                
                # Plot signal
                ax.plot(step_time, data[ch, :len(step_time)], 'b-', linewidth=0.7, alpha=0.8)
                
                # Add noise level line for relevant steps
                if step_key in ['4_rectified', '6_denoised'] and 'noise_levels' in processing_steps:
                    noise_level = processing_steps['noise_levels'][ch]
                    ax.axhline(y=noise_level, color='r', linestyle='--', 
                              linewidth=1, alpha=0.7, label=f'Noise: {noise_level:.1f}')
                    ax.legend(fontsize=8)
                
                # Calculate and display statistics
                mean_val = np.mean(data[ch, :])
                std_val = np.std(data[ch, :])
                
                ax.set_title(f'{step_name}\nCh {ch}', fontsize=10)
                ax.set_xlabel('Time (s)', fontsize=8)
                ax.set_ylabel('Amplitude', fontsize=8)
                ax.grid(True, alpha=0.3)
                
                # Add statistics text
                stats_text = f'μ={mean_val:.1f}\nσ={std_val:.1f}'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', fontsize=7,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            else:
                ax.text(0.5, 0.5, 'Step not\navailable', ha='center', va='center',
                       transform=ax.transAxes)
                ax.set_title(f'{step_name}\nCh {ch}', fontsize=10)
    
    plt.tight_layout()
    plt.show()

def compare_old_vs_new_pipeline(rest_data, mvc_data, sf_rest, sf_mvc):
    """
    Compare noise estimates from old vs new pipeline
    """
    print("=== PIPELINE COMPARISON ===\n")
    
    # Process with NEW (correct) pipeline
    print("Processing with CORRECT pipeline (highpass first):")
    rest_new = process_emg_correct_order(rest_data, sf_rest, show_steps=False)
    mvc_new = process_emg_correct_order(mvc_data, sf_mvc, show_steps=False)
    
    # Simulate OLD pipeline (bandstop first, no proper DC removal)
    print("\nSimulating OLD pipeline (bandstop first, statistical noise on raw):")
    
    if rest_data.shape[0] > rest_data.shape[1]:
        rest_data = rest_data.T
        mvc_data = mvc_data.T
    
    # Cut artifacts only
    rest_cut = rest_data[:, 400:]
    mvc_cut = mvc_data[:, 400:]
    
    # Apply bandstop first (old way)
    notch_rest = BesselFilterArr(numChannels=rest_data.shape[0], order=8, 
                                critFreqs=[58,62], fs=sf_rest, filtType='bandstop')
    rest_notched = notch_rest.filter(rest_cut)
    
    # Then highpass
    hp_rest = BesselFilterArr(numChannels=rest_data.shape[0], order=4, 
                             critFreqs=20, fs=sf_rest, filtType='highpass')
    rest_hp_old = hp_rest.filter(rest_notched)
    
    # Rectify
    rest_rect_old = np.abs(rest_hp_old)
    
    # OLD noise estimation (on data that still had DC issues during estimation)
    noise_old = []
    for ch in range(rest_data.shape[0]):
        # This is what was happening before - statistical estimation on DC-biased data
        mean_val = np.mean(rest_rect_old[ch, :])
        std_val = np.std(rest_rect_old[ch, :])
        noise_old.append(mean_val + 2 * std_val)
    
    noise_old = np.array(noise_old)
    
    # Compare results
    print(f"\nNOISE LEVEL COMPARISON:")
    print(f"{'Channel':<8} {'Old Method':<12} {'New Method':<12} {'Improvement':<12}")
    print("-" * 50)
    
    for ch in range(min(8, rest_data.shape[0])):
        old_val = noise_old[ch]
        new_val = rest_new['noise_levels'][ch]
        improvement = old_val / new_val if new_val > 0 else float('inf')
        
        print(f"{ch:<8} {old_val:<12.1f} {new_val:<12.1f} {improvement:<12.1f}x")
    
    return rest_new, mvc_new, noise_old

def estimate_noise_from_task_data(task_data, sf, mvc_data=None, rest_data=None, method='quiet_periods'):
    """
    Estimate noise levels using actual task data with rest validation
    
    Parameters:
    -----------
    task_data : np.array
        Task EMG data (properly filtered: highpass → bandstop → rectified)
    sf : float
        Sampling frequency
    mvc_data : np.array, optional
        MVC data for reference scaling
    rest_data : np.array, optional
        Rest data for validation (should be ~100% zeros after denoising)
    method : str
        'quiet_periods', 'percentile', or 'hybrid'
    
    Returns:
    --------
    noise_levels : np.array
        Estimated noise level per channel
    noise_info : dict
        Information about noise estimation and validation
    """
    if task_data.shape[0] > task_data.shape[1]:
        task_data = task_data.T
    
    num_channels, num_samples = task_data.shape
    noise_levels = np.zeros(num_channels)
    
    print(f"Estimating noise from task data ({task_data.shape})...")
    
    if method == 'quiet_periods':
        # Find naturally quiet periods during the task
        window_size = int(sf * 2.0)  # 2-second windows
        step_size = int(window_size * 0.5)  # 50% overlap
        
        all_quiet_segments = []
        
        for start in range(0, num_samples - window_size, step_size):
            end = start + window_size
            window_data = task_data[:, start:end]
            
            # Calculate overall activity level (RMS across all channels)
            channel_rms = np.sqrt(np.mean(window_data**2, axis=1))
            overall_activity = np.mean(channel_rms)
            
            # Check if this is a quiet period (bottom 10% of activity)
            if len(all_quiet_segments) == 0:
                # Always include first window for comparison
                all_quiet_segments.append((start, end, overall_activity))
            else:
                # Compare to existing segments
                activities = [seg[2] for seg in all_quiet_segments]
                threshold = np.percentile(activities + [overall_activity], 15)
                
                if overall_activity <= threshold:
                    all_quiet_segments.append((start, end, overall_activity))
        
        # Extract data from quiet periods
        if len(all_quiet_segments) >= 3:  # Need at least 3 quiet periods
            quiet_data_segments = []
            for start, end, _ in all_quiet_segments:
                quiet_data_segments.append(task_data[:, start:end])
            
            # Concatenate all quiet segments
            all_quiet_data = np.concatenate(quiet_data_segments, axis=1)
            
            print(f"  Found {len(all_quiet_segments)} quiet periods")
            print(f"  Total quiet samples: {all_quiet_data.shape[1]} ({all_quiet_data.shape[1]/sf:.1f}s)")
            
            # Calculate initial noise from quiet periods  
            initial_noise = np.zeros(num_channels)
            for ch in range(num_channels):
                quiet_signal = all_quiet_data[ch, :]
                # Use 95th percentile of quiet periods as initial estimate
                initial_noise[ch] = np.percentile(quiet_signal, 95)
                
                if ch < 8:
                    quiet_mean = np.mean(quiet_signal)
                    quiet_95th = initial_noise[ch]
                    print(f"    Ch {ch}: quiet_mean={quiet_mean:.2f}, quiet_95th={quiet_95th:.2f}")
            
            noise_info = {
                'method': 'quiet_periods',
                'num_periods': len(all_quiet_segments),
                'total_quiet_samples': all_quiet_data.shape[1],
                'quiet_duration_s': all_quiet_data.shape[1] / sf,
                'initial_estimates': initial_noise.copy()
            }
            
        else:
            print(f"  Only found {len(all_quiet_segments)} quiet periods, falling back to percentile method")
            method = 'percentile'
    
    if method == 'percentile':
        # Use low percentiles of the entire task as noise estimate
        print("  Using percentile method on full task data")
        initial_noise = np.zeros(num_channels)
        for ch in range(num_channels):
            # Use 10th percentile as baseline noise
            initial_noise[ch] = np.percentile(task_data[ch, :], 10)
            
            if ch < 8:
                signal_median = np.median(task_data[ch, :])
                print(f"    Ch {ch}: 10th_percentile={initial_noise[ch]:.2f}, median={signal_median:.2f}")
        
        noise_info = {
            'method': 'percentile',
            'percentile_used': 10,
            'initial_estimates': initial_noise.copy()
        }
    
    # Start with initial estimates
    noise_levels = initial_noise.copy()
    
    # Apply MVC-based cap if available
    if mvc_data is not None:
        if mvc_data.shape[0] > mvc_data.shape[1]:
            mvc_data = mvc_data.T
        
        print("  Applying MVC-based caps...")
        for ch in range(num_channels):
            mvc_95th = np.percentile(mvc_data[ch, :], 95)
            # max_allowed = mvc_95th * 0.05  # 5% of MVC (less aggressive than 3%)
            max_allowed = mvc_95th   # 5% of MVC (less aggressive than 3%)
            
            original_noise = noise_levels[ch]
            noise_levels[ch] = min(noise_levels[ch], max_allowed)
            
            if ch < 8 and original_noise != noise_levels[ch]:
                print(f"    Ch {ch}: capped {original_noise:.1f} → {noise_levels[ch]:.1f} (5% of MVC)")
    
    # CRITICAL: Validate against rest data - adjust to get ~95-98% zeros
    if rest_data is not None:
        if rest_data.shape[0] > rest_data.shape[1]:
            rest_data = rest_data.T
        
        print(f"\n  VALIDATING against rest data (target: 95-98% zeros)...")
        
        adjusted_noise = noise_levels.copy()
        for ch in range(num_channels):
            rest_signal = rest_data[ch, :]
            
            # Try current threshold
            current_zeros = np.sum(rest_signal <= noise_levels[ch]) / len(rest_signal) * 100
            
            # If not enough zeros, increase threshold
            target_zeros = 96  # Target 96% zeros in rest
            
            if current_zeros < target_zeros:
                # Find threshold that gives target percentage
                target_threshold = np.percentile(rest_signal, target_zeros)
                
                # Don't go too crazy - cap the adjustment
                max_increase = noise_levels[ch] * 3  # Max 3x increase
                adjusted_threshold = min(target_threshold, max_increase)
                
                adjusted_noise[ch] = adjusted_threshold
                
                final_zeros = np.sum(rest_signal <= adjusted_threshold) / len(rest_signal) * 100
                
                if ch < 8:
                    print(f"    Ch {ch}: {current_zeros:.1f}% → {final_zeros:.1f}% zeros "
                          f"(thresh: {noise_levels[ch]:.1f} → {adjusted_threshold:.1f})")
            else:
                if ch < 8:
                    print(f"    Ch {ch}: {current_zeros:.1f}% zeros - OK (thresh: {noise_levels[ch]:.1f})")
        
        noise_levels = adjusted_noise
        noise_info['rest_validation'] = True
        noise_info['target_rest_zeros'] = target_zeros
    
    noise_info['final_estimates'] = noise_levels.copy()
    return noise_levels, noise_info

def track_zeros_per_step(data, step_name, channels_to_show=None, detailed=True):
    """
    Track and print zero percentages for each channel at each processing step
    
    Parameters:
    -----------
    data : np.array, shape [channels, samples]
        EMG data at current processing step
    step_name : str
        Name of the processing step
    channels_to_show : list, optional
        List of channel indices to display (default: first 8)
    detailed : bool
        Whether to show detailed per-channel stats
    
    Returns:
    --------
    zero_stats : dict
        Dictionary with zero statistics for this step
    """
    if channels_to_show is None:
        channels_to_show = list(range(min(8, data.shape[0])))
    
    zero_stats = {
        'step_name': step_name,
        'channel_stats': {},
        'overall_stats': {}
    }
    
    total_zeros = 0
    total_samples = 0
    
    for ch in range(data.shape[0]):
        channel_data = data[ch, :]
        
        # Count exact zeros and near-zeros
        exact_zeros = np.sum(channel_data == 0.0)
        near_zeros = np.sum(np.abs(channel_data) < 1e-10)
        tiny_values = np.sum(np.abs(channel_data) < 1e-6)
        
        total_samples_ch = len(channel_data)
        zero_percentage = 100 * exact_zeros / total_samples_ch
        
        # Store stats
        zero_stats['channel_stats'][ch] = {
            'exact_zeros': exact_zeros,
            'near_zeros': near_zeros,
            'tiny_values': tiny_values,
            'total_samples': total_samples_ch,
            'zero_percentage': zero_percentage,
            'min_value': np.min(channel_data),
            'max_value': np.max(channel_data),
            'min_abs_nonzero': np.min(np.abs(channel_data[channel_data != 0])) if exact_zeros < total_samples_ch else 0.0
        }
        
        total_zeros += exact_zeros
        total_samples += total_samples_ch
    
    # Overall statistics
    overall_zero_percentage = 100 * total_zeros / total_samples
    zero_stats['overall_stats'] = {
        'total_zeros': total_zeros,
        'total_samples': total_samples,
        'zero_percentage': overall_zero_percentage
    }
    
    # Print results
    print(f"\n=== ZERO TRACKING: {step_name} ===")
    
    if detailed:
        for ch in channels_to_show:
            if ch < data.shape[0]:
                stats = zero_stats['channel_stats'][ch]
                min_nonzero = stats['min_abs_nonzero']
                print(f"  Ch {ch:2d}: {stats['exact_zeros']:5d}/{stats['total_samples']} "
                      f"({stats['zero_percentage']:5.1f}%) exact zeros, "
                      f"min_nonzero: {min_nonzero:.2e}")
    
    print(f"  OVERALL: {total_zeros}/{total_samples} ({overall_zero_percentage:.1f}%) exact zeros")
    
    return zero_stats

def print_zero_tracking_summary(tracking_results, channels_to_show):
    """
    Print a comprehensive summary table of zero percentages across all steps
    """
    print(f"\n{'='*100}")
    print(f"COMPREHENSIVE ZERO TRACKING SUMMARY")
    print(f"{'='*100}")
    
    # Header
    print(f"{'Step':<25}", end="")
    for ch in channels_to_show:
        print(f"Ch{ch:2d}%", end="  ")
    print(f"{'Overall%':<10}")
    
    print("-" * 100)
    
    # Data rows
    for result in tracking_results:
        step_name = result['step_name']
        print(f"{step_name:<25}", end="")
        
        for ch in channels_to_show:
            if ch in result['channel_stats']:
                zero_pct = result['channel_stats'][ch]['zero_percentage']
                print(f"{zero_pct:5.1f}", end="  ")
            else:
                print(f"{'--':<5}", end="  ")
        
        overall_pct = result['overall_stats']['zero_percentage']
        print(f"{overall_pct:8.1f}")
    
    print("-" * 100)
    
    # Analyze zero loss between steps
    print(f"\nZERO LOSS ANALYSIS:")
    step_zero_loss = []
    for i in range(1, len(tracking_results)):
        prev_zeros = tracking_results[i-1]['overall_stats']['zero_percentage']
        curr_zeros = tracking_results[i]['overall_stats']['zero_percentage']
        loss = prev_zeros - curr_zeros
        step_zero_loss.append((tracking_results[i]['step_name'], loss, prev_zeros, curr_zeros))
        
        if abs(loss) > 0.1:  # Only show significant changes
            direction = "LOST" if loss > 0 else "GAINED"
            print(f"  {tracking_results[i-1]['step_name']} → {tracking_results[i]['step_name']}: "
                  f"{direction} {abs(loss):.1f}% zeros ({prev_zeros:.1f}% → {curr_zeros:.1f}%)")
    
    if step_zero_loss:
        worst_step = max(step_zero_loss, key=lambda x: x[1])
        if worst_step[1] > 0:
            print(f"\n🔴 BIGGEST ZERO KILLER: {worst_step[0]} (lost {worst_step[1]:.1f}% zeros)")
        
        best_step = min(step_zero_loss, key=lambda x: x[1])
        if best_step[1] < 0:
            print(f"🟢 BIGGEST ZERO CREATOR: {best_step[0]} (gained {abs(best_step[1]):.1f}% zeros)")
    
    # Final zero preservation per channel
    print(f"\nFINAL ZERO PRESERVATION BY CHANNEL:")
    final_result = tracking_results[-1]
    for ch in channels_to_show:
        if ch in final_result['channel_stats']:
            final_stats = final_result['channel_stats'][ch]
            final_zeros = final_stats['zero_percentage']
            min_val = final_stats['min_value']
            print(f"  Ch {ch:2d}: {final_zeros:5.1f}% exact zeros (min_value: {min_val:.2e})")

def process_emg_correct_order_WITH_TRACKING(emg_data, sf, artifact_cut=400, show_steps=True, 
                                          channels_to_show=None, track_zeros=True):
    """
    Process EMG data with correct filter order + comprehensive zero tracking
    
    Parameters:
    -----------
    emg_data : np.array, shape [samples, channels] or [channels, samples]
    sf : float
        Sampling frequency
    artifact_cut : int
        Samples to cut from beginning
    show_steps : bool
        Whether to plot intermediate steps
    channels_to_show : list, optional
        Channels to display in tracking (default: [0,1,2,4,12,13,14,15])
    track_zeros : bool
        Whether to perform zero tracking
    
    Returns:
    --------
    processed_data : dict
        Dictionary containing all processing steps + zero tracking
    """
    
    # Ensure data is [channels, samples]
    if emg_data.shape[0] > emg_data.shape[1]:
        emg_data = emg_data.T
    
    num_channels, num_samples = emg_data.shape
    
    if channels_to_show is None:
        channels_to_show = [0, 1, 2, 4, 12, 13, 14, 15] if num_channels > 15 else list(range(min(8, num_channels)))
    
    print(f"Processing EMG data: {num_channels} channels, {num_samples} samples")
    print(f"Sampling frequency: {sf:.1f} Hz")
    print(f"Duration: {num_samples/sf:.1f} seconds")
    if track_zeros:
        print(f"Tracking zeros for channels: {channels_to_show}")
    print()
    
    # Store all processing steps and zero tracking
    processing_steps = {}
    zero_tracking_results = []
    
    # Step 0: Original data
    filtered_emg = np.copy(emg_data)
    processing_steps['0_original'] = filtered_emg.copy()
    if track_zeros:
        zero_tracking_results.append(track_zeros_per_step(filtered_emg, "0_ORIGINAL", channels_to_show))
    
    # Step 1: Cut initial artifacts
    print("Step 1: Cutting initial artifacts...")
    data_cut = emg_data[:, artifact_cut:]
    processing_steps['1_artifact_cut'] = data_cut.copy()
    print(f"  Removed {artifact_cut} samples ({artifact_cut/sf:.2f}s)")
    print(f"  New shape: {data_cut.shape}")
    
    if track_zeros:
        zero_tracking_results.append(track_zeros_per_step(data_cut, "1_ARTIFACT_CUT", channels_to_show))
    
    # Step 3: BANDSTOP FILTER (removes powerline interference)
    print("\nStep 3: Bandstop filtering (58-62 Hz) - REMOVES POWERLINE...")
    notch_filter = BesselFilterArr(numChannels=num_channels, order=8,
                                  critFreqs=[58, 62], fs=sf, filtType='bandstop')
    data_notch = notch_filter.filter(data_cut)
    
    # Cut additional transients
    transient_cut_notch = int(0.5 * sf)  # 0.5 seconds for notch
    data_notch = data_notch[:, transient_cut_notch:]
    processing_steps['3_bandstop'] = data_notch.copy()

    # Step 2: HIGH-PASS FILTER FIRST (removes DC offset!)
    print("\nStep 2: High-pass filtering (20 Hz) - REMOVES DC OFFSET...")
    hp_filter = BesselFilterArr(numChannels=num_channels, order=4, 
                               critFreqs=20, fs=sf, filtType='highpass')
    data_hp = hp_filter.filter(data_notch)
    
    # Cut filter transients
    transient_cut = int(1.0 * sf)  # 1 second for highpass
    data_hp = data_hp[:, transient_cut:]
    processing_steps['2_highpass'] = data_hp.copy()
    
    print(f"  Applied 4th order Bessel highpass at 20 Hz")
    print(f"  Cut {transient_cut} transient samples ({transient_cut/sf:.1f}s)")
    print(f"  New shape: {data_hp.shape}")
    
    # Check DC removal effectiveness
    # dc_before = np.mean(data_cut, axis=1)
    # dc_after = np.mean(data_hp, axis=1)
    # print(f"  DC offset reduction:")
    # for ch in range(min(4, num_channels)):
    #     print(f"    Ch {ch}: {dc_before[ch]:.1f} → {dc_after[ch]:.3f}")
    
    if track_zeros:
        zero_tracking_results.append(track_zeros_per_step(data_hp, "2_HIGHPASS", channels_to_show))
    
    print(f"  Applied 8th order Bessel bandstop 58-62 Hz")
    print(f"  Cut {transient_cut_notch} additional transient samples")
    print(f"  New shape: {data_notch.shape}")
    
    if track_zeros:
        zero_tracking_results.append(track_zeros_per_step(data_notch, "3_BANDSTOP", channels_to_show))
    
    # Step 4: RECTIFICATION
    print("\nStep 4: Full-wave rectification...")
    data_rect = np.abs(data_hp)
    processing_steps['4_rectified'] = data_rect.copy()
    print(f"  Applied full-wave rectification")
    
    if track_zeros:
        zero_tracking_results.append(track_zeros_per_step(data_hp, "4_RECTIFIED", channels_to_show))
    
    # Step 5: NOISE ESTIMATION (now on properly filtered data!)
    print("\nStep 5: Noise level estimation (on filtered data)...")
    noise_levels = np.zeros(num_channels)
    
    for ch in range(num_channels):
        signal = data_rect[ch, :]
        
        # Now that DC is removed, use robust statistics
        median_val = np.median(signal)
        mad = np.median(np.abs(signal - median_val))
        
        # Conservative noise estimate
        # Use MAD-based approach (more robust than mean+std)
        noise_mad = median_val + 2.0 * mad * 1.4826  # 1.4826 converts MAD to std equivalent
        
        # Alternative: use percentile approach
        noise_percentile = np.percentile(signal, 88)  # 88th percentile
        
        # Take the more conservative estimate
        noise_levels[ch] = max(noise_mad, noise_percentile)
        
        if ch < 8:  # Print for first 8 channels
            print(f"  Ch {ch}: median={median_val:.2f}, MAD={mad:.2f}, "
                  f"noise={noise_levels[ch]:.2f}")
    
    processing_steps['5_noise_levels'] = noise_levels.copy()
    
    # Step 6: NOISE SUBTRACTION
    print("\nStep 6: Noise subtraction...")
    data_denoised = np.clip(data_hp - noise_levels[:, None], 0, None)
    processing_steps['6_denoised'] = data_denoised.copy()
    
    # Calculate zero percentages the old way for comparison
    print("  Zero percentages after denoising (basic count):")
    for ch in range(min(8, num_channels)):
        zero_pct = np.sum(data_denoised[ch, :] == 0) / len(data_denoised[ch, :]) * 100
        print(f"    Ch {ch}: {zero_pct:.1f}%")
    
    if track_zeros:
        zero_tracking_results.append(track_zeros_per_step(data_denoised, "6_NOISE_SUBTRACTED", channels_to_show))
    
    # Step 7: LOWPASS ENVELOPE FILTER
    print("\nStep 7: Lowpass envelope filtering (3 Hz)...")
    lp_filter = BesselFilterArr(numChannels=num_channels, order=4,
                               critFreqs=3, fs=sf, filtType='lowpass')
    data_final = lp_filter.filter(data_denoised)
    processing_steps['7_final'] = data_final.copy()
    
    print(f"  Applied 4th order Bessel lowpass at 3 Hz")
    print(f"  Final processing complete!")
    
    if track_zeros:
        zero_tracking_results.append(track_zeros_per_step(data_final, "7_LOWPASS_FINAL", channels_to_show))
    
    # Create time axis for final data
    time_axis = np.arange(data_final.shape[1]) / sf
    
    # Print comprehensive zero tracking summary
    if track_zeros:
        print_zero_tracking_summary(zero_tracking_results, channels_to_show)
    
    if show_steps:
        plot_processing_steps(processing_steps, time_axis, sf, 
                            channels_to_show=channels_to_show)
    
    return {
        'processed_data': data_final,
        'noise_levels': noise_levels,
        'processing_steps': processing_steps,
        'zero_tracking': zero_tracking_results if track_zeros else None,
        'time_axis': time_axis,
        'sampling_frequency': sf,
        'channels_tracked': channels_to_show
    }

def plot_processing_steps(processing_steps, time_axis, sf, channels_to_show=None):
    """
    Plot the EMG processing pipeline steps with zero tracking annotations
    """
    if channels_to_show is None:
        channels_to_show = [0, 1, 2, 3]
    
    # Select key steps to visualize
    key_steps = [
        ('1_artifact_cut', 'Raw (artifacts cut)'),
        ('2_highpass', 'After Highpass (DC removed)'),
        ('3_bandstop', 'After Bandstop'),
        ('4_rectified', 'Rectified'),
        ('6_denoised', 'Denoised (zeros created!)'),
        ('7_final', 'Final (zeros destroyed?)')
    ]
    
    n_steps = len(key_steps)
    n_channels = len(channels_to_show)
    
    fig, axes = plt.subplots(n_channels, n_steps, figsize=(4*n_steps, 3*n_channels))
    if n_channels == 1:
        axes = axes.reshape(1, -1)
    if n_steps == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle('EMG Processing Pipeline with Zero Tracking', fontsize=16)
    
    for i, ch in enumerate(channels_to_show):
        for j, (step_key, step_name) in enumerate(key_steps):
            ax = axes[i, j]
            
            if step_key in processing_steps:
                data = processing_steps[step_key]
                
                # Adjust time axis length to match data
                if step_key == '1_artifact_cut':
                    # This is the longest signal, create its own time axis
                    step_time = np.arange(data.shape[1]) / sf
                else:
                    # Use the final time axis, but truncate if necessary
                    step_time = time_axis[:min(len(time_axis), data.shape[1])]
                
                # Plot signal
                ax.plot(step_time, data[ch, :len(step_time)], 'b-', linewidth=0.7, alpha=0.8)
                
                # Add noise level line for relevant steps
                if step_key in ['4_rectified', '6_denoised'] and '5_noise_levels' in processing_steps:
                    noise_level = processing_steps['5_noise_levels'][ch]
                    ax.axhline(y=noise_level, color='r', linestyle='--', 
                              linewidth=1, alpha=0.7, label=f'Noise: {noise_level:.1f}')
                    ax.legend(fontsize=8)
                
                # Calculate zero percentage for this step
                zeros = np.sum(data[ch, :] == 0.0)
                total = len(data[ch, :])
                zero_pct = 100 * zeros / total
                
                # Calculate and display statistics
                mean_val = np.mean(data[ch, :])
                std_val = np.std(data[ch, :])
                min_val = np.min(data[ch, :])
                
                ax.set_title(f'{step_name}\nCh {ch}', fontsize=10)
                ax.set_xlabel('Time (s)', fontsize=8)
                ax.set_ylabel('Amplitude', fontsize=8)
                ax.grid(True, alpha=0.3)
                
                # Add statistics text with zero percentage
                stats_text = f'μ={mean_val:.1f}\nσ={std_val:.1f}\nZeros: {zero_pct:.1f}%\nMin: {min_val:.2e}'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', fontsize=6,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            else:
                ax.text(0.5, 0.5, 'Step not\navailable', ha='center', va='center',
                       transform=ax.transAxes)
                ax.set_title(f'{step_name}\nCh {ch}', fontsize=10)
    
    plt.tight_layout()
    plt.show()


def downsample_to_60hz(data, timestamps, target_freq=60.0):
    """
    Downsample data to target frequency using simple decimation
    
    Parameters:
    -----------
    data : np.array, shape [channels, samples]
        Filtered EMG data
    timestamps : np.array
        Timestamp array
    target_freq : float
        Target frequency (default 60Hz)
    
    Returns:
    --------
    downsampled_data : np.array
        Downsampled data
    downsampled_timestamps : np.array
        Downsampled timestamps
    """
    # Calculate current sampling frequency
    current_sf = (len(timestamps) - 1) / (timestamps[-1] - timestamps[0])
    print(f"\nDownsampling from {current_sf:.1f}Hz to {target_freq}Hz")
    
    # Calculate downsample ratio and step
    downsample_ratio = current_sf / target_freq
    downsample_step = int(round(downsample_ratio))
    
    print(f"Downsample ratio: {downsample_ratio:.2f}")
    print(f"Using step size: {downsample_step}")
    
    # Perform downsampling
    downsampled_data = data[:, ::downsample_step]
    downsampled_timestamps = timestamps[::downsample_step]
    
    # Verify final sampling rate
    final_sf = (len(downsampled_timestamps) - 1) / (downsampled_timestamps[-1] - downsampled_timestamps[0])
    print(f"Final sampling frequency: {final_sf:.1f}Hz")
    print(f"Original samples: {data.shape[1]}, Downsampled: {downsampled_data.shape[1]}")
    
    return downsampled_data, downsampled_timestamps


def interpolate_emg_to_angle_time(emg_data, emg_time, angle_time):
    emg_interp = np.zeros((emg_data.shape[0], len(angle_time)))
    for ch in range(emg_data.shape[0]):
        f = interp1d(emg_time, emg_data[ch], kind='linear', bounds_error=False, fill_value='extrapolate')
        emg_interp[ch] = f(angle_time)
    return emg_interp

# Update the main function to use the new tracking version
def main():
    """Main function using task data for noise estimation WITH ZERO TRACKING"""
    
    # Load calibration data
    base_dir = "data/Emanuel/recordings/Calibration/experiments/1"
    rest_data = np.load(os.path.join(base_dir, 'calib_rest_emg.npy'))
    mvc_data = np.load(os.path.join(base_dir, 'calib_mvc_emg.npy'))
    rest_timestamps = np.load(os.path.join(base_dir, 'calib_rest_timestamps.npy'))
    mvc_timestamps = np.load(os.path.join(base_dir, 'calib_mvc_timestamps.npy'))
    
    # Load TASK data
    task_data = np.load("C:/Users/Emanuel Wicki/Documents/MIT/biomech_PCP/data/Emanuel/recordings/mrpFlEx/experiments/1/raw_emg.npy")
    task_timestamps = np.load("C:/Users/Emanuel Wicki/Documents/MIT/biomech_PCP/data/Emanuel/recordings/mrpFlEx/experiments/1/raw_timestamps.npy")
    
    # Calculate sampling frequencies
    sf_rest = (len(rest_timestamps) - 1) / (rest_timestamps[-1] - rest_timestamps[0])
    sf_mvc = (len(mvc_timestamps) - 1) / (mvc_timestamps[-1] - mvc_timestamps[0])
    sf_task = (len(task_timestamps) - 1) / (task_timestamps[-1] - task_timestamps[0])
    
    print("=== TASK-BASED EMG NOISE ESTIMATION WITH ZERO TRACKING ===\n")
    print("Using finger flexion task data for realistic noise estimation!")
    print("Now with comprehensive zero tracking at every step!\n")
    
    print(f"Data loaded:")
    print(f"  Rest: {rest_data.shape} at {sf_rest:.1f} Hz")
    print(f"  MVC: {mvc_data.shape} at {sf_mvc:.1f} Hz") 
    print(f"  Task: {task_data.shape} at {sf_task:.1f} Hz")
    print(f"  Task duration: {task_data.shape[0]/sf_task:.1f} seconds\n")
    
    # Define channels of interest for tracking
    channels_of_interest = [0, 1, 2, 4, 12, 13, 14, 15]
    
    # Process MVC data first (for reference)
    print("Processing MVC data (for reference scaling):")
    mvc_processed = process_emg_correct_order_WITH_TRACKING(
        mvc_data, sf_mvc, 
        show_steps=False, 
        channels_to_show=channels_of_interest,
        track_zeros=True
    )
    
    # Process task data with correct pipeline
    print("\n" + "="*80)
    print("Processing TASK data (finger flexion) with ZERO TRACKING:")
    task_processed = process_emg_correct_order_WITH_TRACKING(
        task_data, sf_task, 
        show_steps=True, 
        channels_to_show=channels_of_interest,
        track_zeros=True
    )

    # Get the filtered data (before noise subtraction for noise estimation)
    task_rectified = task_processed['processing_steps']['4_rectified']
    mvc_rectified = mvc_processed['processing_steps']['4_rectified']
    
    # Process rest data for validation
    print("\n" + "="*80)
    print("Processing REST data (for validation) with ZERO TRACKING:")
    rest_processed = process_emg_correct_order_WITH_TRACKING(
        rest_data, sf_rest, 
        show_steps=False, 
        channels_to_show=channels_of_interest,
        track_zeros=True
    )
    rest_rectified = rest_processed['processing_steps']['4_rectified']
    
    # Estimate noise using task data quiet periods with rest validation
    print("\n=== TASK-BASED NOISE ESTIMATION ===")
    task_noise_levels, noise_info = estimate_noise_from_task_data(
        task_rectified, sf_task, mvc_rectified, rest_rectified, method='quiet_periods'
    )
    
    # Compare with rest-based estimation
    print("\n=== VALIDATION: Rest Data Zero Percentages ===")
    rest_noise_levels = rest_processed['noise_levels']
    
    print(f"Checking rest data denoising with different methods:")
    print(f"{'Channel':<8} {'Task-Based':<12} {'Rest Zeros':<12} {'Target: 95-98%':<15}")
    print("-" * 55)
    
    for ch in channels_of_interest:
        task_noise = task_noise_levels[ch]
        
        # Calculate zero percentage with task-based threshold
        rest_signal = rest_rectified[ch, :]
        rest_zeros = np.sum(rest_signal <= task_noise) / len(rest_signal) * 100
        
        status = "✅" if 95 <= rest_zeros <= 98 else "⚠️" if rest_zeros >= 90 else "❌"
        
        print(f"{ch:<8} {task_noise:<12.1f} {rest_zeros:<12.1f}% {status}")
    
    print(f"\n=== COMPARISON: Task vs Rest Noise Estimation ===")
    print(f"{'Channel':<8} {'Rest-Based':<12} {'Task-Based':<12} {'Difference':<12} {'Recommendation':<15}")
    print("-" * 70)
    
    for ch in channels_of_interest:
        rest_noise = rest_noise_levels[ch]
        task_noise = task_noise_levels[ch]
        diff_ratio = rest_noise / task_noise if task_noise > 0 else float('inf')
        
        if task_noise < rest_noise * 0.5:
            recommendation = "Use Task"
        elif task_noise > rest_noise * 2:
            recommendation = "Use Rest"
        else:
            recommendation = "Similar"
        
        print(f"{ch:<8} {rest_noise:<12.1f} {task_noise:<12.1f} {diff_ratio:<12.1f}x {recommendation:<15}")
    
    # Final recommendation
    print(f"\n=== FINAL NOISE LEVELS (RECOMMENDED) ===")
    final_noise_levels = task_noise_levels.copy()
    
    print(f"Using task-based noise estimation:")
    print(f"Quiet periods method found: {noise_info.get('num_periods', 'N/A')} periods")
    print(f"Total quiet duration: {noise_info.get('quiet_duration_s', 'N/A'):.1f}s")
    print(f"Rest validation applied: {noise_info.get('rest_validation', False)}")
    print(f"\nFinal recommended noise levels:")
    
    for ch in channels_of_interest:
        # Calculate final rest zero percentage
        rest_signal = rest_rectified[ch, :]
        expected_rest_zeros = np.sum(rest_signal <= final_noise_levels[ch]) / len(rest_signal) * 100
        
        # Calculate task zero percentage for comparison
        task_signal = task_rectified[ch, :]
        expected_task_zeros = np.sum(task_signal <= final_noise_levels[ch]) / len(task_signal) * 100
        
        print(f"  Channel {ch}: {final_noise_levels[ch]:.1f}")
        print(f"    Rest: {expected_rest_zeros:.1f}% zeros (target: ~96%)")
        print(f"    Task: {expected_task_zeros:.1f}% zeros")
    
    # Save results
    results = {
        'final_noise_levels': final_noise_levels,
        'task_noise_levels': task_noise_levels,
        'rest_noise_levels': rest_noise_levels,
        'noise_estimation_info': noise_info,
        'sampling_frequencies': {
            'rest': sf_rest,
            'mvc': sf_mvc, 
            'task': sf_task
        },
        'channels_analyzed': channels_of_interest,
        'zero_tracking_results': {
            'task': task_processed['zero_tracking'],
            'rest': rest_processed['zero_tracking'],
            'mvc': mvc_processed['zero_tracking']
        }
    }
    

    mapped_channels = [0, 1, 2, 3, 4, 5, 6, 7]
    labels = [
        'EMG 0 (mapped 0)', 'EMG 1 (mapped 1)', 'EMG 2 (mapped 2)', 'EMG 4 (mapped 3)',
        'EMG 12 (mapped 4)', 'EMG 13 (mapped 5)', 'EMG 14 (mapped 6)', 'EMG 15 (mapped 7)'
    ]
    labels = ['EMG 0 (mapped 0)']

    mvc_data = mvc_processed['processed_data']      # shape: [channels, samples]

    # Use absolute value in case of negative values (rectified/enveloped EMG)
    mvc_95th = np.percentile(mvc_data, 95, axis=1)  # shape: [channels]
    mvc_95th[mvc_95th == 0] = 1e-8
    task_data = task_processed['processed_data']
    task_time = task_processed['time_axis']

    task_data_60hz, task_time_60hz = downsample_to_60hz(task_data, task_time, target_freq=60.0)

    angle_time = np.load("C:/Users/Emanuel Wicki/Documents/MIT/biomech_PCP/data/Emanuel/recordings/mrpFlEx/experiments/1/angle_timestamps.npy")
    angle_data = np.load("C:/Users/Emanuel Wicki/Documents/MIT/biomech_PCP/data/Emanuel/recordings/mrpFlEx/experiments/1/angles.npy")
    angle_data_T = angle_data.T

    print("angle_data shape:", angle_data.shape)
    print("angle_time shape:", angle_time.shape)
    
    angle_data_60hz, angle_time_60hz = downsample_to_60hz(angle_data_T, angle_time, target_freq=60.0)

    emg_aligned = interpolate_emg_to_angle_time(task_data_60hz, task_time_60hz, angle_time_60hz)
    print("emg aligned shape:", emg_aligned.shape)
    emg_norm = emg_aligned / mvc_95th[:, None]

    plt.figure(figsize=(15, 6))
    for ch in range(emg_norm.shape[0]):
        plt.plot(angle_time_60hz, emg_norm[ch], label=f'EMG {ch}', alpha=0.8)
    plt.xlabel('Time (s)')
    plt.ylabel('EMG (a.u.)')
    plt.title('All EMG Channels (Downsampled & Aligned to Angle Time)')
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.show()


    return results, task_processed, mvc_processed

if __name__ == "__main__":
    task_processed, rest_processed, mvc_processed = main()



