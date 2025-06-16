#!/usr/bin/env python3
"""
Raw Timestamp Analysis Tool

Analyzes the raw timestamps from EMG and angle data to understand:
- Start time differences
- Duration differences  
- Sampling rate consistency
- Clock drift patterns
- Synchronization issues from s1.5 recording
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from pathlib import Path

def analyze_raw_timestamps(data_dir):
    """
    Analyze raw timestamps from both EMG and angle systems.
    """
    print(f"Analyzing raw timestamps in: {data_dir}")
    print("=" * 60)
    
    # Load raw data
    try:
        emg_timestamps = np.load(os.path.join(data_dir, 'raw_timestamps.npy'))
        angle_timestamps = np.load(os.path.join(data_dir, 'angle_timestamps.npy'))
        emg_data = np.load(os.path.join(data_dir, 'raw_emg.npy'))
        angle_data = np.load(os.path.join(data_dir, 'angles.npy'))
    except FileNotFoundError as e:
        print(f"❌ Missing file: {e}")
        return None
    
    print(f"📊 DATA SHAPES:")
    print(f"   EMG data: {emg_data.shape} ({len(emg_timestamps)} timestamps)")
    print(f"   Angle data: {angle_data.shape} ({len(angle_timestamps)} timestamps)")
    
    # Basic timing analysis
    print(f"\n⏱️  BASIC TIMING:")
    
    # Start times
    emg_start = emg_timestamps[0]
    angle_start = angle_timestamps[0]
    start_diff = emg_start - angle_start
    
    print(f"   EMG start time: {emg_start:.6f}s")
    print(f"   Angle start time: {angle_start:.6f}s")
    print(f"   Start difference: {start_diff:.6f}s ({start_diff*1000:.3f}ms)")
    
    if abs(start_diff) > 0.1:
        print(f"   ⚠️  Large start time difference! Should be nearly simultaneous from s1.5")
    else:
        print(f"   ✅ Good synchronization start")
    
    # End times
    emg_end = emg_timestamps[-1]
    angle_end = angle_timestamps[-1]
    end_diff = emg_end - angle_end
    
    print(f"   EMG end time: {emg_end:.6f}s")
    print(f"   Angle end time: {angle_end:.6f}s")
    print(f"   End difference: {end_diff:.6f}s ({end_diff*1000:.3f}ms)")
    
    # Durations
    emg_duration = emg_end - emg_start
    angle_duration = angle_end - angle_start
    duration_diff = emg_duration - angle_duration
    
    print(f"\n📏 DURATIONS:")
    print(f"   EMG duration: {emg_duration:.6f}s")
    print(f"   Angle duration: {angle_duration:.6f}s")
    print(f"   Duration difference: {duration_diff:.6f}s ({duration_diff*1000:.3f}ms)")
    
    duration_ratio = emg_duration / angle_duration
    print(f"   Duration ratio (EMG/Angle): {duration_ratio:.8f}")
    
    if abs(duration_ratio - 1.0) > 0.01:
        print(f"   ⚠️  Significant duration difference - indicates clock drift!")
        print(f"   This {abs(duration_ratio - 1.0)*100:.3f}% difference over {emg_duration:.1f}s")
        print(f"   = {duration_diff*1000:.1f}ms total drift")
    else:
        print(f"   ✅ Durations are consistent")
    
    # Sampling rates
    print(f"\n📈 SAMPLING RATES:")
    
    emg_rate = (len(emg_timestamps) - 1) / emg_duration
    angle_rate = (len(angle_timestamps) - 1) / angle_duration
    
    print(f"   EMG sampling rate: {emg_rate:.3f} Hz")
    print(f"   Angle sampling rate: {angle_rate:.3f} Hz")
    
    # Check interval consistency
    emg_intervals = np.diff(emg_timestamps)
    angle_intervals = np.diff(angle_timestamps)
    
    print(f"\n📊 INTERVAL ANALYSIS:")
    print(f"   EMG intervals - Mean: {np.mean(emg_intervals)*1000:.3f}ms, Std: {np.std(emg_intervals)*1000:.3f}ms")
    print(f"   Angle intervals - Mean: {np.mean(angle_intervals)*1000:.3f}ms, Std: {np.std(angle_intervals)*1000:.3f}ms")
    
    # Expected intervals
    expected_emg_interval = 1.0 / 1000  # ~1ms for 1000Hz
    expected_angle_interval = 1.0 / 488  # ~2ms for 488Hz (from hardware test)
    
    emg_interval_error = abs(np.mean(emg_intervals) - expected_emg_interval) * 1000
    angle_interval_error = abs(np.mean(angle_intervals) - expected_angle_interval) * 1000
    
    print(f"   EMG interval error: {emg_interval_error:.3f}ms (expected ~1.0ms)")
    print(f"   Angle interval error: {angle_interval_error:.3f}ms (expected ~2.05ms)")
    
    # Look for timing drift over the recording
    print(f"\n🔄 DRIFT ANALYSIS:")
    
    # Calculate expected vs actual timestamps
    emg_expected = np.linspace(emg_start, emg_start + (len(emg_timestamps)-1)/1000, len(emg_timestamps))
    angle_expected = np.linspace(angle_start, angle_start + (len(angle_timestamps)-1)/488, len(angle_timestamps))
    
    emg_drift = emg_timestamps - emg_expected
    angle_drift = angle_timestamps - angle_expected
    
    print(f"   EMG drift over recording: {(emg_drift[-1] - emg_drift[0])*1000:.3f}ms")
    print(f"   Angle drift over recording: {(angle_drift[-1] - angle_drift[0])*1000:.3f}ms")
    
    # Linear drift rate
    if len(emg_timestamps) > 100:
        # Fit linear trend to drift
        time_points = np.linspace(0, emg_duration, len(emg_timestamps))
        emg_drift_rate = np.polyfit(time_points, emg_drift, 1)[0] * 1000  # ms/s
        
        time_points_angle = np.linspace(0, angle_duration, len(angle_timestamps))
        angle_drift_rate = np.polyfit(time_points_angle, angle_drift, 1)[0] * 1000  # ms/s
        
        print(f"   EMG drift rate: {emg_drift_rate:.3f} ms/s")
        print(f"   Angle drift rate: {angle_drift_rate:.3f} ms/s")
        
        if abs(emg_drift_rate) > 1:
            print(f"   ⚠️  EMG has significant clock drift!")
        if abs(angle_drift_rate) > 1:
            print(f"   ⚠️  Angle system has significant clock drift!")
    
    # Synchronization quality assessment
    print(f"\n🎯 SYNCHRONIZATION ASSESSMENT:")
    
    sync_quality = "✅ EXCELLENT"
    issues = []
    
    if abs(start_diff) > 0.1:
        sync_quality = "⚠️ POOR"
        issues.append(f"Large start difference ({start_diff*1000:.1f}ms)")
    
    if abs(duration_diff) > 0.5:
        sync_quality = "⚠️ POOR" 
        issues.append(f"Large duration difference ({duration_diff*1000:.1f}ms)")
    elif abs(duration_diff) > 0.1:
        sync_quality = "🔶 MODERATE"
        issues.append(f"Moderate duration difference ({duration_diff*1000:.1f}ms)")
    
    if abs(emg_drift_rate) > 5 or abs(angle_drift_rate) > 5:
        sync_quality = "⚠️ POOR"
        issues.append(f"High drift rates (EMG: {emg_drift_rate:.1f}, Angle: {angle_drift_rate:.1f} ms/s)")
    
    print(f"   Overall sync quality: {sync_quality}")
    if issues:
        print(f"   Issues found:")
        for issue in issues:
            print(f"     - {issue}")
    
    return {
        'emg_timestamps': emg_timestamps,
        'angle_timestamps': angle_timestamps,
        'start_diff': start_diff,
        'end_diff': end_diff,
        'duration_diff': duration_diff,
        'duration_ratio': duration_ratio,
        'emg_rate': emg_rate,
        'angle_rate': angle_rate,
        'emg_drift_rate': emg_drift_rate if 'emg_drift_rate' in locals() else 0,
        'angle_drift_rate': angle_drift_rate if 'angle_drift_rate' in locals() else 0,
        'sync_quality': sync_quality,
        'issues': issues
    }

def plot_timestamp_analysis(data_dir, analysis_results=None):
    """
    Create visualizations of timestamp analysis.
    """
    if analysis_results is None:
        analysis_results = analyze_raw_timestamps(data_dir)
        if analysis_results is None:
            return
    
    emg_timestamps = analysis_results['emg_timestamps']
    angle_timestamps = analysis_results['angle_timestamps']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Raw Timestamp Analysis: {os.path.basename(data_dir)}', fontsize=14)
    
    # Plot 1: Timeline comparison
    ax = axes[0, 0]
    
    # Show first 10 seconds
    mask_emg = emg_timestamps <= (emg_timestamps[0] + 10)
    mask_angle = angle_timestamps <= (angle_timestamps[0] + 10)
    
    ax.plot(emg_timestamps[mask_emg] - emg_timestamps[0], 
            np.ones(np.sum(mask_emg)), 'b.', markersize=1, alpha=0.5, label='EMG')
    ax.plot(angle_timestamps[mask_angle] - angle_timestamps[0], 
            np.ones(np.sum(mask_angle)) * 1.1, 'r.', markersize=2, alpha=0.7, label='Angles')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('System')
    ax.set_title('Timeline Comparison (First 10s)')
    ax.legend()
    ax.grid(True)
    
    # Plot 2: Sampling intervals
    ax = axes[0, 1]
    
    emg_intervals = np.diff(emg_timestamps) * 1000  # Convert to ms
    angle_intervals = np.diff(angle_timestamps) * 1000
    
    ax.hist(emg_intervals, bins=50, alpha=0.7, label=f'EMG (mean: {np.mean(emg_intervals):.3f}ms)', color='blue')
    ax.hist(angle_intervals, bins=50, alpha=0.7, label=f'Angles (mean: {np.mean(angle_intervals):.3f}ms)', color='red')
    
    ax.set_xlabel('Interval (ms)')
    ax.set_ylabel('Frequency')
    ax.set_title('Sampling Interval Distribution')
    ax.legend()
    ax.grid(True)
    
    # Plot 3: Cumulative time drift
    ax = axes[1, 0]
    
    # Calculate expected vs actual timestamps
    emg_expected = np.linspace(emg_timestamps[0], 
                               emg_timestamps[0] + (len(emg_timestamps)-1)/1000, 
                               len(emg_timestamps))
    angle_expected = np.linspace(angle_timestamps[0], 
                                 angle_timestamps[0] + (len(angle_timestamps)-1)/488, 
                                 len(angle_timestamps))
    
    emg_drift = (emg_timestamps - emg_expected) * 1000  # ms
    angle_drift = (angle_timestamps - angle_expected) * 1000  # ms
    
    emg_time = emg_timestamps - emg_timestamps[0]
    angle_time = angle_timestamps - angle_timestamps[0]
    
    ax.plot(emg_time, emg_drift, 'b-', alpha=0.7, label='EMG Drift')
    ax.plot(angle_time, angle_drift, 'r-', alpha=0.7, label='Angle Drift')
    
    ax.set_xlabel('Recording Time (s)')
    ax.set_ylabel('Cumulative Drift (ms)')
    ax.set_title('Clock Drift Over Time')
    ax.legend()
    ax.grid(True)
    
    # Plot 4: Duration comparison
    ax = axes[1, 1]
    
    emg_duration = emg_timestamps[-1] - emg_timestamps[0]
    angle_duration = angle_timestamps[-1] - angle_timestamps[0]
    
    durations = [emg_duration, angle_duration]
    labels = ['EMG', 'Angles']
    colors = ['blue', 'red']
    
    bars = ax.bar(labels, durations, color=colors, alpha=0.7)
    
    # Add values on bars
    for bar, duration in zip(bars, durations):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{duration:.3f}s', ha='center', va='bottom')
    
    ax.set_ylabel('Duration (s)')
    ax.set_title(f'Recording Durations\n(Difference: {abs(emg_duration - angle_duration)*1000:.1f}ms)')
    ax.grid(True)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(data_dir, 'timestamp_analysis.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Plot saved to: {plot_path}")
    
    plt.show()

def analyze_multiple_experiments(person_id, out_root='data', movement=None):
    """
    Analyze multiple experiments to see patterns.
    """
    recordings_dir = os.path.join(out_root, person_id, "recordings")
    
    if movement:
        movements = [movement]
    else:
        movements = [d for d in os.listdir(recordings_dir) 
                    if os.path.isdir(os.path.join(recordings_dir, d)) 
                    and d not in {"Calibration", "calibration"}]
    
    all_results = []
    
    print("="*80)
    print("MULTI-EXPERIMENT TIMESTAMP ANALYSIS")
    print("="*80)
    
    for movement in movements:
        experiments_dir = os.path.join(recordings_dir, movement, "experiments")
        if not os.path.exists(experiments_dir):
            continue
            
        for experiment in sorted(os.listdir(experiments_dir)):
            exp_dir = os.path.join(experiments_dir, experiment)
            if not os.path.isdir(exp_dir):
                continue
                
            required_files = ['raw_timestamps.npy', 'angle_timestamps.npy']
            if not all(os.path.exists(os.path.join(exp_dir, f)) for f in required_files):
                continue
            
            print(f"\n📁 {movement}/experiments/{experiment}")
            print("-" * 50)
            
            results = analyze_raw_timestamps(exp_dir)
            if results:
                results['movement'] = movement
                results['experiment'] = experiment
                results['path'] = exp_dir
                all_results.append(results)
    
    # Summary analysis
    if all_results:
        print("\n" + "="*80)
        print("SUMMARY ACROSS ALL EXPERIMENTS")
        print("="*80)
        
        start_diffs = [r['start_diff'] * 1000 for r in all_results]  # ms
        duration_diffs = [r['duration_diff'] * 1000 for r in all_results]  # ms
        emg_rates = [r['emg_rate'] for r in all_results]
        angle_rates = [r['angle_rate'] for r in all_results]
        
        print(f"📊 STATISTICS ACROSS {len(all_results)} EXPERIMENTS:")
        print(f"   Start differences: {np.mean(start_diffs):.1f} ± {np.std(start_diffs):.1f} ms")
        print(f"   Duration differences: {np.mean(duration_diffs):.1f} ± {np.std(duration_diffs):.1f} ms")
        print(f"   EMG sampling rates: {np.mean(emg_rates):.1f} ± {np.std(emg_rates):.1f} Hz")
        print(f"   Angle sampling rates: {np.mean(angle_rates):.1f} ± {np.std(angle_rates):.1f} Hz")
        
        # Identify problematic experiments
        problematic = [r for r in all_results if r['sync_quality'] == "⚠️ POOR"]
        if problematic:
            print(f"\n⚠️  PROBLEMATIC EXPERIMENTS ({len(problematic)}):")
            for r in problematic:
                print(f"   {r['movement']}/{r['experiment']}: {', '.join(r['issues'])}")
    
    return all_results

def main():
    parser = argparse.ArgumentParser(description='Analyze raw timestamps for synchronization issues')
    parser.add_argument('--data_dir', help='Specific experiment directory to analyze')
    parser.add_argument('--person_id', help='Person ID for multi-experiment analysis')
    parser.add_argument('--movement', help='Specific movement to analyze (optional)')
    parser.add_argument('--out_root', default='data', help='Root data directory')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    
    args = parser.parse_args()
    data_dir = "data/Emanuel6/recordings/indexFlEx/experiments/1"
    results = analyze_raw_timestamps(data_dir)
    if results and args.plot:
        plot_timestamp_analysis(args.data_dir, results)
    
    elif args.person_id:
        # Analyze multiple experiments
        analyze_multiple_experiments(args.person_id, args.out_root, args.movement)
    
    else:
        print("Please specify either --data_dir for single analysis or --person_id for multi-experiment analysis")

if __name__ == "__main__":
    main()