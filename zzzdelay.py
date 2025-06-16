# Debug script to analyze your EMG-angle alignment issues
import numpy as np
import matplotlib.pyplot as plt
from advanced_emg_alignment_debug import *

def debug_emg_alignment(data_dir):
    """
    Main debugging function to analyze your specific data.
    
    Args:
        data_dir: Path to experiment directory with aligned data
    """
    print("Loading aligned data...")
    
    # Load your processed data
    emg_data = np.load(f'{data_dir}/aligned_filtered_emg.npy')  # [samples, channels]
    angle_data = np.load(f'{data_dir}/aligned_angles.npy')     # [samples, angles]
    timestamps = np.load(f'{data_dir}/aligned_timestamps.npy')
    
    print(f"Data shapes: EMG {emg_data.shape}, Angles {angle_data.shape}, Time {timestamps.shape}")
    print(f"Recording duration: {timestamps[-1] - timestamps[0]:.1f} seconds")
    
    # Run comprehensive analysis
    print("\nRunning comprehensive analysis...")
    muscle_analysis, drift_analysis, issues = plot_comprehensive_analysis(
        emg_data, angle_data, timestamps
    )
    
    # Print detailed results
    print("\n" + "="*60)
    print("MUSCLE FUNCTION ANALYSIS")
    print("="*60)
    
    for ch, analysis in muscle_analysis.items():
        if np.std(emg_data[:, ch]) > 0.01:  # Only show active channels
            print(f"\nChannel {ch}: {analysis['muscle_type']}")
            print(f"  Position correlation: {analysis['position_correlation']:.3f}")
            print(f"  Velocity correlation: {analysis['velocity_correlation']:.3f}")
            print(f"  Peak/Valley ratio: {analysis['peak_valley_ratio']:.2f}")
    
    print("\n" + "="*60)
    print("TEMPORAL DRIFT ANALYSIS")
    print("="*60)
    
    active_channels = [ch for ch in range(emg_data.shape[1]) 
                      if np.std(emg_data[:, ch]) > 0.01]
    
    for ch in active_channels:
        lags = np.array(drift_analysis['lags_per_channel'][ch])
        if len(lags) > 0:
            print(f"\nChannel {ch}:")
            print(f"  Mean lag: {np.mean(lags):.3f}s")
            print(f"  Lag std: {np.std(lags):.3f}s")
            print(f"  Lag range: {np.max(lags) - np.min(lags):.3f}s")
            
            # Check for systematic drift
            window_centers = np.array(drift_analysis['window_centers'])
            if len(lags) > 3:
                slope = np.polyfit(window_centers, lags, 1)[0]
                print(f"  Drift rate: {slope*1000:.1f} ms/s")
    
    print("\n" + "="*60)
    print("DETECTED ISSUES")
    print("="*60)
    
    if issues:
        for issue in issues:
            print(f"⚠️  {issue}")
    else:
        print("✅ No major issues detected!")
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    recommendations = recommend_alignment_strategy(muscle_analysis, drift_analysis, issues)
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    return muscle_analysis, drift_analysis, issues

def create_muscle_specific_alignment(emg_data, angle_data, timestamps, muscle_analysis):
    """
    Create different alignments for different muscle types.
    """
    print("\nCreating muscle-type-specific alignments...")
    
    # Separate channels by function
    agonist_channels = [ch for ch, data in muscle_analysis.items() 
                       if 'Agonist' in data['muscle_type']]
    antagonist_channels = [ch for ch, data in muscle_analysis.items() 
                          if 'Antagonist' in data['muscle_type']]
    
    aligned_data = {}
    
    if agonist_channels:
        print(f"Agonist channels {agonist_channels}: applying +100ms lead")
        # Agonist muscles should predict movement - add lead time
        lead_time = 0.1  # 100ms lead
        fs = 1.0 / np.mean(np.diff(timestamps))
        lead_samples = int(lead_time * fs)
        
        # Shift EMG forward in time (predict future angles)
        if lead_samples < len(emg_data):
            agonist_emg = emg_data[:-lead_samples, agonist_channels]
            agonist_angles = angle_data[lead_samples:, :]
            agonist_times = timestamps[lead_samples:] - lead_time
            
            aligned_data['agonist'] = {
                'emg': agonist_emg,
                'angles': agonist_angles,
                'timestamps': agonist_times,
                'channels': agonist_channels
            }
    
    if antagonist_channels:
        print(f"Antagonist channels {antagonist_channels}: no lead time")
        # Antagonist muscles - keep current alignment or slight lag
        aligned_data['antagonist'] = {
            'emg': emg_data[:, antagonist_channels],
            'angles': angle_data,
            'timestamps': timestamps,
            'channels': antagonist_channels
        }
    
    return aligned_data

# Example usage:
if __name__ == "__main__":
    # Replace with your actual data directory
    data_dir = "data/Emanuel6/recordings/indexFlEx/experiments/1"
    
    try:
        muscle_analysis, drift_analysis, issues = debug_emg_alignment(data_dir)
        
        # Create muscle-specific alignments if needed
        emg_data = np.load(f'{data_dir}/aligned_filtered_emg.npy')
        angle_data = np.load(f'{data_dir}/aligned_angles.npy')
        timestamps = np.load(f'{data_dir}/aligned_timestamps.npy')
        
        aligned_data = create_muscle_specific_alignment(
            emg_data, angle_data, timestamps, muscle_analysis
        )
        
        # Save the muscle-specific aligned data
        for muscle_type, data in aligned_data.items():
            np.save(f'{data_dir}/aligned_emg_{muscle_type}.npy', data['emg'])
            np.save(f'{data_dir}/aligned_angles_{muscle_type}.npy', data['angles'])
            np.save(f'{data_dir}/aligned_timestamps_{muscle_type}.npy', data['timestamps'])
            print(f"Saved {muscle_type} alignment data")
            
    except FileNotFoundError as e:
        print(f"Data files not found: {e}")
        print("Please check the data_dir path and ensure aligned data exists")
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()