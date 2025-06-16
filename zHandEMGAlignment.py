#!/usr/bin/env python3
"""
Comprehensive EMG-Prosthetic Hand Data Alignment Script
Performs iteration-aware alignment with extensive diagnostics and visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
import os
import argparse
from pathlib import Path
import yaml


class IterationAligner:
    """Main class for aligning EMG and prosthetic hand data using iteration structure"""
    
    def __init__(self, data_dir, expected_iterations=40):
        self.data_dir = data_dir
        self.expected_iterations = expected_iterations
        self.diagnostics = {}
        
        # Define channel mappings
        # self.used_emg_channels = [0, 1, 2, 4, 12, 13, 14, 15]  # Actual channel indices to use
        self.used_emg_channels = [0]  # Actual channel indices to use
        
        # Define angle column indices (from header)
        # index_Pos=26, middle_Pos=27, ring_Pos=28, pinky_Pos=29, thumbFlex_Pos=30, thumbRot_Pos=31
        self.angle_indices = list(range(26, 32))  # Columns 26-31 for the 6 joint positions
        self.angle_names = ['index_Pos', 'middle_Pos', 'ring_Pos', 'pinky_Pos', 'thumbFlex_Pos', 'thumbRot_Pos']
        
    def load_data(self):
        """Load raw EMG and angle data"""
        print("\n=== Loading Data ===")
        
        # Load EMG data - it's already in correct shape [samples, 16 channels]
        emg_all = np.load(os.path.join(self.data_dir, 'raw_emg.npy'))
        # Select only used channels and transpose to [channels, samples]
        self.emg_raw = emg_all[:, self.used_emg_channels].T
        self.emg_timestamps = np.load(os.path.join(self.data_dir, 'raw_timestamps.npy'))
        
        print(f"Loaded EMG data: {emg_all.shape} -> selected {len(self.used_emg_channels)} channels -> {self.emg_raw.shape}")
        
        # Load angle data
        angles_all = np.load(os.path.join(self.data_dir, 'angles.npy'))
        # Extract only the actual position readings (columns 26-31)
        self.angles_raw = angles_all[:, self.angle_indices]
        self.angles_timestamps = np.load(os.path.join(self.data_dir, 'angle_timestamps.npy'))
        
        print(f"Loaded angle data: {angles_all.shape} -> selected columns {self.angle_indices} -> {self.angles_raw.shape}")
        print(f"Angle columns: {self.angle_names}")
        
        # Basic info
        self.emg_duration = self.emg_timestamps[-1] - self.emg_timestamps[0]
        self.angles_duration = self.angles_timestamps[-1] - self.angles_timestamps[0]
        self.emg_fs = len(self.emg_timestamps) / self.emg_duration
        self.angles_fs = len(self.angles_timestamps) / self.angles_duration
        
        print(f"EMG: {self.emg_raw.shape[1]} samples, {self.emg_duration:.2f}s, {self.emg_fs:.1f} Hz")
        print(f"Angles: {len(self.angles_timestamps)} samples, {self.angles_duration:.2f}s, {self.angles_fs:.1f} Hz")
        print(f"Duration difference: {abs(self.emg_duration - self.angles_duration):.3f}s")
        
        # Store diagnostics
        self.diagnostics['initial'] = {
            'emg_samples': self.emg_raw.shape[1],
            'emg_channels': len(self.used_emg_channels),
            'angle_samples': len(self.angles_timestamps),
            'angle_joints': len(self.angle_names),
            'emg_duration': self.emg_duration,
            'angles_duration': self.angles_duration,
            'duration_diff': abs(self.emg_duration - self.angles_duration)
        }
        
    def detect_iterations(self, data, timestamps, data_type='angles'):
        """Detect iteration boundaries using movement patterns"""
        print(f"\n=== Detecting Iterations in {data_type} ===")
        
        if data_type == 'angles':
            # For angles: use velocity to detect movement
            velocities = np.abs(np.diff(data, axis=0))
            movement_signal = np.sum(velocities, axis=1)
        else:  # EMG
            # For EMG: use total activity
            movement_signal = np.sum(data, axis=0)
        
        # Smooth the signal
        movement_smooth = gaussian_filter1d(movement_signal, sigma=int(0.1 * len(movement_signal) / timestamps[-1]))
        
        # Normalize
        movement_norm = (movement_smooth - np.min(movement_smooth)) / (np.max(movement_smooth) - np.min(movement_smooth))
        
        # Find peaks (middle of movements) and valleys (rest positions)
        # Use adaptive thresholds
        peak_threshold = np.percentile(movement_norm, 70)
        valley_threshold = np.percentile(movement_norm, 30)
        
        peaks, peak_props = signal.find_peaks(movement_norm, height=peak_threshold, distance=int(0.5 * len(movement_signal) / timestamps[-1]))
        valleys, valley_props = signal.find_peaks(-movement_norm, height=-valley_threshold, distance=int(0.5 * len(movement_signal) / timestamps[-1]))
        
        # Identify iteration structure
        # Each iteration should have: valley -> peak -> valley
        iterations = []
        
        for i in range(len(peaks)):
            # Find valleys before and after this peak
            valleys_before = valleys[valleys < peaks[i]]
            valleys_after = valleys[valleys > peaks[i]]
            
            if len(valleys_before) > 0 and len(valleys_after) > 0:
                start_idx = valleys_before[-1]
                end_idx = valleys_after[0]
                peak_idx = peaks[i]
                
                iterations.append({
                    'start_idx': start_idx,
                    'peak_idx': peak_idx,
                    'end_idx': end_idx,
                    'start_time': timestamps[start_idx] if data_type == 'angles' else timestamps[start_idx],
                    'peak_time': timestamps[peak_idx] if data_type == 'angles' else timestamps[peak_idx],
                    'end_time': timestamps[end_idx] if data_type == 'angles' else timestamps[end_idx],
                    'duration': timestamps[end_idx] - timestamps[start_idx],
                    'peak_height': movement_norm[peak_idx]
                })
        
        print(f"Detected {len(iterations)} iterations")
        
        # Filter out anomalous iterations
        if len(iterations) > 0:
            durations = [it['duration'] for it in iterations]
            median_duration = np.median(durations)
            std_duration = np.std(durations)
            
            # Keep iterations within 2 std of median duration
            valid_iterations = [
                it for it in iterations 
                if abs(it['duration'] - median_duration) < 2 * std_duration
            ]
            
            print(f"After filtering: {len(valid_iterations)} valid iterations")
            print(f"Median duration: {median_duration:.3f}s ± {std_duration:.3f}s")
            
            return valid_iterations, movement_norm, peaks, valleys
        
        return iterations, movement_norm, peaks, valleys
    
    def find_iteration_correspondence(self, emg_iterations, angle_iterations):
        """Match iterations between EMG and angle data"""
        print("\n=== Finding Iteration Correspondence ===")
        
        correspondences = []
        
        # Strategy: Use the last iterations (most reliable) to establish timing relationship
        n_reference = min(10, len(emg_iterations), len(angle_iterations))
        
        # Use last n_reference iterations
        emg_ref = emg_iterations[-n_reference:]
        angle_ref = angle_iterations[-n_reference:]
        
        # Calculate time offsets for each pair
        offsets = []
        for emg_it, angle_it in zip(emg_ref, angle_ref):
            # Compare peak times
            offset = emg_it['peak_time'] - angle_it['peak_time']
            offsets.append(offset)
            
            correspondences.append({
                'emg_iteration': emg_it,
                'angle_iteration': angle_it,
                'time_offset': offset
            })
        
        # Calculate statistics
        offset_mean = np.mean(offsets)
        offset_std = np.std(offsets)
        
        print(f"Time offset: {offset_mean*1000:.1f} ± {offset_std*1000:.1f} ms")
        
        # Check for linear drift
        if len(offsets) > 3:
            iteration_indices = np.arange(len(offsets))
            slope, intercept, r_value, _, _ = stats.linregress(iteration_indices, offsets)
            print(f"Timing drift: {slope*1000:.3f} ms/iteration (R²={r_value**2:.3f})")
            
            self.diagnostics['drift'] = {
                'slope_ms_per_iter': slope * 1000,
                'intercept_ms': intercept * 1000,
                'r_squared': r_value**2
            }
        
        return correspondences, offset_mean, offset_std
    
    def align_signals(self, offset_mean, offset_std):
        """Apply the calculated alignment"""
        print("\n=== Applying Alignment ===")
        
        # Apply offset to EMG timestamps
        self.emg_timestamps_aligned = self.emg_timestamps - offset_mean
        
        # Find overlapping time region
        start_time = max(self.emg_timestamps_aligned[0], self.angles_timestamps[0])
        end_time = min(self.emg_timestamps_aligned[-1], self.angles_timestamps[-1])
        
        # Trim to overlapping region
        emg_mask = (self.emg_timestamps_aligned >= start_time) & (self.emg_timestamps_aligned <= end_time)
        angle_mask = (self.angles_timestamps >= start_time) & (self.angles_timestamps <= end_time)
        
        self.emg_aligned = self.emg_raw[:, emg_mask]
        self.emg_timestamps_final = self.emg_timestamps_aligned[emg_mask]
        self.angles_aligned = self.angles_raw[angle_mask]
        self.angles_timestamps_final = self.angles_timestamps[angle_mask]
        
        print(f"Aligned duration: {end_time - start_time:.2f}s")
        print(f"EMG samples: {self.emg_aligned.shape[1]}")
        print(f"Angle samples: {len(self.angles_aligned)}")
        
        # Calculate final correlation
        self.calculate_correlation()
        
    def calculate_correlation(self):
        """Calculate correlation between aligned signals"""
        # Create comparable signals
        emg_activity = np.sum(self.emg_aligned, axis=0)
        angle_movement = np.sum(np.abs(np.diff(self.angles_aligned, axis=0)), axis=1)
        
        # Interpolate to common timebase
        common_time = np.linspace(
            max(self.emg_timestamps_final[0], self.angles_timestamps_final[1]),
            min(self.emg_timestamps_final[-1], self.angles_timestamps_final[-1]),
            1000
        )
        
        emg_interp = np.interp(common_time, self.emg_timestamps_final, emg_activity)
        angle_interp = np.interp(common_time, self.angles_timestamps_final[1:], angle_movement)
        
        # Normalize
        emg_norm = (emg_interp - np.mean(emg_interp)) / np.std(emg_interp)
        angle_norm = (angle_interp - np.mean(angle_interp)) / np.std(angle_interp)
        
        # Calculate correlation
        correlation = np.corrcoef(emg_norm, angle_norm)[0, 1]
        
        self.diagnostics['correlation'] = correlation
        print(f"Final correlation: {correlation:.3f}")
        
        return correlation
    
    def visualize_alignment(self, save_path=None):
        """Create comprehensive visualization of alignment results"""
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Raw data overview
        ax1 = plt.subplot(4, 2, 1)
        ax1.plot(self.emg_timestamps, np.sum(self.emg_raw, axis=0), 'b-', alpha=0.5, label='EMG (8 channels)')
        ax1.set_title('Raw EMG Activity')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Total EMG')
        ax1.legend()
        
        ax2 = plt.subplot(4, 2, 2)
        angle_vel = np.sum(np.abs(np.diff(self.angles_raw, axis=0)), axis=1)
        ax2.plot(self.angles_timestamps[1:], angle_vel, 'r-', alpha=0.5, label='Angles (6 joints)')
        ax2.set_title('Raw Angle Movement')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Total Movement')
        ax2.legend()
        
        # 2. Iteration detection
        ax3 = plt.subplot(4, 2, 3)
        emg_iterations, emg_signal, emg_peaks, emg_valleys = self.detect_iterations(self.emg_raw, self.emg_timestamps, 'emg')
        ax3.plot(self.emg_timestamps, emg_signal, 'b-', alpha=0.7)
        ax3.plot(self.emg_timestamps[emg_peaks], emg_signal[emg_peaks], 'bo', markersize=8)
        ax3.plot(self.emg_timestamps[emg_valleys], emg_signal[emg_valleys], 'bs', markersize=6)
        ax3.set_title(f'EMG Iterations ({len(emg_iterations)} detected)')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Normalized Activity')
        
        ax4 = plt.subplot(4, 2, 4)
        angle_iterations, angle_signal, angle_peaks, angle_valleys = self.detect_iterations(self.angles_raw, self.angles_timestamps, 'angles')
        ax4.plot(self.angles_timestamps[1:], angle_signal, 'r-', alpha=0.7)
        ax4.plot(self.angles_timestamps[angle_peaks], angle_signal[angle_peaks], 'ro', markersize=8)
        ax4.plot(self.angles_timestamps[angle_valleys], angle_signal[angle_valleys], 'rs', markersize=6)
        ax4.set_title(f'Angle Iterations ({len(angle_iterations)} detected)')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Normalized Movement')
        
        # 3. Iteration correspondence
        ax5 = plt.subplot(4, 2, 5)
        correspondences, offset_mean, offset_std = self.find_iteration_correspondence(emg_iterations, angle_iterations)
        iteration_numbers = range(len(correspondences))
        offsets = [c['time_offset'] * 1000 for c in correspondences]
        ax5.plot(iteration_numbers, offsets, 'go-', markersize=8)
        ax5.axhline(y=offset_mean * 1000, color='k', linestyle='--', label=f'Mean: {offset_mean*1000:.1f} ms')
        ax5.fill_between(iteration_numbers, 
                         (offset_mean - offset_std) * 1000, 
                         (offset_mean + offset_std) * 1000, 
                         alpha=0.3, color='gray')
        ax5.set_title('Time Offset per Iteration')
        ax5.set_xlabel('Iteration (from end)')
        ax5.set_ylabel('Offset (ms)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 4. Aligned signals
        ax6 = plt.subplot(4, 2, 6)
        # Apply alignment for visualization
        self.align_signals(offset_mean, offset_std)
        
        emg_norm = np.sum(self.emg_aligned, axis=0)
        emg_norm = (emg_norm - np.mean(emg_norm)) / np.std(emg_norm)
        angle_norm = np.sum(np.abs(np.diff(self.angles_aligned, axis=0)), axis=1)
        angle_norm = (angle_norm - np.mean(angle_norm)) / np.std(angle_norm)
        
        ax6.plot(self.emg_timestamps_final, emg_norm, 'b-', alpha=0.7, label='EMG')
        ax6.plot(self.angles_timestamps_final[1:], angle_norm, 'r-', alpha=0.7, label='Angles')
        ax6.set_title(f'Aligned Signals (correlation: {self.diagnostics["correlation"]:.3f})')
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Normalized Signal')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 5. Per-channel view
        ax7 = plt.subplot(4, 2, 7)
        # Show individual EMG channels
        for i, ch_idx in enumerate(self.used_emg_channels[:4]):  # Show first 4 channels
            channel_data = self.emg_aligned[i, :] / np.max(np.abs(self.emg_aligned[i, :])) + i * 2
            ax7.plot(self.emg_timestamps_final, channel_data, label=f'Ch {ch_idx}')
        ax7.set_title('Individual EMG Channels (normalized)')
        ax7.set_xlabel('Time (s)')
        ax7.set_ylabel('Channel (offset for clarity)')
        ax7.legend()
        
        # 6. Per-joint view
        ax8 = plt.subplot(4, 2, 8)
        # Show individual joint angles
        for i, joint_name in enumerate(self.angle_names[:4]):  # Show first 4 joints
            joint_data = self.angles_aligned[:, i] / np.max(np.abs(self.angles_aligned[:, i])) + i * 2
            ax8.plot(self.angles_timestamps_final, joint_data, label=joint_name)
        ax8.set_title('Individual Joint Angles (normalized)')
        ax8.set_xlabel('Time (s)')
        ax8.set_ylabel('Joint (offset for clarity)')
        ax8.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        
        plt.show()
        
    def generate_report(self):
        """Generate comprehensive alignment report"""
        print("\n" + "="*60)
        print("ALIGNMENT REPORT")
        print("="*60)
        
        print("\n1. Data Summary:")
        print(f"   - EMG: {self.diagnostics['initial']['emg_samples']} samples, "
              f"{self.diagnostics['initial']['emg_duration']:.2f}s")
        print(f"   - Angles: {self.diagnostics['initial']['angle_samples']} samples, "
              f"{self.diagnostics['initial']['angles_duration']:.2f}s")
        print(f"   - Duration difference: {self.diagnostics['initial']['duration_diff']:.3f}s")
        
        if 'drift' in self.diagnostics:
            print("\n2. Timing Analysis:")
            print(f"   - Average offset: {self.diagnostics['drift']['intercept_ms']:.1f} ms")
            print(f"   - Drift rate: {self.diagnostics['drift']['slope_ms_per_iter']:.3f} ms/iteration")
            print(f"   - Drift linearity (R²): {self.diagnostics['drift']['r_squared']:.3f}")
        
        print("\n3. Alignment Quality:")
        print(f"   - Final correlation: {self.diagnostics['correlation']:.3f}")
        
        if self.diagnostics['correlation'] > 0.8:
            print("   - Status: EXCELLENT alignment")
        elif self.diagnostics['correlation'] > 0.6:
            print("   - Status: GOOD alignment")
        elif self.diagnostics['correlation'] > 0.4:
            print("   - Status: ACCEPTABLE alignment")
        else:
            print("   - Status: POOR alignment - manual inspection recommended")
        
        print("\n" + "="*60)
        
    def save_aligned_data(self, output_dir=None):
        """Save aligned data"""
        if output_dir is None:
            output_dir = self.data_dir
        
        # Save aligned data
        np.save(os.path.join(output_dir, 'emg_aligned.npy'), self.emg_aligned)
        np.save(os.path.join(output_dir, 'emg_timestamps_aligned.npy'), self.emg_timestamps_final)
        np.save(os.path.join(output_dir, 'angles_aligned.npy'), self.angles_aligned)
        np.save(os.path.join(output_dir, 'angles_timestamps_aligned.npy'), self.angles_timestamps_final)
        
        # Save diagnostics
        with open(os.path.join(output_dir, 'alignment_diagnostics.yaml'), 'w') as f:
            yaml.dump(self.diagnostics, f)
        
        print(f"\nSaved aligned data to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Align EMG and prosthetic hand data using iteration structure')
    # parser.add_argument('data_dir', help='Directory containing raw_emg.npy, raw_timestamps.npy, angles.npy, angle_timestamps.npy')
    parser.add_argument('--iterations', type=int, default=40, help='Expected number of iterations')
    # parser.add_argument('--output_dir', help='Output directory (default: same as input)')
    parser.add_argument('--no_plot', action='store_true', help='Skip visualization')
    parser.add_argument('--save_plot', help='Save plot to file')
    
    args = parser.parse_args()
    data_dir = "data/GG/recordings/handOpCl/experiments/1"
    output_dir = os.getcwd()
    
    # Create aligner
    aligner = IterationAligner(data_dir, args.iterations)
    
    # Run alignment process
    aligner.load_data()
    
    # Detect iterations in both signals
    emg_iterations, _, _, _ = aligner.detect_iterations(aligner.emg_raw, aligner.emg_timestamps, 'emg')
    angle_iterations, _, _, _ = aligner.detect_iterations(aligner.angles_raw, aligner.angles_timestamps, 'angles')
    
    # Find correspondence and align
    correspondences, offset_mean, offset_std = aligner.find_iteration_correspondence(emg_iterations, angle_iterations)
    aligner.align_signals(offset_mean, offset_std)
    
    # Generate report
    aligner.generate_report()
    
    # Visualize
    if not args.no_plot:
        aligner.visualize_alignment(args.save_plot)
    
    # Save aligned data
    output_dir = args.output_dir if args.output_dir else args.data_dir
    aligner.save_aligned_data(output_dir)


if __name__ == "__main__":
    main()
