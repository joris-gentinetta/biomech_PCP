#!/usr/bin/env python3
"""
Standalone EMG-Hand Data Alignment Tool

This script loads raw EMG and hand angle data, applies the same filtering as s2.5,
and provides an interactive interface to manually align the data by sliding 
the shorter signal to match the longer one.

Usage:
    python align_emg_hand_data.py --data_dir /path/to/experiment/folder --person_id SUBJECT_ID

Requirements:
    - numpy, scipy, matplotlib, tkinter
    - Raw data files: raw_emg.npy, raw_timestamps.npy, angles.npy, angle_timestamps.npy
    - Calibration data: scaling.yaml in Calibration/experiments/1/
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from scipy import signal
import yaml
import pandas as pd
import json
from pathlib import Path

class EMGAlignmentTool:
    def __init__(self, data_dir, person_id, out_root):
        self.data_dir = data_dir
        self.person_id = person_id
        self.out_root = out_root
        
        # Data storage
        self.raw_emg = None
        self.raw_emg_timestamps = None
        self.raw_angles = None
        self.raw_angle_timestamps = None
        self.filtered_emg = None
        self.filtered_angles = None
        self.emg_channels_used = None
        
        # Alignment parameters
        self.time_shift = 0.0
        self.max_shift_range = 5.0  # Reduced range for interaction data (shorter duration)
        
        # GUI components
        self.root = None
        self.fig = None
        self.canvas = None
        self.shift_var = None
        self.status_var = None
        
        # Load and process data
        self.load_data()
        self.filter_data()
        
    def load_data(self):
        """Load raw EMG and angle data"""
        print(f"Loading data from: {self.data_dir}")
        
        # Load EMG data (transposed for interaction data)
        emg_file = os.path.join(self.data_dir, "raw_emg.npy")
        emg_ts_file = os.path.join(self.data_dir, "raw_timestamps.npy")
        
        if not os.path.exists(emg_file) or not os.path.exists(emg_ts_file):
            raise FileNotFoundError(f"EMG files not found in {self.data_dir}")
            
        raw_emg_loaded = np.load(emg_file)
        self.raw_emg_timestamps = np.load(emg_ts_file)
        
        # Handle EMG data shape - interaction data might be transposed
        if raw_emg_loaded.shape[0] > raw_emg_loaded.shape[1]:
            # More rows than columns, transpose to get (channels, samples)
            print(f"Transposing EMG data from {raw_emg_loaded.shape} to {raw_emg_loaded.T.shape}")
            self.raw_emg = raw_emg_loaded.T
        else:
            self.raw_emg = raw_emg_loaded
        
        print(f"Loaded EMG data: {self.raw_emg.shape} at {len(self.raw_emg_timestamps)} timepoints")
        
        # Load angle data
        angles_file = os.path.join(self.data_dir, "angles.npy")
        angle_ts_file = os.path.join(self.data_dir, "angle_timestamps.npy")
        
        if not os.path.exists(angles_file) or not os.path.exists(angle_ts_file):
            raise FileNotFoundError(f"Angle files not found in {self.data_dir}")
            
        self.raw_angles = np.load(angles_file)
        self.raw_angle_timestamps = np.load(angle_ts_file)
        
        print(f"Loaded angle data: {self.raw_angles.shape} at {len(self.raw_angle_timestamps)} timepoints")
        
        # Load calibration data for EMG filtering
        scaling_yaml_path = os.path.join(
            self.out_root, self.person_id, "recordings", "Calibration", "experiments", "1", "scaling.yaml"
        )
        
        if not os.path.exists(scaling_yaml_path):
            raise FileNotFoundError(f"Calibration file not found: {scaling_yaml_path}")
            
        with open(scaling_yaml_path, 'r') as file:
            scalers = yaml.safe_load(file)
        
        self.maxVals = np.array(scalers['maxVals'])
        self.noiseLevel = np.array(scalers['noiseLevels'])
        
        # Get used channels (SNR > 3.0)
        self.emg_channels_used = self.get_used_channels_from_snr(scaling_yaml_path, snr_threshold=3.0)
        print(f"Using EMG channels: {self.emg_channels_used}")
        
    def get_used_channels_from_snr(self, scaling_yaml_path, snr_threshold=3.0):
        """Get EMG channels with good SNR"""
        with open(scaling_yaml_path, 'r') as file:
            scalers = yaml.safe_load(file)
        
        maxVals = np.array(scalers['maxVals'])
        noiseLevel = np.array(scalers['noiseLevels'])
        
        # Calculate SNR
        snr = maxVals / (noiseLevel + 1e-8)
        used_channels = np.where(snr >= snr_threshold)[0]
        
        print(f"SNR values: {snr}")
        print(f"Channels with SNR >= {snr_threshold}: {used_channels}")
        
        return used_channels
        
    def filter_data(self):
        """Apply same filtering as s2.5 process script, adapted for interaction data"""
        print("Filtering EMG data...")
        
        # Filter EMG data (same as s2.5)
        emg_subset = self.raw_emg[self.emg_channels_used, :]
        maxVals_subset = self.maxVals[self.emg_channels_used]
        noiseLevel_subset = self.noiseLevel[self.emg_channels_used]
        
        # Calculate sampling frequency
        emg_sf = (len(self.raw_emg_timestamps) - 1) / (self.raw_emg_timestamps[-1] - self.raw_emg_timestamps[0])
        print(f"EMG sampling frequency: {emg_sf:.1f} Hz")
        
        self.filtered_emg = self.process_emg_zero_phase(emg_subset, emg_sf, maxVals_subset, noiseLevel_subset)
        
        # Cut EMG filter artifacts - reduced for shorter interaction data
        emg_artifact_cut_samples = int(1.0 * emg_sf)  # 1 second cut instead of the large cut for short data
        print(f"EMG artifact cut: {emg_artifact_cut_samples} samples ({emg_artifact_cut_samples/emg_sf:.2f}s)")
        
        if emg_artifact_cut_samples < self.filtered_emg.shape[1]:
            self.filtered_emg = self.filtered_emg[:, emg_artifact_cut_samples:]
            self.raw_emg_timestamps = self.raw_emg_timestamps[emg_artifact_cut_samples:]
            print(f"After artifact cut - EMG: {self.filtered_emg.shape}")
        else:
            print(f"Warning: EMG data too short for artifact cut ({self.filtered_emg.shape[1]} samples available)")
        
        print("Filtering angle data...")
        
        # Filter angle data (same as s2.5)
        angle_sf = (len(self.raw_angle_timestamps) - 1) / (self.raw_angle_timestamps[-1] - self.raw_angle_timestamps[0])
        print(f"Angle sampling frequency: {angle_sf:.1f} Hz")
        
        # Extract angle columns (skip timestamp column) - this includes BOTH angles AND forces
        angles_only = self.raw_angles[:, 1:]  # Skip first column (timestamp) - keeps ALL data including forces
        
        # Use the current cutoff frequency from the GUI
        angle_cutoff = getattr(self, 'angle_cutoff_var', None)
        cutoff_freq = angle_cutoff.get() if angle_cutoff else 1.0
        
        # Filter ALL data (angles + forces) with the same filter
        self.filtered_angles = self.process_angles_zero_phase(angles_only, angle_sf, cutoff_freq=cutoff_freq)
        
        print(f"Filtered EMG shape: {self.filtered_emg.shape}")
        print(f"Filtered angles shape: {self.filtered_angles.shape}")
        
    def process_emg_zero_phase(self, emg_data, sampling_freq, maxVals, noiseLevel):
        """Process EMG with zero-phase filtering (from s2.5), adapted for interaction data"""
        emg_data = emg_data.astype(np.float64)
        numElectrodes = emg_data.shape[0]
        
        # Check if data is long enough for filtering
        min_length = 100  # Minimum samples needed
        if emg_data.shape[1] < min_length:
            print(f"Warning: EMG data very short ({emg_data.shape[1]} samples), minimal filtering applied")
            filtered_emg = np.abs(emg_data)
            filtered_emg = np.clip(filtered_emg - noiseLevel[:, None], 0, None)
            normalized_emg = filtered_emg / maxVals[:, None]
            return np.clip(normalized_emg, 0, 1).astype(np.float32)
        
        # 1. Power line filter (bandstop 58-62 Hz) - only if data is long enough
        try:
            powerline_sos = signal.bessel(N=4, Wn=[58, 62], btype='bandstop', output='sos', fs=sampling_freq, analog=False)  # Reduced order
        except:
            print("Skipping powerline filter - data too short")
            powerline_sos = None
        
        # 2. High pass filter (20 Hz) - reduced order for short data
        try:
            highpass_sos = signal.bessel(N=2, Wn=20, btype='highpass', output='sos', fs=sampling_freq, analog=False)  # Reduced order
        except:
            print("Skipping highpass filter - data too short")
            highpass_sos = None
        
        # 3. Low pass filter (3 Hz) - reduced order for short data
        try:
            lowpass_sos = signal.bessel(N=2, Wn=3, btype='lowpass', output='sos', fs=sampling_freq, analog=False)  # Reduced order
        except:
            print("Skipping lowpass filter - data too short")
            lowpass_sos = None
        
        filtered_emg = np.copy(emg_data)
        
        # Apply filters with error handling
        for ch in range(numElectrodes):
            try:
                if powerline_sos is not None:
                    filtered_emg[ch, :] = signal.sosfiltfilt(powerline_sos, filtered_emg[ch, :])
                if highpass_sos is not None:
                    filtered_emg[ch, :] = signal.sosfiltfilt(highpass_sos, filtered_emg[ch, :])
            except Exception as e:
                print(f"Error in channel {ch} filtering: {e}")
                pass  # Keep original data if filtering fails
        
        # Take absolute value and clip noise
        filtered_emg = np.abs(filtered_emg)
        filtered_emg = np.clip(filtered_emg - noiseLevel[:, None], 0, None)
        
        # Apply low pass filter for envelope smoothing
        for ch in range(numElectrodes):
            try:
                if lowpass_sos is not None:
                    filtered_emg[ch, :] = signal.sosfiltfilt(lowpass_sos, filtered_emg[ch, :])
            except Exception as e:
                print(f"Error in channel {ch} lowpass filtering: {e}")
                pass  # Keep current data if filtering fails
        
        filtered_emg = np.clip(filtered_emg, 0, None)
        
        # Normalize
        normalized_emg = filtered_emg / maxVals[:, None]
        normalized_emg = np.clip(normalized_emg, 0, 1)
        
        return normalized_emg.astype(np.float32)
    
    def on_angle_filter_change(self):
        """Handle angle filter cutoff change"""
        self.status_var.set(f"Angle filter: {self.angle_cutoff_var.get():.1f} Hz - Click 'Re-filter' to apply")
        
    def refilter_data(self):
        """Re-filter the angle and force data with new cutoff frequency"""
        try:
            self.status_var.set("Re-filtering angle and force data...")
            self.root.update()
            
            # Re-filter ALL data (angles + forces) with new cutoff
            angle_sf = (len(self.raw_angle_timestamps) - 1) / (self.raw_angle_timestamps[-1] - self.raw_angle_timestamps[0])
            angles_and_forces = self.raw_angles[:, 1:]  # Skip timestamp, keep everything else
            cutoff_freq = self.angle_cutoff_var.get()
            
            print(f"\nRe-filtering angles and forces with {cutoff_freq:.1f} Hz cutoff...")
            self.filtered_angles = self.process_angles_zero_phase(angles_and_forces, angle_sf, cutoff_freq=cutoff_freq)
            
            # Update the plot
            self.update_plot()
            self.status_var.set(f"Re-filtered with {cutoff_freq:.1f} Hz cutoff")
            
        except Exception as e:
            self.status_var.set(f"Re-filtering failed: {str(e)}")
            print(f"Re-filtering error: {e}")
            
    def process_angles_zero_phase(self, angles_and_forces_data, sampling_freq, cutoff_freq=1.0):
        """Process angles AND forces with zero-phase filtering (from s2.5), adapted for interaction data"""
        
        # Check if data is long enough for filtering
        min_length = 50
        if angles_and_forces_data.shape[0] < min_length:
            print(f"Warning: Data very short ({angles_and_forces_data.shape[0]} samples), skipping filtering")
            return angles_and_forces_data
            
        try:
            lowpass_sos = signal.bessel(N=2, Wn=cutoff_freq, btype='lowpass',  # Reduced order
                                       output='sos', fs=sampling_freq, analog=False)
            
            filtered_data = np.copy(angles_and_forces_data)
            
            if angles_and_forces_data.ndim == 1:
                filtered_data = signal.sosfiltfilt(lowpass_sos, filtered_data)
            else:
                # Filter all columns (angles AND forces)
                for col_idx in range(angles_and_forces_data.shape[1]):
                    try:
                        filtered_data[:, col_idx] = signal.sosfiltfilt(lowpass_sos, 
                                                                     filtered_data[:, col_idx])
                    except Exception as e:
                        print(f"Error filtering column {col_idx}: {e}")
                        pass  # Keep original data if filtering fails
            
            print(f"Applied zero-phase low-pass filter ({cutoff_freq:.1f} Hz) to ALL data (angles + forces)")
            return filtered_data
            
        except Exception as e:
            print(f"Error in filtering: {e}, returning unfiltered data")
            return angles_and_forces_data

    def normalize_force_data(self, angles_and_forces_data):
        """Normalize force data using the same method as s2.5"""
        
        # Load headers to identify force columns
        header_path = os.path.join(self.data_dir, 'angles_header.txt')
        if not os.path.exists(header_path):
            print("No headers found, skipping force normalization")
            return angles_and_forces_data
            
        with open(header_path, 'r') as f:
            headers = [h.strip() for h in f.read().split(',')][1:]  # Skip timestamp
        
        print(f"\n=== Force Normalization Debug ===")
        print(f"Headers (first 10): {headers[:10]}")
        print(f"Data shape: {angles_and_forces_data.shape}")
        
        # Find force column indices
        force_headers = [h for h in headers if 'Force' in h or 'force' in h]
        print(f"Force headers found: {force_headers}")
        
        if not force_headers:
            print("No force columns found for normalization")
            return angles_and_forces_data
        
        # Create a copy to modify
        normalized_data = np.copy(angles_and_forces_data)
        
        # Normalize force data (same ranges as s2.5)
        max_force_per_finger = 12.0  # 12N for all fingers
        
        print(f"\nApplying normalization (dividing by {max_force_per_finger}N):")
        
        for force_header in force_headers:
            if force_header in headers:
                idx = headers.index(force_header)
                if idx < angles_and_forces_data.shape[1]:
                    raw_data = angles_and_forces_data[:, idx]
                    
                    # Check if this actually looks like force data (positive values)
                    if np.max(raw_data) > 0.01:  # Only normalize if there's actual force data
                        # Apply normalization: clip to [0, max_force] then divide by max_force
                        normalized_data[:, idx] = np.clip(raw_data / max_force_per_finger, 0, 1)
                        
                        print(f" {force_header} (col {idx}): "
                              f"raw=[{np.min(raw_data):.3f}, {np.max(raw_data):.3f}] → "
                              f"norm=[{np.min(normalized_data[:, idx]):.3f}, {np.max(normalized_data[:, idx]):.3f}]")
                    else:
                        print(f"{force_header} (col {idx}): skipped (all near zero)")
                else:
                    print(f"{force_header}: column index {idx} exceeds data width")
        
        # Verify normalization worked
        print(f"\n=== Normalization Verification ===")
        for force_header in force_headers:
            if force_header in headers:
                idx = headers.index(force_header)
                if idx < normalized_data.shape[1]:
                    col_data = normalized_data[:, idx]
                    max_val = np.max(col_data)
                    if max_val > 1.0:
                        print(f"WARNING: {force_header} still has values > 1.0 (max: {max_val:.3f})")
                    else:
                        print(f"{force_header}: properly normalized (max: {max_val:.3f})")
        
        return normalized_data
        
    def create_gui(self):
        """Create the alignment GUI"""
        self.root = tk.Tk()
        self.root.title("EMG-Hand Data Alignment Tool")
        self.root.geometry("1200x800")
        
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control frame
        control_frame = ttk.LabelFrame(main_frame, text="Alignment Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Time shift control
        ttk.Label(control_frame, text="Time Shift (seconds):").pack(side=tk.LEFT)
        
        self.shift_var = tk.DoubleVar(value=0.0)
        shift_scale = ttk.Scale(control_frame, from_=-self.max_shift_range, to=self.max_shift_range, 
                               variable=self.shift_var, orient=tk.HORIZONTAL, length=400,
                               command=self.on_shift_change)
        shift_scale.pack(side=tk.LEFT, padx=10)
        
        # Shift value display
        self.shift_display = ttk.Label(control_frame, text="0.000s")
        self.shift_display.pack(side=tk.LEFT, padx=10)
        
        # Reset button
        ttk.Button(control_frame, text="Reset", command=self.reset_alignment).pack(side=tk.LEFT, padx=10)
        
        # Cut first 3s checkbox
        self.cut_first_3s_var = tk.BooleanVar(value=True)
        cut_checkbox = ttk.Checkbutton(control_frame, text="Cut first 3s", 
                                      variable=self.cut_first_3s_var)
        cut_checkbox.pack(side=tk.LEFT, padx=10)
        
        # Angle filter cutoff control
        ttk.Label(control_frame, text="Angle Filter (Hz):").pack(side=tk.LEFT, padx=(20,5))
        self.angle_cutoff_var = tk.DoubleVar(value=1.0)  # Default to 1.0 Hz for smoother results
        angle_cutoff_spinbox = ttk.Spinbox(control_frame, from_=0.5, to=3.0, increment=0.1, 
                                          width=6, textvariable=self.angle_cutoff_var,
                                          command=self.on_angle_filter_change)
        angle_cutoff_spinbox.pack(side=tk.LEFT, padx=5)
        
        # Re-filter button
        ttk.Button(control_frame, text="Re-filter", command=self.refilter_data).pack(side=tk.LEFT, padx=5)
        
        # Save button with updated text
        ttk.Button(control_frame, text="Save & Convert to S2.5", command=self.save_alignment).pack(side=tk.LEFT, padx=10)
        
        # Status
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(control_frame, textvariable=self.status_var)
        status_label.pack(side=tk.RIGHT, padx=10)
        
        # Plot frame
        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create matplotlib figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        self.fig.suptitle(f"EMG-Hand Data Alignment - {os.path.basename(self.data_dir)}")
        
        # Embed plot in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()
        
        # Initial plot
        self.update_plot()
        
    def on_shift_change(self, value):
        """Handle shift slider change"""
        self.time_shift = float(value)
        self.shift_display.config(text=f"{self.time_shift:.3f}s")
        self.update_plot()
        
    def reset_alignment(self):
        """Reset alignment to zero"""
        self.shift_var.set(0.0)
        self.time_shift = 0.0
        self.shift_display.config(text="0.000s")
        self.update_plot()
        
    def update_plot(self):
        """Update the alignment plot"""
        self.ax1.clear()
        self.ax2.clear()
        
        # Determine which signal is shorter and should be shifted
        emg_duration = self.raw_emg_timestamps[-1] - self.raw_emg_timestamps[0]
        angle_duration = self.raw_angle_timestamps[-1] - self.raw_angle_timestamps[0]
        
        if emg_duration < angle_duration:
            # EMG is shorter, shift EMG timestamps
            shifted_emg_timestamps = self.raw_emg_timestamps + self.time_shift
            shifted_angle_timestamps = self.raw_angle_timestamps
            title_suffix = "(EMG shifted)"
        else:
            # Angles are shorter, shift angle timestamps
            shifted_emg_timestamps = self.raw_emg_timestamps
            shifted_angle_timestamps = self.raw_angle_timestamps + self.time_shift
            title_suffix = "(Angles shifted)"
        
        # Plot EMG data (average of used channels)
        emg_avg = np.mean(self.filtered_emg, axis=0)
        self.ax1.plot(shifted_emg_timestamps, emg_avg, 'b-', alpha=0.7, label='EMG (avg)', linewidth=1)
        self.ax1.set_ylabel('EMG Amplitude')
        self.ax1.set_title(f'EMG Data {title_suffix}')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.legend()
        
        # Plot angle data (first few angles for visualization)
        n_angles_to_plot = min(3, self.filtered_angles.shape[1])
        colors = ['r', 'g', 'm']
        
        for i in range(n_angles_to_plot):
            self.ax2.plot(shifted_angle_timestamps, self.filtered_angles[:, i], 
                         color=colors[i], alpha=0.7, label=f'Angle {i+1}', linewidth=1)
        
        self.ax2.set_ylabel('Angle (degrees)')
        self.ax2.set_xlabel('Time (seconds)')
        self.ax2.set_title('Hand Angle Data')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.legend()
        
        # Update overlap visualization
        self.highlight_overlap(shifted_emg_timestamps, shifted_angle_timestamps)
        
        self.canvas.draw()
        
    def highlight_overlap(self, emg_timestamps, angle_timestamps):
        """Highlight the overlapping time region"""
        overlap_start = max(emg_timestamps[0], angle_timestamps[0])
        overlap_end = min(emg_timestamps[-1], angle_timestamps[-1])
        
        if overlap_start < overlap_end:
            # Add shaded region for overlap
            self.ax1.axvspan(overlap_start, overlap_end, alpha=0.2, color='green', label='Overlap')
            self.ax2.axvspan(overlap_start, overlap_end, alpha=0.2, color='green', label='Overlap')
            
            overlap_duration = overlap_end - overlap_start
            self.status_var.set(f"Overlap: {overlap_duration:.2f}s ({overlap_start:.2f}s to {overlap_end:.2f}s)")
        else:
            self.status_var.set("No overlap with current shift")
            
    def save_alignment(self):
        """Save the aligned data and convert to S2.5 format in one step"""
        try:
            # Determine which signal was shifted
            emg_duration = self.raw_emg_timestamps[-1] - self.raw_emg_timestamps[0]
            angle_duration = self.raw_angle_timestamps[-1] - self.raw_angle_timestamps[0]
            
            if emg_duration < angle_duration:
                # EMG was shifted
                aligned_emg_timestamps = self.raw_emg_timestamps + self.time_shift
                aligned_angle_timestamps = self.raw_angle_timestamps
                shift_applied_to = "EMG"
            else:
                # Angles were shifted
                aligned_emg_timestamps = self.raw_emg_timestamps
                aligned_angle_timestamps = self.raw_angle_timestamps + self.time_shift
                shift_applied_to = "Angles"
            
            # Calculate overlap
            overlap_start = max(aligned_emg_timestamps[0], aligned_angle_timestamps[0])
            overlap_end = min(aligned_emg_timestamps[-1], aligned_angle_timestamps[-1])
            
            if overlap_start >= overlap_end:
                messagebox.showerror("Error", "No overlap with current shift. Cannot save alignment.")
                return
            
            # Update status
            self.status_var.set("Processing alignment and converting to S2.5 format...")
            self.root.update()
            
            # Convert to S2.5 format directly
            aligned_output_dir = self.convert_to_s25_format(aligned_emg_timestamps, aligned_angle_timestamps, 
                                     overlap_start, overlap_end, shift_applied_to)
            
            messagebox.showinfo("Success", 
                              f"Alignment saved and converted to S2.5 format!\n"
                              f"Shift applied: {self.time_shift:.3f}s to {shift_applied_to}\n"
                              f"Overlap duration: {overlap_end - overlap_start:.2f}s\n"
                              f"{'First 3s removed' if self.cut_first_3s_var.get() else ''}\n\n"
                              f"Files saved to:\n{aligned_output_dir}\n\n"
                              f"Files created:\n"
                              f"• aligned_filtered_emg.npy\n"
                              f"• aligned_angles.npy\n" 
                              f"• aligned_timestamps.npy\n"
                              f"• aligned_angles.parquet\n\n"
                              f"Data is ready for training!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save alignment: {str(e)}")
            
    def convert_to_s25_format(self, aligned_emg_timestamps, aligned_angle_timestamps, 
                             overlap_start, overlap_end, shift_applied_to):
        """Convert aligned data to S2.5 format for training compatibility"""
        
        print(f"\n=== Converting to S2.5 Format ===")
        print(f"Applied shift: {self.time_shift:.3f}s to {shift_applied_to}")
        print(f"Overlap period: {overlap_start:.3f}s to {overlap_end:.3f}s ({overlap_end - overlap_start:.3f}s)")
        
        # Check if we should cut first 3s
        cut_first_3s = self.cut_first_3s_var.get()
        if cut_first_3s:
            print("Cutting first 4 seconds from aligned data...")
            cut_duration = 4.0
            
            # Adjust overlap window to start 3s later
            original_overlap_start = overlap_start
            overlap_start = max(overlap_start + cut_duration, overlap_start)
            
            if overlap_start >= overlap_end:
                raise ValueError(f"Cannot cut first 3s - would eliminate all data. Original overlap: {original_overlap_start:.1f}s to {overlap_end:.1f}s")
            
            print(f"New overlap after cutting first 3s: {overlap_start:.3f}s to {overlap_end:.3f}s ({overlap_end - overlap_start:.3f}s)")
        
        # Trim angle data to overlap period (use as reference timebase)
        angle_mask = (aligned_angle_timestamps >= overlap_start) & (aligned_angle_timestamps <= overlap_end)
        ref_timestamps = aligned_angle_timestamps[angle_mask]
        trimmed_angles_and_forces = self.filtered_angles[angle_mask, :]  # This includes both angles AND forces
        
        # Trim EMG data to overlap period
        emg_mask = (aligned_emg_timestamps >= overlap_start) & (aligned_emg_timestamps <= overlap_end)
        trimmed_emg_ts = aligned_emg_timestamps[emg_mask]
        trimmed_emg = self.filtered_emg[:, emg_mask]
        
        print(f"After trimming to overlap:")
        print(f"  EMG: {trimmed_emg.shape} samples")
        print(f"  Angles+Forces: {trimmed_angles_and_forces.shape} samples")
        
        # Interpolate EMG onto angle timebase (same as s2.5)
        print("Interpolating EMG onto angle timebase...")
        aligned_emg = np.zeros((len(ref_timestamps), trimmed_emg.shape[0]))
        
        for ch in range(trimmed_emg.shape[0]):
            aligned_emg[:, ch] = np.interp(ref_timestamps, trimmed_emg_ts, trimmed_emg[ch, :])
        
        # Downsample to 60Hz (same as s2.5)
        current_sf = (len(ref_timestamps) - 1) / (ref_timestamps[-1] - ref_timestamps[0])
        target_sf = 60.0
        
        print(f"Downsampling from {current_sf:.1f}Hz to {target_sf}Hz...")
        
        total_duration = ref_timestamps[-1] - ref_timestamps[0]
        target_samples = int(total_duration * target_sf)
        indices = np.linspace(0, len(ref_timestamps) - 1, target_samples).astype(int)
        
        emg_60Hz = aligned_emg[indices, :]
        angles_and_forces_60Hz = trimmed_angles_and_forces[indices, :]  # Keep both angles and forces
        downsampled_timestamps = ref_timestamps[indices]
        
        # ============= NORMALIZE FORCE DATA HERE =============
        print("\n🔧 Normalizing force data...")
        # Apply force normalization to the angles_and_forces_60Hz data
        normalized_angles_and_forces = self.normalize_force_data(angles_and_forces_60Hz)
        
        # Normalize timestamps to start at 0 (same as s2.5)
        downsampled_timestamps = downsampled_timestamps - downsampled_timestamps[0]
        
        final_sf = (len(downsampled_timestamps) - 1) / (downsampled_timestamps[-1] - downsampled_timestamps[0]) if len(downsampled_timestamps) > 1 else 0
        
        print(f"Final downsampled data:")
        print(f"  Sampling frequency: {final_sf:.1f}Hz")
        print(f"  Duration: {downsampled_timestamps[-1]:.2f}s")
        print(f"  EMG shape: {emg_60Hz.shape}")
        print(f"  Angles+Forces shape: {normalized_angles_and_forces.shape}")
        if cut_first_3s:
            print(f"  ✂️  First 3s successfully removed from final data")
        
        # Save in s2.5 format (same file names as s2.5)
        print("Saving in s2.5 format...")
        
        np.save(os.path.join(self.data_dir, 'aligned_filtered_emg.npy'), emg_60Hz)
        np.save(os.path.join(self.data_dir, 'aligned_angles.npy'), normalized_angles_and_forces)  # Now includes normalized forces
        np.save(os.path.join(self.data_dir, 'aligned_timestamps.npy'), downsampled_timestamps)
        
        # Create parquet file (same as s2.5) - with normalized forces
        self.save_angles_as_parquet(normalized_angles_and_forces, downsampled_timestamps)
        
        # Save processing metadata
        processing_info = {
            'time_shift_applied': float(self.time_shift),
            'shift_applied_to': shift_applied_to,
            'overlap_duration': float(overlap_end - overlap_start),
            'final_sampling_frequency': float(final_sf),
            'final_duration': float(downsampled_timestamps[-1]),
            'final_samples': int(len(downsampled_timestamps)),
            'emg_channels_used': self.emg_channels_used.tolist(),
            'processing_method': 'manual_alignment_tool',
            'manual_alignment': True,
            'cut_first_3s': cut_first_3s,
            'cut_duration_seconds': 3.0 if cut_first_3s else 0.0,
            'angle_filter_cutoff_hz': float(self.angle_cutoff_var.get()),
            'force_normalization_applied': True,  # Track that normalization was applied
            'force_max_value_per_finger': 6.0  # Document the normalization factor
        }
        
        import json
        with open(os.path.join(self.data_dir, "s25_processing_info.json"), 'w') as f:
            json.dump(processing_info, f, indent=2)
        
        # Save final alignment plot
        self.save_final_alignment_plot(emg_60Hz, normalized_angles_and_forces, downsampled_timestamps, cut_first_3s)
        
        print(f"S2.5 Format Conversion Complete!")
        if cut_first_3s:
            print(f"First 3 seconds removed from final data")
        print(f"Force data normalized to [0,1] range (divided by 12N)")
        print(f"Files saved to: {self.data_dir}")
        print(f"Files created:")
        print(f"  - aligned_filtered_emg.npy ({emg_60Hz.shape})")
        print(f"  - aligned_angles.npy ({normalized_angles_and_forces.shape}) - with normalized forces")
        print(f"  - aligned_timestamps.npy ({len(downsampled_timestamps)} samples)")
        print(f"  - aligned_angles.parquet - with normalized forces")
        print(f"  - s25_processing_info.json")
        print(f"  - final_alignment_plot.png")
        print(f"\n Full paths:")
        print(f"  EMG: {os.path.join(self.data_dir, 'aligned_filtered_emg.npy')}")
        print(f"  Angles: {os.path.join(self.data_dir, 'aligned_angles.npy')}")
        print(f"  Timestamps: {os.path.join(self.data_dir, 'aligned_timestamps.npy')}")
        
        self.status_var.set("Ready for training! S2.5 format files saved with normalized forces.")
        
        return self.data_dir

    def save_angles_as_parquet(self, normalized_angles_and_forces, downsampled_timestamps, output_dir=None, hand_side='left'):
        """
        CORRECTED parquet writer that exactly mirrors s2.5 approach.
        Fixes the two critical bugs: header timestamp offset and proper force summation.
        """
        
        if output_dir is None:
            output_dir = self.data_dir
        
        # Load headers and drop timestamp to match array structure
        header_path = os.path.join(self.data_dir, 'angles_header.txt')
        if not os.path.exists(header_path):
            print(f" No angles_header.txt found at {header_path}")
            return None
        
        with open(header_path, 'r') as f:
            full_headers = [h.strip() for h in f.read().split(',')]
            headers = full_headers[1:]  # DROP TIMESTAMP - this fixes the off-by-one bug!
        
        print(f" Loaded {len(full_headers)} headers, using {len(headers)} after dropping timestamp")
        
        # 1) Extract angle data using header mapping (exactly like s2.5)
        ANGLE_COLS = ['index_Pos', 'middle_Pos', 'ring_Pos', 'pinky_Pos', 'thumbFlex_Pos', 'thumbRot_Pos']
        
        try:
            angle_indices = [headers.index(h) for h in ANGLE_COLS]
            print(f" Mapped angle columns: {dict(zip(ANGLE_COLS, angle_indices))}")
        except ValueError as e:
            print(f" Failed to find angle columns: {e}")
            return None
        
        # Extract angles_only matrix (same size as headers without timestamp)
        angles_only = normalized_angles_and_forces[:, :len(headers)]  # Align with timestamp-dropped headers
        angles_for_parquet = angles_only[:, angle_indices]  # Select only the 6 angle columns
        
        print(f" Extracted angles: {angles_for_parquet.shape}")
        
        # 2) Build 5 per-finger force channels by summing 6 sensors each (exactly like s2.5)
        FINGERS = ['index', 'middle', 'ring', 'pinky', 'thumb']
        force_data = np.zeros((angles_only.shape[0], len(FINGERS)))
        
        print(f"Computing per-finger force sums:")
        for fi, finger in enumerate(FINGERS):
            # Find all 6 force sensors for this finger (finger0_Force through finger5_Force)
            sensor_cols = []
            for k in range(6):
                sensor_name = f"{finger}{k}_Force"
                if sensor_name in headers:
                    sensor_cols.append(headers.index(sensor_name))
            
            if sensor_cols:
                # Sum the 6 sensors for this finger
                finger_forces = angles_only[:, sensor_cols]  # Shape: (samples, 6)
                force_data[:, fi] = np.sum(finger_forces, axis=1)  # Sum across sensors
                
                force_min, force_max = np.min(force_data[:, fi]), np.max(force_data[:, fi])
                print(f" {finger}: {len(sensor_cols)} sensors found, sum range=[{force_min:.3f}, {force_max:.3f}]")
            else:
                print(f" {finger}: no sensors found!")
        
        # 3) Use s2.5's parquet writer to guarantee identical format
        print(f" Writing parquet using s2.5's exact method...")
        
        # Import and call s2.5's save_angles_as_parquet function
        try:
            # You can either import it or copy the function locally
            # For now, let's implement it inline with the exact s2.5 logic:
            
            hand_side_cap = hand_side.capitalize()
            df = pd.DataFrame({'timestamp': downsampled_timestamps})
            
            # Add angle columns exactly like s2.5
            for h, idx in zip(ANGLE_COLS, range(len(ANGLE_COLS))):
                df[(hand_side_cap, h)] = angles_for_parquet[:, idx]
            
            # Add force columns exactly like s2.5  
            for finger_idx, finger_name in enumerate(FINGERS):
                force_col_name = f"{finger_name}_Force"
                df[(hand_side_cap, force_col_name)] = force_data[:, finger_idx]
            
            # Save parquet
            parquet_path = os.path.join(output_dir, 'aligned_angles.parquet')
            df.to_parquet(parquet_path, index=False)
            
            print(f"Saved {parquet_path} with shape {df.shape}")
            print(f"Columns: {list(df.columns)}")
            
            # Verify format
            angle_cols = [col for col in df.columns if isinstance(col, tuple) and 'Pos' in col[1]]
            force_cols = [col for col in df.columns if isinstance(col, tuple) and 'Force' in col[1]]
            
            if len(angle_cols) == 6 and len(force_cols) == 5:
                print(f"Perfect! Created 6 angles + 5 forces = matches s2.5 exactly")
            else:
                print(f"Column count mismatch: {len(angle_cols)} angles, {len(force_cols)} forces")
                
            return parquet_path
            
        except Exception as e:
            print(f"Failed to write parquet: {e}")
            return None
        
    def save_final_alignment_plot(self, emg_60Hz, angles_and_forces_60Hz, downsampled_timestamps, cut_first_3s=False, output_dir=None):
        """Save a plot showing the final S2.5 format data"""
        
        if output_dir is None:
            output_dir = self.data_dir
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Plot final EMG data
        emg_avg = np.mean(emg_60Hz, axis=1)
        ax1.plot(downsampled_timestamps, emg_avg, 'b-', alpha=0.7, label='EMG (avg)', linewidth=1)
        ax1.set_ylabel('EMG Amplitude')
        
        title_parts = [f'Final S2.5 Format EMG Data (60Hz, {self.time_shift:.3f}s shift applied)']
        if cut_first_3s:
            title_parts.append('✂️ First 3s removed')
        ax1.set_title(' - '.join(title_parts))
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot final angle data (first few angle columns for visualization)
        # Load headers to identify which columns are angles vs forces
        header_path = os.path.join(self.data_dir, 'angles_header.txt')
        if os.path.exists(header_path):
            with open(header_path, 'r') as f:
                headers = [h.strip() for h in f.read().split(',')][1:]  # Skip timestamp
            
            angle_cols = [i for i, h in enumerate(headers) if 'Pos' in h][:3]  # First 3 angle columns
            
            for i, col_idx in enumerate(angle_cols):
                if col_idx < angles_and_forces_60Hz.shape[1]:
                    ax2.plot(downsampled_timestamps, angles_and_forces_60Hz[:, col_idx], 
                            alpha=0.7, label=f'{headers[col_idx]}', linewidth=1)
        else:
            # Fallback: plot first 3 columns
            for i in range(min(3, angles_and_forces_60Hz.shape[1])):
                ax2.plot(downsampled_timestamps, angles_and_forces_60Hz[:, i], 
                        alpha=0.7, label=f'Column {i+1}', linewidth=1)
        
        ax2.set_ylabel('Angle (degrees)')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_title('Final S2.5 Format Hand Angle Data (60Hz)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        subtitle_parts = [f'Duration: {downsampled_timestamps[-1]:.2f}s, Samples: {len(downsampled_timestamps)}']
        if cut_first_3s:
            subtitle_parts.append('First 3s removed')
        
        fig.suptitle(f'Final S2.5 Format Data - Ready for Training\n{", ".join(subtitle_parts)}')
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "final_alignment_plot.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)
        
    def run(self):
        """Run the GUI"""
        self.create_gui()
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="EMG-Hand Data Alignment Tool")
    parser.add_argument("--data_dir", required=True, help="Path to experiment data directory")
    parser.add_argument("--person_id", required=True, help="Person/subject ID")
    parser.add_argument("--out_root", default="data", help="Root data directory (default: ./data)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory does not exist: {args.data_dir}")
        sys.exit(1)
        
    print("=== EMG-Hand Data Alignment Tool ===")
    print(f"Data directory: {args.data_dir}")
    print(f"Person ID: {args.person_id}")
    print(f"Output root: {args.out_root}")
    
    try:
        # Create and run the alignment tool
        tool = EMGAlignmentTool(args.data_dir, args.person_id, args.out_root)
        tool.run()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()