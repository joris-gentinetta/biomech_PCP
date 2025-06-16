#!/usr/bin/env python3
"""
Complete EMG Calibration: Noise + MVC Measurement

This script replicates EXACTLY the same processing as your live inference:
1. powerLineFilterArray.filter(emg)  
2. highPassFilters.filter(emg)
3. np.abs(emg) 
4. MEASURE NOISE HERE <- This is where live inference uses it
5. np.clip(emg - self.noiseLevel[:, None], 0, None)
6. lowPassFilters.filter(emg)

Performs complete calibration with both rest (noise) and MVC (scaling) measurements.
"""

import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import yaml
from helpers.EMGClass import EMG
from helpers.BesselFilter import BesselFilterArr


def replicate_live_inference_processing(raw_emg, fs, connected_channels=[0,1,2,4,12,13,14,15]):
    """
    Replicate EXACTLY the same processing as your live inference script
    """
    num_total_channels = 16  # Always process 16 channels like live inference
    
    print(f"Input raw_emg shape: {raw_emg.shape}")
    print(f"Connected channels: {connected_channels}")
    
    # Ensure we have 16 channels (pad with zeros if needed)
    if raw_emg.shape[0] < 16:
        padded_emg = np.zeros((16, raw_emg.shape[1]))
        padded_emg[:raw_emg.shape[0], :] = raw_emg
        emg = padded_emg
    else:
        emg = np.copy(raw_emg)
    
    print(f"Processing shape: {emg.shape}")
    
    # Step 1: Power line filter (exactly like live inference)
    powerLineFilter = BesselFilterArr(numChannels=16, order=8, critFreqs=[58,62], fs=fs, filtType='bandstop')
    emg = powerLineFilter.filter(emg)
    print(f"After powerline filter - range: [{emg.min():.3f}, {emg.max():.3f}]")
    
    # Step 2: High pass filter (exactly like live inference) 
    highPassFilter = BesselFilterArr(numChannels=16, order=4, critFreqs=20, fs=fs, filtType='highpass')
    emg = highPassFilter.filter(emg)
    print(f"After highpass filter - range: [{emg.min():.3f}, {emg.max():.3f}]")
    
    # Step 3: Rectification (exactly like live inference)
    emg = np.abs(emg)
    print(f"After rectification - range: [{emg.min():.3f}, {emg.max():.3f}]")
    
    # THIS IS THE EXACT POINT WHERE LIVE INFERENCE MEASURES/USES NOISE
    # Extract only connected channels for noise measurement
    connected_emg = emg[connected_channels, :]
    print(f"Connected channels EMG shape: {connected_emg.shape}")
    
    return emg, connected_emg  # Return both full array and connected channels


def measure_noise_scientific_approach(connected_emg, connected_channels, target_snr_db=20):
    """
    Improved noise measurement that targets 70-90% zeros during rest
    """
    print(f"\nScientific noise measurement for connected channels: {connected_channels}")
    print(f"Target SNR: {target_snr_db} dB")
    
    noise_levels_connected = []
    
    for i, ch_idx in enumerate(connected_channels):
        ch_signal = connected_emg[i, :]
        
        print(f"\n=== Channel {ch_idx} Scientific Analysis ===")
        print(f"Signal stats: mean={np.mean(ch_signal):.3f}, std={np.std(ch_signal):.3f}")
        print(f"Range: [{np.min(ch_signal):.3f}, {np.max(ch_signal):.3f}]")
        
        # Method 1: Target-based approach - find noise level that gives 75% zeros
        sorted_signal = np.sort(ch_signal)
        target_percentile = 75  # We want 75% of rest signal to be below noise threshold
        noise_target_based = sorted_signal[int(len(sorted_signal) * target_percentile / 100)]
        
        # Method 2: Statistical approach - use mean + 1*std as baseline
        noise_statistical = np.mean(ch_signal) + 1.0 * np.std(ch_signal)
        
        # Method 3: Robust MAD approach (your existing method)
        median_val = np.median(ch_signal)
        mad = np.median(np.abs(ch_signal - median_val))
        noise_robust = mad * 1.4826 * 3.0  # Increased multiplier
        
        # Method 4: RMS-based approach
        noise_rms = np.sqrt(np.mean(ch_signal**2)) * 2.5
        
        print(f"Noise estimation methods:")
        print(f"  Target-based (75th percentile): {noise_target_based:.3f}")
        print(f"  Statistical (mean + 1*std): {noise_statistical:.3f}")
        print(f"  Robust MAD (3x): {noise_robust:.3f}")
        print(f"  RMS-based (2.5x): {noise_rms:.3f}")
        
        # Select the most appropriate method
        # For EMG, we generally want the higher estimates to ensure clean baselines
        candidates = [noise_target_based, noise_statistical, noise_robust, noise_rms]
        
        # Remove outliers (too high or too low)
        median_candidate = np.median(candidates)
        valid_candidates = [c for c in candidates if 0.5 * median_candidate <= c <= 2.0 * median_candidate]
        
        # Choose the median of valid candidates
        final_noise = np.median(valid_candidates) if valid_candidates else median_candidate
        
        print(f"  Selected noise level: {final_noise:.3f}")
        
        # Validate against signal characteristics
        zero_percentage = np.sum(ch_signal < final_noise) / len(ch_signal) * 100
        print(f"  Expected zero percentage: {zero_percentage:.1f}%")
        
        # Adjust if needed to achieve target zero percentage
        if zero_percentage < 60:  # Too few zeros
            print(f"  Adjusting up to achieve better baseline...")
            final_noise *= (75 / zero_percentage)  # Scale up proportionally
            zero_percentage = np.sum(ch_signal < final_noise) / len(ch_signal) * 100
            print(f"  Adjusted noise level: {final_noise:.3f}")
            print(f"  New expected zero percentage: {zero_percentage:.1f}%")
        
        # Bounds checking
        if final_noise < 0.5:
            final_noise = 0.5
            print(f"  Applied minimum bound: {final_noise:.3f}")
        elif final_noise > 50.0:
            final_noise = 50.0  
            print(f"  Applied maximum bound: {final_noise:.3f}")
        
        noise_levels_connected.append(final_noise)
    
    return np.array(noise_levels_connected)
def measure_noise_with_mvc_awareness(connected_rest_emg, connected_mvc_emg, connected_channels):
    """
    Improved noise measurement that considers MVC values to prevent over-subtraction
    """
    print(f"\nRefined noise measurement for connected channels: {connected_channels}")
    
    noise_levels_connected = []
    
    # First, get MVC reference values
    mvc_references = []
    for i in range(len(connected_channels)):
        mvc_signal = connected_mvc_emg[i, :]
        mvc_95 = np.percentile(mvc_signal, 95)
        mvc_references.append(mvc_95)
    
    for i, ch_idx in enumerate(connected_channels):
        ch_signal = connected_rest_emg[i, :]
        mvc_ref = mvc_references[i]
        
        print(f"\n=== Channel {ch_idx} Analysis ===")
        print(f"Rest signal: mean={np.mean(ch_signal):.3f}, std={np.std(ch_signal):.3f}")
        print(f"MVC reference (95th percentile): {mvc_ref:.1f}")
        
        # Method 1: Percentile-based (find level that gives 85-90% zeros)
        target_zero_percentage = 88  # Target percentage
        sorted_signal = np.sort(ch_signal)
        noise_percentile = sorted_signal[int(len(sorted_signal) * target_zero_percentage / 100)]
        
        # Method 2: Statistical approach - mean + k*std where k varies by signal quality
        signal_cv = np.std(ch_signal) / np.mean(ch_signal) if np.mean(ch_signal) > 0 else 10
        if signal_cv < 2:  # Low variability - use 1.5 std
            k_factor = 1.5
        elif signal_cv < 4:  # Medium variability - use 1.0 std
            k_factor = 1.0
        else:  # High variability - use 0.8 std
            k_factor = 0.8
        noise_statistical = np.mean(ch_signal) + k_factor * np.std(ch_signal)
        
        # Method 3: Adaptive threshold based on signal distribution
        # Find the "knee" in the distribution where noise transitions to signal
        hist, bin_edges = np.histogram(ch_signal, bins=100)
        cumsum = np.cumsum(hist)
        cumsum_norm = cumsum / cumsum[-1]
        
        # Find where cumulative distribution reaches 85%
        idx_85 = np.where(cumsum_norm >= 0.85)[0][0]
        noise_adaptive = bin_edges[idx_85]
        
        print(f"Noise estimates:")
        print(f"  Percentile (88th): {noise_percentile:.3f}")
        print(f"  Statistical (mean + {k_factor:.1f}*std): {noise_statistical:.3f}")
        print(f"  Adaptive threshold: {noise_adaptive:.3f}")
        
        # Choose the median of the three methods
        candidates = [noise_percentile, noise_statistical, noise_adaptive]
        base_noise = np.median(candidates)
        
        print(f"  Base noise (median): {base_noise:.3f}")
        
        # CRITICAL: Cap noise level relative to MVC
        # Noise should not exceed 10% of MVC value to preserve dynamic range
        max_allowed_noise = mvc_ref * 0.10
        
        if base_noise > max_allowed_noise:
            print(f"  Capping at 10% of MVC: {max_allowed_noise:.3f}")
            final_noise = max_allowed_noise
        else:
            final_noise = base_noise
        
        # Verify the result
        zero_percentage = np.sum(ch_signal < final_noise) / len(ch_signal) * 100
        snr_estimate = 20 * np.log10(mvc_ref / final_noise) if final_noise > 0 else 999
        
        print(f"  Final noise level: {final_noise:.3f}")
        print(f"  Expected zeros: {zero_percentage:.1f}%")
        print(f"  Estimated SNR: {snr_estimate:.1f} dB")
        
        # Quality check
        if zero_percentage < 70:
            print(f"  ⚠️ Warning: Low zero percentage - may need adjustment")
        elif zero_percentage > 95:
            print(f"  ⚠️ Warning: High zero percentage - may lose weak signals")
        else:
            print(f"  ✓ Good balance")
        
        # Final bounds
        final_noise = np.clip(final_noise, 0.5, 50.0)
        noise_levels_connected.append(final_noise)
    
    return np.array(noise_levels_connected)

def create_full_noise_array(noise_levels_connected, connected_channels):
    """
    Create full 16-channel noise array (like live inference expects)
    """
    full_noise_levels = np.zeros(16)
    
    for i, ch_idx in enumerate(connected_channels):
        full_noise_levels[ch_idx] = noise_levels_connected[i]
    
    print(f"\nFull noise array (16 channels):")
    for ch in range(16):
        if ch in connected_channels:
            print(f"  Channel {ch}: {full_noise_levels[ch]:.3f} ✓")
        else:
            print(f"  Channel {ch}: {full_noise_levels[ch]:.3f} (unused)")
    
    return full_noise_levels


def calculate_mvc_scaling(mvc_emg_processed, connected_channels, percentile=95):
    """
    Calculate maximum voluntary contraction scaling values
    """
    print(f"\n" + "="*60)
    print("CALCULATING MVC SCALING VALUES")
    print("="*60)
    
    mvc_values = []
    
    for i, ch_idx in enumerate(connected_channels):
        ch_signal = mvc_emg_processed[i, :]
        
        # Use percentile instead of max to avoid artifacts
        mvc_value = np.percentile(ch_signal, percentile)
        mvc_values.append(mvc_value)
        
        # Statistics
        signal_mean = np.mean(ch_signal)
        signal_max = np.max(ch_signal)
        signal_std = np.std(ch_signal)
        
        print(f"Channel {ch_idx}:")
        print(f"  Max: {signal_max:.1f}")
        print(f"  {percentile}th percentile: {mvc_value:.1f} ← Using this")
        print(f"  Mean: {signal_mean:.1f}")
        print(f"  Std: {signal_std:.1f}")
    
    return np.array(mvc_values)


def calculate_snr_analysis(rest_emg, mvc_emg, noise_levels, connected_channels):
    """
    Calculate comprehensive SNR analysis
    """
    print(f"\n" + "="*60) 
    print("SIGNAL-TO-NOISE RATIO ANALYSIS")
    print("="*60)
    
    snr_results = []
    
    for i, ch_idx in enumerate(connected_channels):
        rest_signal = rest_emg[i, :]
        mvc_signal = mvc_emg[i, :]
        noise_level = noise_levels[i]
        
        # Calculate signal and noise amplitudes
        rest_rms = np.sqrt(np.mean(rest_signal**2))
        mvc_max = np.percentile(mvc_signal, 95)
        
        # SNR calculations
        snr_db = 20 * np.log10(mvc_max / noise_level) if noise_level > 0 else float('inf')
        
        # Quality assessment
        if snr_db > 20:
            quality = "EXCELLENT"
        elif snr_db > 15:
            quality = "GOOD"
        elif snr_db > 10:
            quality = "ACCEPTABLE"
        else:
            quality = "POOR"
        
        snr_results.append({
            'channel': ch_idx,
            'noise_level': noise_level,
            'rest_rms': rest_rms,
            'mvc_amplitude': mvc_max,
            'snr_db': snr_db,
            'quality': quality
        })
        
        print(f"Channel {ch_idx}:")
        print(f"  Noise level: {noise_level:.3f}")
        print(f"  MVC amplitude: {mvc_max:.1f}")
        print(f"  SNR: {snr_db:.1f} dB")
        print(f"  Quality: {quality}")
    
    return snr_results


def test_noise_subtraction_both_phases(rest_emg, mvc_emg, noise_levels_connected, connected_channels):
    """
    Test noise subtraction on both rest and MVC data
    """
    print(f"\n" + "="*60)
    print("TESTING NOISE SUBTRACTION (Like Live Inference)")
    print("="*60)
    
    print("\n📊 REST DATA ANALYSIS (Should have 70-95% zeros for proper baseline):")
    print("-" * 50)
    
    rest_quality_scores = []
    
    for i, ch_idx in enumerate(connected_channels):
        rest_signal = rest_emg[i, :]
        noise_level = noise_levels_connected[i]
        
        # Apply noise subtraction
        subtracted_rest = np.clip(rest_signal - noise_level, 0, None)
        
        # Calculate statistics
        zero_percentage = np.sum(subtracted_rest == 0) / len(subtracted_rest) * 100
        
        # FIXED QUALITY ASSESSMENT - for proper EMG noise subtraction
        if 70 <= zero_percentage <= 95:
            quality = "EXCELLENT (proper baseline subtraction)"
            score = 3
        elif 50 <= zero_percentage <= 98:
            quality = "GOOD (acceptable baseline)"
            score = 2
        elif zero_percentage < 50:
            quality = "TOO LOW (increase noise level - baseline too high)"
            score = 1
        else:
            quality = "TOO HIGH (over-subtraction - signal loss)"
            score = 0
        
        rest_quality_scores.append(score)
        
        print(f"Channel {ch_idx}: {zero_percentage:.1f}% zeros → {quality}")
    
    print("\n📊 MVC DATA ANALYSIS (Should preserve muscle activation):")
    print("-" * 50)
    
    mvc_quality_scores = []
    
    for i, ch_idx in enumerate(connected_channels):
        mvc_signal = mvc_emg[i, :]
        noise_level = noise_levels_connected[i]
        
        # Apply noise subtraction
        subtracted_mvc = np.clip(mvc_signal - noise_level, 0, None)
        
        # Calculate preservation of signal
        original_max = np.max(mvc_signal)
        subtracted_max = np.max(subtracted_mvc)
        preservation = (subtracted_max / original_max) * 100 if original_max > 0 else 0
        
        # Signal range preservation
        original_range = np.max(mvc_signal) - np.min(mvc_signal)
        subtracted_range = np.max(subtracted_mvc) - np.min(subtracted_mvc)
        range_preservation = (subtracted_range / original_range) * 100 if original_range > 0 else 0
        
        # Quality assessment for MVC
        if preservation > 95 and range_preservation > 90:
            mvc_score = 3
        elif preservation > 85 and range_preservation > 80:
            mvc_score = 2
        elif preservation > 75:
            mvc_score = 1
        else:
            mvc_score = 0
        
        mvc_quality_scores.append(mvc_score)
        
        print(f"Channel {ch_idx}: {preservation:.1f}% signal preserved, {range_preservation:.1f}% range preserved")
    
    # Overall assessment
    avg_rest_quality = np.mean(rest_quality_scores)
    avg_mvc_quality = np.mean(mvc_quality_scores)
    avg_quality = (avg_rest_quality + avg_mvc_quality) / 2
    
    print(f"\n📈 REST QUALITY SCORE: {avg_rest_quality:.1f}/3.0")
    print(f"📈 MVC QUALITY SCORE: {avg_mvc_quality:.1f}/3.0")
    print(f"📈 OVERALL QUALITY SCORE: {avg_quality:.1f}/3.0")
    
    if avg_quality >= 2.5:
        print("✅ EXCELLENT: Noise levels are well calibrated!")
    elif avg_quality >= 2.0:
        print("✅ GOOD: Noise levels are acceptable")
    elif avg_quality >= 1.5:
        print("⚠️  FAIR: Consider slight noise level adjustment")
    else:
        print("❌ POOR: Noise levels need significant adjustment")
    
    return avg_quality


def record_rest_and_mvc_emg(rest_duration=15, mvc_duration=10, connected_channels=[0,1,2,4,12,13,14,15]):
    """
    Record both rest EMG (for noise) and MVC EMG (for maximum values)
    """
    print("="*80)
    print("EMG CALIBRATION: REST + MVC RECORDING")
    print("="*80)
    
    emg_device = EMG()
    
    # ===================
    # Phase 1: REST EMG
    # ===================
    print(f"\nPhase 1: Recording {rest_duration}s of REST EMG for noise measurement...")
    print("🟡 PLEASE RELAX COMPLETELY - Do not move or contract any muscles!")
    print("This measures baseline noise levels.")
    input("Press Enter when ready to start rest recording...")
    
    # Wait for first packet
    emg_device.readEMG()
    first_time = emg_device.OS_time
    
    rest_data_buf = []
    rest_ts_buf = []
    
    print("Starting rest recording in 3... 2... 1...")
    time.sleep(1)
    
    t0 = time.time()
    while time.time() - t0 < rest_duration:
        emg_device.readEMG()
        rest_data_buf.append(list(emg_device.rawEMG))
        rest_ts_buf.append((emg_device.OS_time - first_time) / 1e6)
        
        # Progress indicator
        remaining = rest_duration - (time.time() - t0)
        if remaining > 0:
            print(f"Recording rest... {remaining:.1f}s remaining", end='\r')
    
    print(f"\n✅ Rest recording complete! Recorded {len(rest_data_buf)} samples")
    
    rest_data = np.vstack(rest_data_buf)
    rest_timestamps = np.array(rest_ts_buf)
    
    # ===================
    # Phase 2: MVC EMG  
    # ===================
    print(f"\n" + "="*50)
    print("Phase 2: MVC (Maximum Voluntary Contraction) Recording")
    print("="*50)
    print(f"Recording {mvc_duration}s of MAXIMUM EMG for scaling...")
    print("🔴 PERFORM SUSTAINED MAXIMUM MUSCLE CONTRACTIONS:")
    print("   - Make tight fists and HOLD for 3-4 seconds")
    print("   - Flex all forearm muscles as hard as possible") 
    print("   - Take 2-3 second breaks between contractions")
    print("   - Aim for 2-3 maximum effort periods during recording")
    print("   - Focus on maximum force, not speed")
    input("Press Enter when ready to start MVC recording...")
    
    mvc_data_buf = []
    mvc_ts_buf = []
    
    print("Starting MVC recording in 3... 2... 1...")
    print("🔴 CONTRACT AND HOLD! (3-4 seconds)")
    time.sleep(1)
    
    # Reset timestamp reference for MVC
    emg_device.readEMG()
    mvc_first_time = emg_device.OS_time
    
    t0 = time.time()
    contraction_phase = True
    last_instruction_time = t0
    instruction_interval = 4.0  # Change instruction every 4 seconds
    
    while time.time() - t0 < mvc_duration:
        emg_device.readEMG()
        mvc_data_buf.append(list(emg_device.rawEMG))
        mvc_ts_buf.append((emg_device.OS_time - mvc_first_time) / 1e6)
        
        # Progress indicator with realistic instructions
        elapsed = time.time() - t0
        remaining = mvc_duration - elapsed
        
        # Change instruction every 4 seconds
        if elapsed - last_instruction_time >= instruction_interval:
            contraction_phase = not contraction_phase
            last_instruction_time = elapsed
        
        if remaining > 0:
            if contraction_phase:
                instruction = "🔴 CONTRACT AND HOLD!"
            else:
                instruction = "🟡 relax... (prepare for next)"
            print(f"Recording MVC... {remaining:.1f}s remaining - {instruction}", end='\r')
    
    print(f"\n✅ MVC recording complete! Recorded {len(mvc_data_buf)} samples")
    
    mvc_data = np.vstack(mvc_data_buf)
    mvc_timestamps = np.array(mvc_ts_buf)
    
    # Calculate sampling frequency
    rest_elapsed = rest_timestamps[-1] - rest_timestamps[0]
    sf = (len(rest_timestamps) - 1) / rest_elapsed if rest_elapsed > 1 else 1000
    
    print(f"📊 Sampling frequency: {sf:.1f} Hz")
    
    # Clean up EMG
    emg_device.exitEvent.set()
    time.sleep(0.5)
    if hasattr(emg_device, 'emgThread'):
        emg_device.shutdown()
    else:
        if hasattr(emg_device, 'sock'):
            emg_device.sock.close()
        if hasattr(emg_device, 'ctx'):
            emg_device.ctx.term()
    
    # Remove initial artifacts
    artifact_cut = min(400, len(rest_data) // 4)
    clean_rest_data = rest_data[artifact_cut:]
    clean_rest_timestamps = rest_timestamps[artifact_cut:artifact_cut+len(clean_rest_data)]
    
    artifact_cut_mvc = min(400, len(mvc_data) // 4)
    clean_mvc_data = mvc_data[artifact_cut_mvc:]
    clean_mvc_timestamps = mvc_timestamps[artifact_cut_mvc:artifact_cut_mvc+len(clean_mvc_data)]
    
    print(f"📊 After artifact removal:")
    print(f"   Rest: {len(clean_rest_data)} samples")
    print(f"   MVC:  {len(clean_mvc_data)} samples")
    
    return (clean_rest_data.T, clean_rest_timestamps, 
            clean_mvc_data.T, clean_mvc_timestamps, sf)


def main():
    parser = argparse.ArgumentParser(description="Complete EMG Calibration: Noise + MVC")
    parser.add_argument("--person_id", "-p", required=True, help="Person ID")
    parser.add_argument("--rest_duration", "-r", type=float, default=15, help="Rest recording duration")
    parser.add_argument("--mvc_duration", "-m", type=float, default=10, help="MVC recording duration")
    parser.add_argument("--target_noise", "-t", type=float, default=10.0, help="Target noise level")
    parser.add_argument("--output_dir", "-o", default="data", help="Output directory")
    
    args = parser.parse_args()
    
    # Connected channels (from your setup)
    connected_channels = [0, 1, 2, 4, 12, 13, 14, 15]
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, args.person_id, "complete_emg_calibration")
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("COMPLETE EMG CALIBRATION: NOISE + MVC")
    print("="*80)
    print("This script will:")
    print("1. Record REST EMG (for noise measurement)")
    print("2. Record MVC EMG (for maximum scaling)")
    print("3. Process both through live inference pipeline")
    print("4. Calculate optimal noise levels and scaling")
    print("5. Validate with SNR analysis")
    print(f"Connected channels: {connected_channels}")
    print("="*80)
    
    try:
        # Step 1: Record both rest and MVC data
        print("\nStep 1: Recording EMG data...")
        rest_emg, rest_timestamps, mvc_emg, mvc_timestamps, sf = record_rest_and_mvc_emg(
            args.rest_duration, args.mvc_duration, connected_channels
        )
        
        # Step 2: Process both through live inference pipeline
        print("\nStep 2: Processing through live inference pipeline...")
        
        print("\n🔄 Processing REST data...")
        full_rest_emg, connected_rest_emg = replicate_live_inference_processing(rest_emg, sf, connected_channels)
        
        print("\n🔄 Processing MVC data...")
        full_mvc_emg, connected_mvc_emg = replicate_live_inference_processing(mvc_emg, sf, connected_channels)
        
        # Step 3: Measure noise from rest data
        print("\nStep 3: Scientific noise measurement from rest data...")
        # noise_levels_connected = measure_noise_scientific_approach(connected_rest_emg, connected_channels, target_snr_db=20)
        noise_levels_connected = measure_noise_with_mvc_awareness(connected_rest_emg, connected_mvc_emg, connected_channels)
        
        # Step 4: Calculate MVC scaling from MVC data
        print("\nStep 4: Calculating MVC scaling...")
        mvc_values_connected = calculate_mvc_scaling(connected_mvc_emg, connected_channels)
        
        # Step 5: Create full arrays
        print("\nStep 5: Creating full 16-channel arrays...")
        full_noise_levels = create_full_noise_array(noise_levels_connected, connected_channels)
        
        # Create full MVC array
        full_mvc_values = np.ones(16)  # Default to 1 for unused channels
        for i, ch_idx in enumerate(connected_channels):
            full_mvc_values[ch_idx] = mvc_values_connected[i]
        
        # Step 6: Test noise subtraction on both datasets
        print("\nStep 6: Testing noise subtraction...")
        quality_score = test_noise_subtraction_both_phases(
            connected_rest_emg, connected_mvc_emg, noise_levels_connected, connected_channels
        )
        
        # Step 7: SNR Analysis
        print("\nStep 7: SNR Analysis...")
        snr_results = calculate_snr_analysis(
            connected_rest_emg, connected_mvc_emg, noise_levels_connected, connected_channels
        )
        
        # Step 8: Save results
        print(f"\nStep 8: Saving results...")
        
        # Save all data
        np.save(os.path.join(output_dir, "rest_emg_raw.npy"), rest_emg)
        np.save(os.path.join(output_dir, "mvc_emg_raw.npy"), mvc_emg)
        np.save(os.path.join(output_dir, "rest_emg_processed.npy"), full_rest_emg)
        np.save(os.path.join(output_dir, "mvc_emg_processed.npy"), full_mvc_emg)
        np.save(os.path.join(output_dir, "rest_timestamps.npy"), rest_timestamps)
        np.save(os.path.join(output_dir, "mvc_timestamps.npy"), mvc_timestamps)
        
        # Convert numpy types to Python types for YAML serialization
        snr_results_serializable = []
        for result in snr_results:
            snr_results_serializable.append({
                'channel': int(result['channel']),
                'noise_level': float(result['noise_level']),
                'rest_rms': float(result['rest_rms']),
                'mvc_amplitude': float(result['mvc_amplitude']),
                'snr_db': float(result['snr_db']) if result['snr_db'] != float('inf') else 999.0,
                'quality': result['quality']
            })
        
        # Save comprehensive results
        results = {
            'noise_levels_full': [float(x) for x in full_noise_levels],
            'mvc_values_full': [float(x) for x in full_mvc_values],
            'noise_levels_connected': [float(x) for x in noise_levels_connected],
            'mvc_values_connected': [float(x) for x in mvc_values_connected],
            'connected_channels': connected_channels,
            'sampling_frequency': float(sf),
            'quality_score': float(quality_score),
            'snr_analysis': snr_results_serializable,
            'recording_parameters': {
                'rest_duration': float(args.rest_duration),
                'mvc_duration': float(args.mvc_duration),
                'target_noise': float(args.target_noise)
            }
        }
        
        results_file = os.path.join(output_dir, "complete_calibration_results.yaml")
        with open(results_file, 'w') as f:
            yaml.safe_dump(results, f)
        
        # Save in scaling.yaml format (for your calibration script)
        scaling_dict = {
            'noiseLevels': [float(x) for x in full_noise_levels],
            'maxVals': [float(x) for x in full_mvc_values],
            'connectedChannels': connected_channels,
            'method': 'complete_calibration_pipeline',
            'qualityScore': float(quality_score)
        }
        scaling_file = os.path.join(output_dir, "scaling.yaml")
        with open(scaling_file, 'w') as f:
            yaml.safe_dump(scaling_dict, f)
        
        print(f"\nResults saved to: {results_file}")
        print(f"Scaling file saved to: {scaling_file}")
        
        # Final Summary
        print(f"\n" + "="*80)
        print("CALIBRATION COMPLETE - SUMMARY")
        print("="*80)
        
        print("📊 Noise Levels (connected channels):")
        for i, ch_idx in enumerate(connected_channels):
            snr = next((r['snr_db'] for r in snr_results if r['channel'] == ch_idx), 0)
            quality = next((r['quality'] for r in snr_results if r['channel'] == ch_idx), "UNKNOWN")
            print(f"  Channel {ch_idx}: {noise_levels_connected[i]:.3f} (SNR: {snr:.1f} dB, {quality})")
        
        print(f"\n📊 MVC Values (connected channels):")
        for i, ch_idx in enumerate(connected_channels):
            print(f"  Channel {ch_idx}: {mvc_values_connected[i]:.1f}")
        
        print(f"\n📈 Overall Quality Score: {quality_score:.1f}/3.0")
        
        valid_snrs = [r['snr_db'] for r in snr_results if r['snr_db'] != float('inf') and r['snr_db'] < 999]
        avg_snr = np.mean(valid_snrs) if valid_snrs else 0
        print(f"📈 Average SNR: {avg_snr:.1f} dB")
        
        excellent_channels = sum(1 for r in snr_results if r['quality'] == 'EXCELLENT')
        print(f"📈 Excellent Channels: {excellent_channels}/{len(connected_channels)}")
        
        if quality_score >= 2.5 and avg_snr >= 15:
            print("\n🎉 CALIBRATION SUCCESS! Ready for live inference.")
        elif quality_score >= 2.0:
            print("\n✅ CALIBRATION GOOD! Should work well for live inference.")
        else:
            print("\n⚠️  CALIBRATION NEEDS IMPROVEMENT. Consider re-running or adjusting parameters.")
        
        print(f"\n📁 Use scaling.yaml in your calibration script:")
        print(f"   {scaling_file}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("Calibration failed. Check connections and try again.")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())