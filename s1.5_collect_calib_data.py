#!/usr/bin/env python3
import argparse
import os
import time
import threading
import numpy as np
import cv2
import yaml
from os.path import join
from helpers.hand_poses import hand_poses
from psyonicHand import psyonicArm
from scipy.interpolate import CubicSpline
from helpers.EMGClass import EMG
from scipy.signal import butter, filtfilt
from helpers.BesselFilter import BesselFilterArr

def make_timestamps_unique(timestamps):
    timestamps = np.array(timestamps)
    for i in range(1, len(timestamps)):
        if timestamps[i] <= timestamps[i - 1]:
            timestamps[i] = timestamps[i - 1] + 1e-6  # add 1 microsecond
    return timestamps

def filter_emg_pipeline_bessel(raw_emg, fs, noise_level=None):
    # raw_emg: shape [channels, samples] (transpose if needed)
    num_channels = raw_emg.shape[0]

    # 1. Bandstop (powerline)
    notch = BesselFilterArr(numChannels=num_channels, order=8, critFreqs=[58,62], fs=fs, filtType='bandstop')
    emg = notch.filter(raw_emg)
    # 2. Highpass (20 Hz)
    hp = BesselFilterArr(numChannels=num_channels, order=4, critFreqs=20, fs=fs, filtType='highpass')
    emg = hp.filter(emg)
    # 3. Rectification
    emg = np.abs(emg)
    # 4. Noise subtraction and clipping
    if noise_level is not None:
        emg = np.clip(emg - noise_level[:, None], 0, None)
    # 5. Lowpass (envelope, 3 Hz)
    lp = BesselFilterArr(numChannels=num_channels, order=4, critFreqs=3, fs=fs, filtType='lowpass')
    emg = lp.filter(emg)

    return emg  # same shape as input


def robust_mvc(mvc_data, percentile=93):
    # mvc_data: [samples, channels]
    robust_max = []
    for ch in range(mvc_data.shape[1]):
        channel_data = mvc_data[:, ch]
        # Calculate the percentile value
        robust_val = np.percentile(channel_data, percentile)
        robust_max.append(robust_val)
    return np.array(robust_max)

def measure_noise_levels_refined(rest_data, mvc_data, sf_rest, sf_mvc, artifact_cut=400):
    """
    Refined noise measurement with MVC-aware bounds and multiple estimation methods
    """
    num_channels = rest_data.shape[0]
    
    # First, process the rest data through the pipeline up to noise measurement point
    # 1. Bandstop (powerline)
    notch = BesselFilterArr(numChannels=num_channels, order=8, critFreqs=[58,62], fs=sf_rest, filtType='bandstop')
    emg = notch.filter(rest_data)
    
    # 2. Highpass (20 Hz)
    hp = BesselFilterArr(numChannels=num_channels, order=4, critFreqs=20, fs=sf_rest, filtType='highpass')
    emg = hp.filter(emg)
    
    # 3. Rectification (same as live: np.abs(emg))
    emg = np.abs(emg)
    
    # Cut artifacts from beginning
    emg_cut = emg[:, artifact_cut:]
    
    # Process MVC data to get reference values
    notch_mvc = BesselFilterArr(numChannels=num_channels, order=8, critFreqs=[58,62], fs=sf_mvc, filtType='bandstop')
    mvc_filtered = notch_mvc.filter(mvc_data)
    hp_mvc = BesselFilterArr(numChannels=num_channels, order=4, critFreqs=20, fs=sf_mvc, filtType='highpass')
    mvc_filtered = hp_mvc.filter(mvc_filtered)
    mvc_filtered = np.abs(mvc_filtered)
    mvc_filtered_cut = mvc_filtered[:, artifact_cut:]
    
    # Get MVC reference values (95th percentile for each channel)
    mvc_references = np.percentile(mvc_filtered_cut, 95, axis=1)
    
    noise_levels = []
    
    print("\n=== Refined Noise Measurement ===")
    for ch in range(num_channels):
        ch_signal = emg_cut[ch, :]
        mvc_ref = mvc_references[ch]
        
        print(f"\nChannel {ch}:")
        print(f"  Rest signal: mean={np.mean(ch_signal):.3f}, std={np.std(ch_signal):.3f}")
        print(f"  MVC reference (95th percentile): {mvc_ref:.1f}")
        
        # Method 1: Percentile-based (find level that gives 88% zeros)
        target_zero_percentage = 88
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
        hist, bin_edges = np.histogram(ch_signal, bins=100)
        cumsum = np.cumsum(hist)
        cumsum_norm = cumsum / cumsum[-1]
        
        # Find where cumulative distribution reaches 85%
        idx_85 = np.where(cumsum_norm >= 0.85)[0]
        if len(idx_85) > 0:
            noise_adaptive = bin_edges[idx_85[0]]
        else:
            noise_adaptive = noise_percentile  # fallback
        
        print(f"  Noise estimates:")
        print(f"    Percentile (88th): {noise_percentile:.3f}")
        print(f"    Statistical (mean + {k_factor:.1f}*std): {noise_statistical:.3f}")
        print(f"    Adaptive threshold: {noise_adaptive:.3f}")
        
        # Choose the median of the three methods
        candidates = [noise_percentile, noise_statistical, noise_adaptive]
        base_noise = np.median(candidates)
        
        print(f"    Base noise (median): {base_noise:.3f}")
        
        # CRITICAL: Cap noise level relative to MVC
        # Noise should not exceed 10% of MVC value to preserve dynamic range
        max_allowed_noise = mvc_ref * 0.10
        
        if base_noise > max_allowed_noise:
            print(f"    Capping at 10% of MVC: {max_allowed_noise:.3f}")
            final_noise = max_allowed_noise
        else:
            final_noise = base_noise
        
        # Verify the result
        zero_percentage = np.sum(ch_signal < final_noise) / len(ch_signal) * 100
        snr_estimate = 20 * np.log10(mvc_ref / final_noise) if final_noise > 0 else 999
        
        print(f"    Final noise level: {final_noise:.3f}")
        print(f"    Expected zeros: {zero_percentage:.1f}%")
        print(f"    Estimated SNR: {snr_estimate:.1f} dB")
        
        # Quality check
        if zero_percentage < 70:
            print(f"    ⚠️ Warning: Low zero percentage - may need adjustment")
        elif zero_percentage > 95:
            print(f"    ⚠️ Warning: High zero percentage - may lose weak signals")
        else:
            print(f"    ✓ Good balance")
        
        # Final bounds
        final_noise = np.clip(final_noise, 0.5, 50.0)
        noise_levels.append(final_noise)
    
    return np.array(noise_levels)

def calibrate_emg(base_dir, rest_time=10, mvc_time=10, free_time=8, target_free_ratio=0.3):
    print("EMG Calibration Routine")
    emg = EMG()
    # emg.startCommunication()

    # 1. REST 
    print("\n=== Phase 1: REST Recording ===")
    print("🟡 PLEASE RELAX COMPLETELY - Do not move or contract any muscles!")
    print("This measures baseline noise levels.")
    input("Press Enter when ready to start rest recording...")
    
    emg.readEMG()
    first_rest_time = emg.OS_time

    rest_data_buf = []
    rest_ts_buf = []
    t0 = time.time()
    print(f"Recording rest for {rest_time} seconds...")
    while time.time() - t0 < rest_time:
        emg.readEMG()
        rest_data_buf.append(list(emg.rawEMG))
        rest_ts_buf.append((emg.OS_time - first_rest_time) / 1e6) # convert from microseconds
    rest_data = np.vstack(rest_data_buf)
    rest_timestamps = np.array(rest_ts_buf)
    np.save(os.path.join(base_dir, "calib_rest_emg.npy"), rest_data)
    np.save(os.path.join(base_dir, "calib_rest_timestamps.npy"), rest_timestamps)
    print(f"✅ Rest recording complete! Recorded {len(rest_data)} samples")

    elapsed = rest_timestamps[-1] - rest_timestamps[0]
    sf_rest = (len(rest_timestamps)-1)/ elapsed if elapsed > 1 else 1000
    print(f"Sampling frequency (rest): {sf_rest:.1f} Hz")

    # 2. MVC
    print("\n=== Phase 2: MVC Recording ===")
    print("🔴 PERFORM SUSTAINED MAXIMUM MUSCLE CONTRACTIONS:")
    print("   - Make tight fists and HOLD for 3-4 seconds")
    print("   - Flex all forearm muscles as hard as possible")
    print("   - Take 2-3 second breaks between contractions")
    print("   - Aim for 2-3 maximum effort periods during recording")
    input("Press Enter when ready to start MVC recording...")
    
    emg.readEMG()
    first_mvc_time = emg.OS_time

    mvc_data_buf = []
    mvc_ts_buf = []
    t0 = time.time()
    print(f"Recording MVC for {mvc_time} seconds...")
    print("🔴 CONTRACT AND HOLD! (3-4 seconds)")
    while time.time() - t0 < mvc_time:
        emg.readEMG()
        mvc_data_buf.append(list(emg.rawEMG))
        mvc_ts_buf.append((emg.OS_time - first_mvc_time) / 1e6) # Convert from microseconds
    mvc_data = np.vstack(mvc_data_buf)
    mvc_timestamps = np.array(mvc_ts_buf)
    np.save(os.path.join(base_dir, "calib_mvc_emg.npy"), mvc_data)
    np.save(os.path.join(base_dir, "calib_mvc_timestamps.npy"), mvc_timestamps)
    print(f"✅ MVC recording complete! Recorded {len(mvc_data)} samples")

    sf_mvc = (len(mvc_timestamps) - 1) / (mvc_timestamps[-1] - mvc_timestamps[0]) if len(mvc_timestamps) > 1 else 1000
    print(f"Sampling frequency (MVC): {sf_mvc:.1f} Hz")
    
    # COMPARISON: Simple vs Refined Noise Calculation
    artifact_cut = 400
    
    # === SIMPLE NOISE CALCULATION (CORRECTED) ===
    # Apply same preprocessing as refined method for fair comparison
    num_channels = rest_data.shape[1]  # rest_data is [samples, channels]
    
    # 1. Process rest data through same initial filters
    notch = BesselFilterArr(numChannels=num_channels, order=8, critFreqs=[58,62], fs=sf_rest, filtType='bandstop')
    rest_filtered = notch.filter(rest_data.T)  # transpose to [channels, samples]
    
    hp = BesselFilterArr(numChannels=num_channels, order=4, critFreqs=20, fs=sf_rest, filtType='highpass')
    rest_filtered = hp.filter(rest_filtered)
    
    rest_filtered = np.abs(rest_filtered)  # rectify
    rest_filtered_cut = rest_filtered[:, artifact_cut:]  # remove artifacts
    
    # Simple method: mean + 2*std per channel
    simple_noise_levels = []
    for ch in range(num_channels):
        ch_data = rest_filtered_cut[ch, :]
        simple_noise = np.mean(ch_data) + 2 * np.std(ch_data)
        simple_noise_levels.append(simple_noise)
    simple_noise_levels = np.array(simple_noise_levels)
    
    # === REFINED NOISE CALCULATION ===
    refined_noise_levels = measure_noise_levels_refined(rest_data.T, mvc_data.T, sf_rest, sf_mvc, artifact_cut=artifact_cut)
    
    # === COMPARISON RESULTS ===
    print(f"\n=== NOISE CALCULATION COMPARISON ===")
    print(f"{'Channel':<8} {'Simple':<10} {'Refined':<10} {'Difference':<12} {'Ratio':<8}")
    print("-" * 55)
    
    for ch in range(len(simple_noise_levels)):
        diff = refined_noise_levels[ch] - simple_noise_levels[ch]
        ratio = refined_noise_levels[ch] / simple_noise_levels[ch] if simple_noise_levels[ch] > 0 else 0
        print(f"{ch:<8} {simple_noise_levels[ch]:<10.3f} {refined_noise_levels[ch]:<10.3f} {diff:<+12.3f} {ratio:<8.2f}")
    
    print(f"\nSummary:")
    print(f"Simple method mean:    {np.mean(simple_noise_levels):.3f}")
    print(f"Refined method mean:   {np.mean(refined_noise_levels):.3f}")
    print(f"Average ratio (R/S):   {np.mean(refined_noise_levels/simple_noise_levels):.2f}")
    
    # Use refined method for actual calibration
    noise_levels = refined_noise_levels
    
    # Filter MVC for max values
    filtered_mvc = filter_emg_pipeline_bessel(mvc_data.T, sf_mvc, noise_level=noise_levels)
    filtered_mvc_cut = filtered_mvc[:, artifact_cut:]
    np.save(os.path.join(base_dir, "calib_mvc_filtered.npy"), filtered_mvc_cut)
    
    # Use percentile for robust max values
    maxVals = np.percentile(filtered_mvc_cut, 95, axis=1)

    print(f"\n=== Calibration Results ===")
    print(f"maxVals: {maxVals}")
    print(f"noiseLevels (refined): {noise_levels}")
    
    # Calculate and display SNR for each channel using both methods
    print(f"\n=== Signal-to-Noise Ratio Comparison ===")
    print(f"{'Channel':<8} {'SNR Simple':<12} {'SNR Refined':<12} {'Quality Simple':<15} {'Quality Refined':<15}")
    print("-" * 80)
    
    for ch in range(len(noise_levels)):
        if simple_noise_levels[ch] > 0 and noise_levels[ch] > 0:
            snr_simple = 20 * np.log10(maxVals[ch] / simple_noise_levels[ch])
            snr_refined = 20 * np.log10(maxVals[ch] / noise_levels[ch])
            
            def get_quality(snr):
                return "EXCELLENT" if snr > 20 else "GOOD" if snr > 15 else "ACCEPTABLE" if snr > 10 else "POOR"
            
            qual_simple = get_quality(snr_simple)
            qual_refined = get_quality(snr_refined)
            
            print(f"{ch:<8} {snr_simple:<12.1f} {snr_refined:<12.1f} {qual_simple:<15} {qual_refined:<15}")

    # Save minimal scaling.yaml with refined values
    scaling_dict = {
        'maxVals': maxVals.tolist(),
        'noiseLevels': noise_levels.tolist(),
        'noiseLevels_simple': simple_noise_levels.tolist()  # save for comparison
    }
    yaml_path = os.path.join(base_dir, 'scaling.yaml')
    with open(yaml_path, 'w') as f:
        yaml.safe_dump(scaling_dict, f)
    print(f"\n✅ Calibration complete! Saved scaling.yaml to {yaml_path}")

    emg.exitEvent.set()
    time.sleep(0.5)
    if hasattr(emg, 'emgThread'):
        emg.shutdown()
    else:
        emg.sock.close()
        emg.ctx.term()

def start_raw_emg_recorder(base_dir, enable_video=False, sync_event=None):
    """
    Start background threads to record raw EMG and (optionally) webcam video.
    Returns: stop_event, emg_thread, video_thread (or None), raw_history, raw_timestamps, video_timestamps
    """
    emg = EMG()  # connect directly to ADS1299 via EMGClass (records all 16 channels by default)

    raw_history = []
    raw_timestamps = []
    video_timestamps = []
    stop_event = threading.Event()

    # Optional video recorder thread with sync_event
    video_thread = None
    def video_loop():
        cap = cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_path = join(base_dir, 'webcam.mp4')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        # Wait for main event to start
        if sync_event is not None:
            sync_event.wait()
        video_first_time = None
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            now = time.time()
            if video_first_time is None:
                video_first_time = now
            video_timestamps.append(now - video_first_time)
            writer.write(frame)
            time.sleep(1.0 / fps)
        cap.release()
        writer.release()

    if enable_video:
        video_thread = threading.Thread(target=video_loop, daemon=True)
        video_thread.start()

    # EMG capture thread with sync_event - PROPERLY INDENTED
    def capture_loop():
        try:
            # Start the EMG communication pipeline (like the original)
            emg.startCommunication()  # This starts the internal emgThread
            
            # Wait for first data
            while getattr(emg, 'OS_time', None) is None:
                time.sleep(0.001)

            print("EMG thread: Keeping connection alive, waiting for sync event...")
            
            # Wait for sync while EMG pipeline runs in background
            while sync_event is not None and not sync_event.is_set() and not stop_event.is_set():
                time.sleep(0.001)

            if stop_event.is_set():
                return

            raw_history.clear()
            raw_timestamps.clear()

            # Get first timestamp like the original
            first_emg_time = emg.OS_time
            last_time = emg.OS_time
            
            while not stop_event.is_set():
                time_sample = emg.OS_time
                
                # Add timing validation like the original
                if (time_sample - last_time)/1e6 > 0.1:
                    print(f'Read time: {time_sample}, expected time: {last_time}')
                    raise ValueError('EMG alignment lost. Please restart the EMG board and the script.')
                
                # Only collect data when we have new timestamps
                elif time_sample > last_time:
                    emg_sample = np.asarray(emg.rawEMG)
                    raw_history.append(list(emg_sample))
                    raw_timestamps.append((time_sample - first_emg_time) / 1e6)
                
                last_time = time_sample
                time.sleep(0.001)  # Small delay to prevent busy waiting
                
        except Exception as e:
            print(f"EMG capture error: {e}")
        finally:
            emg.exitEvent.set()
            time.sleep(0.1)

    # Create and start the thread
    emg_thread = threading.Thread(target=capture_loop, daemon=True)
    emg_thread.start()

    return stop_event, emg_thread, video_thread, raw_history, raw_timestamps, video_timestamps

def get_movement_sequences(movement_name):
    """
    Get movement sequence(s) for individual or combined movements
    Returns: (sequences, is_combined)
    """
    if movement_name not in hand_poses:
        available = [k for k in hand_poses.keys() if not isinstance(hand_poses[k][0], str)]
        combined = [k for k in hand_poses.keys() if isinstance(hand_poses[k][0], str)]
        raise ValueError(f"Movement '{movement_name}' not found.\nAvailable individual: {available}\nAvailable combined: {combined}")
    
    movement_data = hand_poses[movement_name]
    
    # Check if this is a combined movement (list of strings)
    if isinstance(movement_data, list) and len(movement_data) > 0 and isinstance(movement_data[0], str):
        print(f"🔄 Combined movement '{movement_name}' → {movement_data}")
        sequences = []
        for sub_movement in movement_data:
            if sub_movement in hand_poses:
                sequences.append(hand_poses[sub_movement])
            else:
                raise ValueError(f"Sub-movement '{sub_movement}' not found")
        return sequences, True
    else:
        # Regular single movement
        return [movement_data], False

def calculate_angular_distance(pos1, pos2):
    """
    Calculate the total angular distance between two joint positions
    """
    return np.linalg.norm(np.array(pos2) - np.array(pos1))

def create_constant_velocity_trajectory(positions, num_steps):
    """
    Create trajectory with constant angular velocity (same angle change per step)
    """
    if len(positions) < 2:
        return np.array(positions)
    
    positions = [np.array(p) for p in positions]
    
    # Calculate cumulative distances along the path
    distances = [0.0]
    for i in range(1, len(positions)):
        dist = calculate_angular_distance(positions[i-1], positions[i])
        distances.append(distances[-1] + dist)
    
    total_distance = distances[-1]
    
    if total_distance == 0:
        # All positions are the same, just return repeated positions
        return np.tile(positions[0], (num_steps, 1))
    
    # Create evenly spaced points along the total distance
    target_distances = np.linspace(0, total_distance, num_steps)
    
    # Interpolate positions at these evenly spaced distances
    trajectory = []
    
    for target_dist in target_distances:
        # Find which segment this distance falls into
        segment_idx = 0
        for i in range(len(distances) - 1):
            if distances[i] <= target_dist <= distances[i + 1]:
                segment_idx = i
                break
        
        # Interpolate within this segment
        if segment_idx == len(distances) - 1:
            # We're at the end
            trajectory.append(positions[-1])
        else:
            # Linear interpolation based on distance
            seg_start_dist = distances[segment_idx]
            seg_end_dist = distances[segment_idx + 1]
            seg_length = seg_end_dist - seg_start_dist
            
            if seg_length > 0:
                alpha = (target_dist - seg_start_dist) / seg_length
            else:
                alpha = 0
            
            pos_start = positions[segment_idx]
            pos_end = positions[segment_idx + 1]
            
            interpolated_pos = pos_start * (1 - alpha) + pos_end * alpha
            trajectory.append(interpolated_pos)
    
    return np.array(trajectory)

def build_constant_velocity_trajectory(pose, movement_name):
    """
    Build trajectory with constant angular velocity for multi-step movements
    """
    if not isinstance(pose, list):
        raise ValueError(f"Movement {movement_name} should be a list of positions")
    
    positions = [np.array(p) for p in pose]
    total_steps = 600  # 1 second at 600Hz
    
    if len(positions) == 2:
        # Simple 2-position movement: use regular linear interpolation
        pos1, pos2 = positions
        steps_movement = 400
        steps_pause = 200
        
        movement_traj = create_smooth_segment(pos1, pos2, steps_movement)
        pause_traj = np.tile(pos2, (steps_pause, 1))
        complete_traj = np.vstack([movement_traj, pause_traj])
        
        print(f"{movement_name}: {steps_movement} move + {steps_pause} pause = {complete_traj.shape[0]} steps → 1.0s")
        
    else:
        # Multi-step movement: use constant angular velocity
        pause_steps = 150
        movement_steps = total_steps - pause_steps
        
        # Create constant velocity trajectory through all positions
        movement_traj = create_constant_velocity_trajectory(positions, movement_steps)
        
        # Pause at final position
        pause_traj = np.tile(positions[-1], (pause_steps, 1))
        
        complete_traj = np.vstack([movement_traj, pause_traj])
        
        # Calculate actual distances for verification
        total_angular_distance = sum(calculate_angular_distance(positions[i], positions[i+1]) 
                                   for i in range(len(positions)-1))
        angular_velocity = total_angular_distance / (movement_steps / 600.0)  # degrees per second
        
        print(f"{movement_name}: {len(positions)} positions, {movement_steps} steps + {pause_steps} pause")
        print(f"  Total angular distance: {total_angular_distance:.1f}°")
        print(f"  Angular velocity: {angular_velocity:.1f}°/s")
        print(f"  Distance per step: {total_angular_distance/movement_steps:.3f}°")
    
    return complete_traj

def build_combined_constant_velocity_trajectory(sequences, movement_name):
    """
    Build combined trajectory where each part has constant angular velocity
    """
    print(f"📋 Building combined movement '{movement_name}' with constant angular velocity...")
    
    all_trajectories = []
    
    for i, seq in enumerate(sequences):
        part_name = f"{movement_name}_part{i+1}"
        print(f"\n   Part {i+1}/{len(sequences)}: {len(seq)} positions")
        
        # Build constant velocity trajectory for this part
        traj = build_constant_velocity_trajectory(seq, part_name)
        all_trajectories.append(traj)
        
        # Add pause between movements
        if i < len(sequences) - 1:
            pause_between = 120  # 0.2s pause
            pause_traj = np.tile(traj[-1], (pause_between, 1))
            all_trajectories.append(pause_traj)
            print(f"   + 0.2s pause")
    
    combined_traj = np.vstack(all_trajectories)
    total_time = combined_traj.shape[0] / 600.0
    
    print(f"\n🎯 Total combined trajectory: {combined_traj.shape[0]} steps → {total_time:.1f}s")
    print("✅ Each movement part has constant angular velocity")
    
    return combined_traj

def create_smooth_segment(start_pos, end_pos, num_steps):
    """
    Create smooth trajectory segment between two positions with constant speed
    """
    if num_steps <= 1:
        return np.array([end_pos])
    
    # Simple linear interpolation for consistent timing
    trajectory = []
    for i in range(num_steps):
        alpha = i / (num_steps - 1)  # 0 to 1
        pos = start_pos * (1 - alpha) + end_pos * alpha
        trajectory.append(pos)
    
    return np.array(trajectory)

def build_movement_trajectory(pose, movement_name):
    """
    Build trajectory - executes the pose as designed WITHOUT forcing return to start
    """
    if not isinstance(pose, list):
        raise ValueError(f"Movement {movement_name} should be a list of positions")
    
    # Convert all positions to numpy arrays
    positions = [np.array(p) for p in pose]
    
    # FIXED TIMING: Always create exactly 600 steps (1 second)
    total_steps = 600
    
    if len(positions) == 2:
        # Simple 2-position movement: JUST execute pos1 → pos2 with pause
        pos1, pos2 = positions
        
        # Split into 2 parts: movement (400 steps) + pause (200 steps)
        steps_movement = 400  # 2/3 of time for movement
        steps_pause = 200     # 1/3 of time for pause at end
        
        # Part 1: pos1 → pos2 (smooth movement)
        movement_traj = create_smooth_segment(pos1, pos2, steps_movement)
        
        # Part 2: pause at pos2 (hold final position)
        pause_traj = np.tile(pos2, (steps_pause, 1))
        
        # Combine (total = 600 steps)
        complete_traj = np.vstack([movement_traj, pause_traj])
        
        print(f"{movement_name}: {steps_movement} move + {steps_pause} pause = {complete_traj.shape[0]} steps → 1.0s")
        
    else:
        # Multi-step movement: go through all positions sequentially with pause at end
        num_segments = len(positions) - 1
        
        # Reserve steps for final pause
        pause_steps = 150  # 0.25s pause at end
        movement_steps = total_steps - pause_steps
        steps_per_segment = movement_steps // num_segments
        
        trajectory_parts = []
        
        # Go through all positions in sequence (no return journey)
        for i in range(num_segments):
            segment = create_smooth_segment(positions[i], positions[i+1], steps_per_segment)
            trajectory_parts.append(segment)
        
        # Pause at final position
        pause_traj = np.tile(positions[-1], (pause_steps, 1))
        trajectory_parts.append(pause_traj)
        
        # Combine all parts
        complete_traj = np.vstack(trajectory_parts)
        
        # Ensure exactly 600 steps
        if complete_traj.shape[0] < total_steps:
            padding_needed = total_steps - complete_traj.shape[0]
            padding = np.tile(complete_traj[-1], (padding_needed, 1))
            complete_traj = np.vstack([complete_traj, padding])
        elif complete_traj.shape[0] > total_steps:
            complete_traj = complete_traj[:total_steps]
        
        print(f"{movement_name}: {num_segments} segments + pause = {complete_traj.shape[0]} steps → 1.0s")
    
    return complete_traj

def build_combined_trajectory(sequences, movement_name):
    """
    Build trajectory for combined movements with smooth transitions
    """
    print(f"📋 Building combined movement '{movement_name}' with {len(sequences)} parts...")
    
    all_trajectories = []
    
    for i, seq in enumerate(sequences):
        print(f"   Part {i+1}/{len(sequences)}: {len(seq)} positions")
        
        # Build trajectory for this part
        traj = build_movement_trajectory(seq, f"{movement_name}_part{i+1}")
        all_trajectories.append(traj)
        
        # Add SHORT pause between movements (except after last one)
        if i < len(sequences) - 1:
            pause_between = 120  # 0.2s pause between movements (shorter!)
            pause_traj = np.tile(traj[-1], (pause_between, 1))
            all_trajectories.append(pause_traj)
            print(f"   + 0.2s pause")
    
    combined_traj = np.vstack(all_trajectories)
    total_time = combined_traj.shape[0] / 600.0
    
    print(f"🎯 Total combined trajectory: {combined_traj.shape[0]} steps → {total_time:.1f}s")
    
    return combined_traj

def main():
    parser = argparse.ArgumentParser(
        description="Record synchronized EMG + prosthetic hand pose data"
    )
    parser.add_argument("--person_id", "-p", required=True,
                        help="Person ID (folder under data/)")
    parser.add_argument("--movement", "-m", required=True,
                        help="Movement name (subfolder under recordings/)")
    parser.add_argument("--out_root", "-o", default="data",
                        help="Root data directory (default: ./data)")
    parser.add_argument("--no_emg", action="store_true",
                        help="Disable EMG capture for testing without hardware")
    parser.add_argument("--no_prosthesis", action="store_true",
                        help="Disable prosthetic arm control; EMG-only recording")
    parser.add_argument("--hand_side", "-s", choices=["left", "right"],
                        default="left", help="Side of the prosthetic hand")
    parser.add_argument("--sync_iterations", type=int, default=5,
                        help="Warm-up sync iterations (default: 5)")
    parser.add_argument("--record_iterations", "-r", type=int, default=40,
                        help="Number of recording iterations (default: 40)")
    parser.add_argument("--video", action="store_true",
                        help="Enable simultaneous webcam video recording")
    parser.add_argument("--calibrate_emg", action="store_true",
                    help="Run EMG calibration (noise level and MVC detection)")
    args = parser.parse_args()

    # Prepare output directory
    exp_parent = os.path.join(
        args.out_root,
        args.person_id,
        "recordings",
        args.movement,
        "experiments"
    )

    exp_idx = 1
    while os.path.exists(os.path.join(exp_parent, str(exp_idx))):
        exp_idx += 1
    base_dir = os.path.join(exp_parent, str(exp_idx))
    os.makedirs(base_dir, exist_ok=True)

    # Calibration mode
    if args.calibrate_emg:
        calibrate_emg(base_dir, rest_time=10, mvc_time=10)
        return

    # Synchronization event for EMG and video threads
    sync_event = threading.Event() if (not args.no_emg or args.video) else None

    # Start raw EMG (and optional video) recorder in background
    # IMPORTANT: They will wait for sync_event before recording
    if not args.no_emg:
        stop_event, raw_thread, video_thread, raw_history, raw_timestamps, video_timestamps = \
            start_raw_emg_recorder(base_dir, enable_video=args.video, sync_event=sync_event)
    else:
        stop_event = raw_thread = video_thread = None
        raw_history = []
        raw_timestamps = []
        video_timestamps = []

    # Raw-EMG-only mode (no prosthesis movement, just record)
    if args.no_prosthesis:
        raw_seconds = 20.0
        print(f"Raw-only mode: recording raw EMG for {raw_seconds:.1f} seconds…")
        if sync_event is not None:
            sync_event.set()  # Start EMG/video capture immediately in raw-only mode
        try:
            time.sleep(raw_seconds)
        except KeyboardInterrupt:
            print("Interrupted early by user")
        stop_event.set()
        raw_thread.join()
        if video_thread:
            video_thread.join()
        if raw_history:
            np.save(join(base_dir, "raw_emg.npy"), np.vstack(raw_history))
            np.save(join(base_dir, "raw_timestamps.npy"),
                    np.array(raw_timestamps))
            if video_timestamps:
                np.save(join(base_dir, "video_timestamps.npy"), np.array(video_timestamps))
            print(f"Saved raw_emg.npy with {len(raw_history)} samples.")
        else:
            print("No raw data captured.")
        print("Done.")
        return

    # PROSTHESIS MODE
    # Initialize prosthetic arm
    arm = psyonicArm(hand=args.hand_side)
    arm.initSensors()
    arm.startComms()

    # Get movement sequences (handles both single and combined movements)
    try:
        sequences, is_combined = get_movement_sequences(args.movement)
    except ValueError as e:
        print(f"Error: {e}")
        arm.close()
        return

    if is_combined:
        smooth_traj = build_combined_constant_velocity_trajectory(sequences, args.movement)
    else:
        smooth_traj = build_constant_velocity_trajectory(sequences[0], args.movement)

    # Apply safety clipping
    joint_mins = np.array([0, 0, 0, 0, 0, -120])
    joint_maxs = np.array([120, 120, 120, 120, 120, 0])
    smooth_traj = np.clip(smooth_traj, joint_mins, joint_maxs)

    # Display trajectory info
    print(f"Final trajectory shape: {smooth_traj.shape}")
    print(f"Estimated execution time: {smooth_traj.shape[0]/600:.2f}s")
    print(f"Start pose: {smooth_traj[0]}")
    print(f"End pose: {smooth_traj[-1]}")

    # CRITICAL FIX: Warm-up iterations WITHOUT recording
    print(f"\n=== Running {args.sync_iterations} warm-up iterations (NOT recording) ===")
    print("User should synchronize with the hand movement during this phase...")
    
    for i in range(args.sync_iterations):
        print(f"Warm-up iteration {i+1}/{args.sync_iterations}")
        arm.mainControlLoop(posDes=smooth_traj, period=1, emg=None)

    print("\n=== Warm-up complete. Starting synchronized recording NOW ===")
    
    # CRITICAL FIX: Set sync event AFTER warm-up, RIGHT BEFORE recording iterations
    recording_start_time = time.time()
    if sync_event is not None:
        sync_event.set()  # Signal EMG/video threads to start recording NOW
    
    # Small delay to ensure threads have received the signal
    time.sleep(0.001)

    # Recording loop - EMG is now recording in sync
    all_records = []
    headers = None

    for itr in range(1, args.record_iterations + 1):
        print(f"Recording iteration {itr}/{args.record_iterations}...")
        arm.resetRecording()
        arm.recording = True
        arm.mainControlLoop(posDes=smooth_traj, period=1, emg=None)
        arm.recording = False

        # Store angle data
        raw_data = arm.recordedData
        if headers is None:
            headers = raw_data[0]  # first row is header names
        data_rows = raw_data[1:]
        all_records.extend(data_rows)

    # Recording complete - stop EMG/video immediately
    recording_end_time = time.time()
    print(f"\nRecording complete. Duration: {recording_end_time - recording_start_time:.2f}s")

    # Stop raw EMG (and video) recorder
    if not args.no_emg:
        stop_event.set()
        raw_thread.join()
        if video_thread:
            video_thread.join()

    # Save all data
    if raw_history:
        np.save(join(base_dir, "raw_emg.npy"), np.vstack(raw_history))
        raw_timestamps_unique = make_timestamps_unique(raw_timestamps)
        np.save(join(base_dir, "raw_timestamps.npy"), np.array(raw_timestamps_unique))
        print(f"Saved raw_emg.npy with {len(raw_history)} samples.")
        print(f"EMG duration: {raw_timestamps[-1] - raw_timestamps[0]:.2f}s")
    
    if video_timestamps:
        np.save(join(base_dir, "video_timestamps.npy"), np.array(video_timestamps))
        print(f"Saved video_timestamps.npy with {len(video_timestamps)} frames.")

    # Save angle data
    if all_records:
        rec = np.array(all_records, dtype=float)
        ts = rec[:, 0]
        ts -= ts[0]  # normalize timestamps
        np.save(join(base_dir, "angles.npy"), rec)
        angle_timestamps_unique = make_timestamps_unique(ts)
        np.save(join(base_dir, "angle_timestamps.npy"), angle_timestamps_unique)
        with open(join(base_dir, "angles_header.txt"), "w") as f:
            f.write(",".join(headers))
        print(f"Saved angles.npy with {len(rec)} frames.")
        print(f"Angles duration: {ts[-1]:.2f}s")
        
        # Quick sync check
        if raw_history:
            emg_duration = raw_timestamps[-1] - raw_timestamps[0]
            angles_duration = ts[-1]
            print(f"\n=== Sync Quality ===")
            print(f"Duration difference: {abs(emg_duration - angles_duration):.2f}s")
            if abs(emg_duration - angles_duration) < 1.0:
                print("✅ Excellent synchronization!")
            else:
                print("⚠️  Check synchronization - durations differ significantly")

    arm.close()

if __name__ == "__main__":
    main()