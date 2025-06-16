#!/usr/bin/env python3
"""
Hardware Synchronization Test Script

This script tests the timing consistency between EMG board and psyonic hand
to identify hardware clock drift, buffer issues, or communication delays.
"""

import time
import threading
import numpy as np
import matplotlib.pyplot as plt
from helpers.EMGClass import EMG
from psyonicHand import psyonicArm
import argparse
import os
from datetime import datetime

class SynchronizationTester:
    def __init__(self, hand_side='left', test_duration=60):
        self.hand_side = hand_side
        self.test_duration = test_duration
        
        # Data storage
        self.emg_data = []
        self.emg_timestamps = []
        self.emg_os_timestamps = []  # Raw OS timestamps from EMG
        
        self.hand_data = []
        self.hand_timestamps = []
        
        # System timestamps (computer clock)
        self.emg_system_times = []
        self.hand_system_times = []
        
        # Control flags
        self.stop_event = threading.Event()
        self.sync_start_time = None
        
        # Statistics
        self.emg_packet_intervals = []
        self.hand_packet_intervals = []
        
    def initialize_hardware(self):
        """Initialize both EMG and hand hardware."""
        print("Initializing hardware...")
        
        try:
            # Initialize EMG
            print("  Connecting to EMG board...")
            self.emg = EMG()
            # Wait for first packet to establish connection
            while getattr(self.emg, 'OS_time', None) is None:
                self.emg.readEMG()
            print("  ✅ EMG connected")
            
            # Initialize prosthetic hand
            print("  Connecting to prosthetic hand...")
            self.arm = psyonicArm(hand=self.hand_side)
            print("  Initializing hand sensors...")
            self.arm.initSensors()
            print("  Starting hand communications...")
            self.arm.startComms()
            
            # Wait a moment for the hand to stabilize
            time.sleep(2)
            
            # Verify hand is responding
            current_pos = self.arm.getCurPos()
            if all(pos == -1 for pos in current_pos):
                print("  ⚠️ Hand not responding properly, but proceeding...")
            else:
                print(f"  ✅ Hand connected - current position: {[f'{p:.1f}' for p in current_pos]}")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Hardware initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def emg_capture_thread(self):
        """Thread function for capturing EMG data with precise timing."""
        print("EMG capture thread started")
        
        # Wait for synchronization signal
        self.sync_start_time = time.time()
        
        # Reset data storage
        self.emg_data.clear()
        self.emg_timestamps.clear()
        self.emg_os_timestamps.clear()
        self.emg_system_times.clear()
        
        # Get initial EMG timestamp
        self.emg.readEMG()
        emg_start_time = self.emg.OS_time
        
        last_system_time = time.time()
        packet_count = 0
        
        while not self.stop_event.is_set():
            try:
                # Capture system time before reading
                system_time_before = time.time()
                
                # Read EMG data
                self.emg.readEMG()
                
                # Capture system time after reading
                system_time_after = time.time()
                read_duration = system_time_after - system_time_before
                
                # Store data
                self.emg_data.append(list(self.emg.rawEMG))
                emg_relative_time = (self.emg.OS_time - emg_start_time) / 1e6  # Convert to seconds
                self.emg_timestamps.append(emg_relative_time)
                self.emg_os_timestamps.append(self.emg.OS_time)
                self.emg_system_times.append(system_time_after - self.sync_start_time)
                
                # Calculate packet interval
                current_system_time = system_time_after
                if packet_count > 0:
                    interval = current_system_time - last_system_time
                    self.emg_packet_intervals.append(interval)
                
                last_system_time = current_system_time
                packet_count += 1
                
                # Log communication delays
                if read_duration > 0.01:  # More than 10ms to read
                    print(f"EMG: Long read time: {read_duration*1000:.1f}ms")
                    
            except Exception as e:
                print(f"EMG capture error: {e}")
                break
        
        print(f"EMG capture thread finished. Captured {len(self.emg_data)} packets")
    
    def hand_capture_thread(self):
        """Thread function for capturing hand data with precise timing."""
        print("Hand capture thread started")
        
        # Wait for synchronization (same start time as EMG)
        while self.sync_start_time is None:
            time.sleep(0.001)
        
        # Reset data storage
        self.hand_data.clear()
        self.hand_timestamps.clear()
        self.hand_system_times.clear()
        
        last_system_time = time.time()
        packet_count = 0
        
        # Start recording on the hand
        self.arm.resetRecording()
        self.arm.recording = True
        
        # Get current position to maintain
        current_pos = self.arm.getCurPos()
        
        while not self.stop_event.is_set():
            try:
                # Capture system time before reading
                system_time_before = time.time()
                
                # Manually trigger one iteration of the hand's communication
                # This is similar to what happens inside mainControlLoop
                if self.arm.isValidCommand(current_pos):
                    self.arm.handCom = current_pos
                
                # Update timestamp and add log entry (this captures the hand's internal timing)
                self.arm.timestamp = time.time()
                if self.arm.recording:
                    self.arm.addLogEntry(emg=None)
                
                # Capture system time after reading
                system_time_after = time.time()
                read_duration = system_time_after - system_time_before
                
                # Store system timestamp
                self.hand_system_times.append(system_time_after - self.sync_start_time)
                
                # Calculate packet interval
                current_system_time = system_time_after
                if packet_count > 0:
                    interval = current_system_time - last_system_time
                    self.hand_packet_intervals.append(interval)
                
                last_system_time = current_system_time
                packet_count += 1
                
                # Log communication delays
                if read_duration > 0.01:  # More than 10ms to read
                    print(f"Hand: Long read time: {read_duration*1000:.1f}ms")
                
                # Match the hand's expected loop rate
                time.sleep(1/(self.arm.loopRate * self.arm.Hz))
                
            except Exception as e:
                print(f"Hand capture error: {e}")
                break
        
        # Stop recording and collect data
        self.arm.recording = False
        
        # Extract recorded data
        if hasattr(self.arm, 'recordedData') and len(self.arm.recordedData) > 1:
            headers = self.arm.recordedData[0]
            data_rows = self.arm.recordedData[1:]
            
            if len(data_rows) > 0:
                hand_array = np.array(data_rows, dtype=float)
                hand_timestamps = hand_array[:, 0] - hand_array[0, 0]  # Normalize timestamps
                
                self.hand_data = hand_array
                self.hand_timestamps = hand_timestamps
                
                print(f"Hand data shape: {hand_array.shape}")
                print(f"Hand timestamp range: {hand_timestamps[0]:.3f} to {hand_timestamps[-1]:.3f}s")
        
        print(f"Hand capture thread finished. Captured {len(self.hand_data)} packets")
    
    def run_synchronization_test(self):
        """Run the complete synchronization test."""
        print(f"\n{'='*60}")
        print(f"HARDWARE SYNCHRONIZATION TEST")
        print(f"Duration: {self.test_duration} seconds")
        print(f"{'='*60}")
        
        if not self.initialize_hardware():
            return False
        
        # Start capture threads
        emg_thread = threading.Thread(target=self.emg_capture_thread, daemon=True)
        hand_thread = threading.Thread(target=self.hand_capture_thread, daemon=True)
        
        print(f"\nStarting synchronized data capture...")
        start_time = time.time()
        
        emg_thread.start()
        hand_thread.start()
        
        # Run for specified duration
        try:
            time.sleep(self.test_duration)
        except KeyboardInterrupt:
            print("\nTest interrupted by user")
        
        # Stop capture
        print("Stopping capture...")
        self.stop_event.set()
        
        # Wait for threads to finish
        emg_thread.join(timeout=2)
        hand_thread.join(timeout=2)
        
        actual_duration = time.time() - start_time
        print(f"Actual test duration: {actual_duration:.3f} seconds")
        
        return True
    
    def analyze_timing_consistency(self):
        """Analyze timing consistency and drift between systems."""
        print(f"\n{'='*60}")
        print("TIMING ANALYSIS")
        print(f"{'='*60}")
        
        # Convert data to numpy arrays
        emg_timestamps = np.array(self.emg_timestamps)
        emg_system_times = np.array(self.emg_system_times)
        hand_system_times = np.array(self.hand_system_times)
        hand_timestamps = np.array(self.hand_timestamps)
        
        # Calculate sampling rates
        if len(emg_timestamps) > 1:
            emg_duration = emg_timestamps[-1] - emg_timestamps[0]
            emg_rate = (len(emg_timestamps) - 1) / emg_duration if emg_duration > 0 else 0
            emg_system_duration = emg_system_times[-1] - emg_system_times[0]
            emg_system_rate = (len(emg_system_times) - 1) / emg_system_duration if emg_system_duration > 0 else 0
        else:
            emg_rate = emg_system_rate = 0
        
        if len(hand_timestamps) > 1:
            hand_duration = hand_timestamps[-1] - hand_timestamps[0]
            hand_rate = (len(hand_timestamps) - 1) / hand_duration if hand_duration > 0 else 0
            hand_system_duration = hand_system_times[-1] - hand_system_times[0]
            hand_system_rate = (len(hand_system_times) - 1) / hand_system_duration if hand_system_duration > 0 else 0
        else:
            hand_rate = hand_system_rate = 0
        
        print(f"\nSAMPLING RATES:")
        print(f"  EMG (internal clock):  {emg_rate:.3f} Hz")
        print(f"  EMG (system clock):    {emg_system_rate:.3f} Hz")
        print(f"  Hand (internal clock): {hand_rate:.3f} Hz")
        print(f"  Hand (system clock):   {hand_system_rate:.3f} Hz")
        
        # Calculate packet interval statistics
        if self.emg_packet_intervals:
            emg_intervals = np.array(self.emg_packet_intervals)
            print(f"\nEMG PACKET INTERVALS:")
            print(f"  Mean: {np.mean(emg_intervals)*1000:.3f} ms")
            print(f"  Std:  {np.std(emg_intervals)*1000:.3f} ms")
            print(f"  Min:  {np.min(emg_intervals)*1000:.3f} ms")
            print(f"  Max:  {np.max(emg_intervals)*1000:.3f} ms")
        
        if self.hand_packet_intervals:
            hand_intervals = np.array(self.hand_packet_intervals)
            print(f"\nHAND PACKET INTERVALS:")
            print(f"  Mean: {np.mean(hand_intervals)*1000:.3f} ms")
            print(f"  Std:  {np.std(hand_intervals)*1000:.3f} ms")
            print(f"  Min:  {np.min(hand_intervals)*1000:.3f} ms")
            print(f"  Max:  {np.max(hand_intervals)*1000:.3f} ms")
        
        # Check for clock drift
        print(f"\nCLOCK DRIFT ANALYSIS:")
        
        if len(emg_timestamps) > 10 and len(emg_system_times) > 10:
            # Compare EMG internal clock vs system clock
            emg_drift_slope = np.polyfit(emg_system_times, emg_timestamps, 1)[0]
            emg_drift_rate = (emg_drift_slope - 1) * 1000  # ms drift per second
            print(f"  EMG clock drift: {emg_drift_rate:.3f} ms/s")
            
            if abs(emg_drift_rate) > 1:
                print(f"  ⚠️  EMG has significant clock drift!")
        
        if len(hand_timestamps) > 10 and len(hand_system_times) > 10:
            # Compare Hand internal clock vs system clock
            hand_drift_slope = np.polyfit(hand_system_times, hand_timestamps, 1)[0]
            hand_drift_rate = (hand_drift_slope - 1) * 1000  # ms drift per second
            print(f"  Hand clock drift: {hand_drift_rate:.3f} ms/s")
            
            if abs(hand_drift_rate) > 1:
                print(f"  ⚠️  Hand has significant clock drift!")
        
        # Cross-system synchronization check
        if len(emg_system_times) > 1 and len(hand_system_times) > 1:
            # Find overlapping time range
            start_time = max(emg_system_times[0], hand_system_times[0])
            end_time = min(emg_system_times[-1], hand_system_times[-1])
            
            if end_time > start_time:
                print(f"  Overlapping capture time: {end_time - start_time:.3f} seconds")
                
                # Calculate relative timing drift between systems
                relative_duration_emg = emg_system_times[-1] - emg_system_times[0]
                relative_duration_hand = hand_system_times[-1] - hand_system_times[0]
                
                if relative_duration_emg > 0 and relative_duration_hand > 0:
                    relative_drift = (relative_duration_emg - relative_duration_hand) / max(relative_duration_emg, relative_duration_hand) * 100
                    print(f"  Relative duration drift: {relative_drift:.3f}%")
            else:
                print(f"  ⚠️ No overlapping capture time - synchronization failed!")
        
        return {
            'emg_rate': emg_rate,
            'hand_rate': hand_rate,
            'emg_system_rate': emg_system_rate,
            'hand_system_rate': hand_system_rate,
            'emg_intervals': self.emg_packet_intervals,
            'hand_intervals': self.hand_packet_intervals
        }
    
    def create_timing_plots(self):
        """Create detailed timing analysis plots."""
        if len(self.emg_timestamps) == 0 and len(self.hand_timestamps) == 0:
            print("No data to plot")
            return
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        # Plot 1: Internal timestamps over time
        if len(self.emg_timestamps) > 0:
            axes[0, 0].plot(self.emg_system_times, self.emg_timestamps, 'b.-', alpha=0.7, label='EMG')
        if len(self.hand_timestamps) > 0:
            axes[0, 0].plot(self.hand_system_times, self.hand_timestamps, 'r.-', alpha=0.7, label='Hand')
        axes[0, 0].set_xlabel('System Time (s)')
        axes[0, 0].set_ylabel('Internal Timestamp (s)')
        axes[0, 0].set_title('Internal Timestamps vs System Time')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot 2: Packet intervals
        if self.emg_packet_intervals:
            axes[0, 1].plot(self.emg_packet_intervals[::10], 'b.-', alpha=0.7, label='EMG')
        if self.hand_packet_intervals:
            axes[0, 1].plot(self.hand_packet_intervals[::10], 'r.-', alpha=0.7, label='Hand')
        axes[0, 1].set_xlabel('Packet Number (every 10th)')
        axes[0, 1].set_ylabel('Interval (s)')
        axes[0, 1].set_title('Packet Interval Consistency')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot 3: Clock drift (internal vs system)
        if len(self.emg_timestamps) > 1:
            emg_expected = np.array(self.emg_system_times) - self.emg_system_times[0]
            emg_actual = np.array(self.emg_timestamps)
            emg_drift = emg_actual - emg_expected
            axes[1, 0].plot(self.emg_system_times, emg_drift * 1000, 'b.-', alpha=0.7)
        axes[1, 0].set_xlabel('System Time (s)')
        axes[1, 0].set_ylabel('Clock Drift (ms)')
        axes[1, 0].set_title('EMG Clock Drift Over Time')
        axes[1, 0].grid(True)
        
        if len(self.hand_timestamps) > 1:
            hand_expected = np.array(self.hand_system_times) - self.hand_system_times[0]
            hand_actual = np.array(self.hand_timestamps)
            hand_drift = hand_actual - hand_expected
            axes[1, 1].plot(self.hand_system_times, hand_drift * 1000, 'r.-', alpha=0.7)
        axes[1, 1].set_xlabel('System Time (s)')
        axes[1, 1].set_ylabel('Clock Drift (ms)')
        axes[1, 1].set_title('Hand Clock Drift Over Time')
        axes[1, 1].grid(True)
        
        # Plot 4: Interval histograms
        if self.emg_packet_intervals:
            axes[2, 0].hist(np.array(self.emg_packet_intervals) * 1000, bins=50, alpha=0.7, color='blue')
        axes[2, 0].set_xlabel('Packet Interval (ms)')
        axes[2, 0].set_ylabel('Frequency')
        axes[2, 0].set_title('EMG Packet Interval Distribution')
        axes[2, 0].grid(True)
        
        if self.hand_packet_intervals:
            axes[2, 1].hist(np.array(self.hand_packet_intervals) * 1000, bins=50, alpha=0.7, color='red')
        axes[2, 1].set_xlabel('Packet Interval (ms)')
        axes[2, 1].set_ylabel('Frequency')
        axes[2, 1].set_title('Hand Packet Interval Distribution')
        axes[2, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def save_test_results(self, output_dir):
        """Save test results for further analysis."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save timing data
        np.save(os.path.join(output_dir, f'emg_timestamps_{timestamp}.npy'), self.emg_timestamps)
        np.save(os.path.join(output_dir, f'emg_system_times_{timestamp}.npy'), self.emg_system_times)
        np.save(os.path.join(output_dir, f'hand_timestamps_{timestamp}.npy'), self.hand_timestamps)
        np.save(os.path.join(output_dir, f'hand_system_times_{timestamp}.npy'), self.hand_system_times)
        
        # Save interval data
        if self.emg_packet_intervals:
            np.save(os.path.join(output_dir, f'emg_intervals_{timestamp}.npy'), self.emg_packet_intervals)
        if self.hand_packet_intervals:
            np.save(os.path.join(output_dir, f'hand_intervals_{timestamp}.npy'), self.hand_packet_intervals)
        
        print(f"✅ Test results saved to {output_dir}")
    
    def cleanup(self):
        """Clean up hardware connections."""
        try:
            if hasattr(self, 'emg'):
                self.emg.exitEvent.set()
                if hasattr(self.emg, 'shutdown'):
                    self.emg.shutdown()
                else:
                    if hasattr(self.emg, 'sock'):
                        self.emg.sock.close()
                    if hasattr(self.emg, 'ctx'):
                        self.emg.ctx.term()
            
            if hasattr(self, 'arm'):
                self.arm.close()
                
        except Exception as e:
            print(f"Cleanup error: {e}")

def main():
    parser = argparse.ArgumentParser(description='Test hardware synchronization between EMG and prosthetic hand')
    parser.add_argument('--duration', '-d', type=int, default=30, 
                       help='Test duration in seconds (default: 30)')
    parser.add_argument('--hand_side', '-s', choices=['left', 'right'], default='left',
                       help='Side of prosthetic hand')
    parser.add_argument('--output_dir', '-o', default='sync_test_results',
                       help='Output directory for results')
    args = parser.parse_args()
    
    tester = SynchronizationTester(hand_side=args.hand_side, test_duration=args.duration)
    
    try:
        if tester.run_synchronization_test():
            results = tester.analyze_timing_consistency()
            tester.create_timing_plots()
            tester.save_test_results(args.output_dir)
            
            print(f"\n{'='*60}")
            print("TEST RECOMMENDATIONS")
            print(f"{'='*60}")
            
            # Provide specific recommendations based on results
            if results['emg_rate'] > 0 and results['hand_rate'] > 0:
                rate_diff = abs(results['emg_rate'] - results['hand_rate'])
                if rate_diff > 10:
                    print("⚠️  Large sampling rate difference detected!")
                    print("   Consider using a common time base or interpolation")
                
                if len(tester.emg_packet_intervals) > 0:
                    emg_jitter = np.std(tester.emg_packet_intervals) * 1000
                    if emg_jitter > 5:
                        print(f"⚠️  High EMG timing jitter: {emg_jitter:.1f}ms")
                        print("   Check EMG board connection and USB bandwidth")
                
                if len(tester.hand_packet_intervals) > 0:
                    hand_jitter = np.std(tester.hand_packet_intervals) * 1000
                    if hand_jitter > 5:
                        print(f"⚠️  High Hand timing jitter: {hand_jitter:.1f}ms")
                        print("   Check prosthetic hand communication settings")
            
            print("✅ Hardware synchronization test complete!")
        else:
            print("❌ Hardware synchronization test failed!")
            
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Test error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        tester.cleanup()

if __name__ == "__main__":
    main()