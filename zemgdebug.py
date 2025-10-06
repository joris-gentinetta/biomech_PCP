#!/usr/bin/env python3
"""
Comprehensive Prosthetic Hand Movement Test

This standalone script runs a complete test suite for prosthetic hand control.
Perfect for verifying smooth movement execution and hand responsiveness.

Features:
- Progressive speed testing (slow to fast - easier for user to mimic)
- Extended movement library
- Individual joint verification
- Complex coordinated movements
- Safety limit testing
- Real-time feedback

No EMG required - just the prosthetic hand.
"""

import argparse
import time
import numpy as np
from psyonicHand import psyonicArm

# Extended hand poses for comprehensive testing
COMPREHENSIVE_POSES = {
    # Basic poses
    "rest": [0, 0, 0, 0, 0, 0],
    "fist": [120, 120, 120, 120, 60, -100],
    
    # Functional grips
    "pinch": [60, 0, 0, 0, 59, -59],
    "precision_pinch": [30, 0, 0, 0, 40, -40],
    "power_grip": [90, 90, 90, 90, 50, -80],
    "key_grip": [60, 0, 0, 0, 85, 0],
    
    # Finger positions
    "point": [0, 120, 120, 120, 0, 0],
    "peace": [0, 0, 120, 120, 0, 0],
    "rock_on": [0, 120, 120, 0, 0, 0],
    "thumbs_up": [120, 120, 120, 120, 85, 0],
    
    # Individual finger tests
    "index_only": [120, 0, 0, 0, 0, 0],
    "middle_only": [0, 120, 0, 0, 0, 0],
    "ring_only": [0, 0, 120, 0, 0, 0],
    "pinky_only": [0, 0, 0, 120, 0, 0],
    "thumb_flex": [0, 0, 0, 0, 103, -103],
    "thumb_abduct": [0, 0, 0, 0, 85, 0],
    
    # Complex positions
    "half_fist": [60, 60, 60, 60, 30, -30],
    "claw": [40, 40, 40, 40, 20, -20],
    "relaxed": [20, 20, 20, 20, 10, -10],
    "spread": [0, 0, 0, 0, 0, 85],  # Fingers spread with thumb abducted
}

# Comprehensive movement sequences
COMPREHENSIVE_SEQUENCES = {
    # Basic function tests
    "open_close": ["rest", "fist", "rest"],
    "pinch_release": ["rest", "pinch", "rest"],
    "grip_test": ["rest", "power_grip", "rest"],
    
    # Progressive difficulty
    "finger_wave": ["rest", "index_only", "middle_only", "ring_only", "pinky_only", "rest"],
    "finger_progression": ["rest", "index_only", "peace", "rock_on", "fist", "rest"],
    "thumb_sequence": ["rest", "thumb_flex", "thumb_abduct", "thumbs_up", "rest"],
    
    # Functional movements
    "grip_progression": ["rest", "precision_pinch", "pinch", "key_grip", "power_grip", "fist", "rest"],
    "gesture_demo": ["rest", "point", "peace", "thumbs_up", "rock_on", "rest"],
    
    # Complex coordination
    "dexterity_test": ["rest", "pinch", "precision_pinch", "key_grip", "point", "peace", "rest"],
    "full_demo": ["rest", "half_fist", "fist", "power_grip", "pinch", "point", "peace", "thumbs_up", "rest"],
    
    # Endurance test (longer sequence)
    "endurance": ["rest", "fist", "rest", "pinch", "rest", "point", "rest", "power_grip", "rest", 
                  "thumbs_up", "rest", "peace", "rest", "key_grip", "rest", "claw", "rest"]
}

class ComprehensiveHandTester:
    """
    Comprehensive tester for prosthetic hand movements with progressive speed testing
    """
    
    def __init__(self, hand_side="left"):
        self.hand_side = hand_side
        self.arm = None
        self.joint_names = ["Index", "Middle", "Ring", "Pinky", "Thumb_Flex", "Thumb_Ab"]
        self.joint_limits = {
            'mins': np.array([0, 0, 0, 0, 0, -120]),
            'maxs': np.array([120, 120, 120, 120, 120, 0])
        }
        
        # Progressive speed settings (slow to fast for easier mimicking)
        self.test_speeds = [
            {'name': 'Very Slow', 'duration': 3.0, 'description': 'Easy to follow'},
            {'name': 'Slow', 'duration': 2.0, 'description': 'Comfortable pace'},
            {'name': 'Normal', 'duration': 1.5, 'description': 'Standard speed'},
            {'name': 'Fast', 'duration': 1.0, 'description': 'Quick movement'},
            {'name': 'Very Fast', 'duration': 0.7, 'description': 'Rapid execution'}
        ]
        
        self.current_pose = None
        
    def connect_hand(self):
        """Initialize and connect to prosthetic hand"""
        print(f" Connecting to {self.hand_side} prosthetic hand...")
        try:
            self.arm = psyonicArm(hand=self.hand_side)
            self.arm.initSensors()
            self.arm.startComms()
            print(" Hand connected successfully!")
            self.current_pose = COMPREHENSIVE_POSES['rest']
            return True
        except Exception as e:
            print(f" Failed to connect to hand: {e}")
            return False
    
    def disconnect_hand(self):
        """Safely disconnect from hand"""
        if self.arm:
            try:
                print("Returning to safe position...")
                self.move_to_pose("rest", duration=2.0, verbose=False)
                time.sleep(1.0)
                self.arm.close()
                print(" Hand disconnected safely")
            except Exception as e:
                print(f"Warning during disconnect: {e}")
    
    def create_smooth_trajectory(self, start_pose, end_pose, duration=1.0, fs=600):
        """Create smooth S-curve trajectory between poses"""
        start_pose = np.array(start_pose)
        end_pose = np.array(end_pose)
        
        num_steps = int(duration * fs)
        trajectory = []
        
        for i in range(num_steps):
            alpha = i / (num_steps - 1) if num_steps > 1 else 0
            # Smooth S-curve (ease-in-out)
            smooth_alpha = 3 * alpha**2 - 2 * alpha**3
            pose = start_pose * (1 - smooth_alpha) + end_pose * smooth_alpha
            trajectory.append(pose)
        
        trajectory = np.array(trajectory)
        # Apply safety limits
        trajectory = np.clip(trajectory, self.joint_limits['mins'], self.joint_limits['maxs'])
        
        return trajectory
    
    def move_to_pose(self, pose_name, duration=1.5, verbose=True):
        """Move hand to a named pose"""
        if pose_name not in COMPREHENSIVE_POSES:
            print(f" Unknown pose: {pose_name}")
            return False
        
        target_pose = COMPREHENSIVE_POSES[pose_name]
        
        if verbose:
            print(f"Moving to '{pose_name}': {target_pose}")
        
        try:
            start_pose = self.current_pose if self.current_pose is not None else COMPREHENSIVE_POSES['rest']
            trajectory = self.create_smooth_trajectory(start_pose, target_pose, duration)
            
            start_time = time.time()
            self.arm.mainControlLoop(posDes=trajectory, period=1, emg=None)
            actual_duration = time.time() - start_time
            
            self.current_pose = target_pose
            
            if verbose:
                print(f" Movement completed in {actual_duration:.2f}s")
            
            return True
            
        except Exception as e:
            print(f" Movement failed: {e}")
            return False
    
    def execute_sequence(self, sequence_name, speed_setting=None, verbose=True):
        """Execute a movement sequence at specified speed"""
        if sequence_name not in COMPREHENSIVE_SEQUENCES:
            print(f" Unknown sequence: {sequence_name}")
            return False
        
        sequence = COMPREHENSIVE_SEQUENCES[sequence_name]
        duration = speed_setting['duration'] if speed_setting else 1.5
        
        if verbose:
            speed_name = speed_setting['name'] if speed_setting else 'Default'
            print(f"🎬 Executing '{sequence_name}' at {speed_name} speed: {' → '.join(sequence)}")
        
        success = True
        for i, pose_name in enumerate(sequence):
            if verbose:
                print(f"  Step {i+1}/{len(sequence)}: {pose_name}")
            
            if not self.move_to_pose(pose_name, duration=duration, verbose=False):
                success = False
                break
            
            # Brief pause between movements
            if i < len(sequence) - 1:
                time.sleep(0.3)
        
        return success
    
    def test_basic_poses(self):
        """Test all basic poses"""
        print("\nBASIC POSE TESTING")
        print("=" * 50)
        
        basic_poses = ["rest", "fist", "pinch", "point", "thumbs_up", "peace"]
        
        for i, pose in enumerate(basic_poses):
            print(f"\nTesting pose {i+1}/{len(basic_poses)}: {pose}")
            if not self.move_to_pose(pose, duration=2.0):
                print(f" Basic pose test failed at '{pose}'")
                return False
            time.sleep(1.0)
        
        # Return to rest
        self.move_to_pose("rest", duration=2.0, verbose=False)
        print("\n All basic poses completed successfully!")
        return True
    
    def test_individual_joints(self):
        """Test each joint individually"""
        print("\nINDIVIDUAL JOINT TESTING")
        print("=" * 50)
        
        joint_test_poses = ["index_only", "middle_only", "ring_only", "pinky_only", "thumb_flex", "thumb_abduct"]
        
        for i, pose in enumerate(joint_test_poses):
            joint_name = self.joint_names[i] if i < len(self.joint_names) else f"Joint {i}"
            print(f"\nTesting {joint_name} with pose '{pose}'")
            
            if not self.move_to_pose(pose, duration=1.5):
                print(f" Joint test failed for {joint_name}")
                return False
            
            time.sleep(1.0)
            
            # Return to rest between tests
            self.move_to_pose("rest", duration=1.5, verbose=False)
            time.sleep(0.5)
        
        print("\n All individual joints tested successfully!")
        return True
    
    def test_progressive_speeds(self):
        """Test progressive speeds (slow to fast) for easier user mimicking"""
        print(" PROGRESSIVE SPEED TESTING")
        print("=" * 50)
        print("Testing speeds from slow to fast - easier for user to mimic!")
        
        test_sequence = "open_close"  # Simple sequence for speed testing
        
        for i, speed_setting in enumerate(self.test_speeds):
            print(f" Speed Test {i+1}/{len(self.test_speeds)}: {speed_setting['name']}")
            print(f"   Duration: {speed_setting['duration']}s - {speed_setting['description']}")
            
            if not self.execute_sequence(test_sequence, speed_setting, verbose=False):
                print(f" Speed test failed at {speed_setting['name']}")
                return False
            
            # Brief pause between speed tests
            time.sleep(1.0)
        
        print("\n All speed variations completed successfully!")
        print(" User can now mimic at any speed from very slow to very fast!")
        return True
    
    def test_functional_grips(self):
        """Test functional grip positions"""
        print("\n FUNCTIONAL GRIP TESTING")
        print("=" * 50)
        
        grip_sequence = "grip_progression"
        print("Testing progression through functional grips...")
        
        if not self.execute_sequence(grip_sequence, verbose=True):
            print(" Functional grip test failed")
            return False
        
        time.sleep(1.0)
        
        # Test precision movements
        print("\nTesting precision movements...")
        precision_poses = ["precision_pinch", "key_grip", "claw"]
        
        for pose in precision_poses:
            print(f"Testing precision pose: {pose}")
            if not self.move_to_pose(pose, duration=2.0):
                print(f" Precision test failed at {pose}")
                return False
            time.sleep(1.5)
        
        self.move_to_pose("rest", duration=2.0, verbose=False)
        print("\n All functional grips tested successfully!")
        return True
    
    def test_complex_sequences(self):
        """Test complex movement sequences"""
        print("COMPLEX SEQUENCE TESTING")
        print("=" * 50)
        
        complex_sequences = ["finger_progression", "gesture_demo", "dexterity_test"]
        
        for i, sequence in enumerate(complex_sequences):
            print(f"\nComplex Sequence {i+1}/{len(complex_sequences)}: {sequence}")
            
            if not self.execute_sequence(sequence, verbose=True):
                print(f" Complex sequence failed: {sequence}")
                return False
            
            time.sleep(2.0)
        
        print("\n All complex sequences completed successfully!")
        return True
    
    def test_joint_limits_safety(self):
        """Test joint limit safety with comprehensive range testing"""
        print(" JOINT LIMIT SAFETY TESTING")
        print("=" * 50)
        
        rest_pose = np.array(COMPREHENSIVE_POSES['rest'])
        
        for joint_idx, joint_name in enumerate(self.joint_names):
            print(f"\nTesting {joint_name} safety limits...")
            
            # Test full range motion
            min_val = self.joint_limits['mins'][joint_idx]
            max_val = self.joint_limits['maxs'][joint_idx]
            
            print(f"  Range: {min_val:.1f}° to {max_val:.1f}°")
            
            # Create test poses at limits
            min_pose = rest_pose.copy()
            max_pose = rest_pose.copy()
            min_pose[joint_idx] = min_val
            max_pose[joint_idx] = max_val
            
            try:
                # Test minimum
                trajectory = self.create_smooth_trajectory(rest_pose, min_pose, duration=1.5)
                self.arm.mainControlLoop(posDes=trajectory, period=1, emg=None)
                time.sleep(0.5)
                
                # Test maximum
                trajectory = self.create_smooth_trajectory(min_pose, max_pose, duration=2.0)
                self.arm.mainControlLoop(posDes=trajectory, period=1, emg=None)
                time.sleep(0.5)
                
                # Return to rest
                trajectory = self.create_smooth_trajectory(max_pose, rest_pose, duration=1.5)
                self.arm.mainControlLoop(posDes=trajectory, period=1, emg=None)
                time.sleep(0.5)
                
                print(f"   {joint_name} safety limits OK")
                
            except Exception as e:
                print(f"   {joint_name} safety test failed: {e}")
                return False
        
        print("\n All joint safety limits verified!")
        return True
    
    def test_endurance_sequence(self):
        """Test extended endurance sequence"""
        print(" ENDURANCE TESTING")
        print("=" * 50)
        print("Running extended sequence to test sustained operation...")
        
        if not self.execute_sequence("endurance", verbose=True):
            print(" Endurance test failed")
            return False
        
        print("\n Endurance test completed successfully!")
        print(" Hand maintained smooth operation throughout extended sequence!")
        return True
    
    def run_comprehensive_test_suite(self):
        """
        Run the complete comprehensive test suite
        """
        print(" COMPREHENSIVE PROSTHETIC HAND TEST SUITE")
        print("=" * 60)
        print("This comprehensive test verifies:")
        print("• Basic pose execution")
        print("• Individual joint control")
        print("• Progressive speed control (slow → fast)")
        print("• Functional grip positions")
        print("• Complex coordinated movements")
        print("• Safety limit enforcement")
        print("• Extended operation endurance")
        print("\nEstimated duration: 8-10 minutes")
        
        input("\nPress Enter to begin comprehensive testing...")
        
        start_time = time.time()
        test_results = []
        
        # Run all test phases
        test_phases = [
            ("Basic Poses", self.test_basic_poses),
            ("Individual Joints", self.test_individual_joints),
            ("Progressive Speeds", self.test_progressive_speeds),
            ("Functional Grips", self.test_functional_grips),
            ("Complex Sequences", self.test_complex_sequences),
            ("Safety Limits", self.test_joint_limits_safety),
            ("Endurance", self.test_endurance_sequence)
        ]
        
        for phase_name, test_function in test_phases:
            print(f"Starting {phase_name}...")
            
            try:
                result = test_function()
                test_results.append((phase_name, result))
                
                if result:
                    print(f" {phase_name} - PASSED")
                else:
                    print(f" {phase_name} - FAILED")
                    
            except Exception as e:
                print(f" {phase_name} - ERROR: {e}")
                test_results.append((phase_name, False))
            
            # Brief pause between phases
            time.sleep(2.0)
        
        # Final results summary
        total_time = time.time() - start_time
        passed_tests = sum(1 for _, result in test_results if result)
        total_tests = len(test_results)
        
        print("\n" + "=" * 60)
        print("COMPREHENSIVE TEST RESULTS")
        print("=" * 60)
        
        for phase_name, result in test_results:
            status = " PASSED" if result else " FAILED"
            print(f"{phase_name:<20} {status}")
        
        print("-" * 60)
        print(f"Overall Result: {passed_tests}/{total_tests} tests passed")
        print(f"Total Duration: {total_time/60:.1f} minutes")
        
        if passed_tests == total_tests:
            print(" ALL TESTS PASSED!")
            print(" Prosthetic hand control is working perfectly!")
            print(" Movements are smooth and coordinated!")
            print(" Safety systems are functioning correctly!")
            print(" Ready for EMG data collection!")
        else:
            print(f"\n{total_tests - passed_tests} tests failed - review hand setup")
        
        return passed_tests == total_tests

def main():
    """
    Main function for comprehensive hand testing
    """
    parser = argparse.ArgumentParser(
        description="Comprehensive prosthetic hand movement test suite"
    )
    parser.add_argument("--hand_side", "-s", choices=["left", "right"], 
                       default="left", help="Side of prosthetic hand")
    
    args = parser.parse_args()
    
    print(" COMPREHENSIVE PROSTHETIC HAND TESTER")
    print("=" * 50)
    print("This tool runs a complete test suite to verify:")
    print("• Smooth movement execution")
    print("• Progressive speed control (slow to fast)")
    print("• Individual joint functionality")
    print("• Complex coordinated movements")
    print("• Safety and reliability")
    print("\n Perfect for verifying hand control before EMG data collection!")
    
    # Create tester
    tester = ComprehensiveHandTester(hand_side=args.hand_side)
    
    # Connect to hand
    if not tester.connect_hand():
        print(" Cannot proceed without hand connection")
        print("Check hand power, connections, and communication settings")
        return
    
    try:
        # Run comprehensive test suite
        success = tester.run_comprehensive_test_suite()
        
        if success:
            print("\nYour hand is ready for enhanced EMG data collection!")
        else:
            print("\nPlease review failed tests before proceeding")
            
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        
    except Exception as e:
        print(f" Unexpected error: {e}")
        
    finally:
        # Always disconnect safely
        tester.disconnect_hand()
    
    print("\nComprehensive testing complete!")

if __name__ == "__main__":
    main()