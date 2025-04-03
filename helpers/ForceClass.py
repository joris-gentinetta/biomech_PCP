import os
import sys
import time
import numpy as np
import threading
from helpers.BesselFilter import BesselFilterArr

# Import the connection module.
from s0_forceInterface import ForceInterface

class Force():
    def __init__(self, hand='left', frequency=60, numChannels=30, samplingFreq=1000, maxForce=5):
        """
        Uses ForceInterface to connect to the prosthetic hand, then applies low-pass filtering
        and normalization based on a two-point calibration (baseline and known maximum force).
        
        Parameters:
            hand (str): 'left' or 'right'
            frequency (int): Acquisition frequency for the connection (Hz).
            numChannels (int): Total number of force sensor channels.
            samplingFreq (int): Sampling frequency for processing (Hz).
            maxForce (float): Known maximum force value (default ~4.5).
        """
        self.hand_side = hand
        self.frequency = frequency
        self.numChannels = numChannels
        self.samplingFreq = samplingFreq
        self.maxForce = maxForce
        
        self.forceHistory = np.empty((numChannels, 0))
        self.normForceHistory = np.empty((numChannels, 0))
        self.baseline = None  # To be computed via calibrateBaseline
        
        self.exitEvent = threading.Event()
        
        # Connect using the ForceInterface from s0_forceInterface.
        self.force_interface = ForceInterface(hand=self.hand_side, frequency=self.frequency, sensor_source='prosthesis')
        
        # Initialize the low-pass filter.
        self.initFilters()

    def initFilters(self):
        self.lowPassFilters = BesselFilterArr(
            numChannels=self.numChannels,
            order=4,
            critFreqs=5,
            fs=self.samplingFreq,
            filtType='lowpass'
        )
    
    def readForceSensors(self):
        """Returns raw force sensor data using the connection module."""
        return self.force_interface.read_force_sensors()
    
    def pipelineForce(self):
        # Automatically calibrate baseline if it is not already set.
        if self.baseline is None:
            print("Baseline not set. Automatically calibrating baseline. Ensure no force is applied!")
            self.calibrateBaseline(num_samples=100)


        while not self.exitEvent.is_set():
            raw_force = self.readForceSensors()
            # print(f"Raw force: {raw_force}")
            filtered_force = self.lowPassFilters.filter(np.array(raw_force).reshape(-1, 1)).flatten()
            # print(f"Filtered Force: {filtered_force}")
            self.forceHistory = np.column_stack((self.forceHistory, filtered_force))
            try:
                normalized_force = self.normalizeForce(filtered_force)
            except Exception:
                normalized_force = filtered_force  # Fallback if baseline not set.
            self.normForceHistory = np.column_stack((self.normForceHistory, normalized_force))
            time.sleep(1.0 / self.samplingFreq)
    
    def startCommunication(self):
        self.forceThread = threading.Thread(target=self.pipelineForce, daemon=True)
        self.forceThread.start()
    
    def shutdown(self):
        self.exitEvent.set()
        self.forceThread.join()
        if self.force_interface.hand is not None:
            self.force_interface.hand.close()
    
    def calibrateBaseline(self, num_samples=100):
        """
        Collects 'num_samples' filtered force readings (with no applied force) and computes
        the per-channel baseline (B). Later, normalization will use (raw_force - B)/(maxForce - B).
        
        Returns:
            baseline (np.ndarray): Array of baseline values (one per channel).
        """
        print("Calibrating baseline... Please ensure no force is applied.")
        samples = []
        for _ in range(num_samples):
            raw_force = self.readForceSensors()
            filtered_force = self.lowPassFilters.filter(np.array(raw_force).reshape(-1, 1)).flatten()
            samples.append(filtered_force)
            time.sleep(1.0 / self.samplingFreq)
        samples = np.array(samples)  # Shape: (num_samples, numChannels)
        baseline = np.mean(samples, axis=0)
        self.baseline = baseline
        print("Baseline calibration complete. Baseline:", baseline)
        return baseline
    
    def normalizeForce(self, raw_force):
        """
        Normalizes the raw_force reading using:
            normalized = (raw_force - baseline) / (maxForce - baseline)
        Values are clipped between 0 and 1.
        """
        if self.baseline is None:
            raise ValueError("Baseline has not been calibrated.")
        normalized = (np.array(raw_force) - self.baseline) / (self.maxForce - self.baseline)
        return np.clip(normalized, 0, 1)
    
    def printCurrentSample(self):
        print("Latest raw sample:", self.forceHistory[:, -1])
        print("Latest normalized sample:", self.normForceHistory[:, -1])

# if __name__ == '__main__':
#     # Test routine.
#     force = Force(hand='left', frequency=60, numChannels=30, samplingFreq=1000, maxForce=100)
#     force.calibrateBaseline(num_samples=100)
#     force.startCommunication()
#     time.sleep(5)
#     force.shutdown()
#     print("Raw force shape:", force.forceHistory.shape)
#     print("Normalized force shape:", force.normForceHistory.shape)
#     force.printCurrentSample()
