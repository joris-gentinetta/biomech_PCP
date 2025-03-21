# ForceClass.py

# Provides interface for acquiring and processing analog force sensor data.
# It is analogous to the EMGClass but tailored for force sensor readings.
#
# Key features:
#  - Reads raw force data via a ZMQ socket.
#  - Provides calibration routines to compute sensor bounds (max and min) and deltas.
#  - Normalizes incoming force data based on these calibration values.
#  - Dynamically estimates the sampling frequency.
#  - Supports both live and offline data streams via an offline data generator.
#  - Continuously acquires data in a dedicated thread and stores history.
#  - Includes debugging print statements for diagnostics.

import struct
import os
import sys
import zmq
import numpy as np
import threading
import platform
import time

class Force():
    def __init__(self, socketAddr='tcp://127.0.0.1:1236', numChannels=30, samplingFreq=1000, offlineData=None):
        """
        Initialize the Force sensor interface.
        """
        # If offline data is provided, wrap it into a generator.
        if offlineData is not None:
            self.offlineData = self.offline_data_gen(offlineData)
        else:
            self.offlineData = None

        self.numChannels = numChannels
        self.samplingFreq = samplingFreq
        # History arrays to store raw and normalized force data.
        self.forceHistory = np.empty((self.numChannels, 1))
        self.normForceHistory = np.empty((self.numChannels, 1))
        self.OS_time = None
        
        # Calibration attributes.
        self.maxVals = None
        self.minVals = None
        self.bounds = None
        self.deltas = None
        
        # Setup ZMQ to subscribe to force data.
        self.socketAddr = socketAddr
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.connect(self.socketAddr)
        self.sock.subscribe('')  # Subscribe to all messages

        # OPTIONAL: Set a receive timeout (in milliseconds) to prevent blocking indefinitely in live mode.
        # Uncomment the next line if you want a 1-second timeout.
        # self.sock.RCVTIMEO = 1000  

        # For dynamic frequency calculation.
        self.last_time = None
        
        # Event for signaling exit.
        self.exitEvent = threading.Event()
    
    def resetForce(self):
        """
        Resets the internal state of the force sensor data.
        """
        self.forceHistory = np.empty((self.numChannels, 0))  # No data columns.
        self.normForceHistory = np.empty((self.numChannels, 0))
        self.OS_time = None
        self.last_time = None
        print(" Force state has been reset.")

    def offline_data_gen(self, data):
        """
        Generates a stream of force data samples from pre-recorded data.
        Pads the data at the beginning and end to simulate a live stream.
        
        Parameters:
            data (np.array): Array of shape (numChannels, num_samples).
        
        Yields:
            np.array: A column vector (shape: (numChannels,)) representing a single force sample.
        """
        pad_width = 100  # Number of samples to pad.
        padded = np.pad(data, ((0, 0), (pad_width, pad_width)), mode='edge')
        total_samples = padded.shape[1]
        print(f"Offline generator: total_samples = {total_samples}")
        count = 0
        for i in range(total_samples):
            # Debug: print each time a sample is yielded.
            print(f"offline_data_gen yielding sample {i}")
            yield padded[:, i]
            count += 1
            # Optionally, stop after a maximum number of samples.
            # Uncomment and adjust the next lines if needed:
            # if count >= 2000:
            #     print(f" offline_data_gen reached max_samples: 2000. Setting exitEvent.")
            #     self.exitEvent.set()
            #     break
            # Signal exit near the end.
            if i == total_samples - pad_width:
                print("   Offline generator near end of data; exitEvent set.")
                self.exitEvent.set()
                break

    def calibrateBounds(self, calibration_data):
        """
        Calibrates the force sensor bounds from calibration data.
        """
        self.maxVals = np.max(calibration_data, axis=1)
        self.minVals = np.min(calibration_data, axis=1)
        self.bounds = np.concatenate((self.maxVals, self.minVals))
        print(f"   Calibration complete. Max values: {self.maxVals}, Min values: {self.minVals}")

    def computeDeltas(self):
        """
        Computes the range (delta) between max and min for each sensor.
        """
        if self.maxVals is None or self.minVals is None:
            raise ValueError("Calibration bounds not set!")
        self.deltas = self.maxVals - self.minVals
        print(f"   Deltas computed: {self.deltas}")

    def normalizeForce(self, raw_force):
        """
        Normalizes a raw force reading using calibration bounds.
        """
        if self.maxVals is None or self.minVals is None:
            raise ValueError("Calibration bounds must be set before normalization.")
        ranges = self.maxVals - self.minVals
        ranges[ranges == 0] = 1  # Prevent division by zero.
        normalized = (np.array(raw_force) - self.minVals) / ranges
        normalized = np.clip(normalized, 0, 1)
        return normalized

    def readForcePacket(self):
        """
        Reads a single force data packet from the ZMQ socket.
        """
        try:
            packet = self.sock.recv()
            fmt = 'f' * (1 + self.numChannels)
            data = struct.unpack(fmt, packet)
            self.OS_time = data[0]
            raw_force = data[1:]
            print(f"   Packet received at {self.OS_time}: {raw_force}")
            return raw_force, self.OS_time
        except Exception as e:
            print(f"[ERROR] readForcePacket: {e}")
            return (0,) * self.numChannels, None

    def readForcePacketBatch(self, numPackets):
        """
        Reads a batch of force packets and returns them in an array.
        """
        batch = np.empty((self.numChannels, numPackets))
        fmt = 'f' * (1 + self.numChannels)
        expected_size = struct.calcsize(fmt)
        for i in range(numPackets):
            if self.offlineData is None:
                packet = self.sock.recv()
                if len(packet) != expected_size:
                    raise ValueError(f"Received packet size {len(packet)} does not match expected size {expected_size}")
                data = struct.unpack(fmt, packet)
                self.OS_time = data[0]
                raw_force = data[1:]
            else:
                raw_force = next(self.offlineData)
                self.OS_time = time.time()
            batch[:, i] = raw_force
            print(f"   Batch packet {i+1}/{numPackets} read: {raw_force}")
        return batch

    def pipelineForce(self):
        """
        Continuously reads force data packets, updates sampling frequency dynamically,
        normalizes readings, and appends both raw and normalized data to history arrays.
        """
        while not self.exitEvent.is_set():
            # Debug: print at start of each loop iteration.
            print(f"   pipelineForce loop iteration. OS_time: {self.OS_time}")
            if self.offlineData is None:
                raw_force, timestamp = self.readForcePacket()
            else:
                raw_force = next(self.offlineData)
                timestamp = time.time()
                self.OS_time = timestamp
            
            # Dynamic frequency calculation.
            if self.last_time is not None and timestamp is not None:
                dt = timestamp - self.last_time
                if dt > 0:
                    current_freq = 1.0 / dt
                    self.samplingFreq = 0.9 * self.samplingFreq + 0.1 * current_freq
                    print(f"   dt: {dt:.4f}s, current_freq: {current_freq:.2f}Hz, updated samplingFreq: {self.samplingFreq:.2f}Hz")
            self.last_time = timestamp
            
            # Append raw force data.
            sample = np.array(raw_force).reshape(self.numChannels, 1)
            self.forceHistory = np.concatenate((self.forceHistory, sample), axis=1)
            
            # Normalize the sample.
            try:
                norm_sample = self.normalizeForce(raw_force)
            except ValueError as e:
                print(f"[ERROR] normalization failed: {e}")
                norm_sample = np.array(raw_force)
            norm_sample = norm_sample.reshape(self.numChannels, 1)
            self.normForceHistory = np.concatenate((self.normForceHistory, norm_sample), axis=1)
            
            # Debug prints for the current sample.
            print(f"   Raw sample: {raw_force}")
            print(f"   Normalized sample: {norm_sample.flatten()}")
            
            time.sleep(1.0 / self.samplingFreq)
    
    def startCommunication(self):
        """
        Starts the force data acquisition thread.
        """
        self.forceThread = threading.Thread(target=self.pipelineForce, name='pipelineForce')
        self.forceThread.daemon = False
        self.forceThread.start()
        print("   Force data acquisition thread started.")

    # Additional debugging getters.
    def printBounds(self):
        if self.bounds is not None:
            print(f"   Bounds: {self.bounds}")
        else:
            print("   Bounds not calibrated yet.")
    
    def printDeltas(self):
        if self.deltas is not None:
            print(f"   Deltas: {self.deltas}")
        else:
            print("   Deltas not computed yet.")
    
    def printCurrentSample(self):
        print(f"   OS_time: {self.OS_time}")
        if self.forceHistory.shape[1] > 0:
            print(f"   Latest raw force sample: {self.forceHistory[:, -1]}")
        if self.normForceHistory.shape[1] > 0:
            print(f"   Latest normalized force sample: {self.normForceHistory[:, -1]}")

    def shutdown(self):
        try:
            self.exitEvent.set()
            self.forceThread.join()
            self.sock.close()
            self.ctx.term()
        except Exception as e:
            print(f'Force shutdown error: {e}')            


# Test block (if run as a script)
if __name__ == '__main__':
    # Generated data for pretraining
    offline_data = np.load(f"C:/Users/Emanuel Wicki/Documents/MIT/biomech_PCP/helpers/offline_force_data.npy")
    # Create a Force object.
    force_sensor = Force(socketAddr='tcp://127.0.0.1:1236', numChannels=30, samplingFreq=1000, offlineData=offline_data)
    force_sensor.resetForce()
    
    # Simulate calibration data (dummy data for 30 channels and 1000 samples).
    dummy_calib = np.random.uniform(low=0.2, high=5.0, size=(30, 1000))
    force_sensor.calibrateBounds(dummy_calib)
    force_sensor.computeDeltas()
    force_sensor.printBounds()
    force_sensor.printDeltas()
    
    # Start data acquisition.
    force_sensor.startCommunication()
    
    # Run acquisition for 5 seconds.
    time.sleep(5)
    force_sensor.shutdown()
    
    # Print shapes of the collected data.
    print("Raw Force History shape:", force_sensor.forceHistory.shape)
    print("Normalized Force History shape:", force_sensor.normForceHistory.shape)
    
    # Print the last sample for debugging.
    force_sensor.printCurrentSample()

