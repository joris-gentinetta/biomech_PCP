# s0_forceInterface.py
# Interface for reading force sensor data from the Psyonic Ability Hand or an external source.
# When using an external source, a placeholder is used for intact limb training.

import time
import numpy as np
from psyonicHand import psyonicArm
import multiprocessing
import signal
import argparse
import os

class ForceInterface:
    def __init__(self, hand='left', frequency=60, sensor_source='prosthesis'):
        """
        sensor_source: 'prosthesis' (default) or 'external'
        """
        self.hand_side = hand
        self.frequency = frequency
        self.sensor_source = sensor_source
        self.force_sensor_data = []
        self.timestamps = []
        if self.sensor_source == 'prosthesis':
            self.hand = psyonicArm(hand=self.hand_side, stuffing=False, usingEMG=False)
            self.hand.initSensors()
            self.hand.startComms()
        else:
            # External sensor integration not implemented yet.
            self.hand = None
            print("External force sensor integration for intact limb training not implemented yet.")
        
        self.stop_flag = multiprocessing.Value('b', False)
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, sig, frame):
        print("Graceful shutdown...")
        self.stop_flag.value = True

    def read_force_sensors(self):
        if self.sensor_source == 'prosthesis':
            # sensor_data = self.hand.get_sensor_readings()
            sensor_data = {'sensorForces': [self.hand.sensors[force] for force in self.hand.sensorForce]}
            # forces = sensor_data['sensorForces']
            # return np.array(forces)
            forces = np.array(sensor_data['sensorForces'])
            forces[forces < 0.01] = 0  # Set all values < 0.01 to zero
            return forces

        else:
            # Placeholder: External sensor integration to be implemented.
            print("External force sensor integration not implemented. Returning dummy zero values.")
            # 30 force channels for the prosthetic hand; adjust if using external force sensors for training with intact limb.
            return np.zeros(30)

    def start_acquisition(self, duration=None):
        print(f"Starting force data acquisition on {self.hand_side} hand using {self.sensor_source} sensor source...")
        interval = 1.0 / self.frequency
        start_time = time.time()
        
        while not self.stop_flag.value:
            current_time = time.time()
            forces = self.read_force_sensors()
            print (f"Forces measured: {forces}")
            self.force_sensor_data.append(forces)
            self.timestamps.append(current_time)
            
            if duration and (current_time - start_time >= duration):
                break

            sleep_time = interval - (time.time() - current_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        if self.hand is not None:
            self.hand.close()
        print("Force data acquisition stopped.")

    def save_data(self, output_dir="."):
        np.save(os.path.join(output_dir, "force_sensor_data.npy"), np.array(self.force_sensor_data))
        np.save(os.path.join(output_dir, "force_timestamps.npy"), np.array(self.timestamps))
        print(f"Data saved to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Acquire force data from Psyonic Ability Hand or an external sensor.')
    parser.add_argument('--duration', type=int, default=None, help='Duration of acquisition in seconds')
    parser.add_argument('--hand', type=str, default='left', choices=['left', 'right'], help='Hand side to use')
    parser.add_argument('--frequency', type=int, default=60, help='Acquisition frequency in Hz')
    parser.add_argument('--output_dir', type=str, default='.', help='Directory to save force sensor data')
    parser.add_argument('--external', action='store_true', help='Use external force sensor (not implemented yet)')
    parser.add_argument('-p', '--port', type=str, default='COM5', help='Serial port to use')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Create an instance of psyonicArm for the left hand (or 'right' if needed)
    # hand = psyonicArm(hand='left', stuffing=False, usingEMG=False)

    # Initialize sensors
    # hand.initSensors()
    # print("Current control mode:", hand.getCurControlMode())


    # port = 'COM5'
    # baudrate = 921600 
    

    sensor_source = 'external' if args.external else 'prosthesis'
    force_interface = ForceInterface(hand=args.hand, frequency=args.frequency, sensor_source=sensor_source)
    force_interface.start_acquisition(duration=args.duration)
    force_interface.save_data(args.output_dir)
