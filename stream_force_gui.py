#!/usr/bin/env python
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Import the updated Force class from the helpers directory.
from helpers.ForceClass import Force
from config import NUM_CHANNELS, SENSORS_PER_FINGER, NUM_FINGERS, BUFFER_LENGTH, ACQ_FREQUENCY


# Initialize a circular buffer for sensor data (shape: NUM_CHANNELS x BUFFER_LENGTH).
sensor_buffer = np.zeros((NUM_CHANNELS, BUFFER_LENGTH))

# -------------------- Helper Functions --------------------
def update_buffer(new_sample):
    """
    Shift the sensor_buffer one column to the left and append new_sample as the new last column.
    """
    global sensor_buffer
    sensor_buffer = np.hstack((sensor_buffer[:, 1:], new_sample.reshape(-1, 1)))

# -------------------- Main Streaming and Visualization --------------------
def main():
    # Create a Force object using the new connection via s0_forceInterface.
    force_sensor = Force(hand='left', frequency=ACQ_FREQUENCY, numChannels=NUM_CHANNELS, samplingFreq=ACQ_FREQUENCY)
    # Calibrate baseline using the integrated method in the Force class.
    baseline = force_sensor.calibrateBaseline(num_samples=100)
    force_sensor.startCommunication()

    # Set up the matplotlib figure: one subplot per finger.
    fig, axs = plt.subplots(NUM_FINGERS, 1, figsize=(10, 3 * NUM_FINGERS), sharex=True)
    finger_lines = []  # List to store line objects for each finger's sensor channels.
    x_data = np.arange(BUFFER_LENGTH)
    finger_names = ["index", "middle", "ring", "pinky", "thumb"]

    for i in range(NUM_FINGERS):
        ax = axs[i] if NUM_FINGERS > 1 else axs
        lines = []
        start_channel = i * SENSORS_PER_FINGER
        for j in range(SENSORS_PER_FINGER):
            line, = ax.plot(x_data, sensor_buffer[start_channel + j, :],
                            label=f"{finger_names[i].capitalize()} sensor {j+1}")
            lines.append(line)
        finger_lines.append(lines)
        ax.set_ylabel("Force reading")
        ax.set_title(f"{finger_names[i].capitalize()} Finger Force Sensor Data")
        ax.legend(loc="upper right")
        ax.grid(True)
    axs[-1].set_xlabel("Sample Index")
    plt.tight_layout()

    def update(frame):
        """
        Animation update function:
         - Retrieves the latest normalized sample from the force sensor.
         - Subtracts the baseline (clipping negatives to zero).
         - Updates the circular buffer.
         - Rescales the y-axis and refreshes each subplot.
        """
        if force_sensor.normForceHistory.shape[1] > 0:
            new_sample = force_sensor.normForceHistory[:, -1]
        else:
            new_sample = np.zeros(NUM_CHANNELS)
        new_sample_corrected = np.clip(new_sample - baseline, 0, None)
        update_buffer(new_sample_corrected)

        for i in range(NUM_FINGERS):
            start = i * SENSORS_PER_FINGER
            for j, line in enumerate(finger_lines[i]):
                line.set_ydata(sensor_buffer[start + j, :])
            axs[i].relim()
            axs[i].autoscale_view(scaley=True, scalex=False)
        return [line for sublist in finger_lines for line in sublist]

    ani = FuncAnimation(fig, update, interval=1000/ACQ_FREQUENCY, blit=False)

    print("Starting real-time GUI using ForceClass with s0_forceInterface connection.")
    print("Close the window to stop acquisition.")
    plt.show()

    force_sensor.shutdown()
    print("Acquisition stopped.")

if __name__ == "__main__":
    main()
