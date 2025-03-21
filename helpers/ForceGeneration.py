import numpy as np

# Configuration for the offline data.
numChannels = 30    # For example, 30 force sensor channels.
num_samples = 1000  # Number of samples in the offline data.

# Generate random data between 0.2 and 5.0 (simulate force readings).
offline_data = np.random.uniform(low=0.2, high=5.0, size=(numChannels, num_samples))

# Save the data to a .npy file.
np.save("offline_force_data.npy", offline_data)

print("Saved offline_force_data.npy with shape:", offline_data.shape)
