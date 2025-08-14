import time
import struct
import zmq
import numpy as np

# ————— CONFIG —————
SOCKET_ADDR = "tcp://127.0.0.1:1236"   # where your EMGStreamer is PUBlishing
DURATION    = 10                     # seconds to record
OUT_DATA    = "data/raw_stream.npy"
OUT_TS      = "data/raw_timestamps.npy"

# packet format used by EMGStreamer.pack():
#   "BBBBIHH"        → 1 byte ×3 hdr + 1 byte freqScalar + 4×osTime_us(I) + 2×osTime_ms(H) + 2×processingTime(H)
#   "f"*16           → 16 floats of dataBuf
#   "BBBB"           → gpioTimedState, sw1, sw2, newline
FMT = "BBBBIHH" + "f"*16 + "BBBB"
SIZE = struct.calcsize(FMT)

def main():
    ctx  = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect(SOCKET_ADDR)
    sock.setsockopt_string(zmq.SUBSCRIBE, "")  # subscribe to all

    data_buf = []
    ts_buf   = []

    t0 = time.time()
    print(f"Recording {DURATION:.1f}s of EMG → {OUT_DATA}, {OUT_TS}")
    while time.time() - t0 < DURATION:
        msg = sock.recv()         # blocks until one packet arrives
        if len(msg) != SIZE:
            # unexpected packet size?
            continue

        pkt = struct.unpack(FMT, msg)
        # pkt[4] is osTime_us, pkt[5] osTime_ms; choose one or use local clock:
        ts = pkt[4] / 1e6         # convert microsecond counter → seconds
        raw16 = pkt[7:7+16]       # the 16 floats

        ts_buf.append(ts)
        data_buf.append(raw16)

    data = np.vstack(data_buf)
    ts   = np.array(ts_buf)

    np.save(OUT_DATA, data)
    np.save(OUT_TS, ts)
    print(f"Done. Saved {data.shape[0]} samples × {data.shape[1]} channels.")

if __name__ == "__main__":
    main()

    
############################################
'''
import time
import struct
import zmq
import numpy as np
import os

# ————— CONFIG —————
SOCKET_ADDR    = "tcp://127.0.0.1:1237"   # your filtered EMG PUB endpoint
DURATION       = 6.0                     # seconds to record
OUT_DATA       = "data/filtered_stream.npy"
OUT_TS         = "data/filtered_timestamps.npy"
USED_CHANNELS  = [0]  # must match modular.yaml
N_CHANNELS     = len(USED_CHANNELS)
FMT            = "f" * N_CHANNELS         # each packet is N_CHANNELS float32s
SIZE           = struct.calcsize(FMT)

def main():
    # Make sure output folder exists
    os.makedirs(os.path.dirname(OUT_DATA), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_TS), exist_ok=True)

    ctx  = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect(SOCKET_ADDR)
    sock.setsockopt_string(zmq.SUBSCRIBE, "")  # subscribe to all

    data_buf = []
    ts_buf   = []

    t0 = time.time()
    print(f"Recording {DURATION:.1f}s of filtered EMG → {OUT_DATA}, {OUT_TS}")
    while time.time() - t0 < DURATION:
        msg = sock.recv()  # blocks until one packet arrives
        if len(msg) != SIZE:
            # If we get an unexpected‐size packet, skip it
            print(f"Skipping packet of size {len(msg)} (expected {SIZE})")
            continue

        # Unpack exactly N_CHANNELS floats:
        values = struct.unpack(FMT, msg)

        # Timestamp however you like. Here we use local clock:
        ts = time.time()
        ts_buf.append(ts)
        data_buf.append(values)

    # Convert to NumPy arrays
    data = np.vstack(data_buf)       # shape = (num_samples, N_CHANNELS)
    ts   = np.array(ts_buf)          # shape = (num_samples,)

    np.save(OUT_DATA, data)
    np.save(OUT_TS, ts)
    print(f"Done. Saved {data.shape[0]} samples × {data.shape[1]} channels.")
    sock.close()
    ctx.term()

if __name__ == "__main__":
    main()

    

#############################################


import zmq
import struct
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import time

# Calibration arrays (paste from your YAML)
maxVals = np.array([
    68.41610183342885, 12.641076312411426, 20.814204946451706, 30.411780574818238,
    25.676988070220734, 14.619387051433721, 31.775786316500533, 13.551328392737172,
    125.0298956695996, 10.976562983723314, 15.0264968037861, 10.828753372755706,
    21.885527340001715, 22.559317849632325, 33.83648210180887, 23.27721835345018
], dtype=np.float32)

noiseLevels = np.array([
    3.546619434893476, 139.6056897823201, 122.80153117061415, 148.6314335080685,
    132.07512525367812, 104.69938548028595, 190.74780813184302, 98.49462605127962,
    255.79306257688748, 116.46982878951346, 100.59551217790073, 98.0170178269472,
    154.1908571736336, 174.28007366131962, 161.92121783343444, 144.7611179581469
], dtype=np.float32)

def process_emg_zero_phase(emg_data, sampling_freq, maxVals, noiseLevel):
    numElectrodes = emg_data.shape[0]
    powerline_sos = signal.bessel(N=8, Wn=[58, 62], btype='bandstop', output='sos', fs=sampling_freq)
    highpass_sos = signal.bessel(N=4, Wn=20, btype='highpass', output='sos', fs=sampling_freq)
    lowpass_sos = signal.bessel(N=4, Wn=3, btype='lowpass', output='sos', fs=sampling_freq)
    filtered_emg = np.copy(emg_data)
    for ch in range(numElectrodes):
        filtered_emg[ch, :] = signal.sosfiltfilt(powerline_sos, filtered_emg[ch, :])
    for ch in range(numElectrodes):
        filtered_emg[ch, :] = signal.sosfiltfilt(highpass_sos, filtered_emg[ch, :])
    filtered_emg = np.abs(filtered_emg)
    filtered_emg = np.clip(filtered_emg - noiseLevel[:, None], 0, None)
    for ch in range(numElectrodes):
        filtered_emg[ch, :] = signal.sosfiltfilt(lowpass_sos, filtered_emg[ch, :])
    filtered_emg = np.clip(filtered_emg, 0, None)
    normalized_emg = filtered_emg / maxVals[:, None]
    normalized_emg = np.clip(normalized_emg, 0, 1)
    return normalized_emg

# --- Streaming parameters ---
sampling_freq = 1000  # Hz
emg_channels = 16
frame_size = 92
record_secs = 10

# --- Connect to ZMQ ---
context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://127.0.0.1:1235")
socket.subscribe(b'')

# --- Collect data for 10 seconds ---
emg_buffer = []
start_time = time.time()
print(f"Collecting EMG for {record_secs} seconds...")
while (time.time() - start_time) < record_secs:
    msg = socket.recv()
    if len(msg) != frame_size:
        continue
    unpacked = struct.unpack("ffffffffffffffffffIIIIf", msg)
    emg = np.array(unpacked[:emg_channels])
    emg_buffer.append(emg)

emg_array = np.stack(emg_buffer, axis=1)  # shape: [16, N]
print("Finished recording. Processing...")

# --- Filter and normalize ---
filtered_emg = process_emg_zero_phase(emg_array, sampling_freq, maxVals, noiseLevels)

# --- Plot (example: first 4 channels) ---
plt.figure(figsize=(15, 8))
for ch in range(4):
    plt.plot(filtered_emg[ch], label=f"EMG {ch}")
plt.xlabel("Samples (ms)" if sampling_freq==1000 else "Samples")
plt.ylabel("Normalized EMG")
plt.title("Filtered, normalized EMG (channels 0–3, 10s recording)")
plt.legend()
plt.tight_layout()
plt.show()
'''