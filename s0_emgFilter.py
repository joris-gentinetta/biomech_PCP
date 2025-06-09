import argparse
import os
import yaml
import zmq
import struct
import numpy as np
import sys

from helpers.emgDef import emgDataFull, USB_PACKET_LENGTH  # emgDataFull has .format

def load_scaling(person_dir, used_channels):
    """
    Reads data/<person_dir>/recordings/calibration/experiments/1/scaling.yaml
    and returns (noiseLevels_subset, maxVals_subset) as float32 numpy arrays 
    for only the used_channels.
    """
    yaml_path = os.path.join(
        "data", person_dir, "recordings", "calibration", "experiments", "1", "scaling.yaml"
    )
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"Scaling file not found: {yaml_path}")

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    if 'noise' in data:
        noise = np.array(data['noise'], dtype=np.float32)
    elif 'noiseLevels' in data:
        noise = np.array(data['noiseLevels'], dtype=np.float32)
    else:
        raise KeyError("Key 'noise' or 'noiseLevels' not found in scaling.yaml")

    if 'maxVals' in data:
        maxvals = np.array(data['maxVals'], dtype=np.float32)
    else:
        raise KeyError("Key 'maxVals' not found in scaling.yaml")

    noise_subset = noise[used_channels]
    maxvals_subset = maxvals[used_channels]
    return noise_subset, maxvals_subset

def load_used_channels_from_modular(person_dir):
    """
    Reads data/<person_dir>/configs/modular.yaml and extracts the EMG channel indices 
    from parameters.features.value (list of [emg, '<index>'] pairs).
    Returns a sorted list of ints.
    """
    modular_path = os.path.join("data", person_dir, "configs", "modular.yaml")
    if not os.path.isfile(modular_path):
        raise FileNotFoundError(f"Modular config not found: {modular_path}")

    with open(modular_path, 'r') as f:
        mod = yaml.safe_load(f)

    try:
        feats = mod['parameters']['features']['value']
    except KeyError:
        raise KeyError("Could not find parameters.features.value in modular.yaml")

    used_channels = []
    for item in feats:
        # Expect item == ['emg', '<digit_string>']
        if isinstance(item, list) and len(item) == 2 and item[0] == 'emg':
            try:
                idx = int(item[1])
                used_channels.append(idx)
            except:
                continue

    if not used_channels:
        raise ValueError("No valid EMG channel indices found under parameters.features.value")

    return sorted(used_channels)

def main():
    parser = argparse.ArgumentParser(
        description="EMG filter+downsample node (auto‐reads used channels from modular.yaml)."
    )
    parser.add_argument(
        '-i', '--in_addr', type=str, default='tcp://127.0.0.1:1236',
        help='ZMQ SUB address of raw EMG streamer (full‐packet).'
    )
    parser.add_argument(
        '-o', '--out_addr', type=str, default='tcp://127.0.0.1:1237',
        help='ZMQ PUB address for filtered & downsampled EMG.'
    )
    parser.add_argument(
        '-d', '--person_dir', type=str, required=True,
        help='Person directory (under data/) that contains both scaling.yaml and configs/modular.yaml.'
    )

    args = parser.parse_args()
    person_dir = args.person_dir

    # 1) Load used_channels from modular.yaml
    used_channels = load_used_channels_from_modular(person_dir)
    print(f"Using EMG channels from modular.yaml: {used_channels}")

    # 2) Load noiseLevels and maxVals (only for those channels)
    noiseLevels, maxVals = load_scaling(person_dir, used_channels)

    # 3) Set up ZMQ: subscribe to raw EMG (full packet) and publish filtered→downsampled
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.connect(args.in_addr)
    sub.setsockopt_string(zmq.SUBSCRIBE, '')

    pub = ctx.socket(zmq.PUB)
    pub.bind(args.out_addr)

    # 4) Determine expected packet size from emgDataFull.format via an instance
    tmp = emgDataFull()
    fmt = tmp.format
    expected_struct_size = struct.calcsize(fmt)
    if expected_struct_size != USB_PACKET_LENGTH:
        print(f"Warning: struct.calcsize(emgDataFull.format) = {expected_struct_size}, "
              f"but USB_PACKET_LENGTH = {USB_PACKET_LENGTH}.")
    expected_size = USB_PACKET_LENGTH

    print(f"Filter node listening on {args.in_addr}, publishing to {args.out_addr}")
    print(f"Expecting {expected_size} bytes per incoming packet (full header + 16 floats)")

    # 5) Downsampling parameters (60 Hz)
    target_hz = 60.0
    ms_per_sample = 1000.0 / target_hz  # ~16.6667 ms
    next_output_ts = None

    try:
        while True:
            msg = sub.recv()  # blocks until one packet arrives
            print(f"[filter-node] Received raw packet of {len(msg)} bytes")  # debug

            if len(msg) != expected_size:
                print(f"[filter-node] Unexpected size {len(msg)}, expected {expected_size}. Skipping.")
                continue

            unpacked = struct.unpack(fmt, msg)
            device_ts_ms = float(unpacked[5])  # osTime_ms
            print(f"[filter-node] device_ts_ms = {device_ts_ms:.1f} ms")  # debug

            # Initialize next_output_ts on the very first packet
            if next_output_ts is None:
                next_output_ts = device_ts_ms
                print(f"[filter-node] Initialized next_output_ts = {next_output_ts:.1f} ms")  # debug

            # Extract the 16‐channel float array (indices 7..22)
            all_floats = np.array(unpacked[7:23], dtype=np.float32)
            subset = all_floats[used_channels]
            print(f"[filter-node] raw subset = {subset}")  # debug

            calibrated = np.clip(subset - noiseLevels, 0, None)
            normalized = calibrated / maxVals
            print(f"[filter-node] normalized = {normalized}")  # debug

            # Downsample: publish only if device_ts_ms ≥ next_output_ts
            if device_ts_ms >= next_output_ts:
                out_packet = struct.pack("f" * len(used_channels), *normalized.tolist())
                pub.send(out_packet)
                print(f"[filter-node] → published {len(used_channels)} floats")  # debug

                next_output_ts += ms_per_sample
                # Catch up if we’ve fallen behind
                if device_ts_ms > next_output_ts:
                    intervals_missed = np.floor((device_ts_ms - next_output_ts) / ms_per_sample)
                    next_output_ts += intervals_missed * ms_per_sample
                    print(f"[filter-node] Catching up, next_output_ts = {next_output_ts:.1f} ms")  # debug

    except KeyboardInterrupt:
        print("Interrupted by user. Exiting.")
    finally:
        sub.close()
        pub.close()
        ctx.term()

if __name__ == "__main__":
    main()
