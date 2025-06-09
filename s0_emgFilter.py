import argparse
import os
import yaml
import zmq
import struct
import numpy as np
import math
import sys

from helpers.emgDef import emgDataFull, USB_PACKET_LENGTH
from helpers.BesselFilter import BesselFilterArr

def load_scaling(person_dir, used_channels):
    yaml_path = os.path.join(
        "data", person_dir, "recordings", "calibration", "experiments", "1", "scaling.yaml"
    )
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"Scaling file not found: {yaml_path}")
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    noise = np.array(data.get('noise', data.get('noiseLevels')), dtype=np.float32)
    maxvals = np.array(data['maxVals'], dtype=np.float32)
    return noise[used_channels], maxvals[used_channels]

def load_used_channels_from_modular(person_dir):
    modular_path = os.path.join("data", person_dir, "configs", "modular.yaml")
    if not os.path.isfile(modular_path):
        raise FileNotFoundError(f"Modular config not found: {modular_path}")
    with open(modular_path, 'r') as f:
        mod = yaml.safe_load(f)
    feats = mod['parameters']['features']['value']
    used = sorted(int(idx) for tag, idx in feats if tag=='emg')
    if not used:
        raise ValueError("No EMG channels in modular.yaml")
    return used

def main():
    p = argparse.ArgumentParser("EMG filter+downsample (with anti-aliasing)")
    p.add_argument('-i','--in_addr', default='tcp://127.0.0.1:1236')
    p.add_argument('-o','--out_addr', default='tcp://127.0.0.1:1237')
    p.add_argument('-d','--person_dir', required=True)
    args = p.parse_args()

    used_ch = load_used_channels_from_modular(args.person_dir)
    noiseLv, maxV = load_scaling(args.person_dir, used_ch)
    nchan = len(used_ch)

    # ZMQ setup
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.connect(args.in_addr)
    sub.setsockopt_string(zmq.SUBSCRIBE, '')
    pub = ctx.socket(zmq.PUB)
    pub.bind(args.out_addr)

    # packet format check
    tmp = emgDataFull()
    fmt = tmp.format
    if struct.calcsize(fmt)!=USB_PACKET_LENGTH:
        print("⚠️ format size mismatch")

    # --- 1) estimate sampling frequency from first N packets ---
    N = 200
    ts_buf = []
    print(f"Estimating sampling rate from first {N} packets…")
    for _ in range(N):
        msg = sub.recv()
        unpacked = struct.unpack(fmt, msg)
        ts_buf.append(float(unpacked[5]))
    dt = ts_buf[-1] - ts_buf[0]  # in ms
    fs = (N-1)*1000.0 / dt
    print(f"→ Estimated fs = {fs:.1f} Hz")

    # 2) initialize filters at raw rate
    powerFilt = BesselFilterArr(nchan, order=8, critFreqs=[58,62], fs=fs, filtType='bandstop')
    hpFilt    = BesselFilterArr(nchan, order=4, critFreqs=20,     fs=fs, filtType='highpass')
    lpFilt    = BesselFilterArr(nchan, order=4, critFreqs=3,      fs=fs, filtType='lowpass')

    # 3) setup downsampling to 60 Hz
    target_hz = 60.0
    ms_per_sample = 1000.0 / target_hz
    next_ts = ts_buf[0]

    # 4) buffer for raw history
    numPackets = math.ceil(fs/target_hz)
    rawHistory = np.zeros((nchan, numPackets), dtype=np.float32)

    print(f"Processing… publishing filtered {target_hz}Hz EMG on {args.out_addr}")

    try:
        for ts in ts_buf:
            # preload buffer with initial samples
            pass
        # now main loop
        while True:
            msg = sub.recv()
            unpacked = struct.unpack(fmt, msg)
            device_ts = float(unpacked[5])  # ms

            # extract subset of channels
            all_vals = np.array(unpacked[7:23], dtype=np.float32)
            subset = all_vals[used_ch]

            # shift buffer and append new raw sample
            rawHistory[:,:-1] = rawHistory[:,1:]
            rawHistory[:,-1] = subset

            # downsample by device_ts
            if device_ts >= next_ts:
                # 1) anti-alias & envelope pipeline
                x = powerFilt.filter(rawHistory)
                x = hpFilt.filter(x)
                x = np.abs(x)
                x = np.clip(x - noiseLv[:,None], 0, None)
                env = lpFilt.filter(x)

                iemg = env[:,-1]
                norm = iemg / maxV
                norm = np.clip(norm, 0, 1)

                # publish
                pub.send(struct.pack("f"*nchan, *norm.tolist()))

                # schedule next
                next_ts += ms_per_sample
                if device_ts > next_ts:
                    missed = math.floor((device_ts - next_ts)/ms_per_sample)
                    next_ts += missed*ms_per_sample

    except KeyboardInterrupt:
        print("Interrupted, exiting…")
    finally:
        sub.close(); pub.close(); ctx.term()

if __name__=="__main__":
    main()
