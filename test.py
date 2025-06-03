import time
import struct
import zmq
import numpy as np

# ————— CONFIG —————
SOCKET_ADDR = "tcp://127.0.0.1:1236"   # where your EMGStreamer is PUBlishing
DURATION    = 10.0                     # seconds to record
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