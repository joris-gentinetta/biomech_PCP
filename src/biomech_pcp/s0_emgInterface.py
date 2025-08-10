# Mikey Fernandez
# 12/08/2022
# Python interface to pull EMG board data over USB

import os
import struct
import threading
from time import sleep, time

import numpy as np
import serial
import zmq
from helpers.BCI_Data_Receiver import BCI_Data_Receiver
from helpers.emgDef import (
    ACTIVE_DATA,
    BUFFER_FULL,
    BUFFER_FULL_LENGTH,
    BUFFER_HALF,
    BUFFER_HALF_LENGTH,
    COMM_COMMAND_CONFIG_USB,
    COMM_TX_DONE,
    COMM_TX_INACTIVE,
    DSP_DATA_AUX3,
    USB_PACKET_LENGTH,
    emgDataFull,
    emgDataHalf,
)
from helpers.fakeDataSource import FakeStreamer
from serial.tools import list_ports

os.environ["PYQTGRAPH_QT_LIB"] = "PyQt5"
import argparse
import sys


class EMGHardware:
    def __init__(self, baudrate=921600, port="COM9"):
        self.port = port
        self.baudrate = baudrate
        self.EMGPort = serial.Serial(port=port, baudrate=self.baudrate, timeout=None)

        self.mEMGData = emgDataHalf()
        self.mEMGDataFull = emgDataFull()

        self.usb_receive_buf = bytearray(USB_PACKET_LENGTH)
        self.usb_send_buf = bytearray(USB_PACKET_LENGTH)

        self.aligned = False

    def _semg_usb_init(self):
        # turn off board, close port, open port, turn on board
        self._semg_stream_off()
        self.EMGPort.close()
        sleep(0.0005)
        self.EMGPort.open()
        sleep(0.0005)

        self.EMGPort.flush()
        # print('Reset and flushed USB port')

        self._semg_stream_aux_on()
        # print('Opened USB port')

    def _semg_usb_readHalf(self):
        cnt = 0
        while not self.aligned:
            buf = self.EMGPort.read(USB_PACKET_LENGTH)
            self._semg_decodeBufferHalf(buf)

            print(
                f"tick: {self.mEMGData.emg_os_tick}, error: {cnt}, idx: {self.mEMGData.idx}, data[0]: {self.mEMGData.dataBuf[0]}"
            )

            if self.mEMGData.dataBuf[0] == -800:
                # self._semg_stream_on()
                sleep(0.0005)

                while self.mEMGData.dataBuf[0] == -800:
                    buf = self.EMGPort.read(USB_PACKET_LENGTH)
                    self._semg_decodeBufferHalf(buf)

                    sleep(0.01)
                    print("Requesting EMG stream...")

                self.aligned = True
                print("EMG data successfully aligned.")
            else:
                self._semg_stream_off()

                self.EMGPort.close()
                sleep(0.0005)

                self.EMGPort.open()
                sleep(0.0005)

                self._semg_stream_aux_on()
                # print('Attempting EMG data alignment...')
                cnt += 1

        buf = self.EMGPort.read(USB_PACKET_LENGTH)
        self._semg_decodeBufferHalf(buf)

        return 0

    def _semg_usb_readFull(self):
        cnt = 0
        while not self.aligned:
            buf = self.EMGPort.read(USB_PACKET_LENGTH)
            self._semg_decodeBufferFull(buf)
            print(
                f"{len(buf)} bytes read ({buf[0:20]})"
            )  # print up to the first EMG channel reading
            assert len(buf) == USB_PACKET_LENGTH, (
                "Read incorrect number of bytes in _semg_usb_readFull"
            )

            print(
                f"\ttick: {self.mEMGDataFull.osTime_ms:08d}, error: {cnt:03d}, data[0]: {self.mEMGDataFull.dataBuf[0]:07.2f}\n"
            )

            if (
                self.mEMGDataFull.dataBuf[0] == -800
                and self.mEMGDataFull.dataBuf[1] == -700
            ):
                self._semg_stream_on()
                self.mEMGDataFull.first_timestamp = self.mEMGDataFull.osTime_ms
                self.mEMGDataFull.last_timestamp = self.mEMGDataFull.osTime_ms
                sleep(0.0005)

                while (
                    self.mEMGDataFull.dataBuf[0] == -800
                    and self.mEMGDataFull.dataBuf[1] == -700
                ):
                    buf = self.EMGPort.read(USB_PACKET_LENGTH)
                    self._semg_decodeBufferFull(buf)
                    self.mEMGDataFull.last_timestamp = self.mEMGDataFull.osTime_ms
                    sleep(0.0005)
                    self._semg_stream_on()
                    # print('Requesting EMG stream...')
                    print(
                        self.mEMGDataFull.last_timestamp
                        - self.mEMGDataFull.first_timestamp
                    )

                self._semg_stream_on()
                print(self.mEMGDataFull.dataBuf)

                self.aligned = True
                print("EMG data successfully aligned.")
            else:
                self._semg_stream_off()

                self.EMGPort.close()
                sleep(0.0005)

                self.EMGPort.open()
                sleep(0.0005)

                # self.EMGPort.reset_input_buffer()
                # self.EMGPort.reset_output_buffer()
                self._semg_stream_aux_on()

                cnt += 1

        buf = self.EMGPort.read(USB_PACKET_LENGTH)
        self._semg_decodeBufferFull(buf)

        return 0

    def _semg_decodeBufferHalf(self, bufferData):
        assert struct.calcsize(self.mEMGData.format) == len(bufferData), (
            "Wrong buffer size in _semg_decodeBufferHalf"
        )
        buf = struct.unpack(self.mEMGData.format, bufferData)

        self.mEMGData.idx = buf[0]
        self.mEMGData.dataType = buf[1]
        self.mEMGData.emg_os_tick = buf[2]
        self.mEMGData.syncButton = buf[3]
        self.mEMGData.buttonCounter = buf[4]
        self.mEMGData.sw1 = buf[5]
        self.mEMGData.sw2 = buf[6]

        self.mEMGData.dataBuf = np.asarray(buf[7:23])

        # if self.aligned: self.mEMGData.emgDataPrint()

    def _semg_decodeBufferFull(self, bufferData):
        assert struct.calcsize(self.mEMGDataFull.format) == len(bufferData), (
            "Wrong buffer size in _semg_decodeBufferFull"
        )
        buf = struct.unpack(self.mEMGDataFull.format, bufferData)

        self.mEMGDataFull.startCode = buf[0]
        self.mEMGDataFull.gpioState = buf[1]
        self.mEMGDataFull.dataType = buf[2]
        self.mEMGDataFull.freqScalar = buf[3]
        self.mEMGDataFull.osTime_us = buf[4]
        self.mEMGDataFull.osTime_ms = buf[5]
        self.mEMGDataFull.processingTime = buf[6]

        self.mEMGDataFull.dataBuf = np.asarray(buf[7:23])

        self.mEMGDataFull.gpioTimedState = buf[23]
        self.mEMGDataFull.sw1 = buf[24]
        self.mEMGDataFull.sw2 = buf[25]
        self.mEMGDataFull.newline = buf[26]

        # if self.aligned: self.mEMGDataFull.emgDataFullPrint()

    def _semg_stream_off(self):
        bytesToSend = 8
        buf = bytearray(bytesToSend)

        buf[0] = 0x00
        buf[1] = COMM_COMMAND_CONFIG_USB
        buf[2] = COMM_TX_INACTIVE

        bytes_written = self.EMGPort.write(buf)
        assert bytes_written == bytesToSend, (
            "Wrong number of bytes send in _semg_stream_off"
        )
        # print('EMG Stream: OFF')

    def _semg_stream_on(self):
        bytesToSend = 8
        buf = bytearray(bytesToSend)

        buf[0] = 0x00
        buf[1] = COMM_COMMAND_CONFIG_USB  # comm port type
        buf[2] = COMM_TX_DONE  # start/stop stream
        buf[3] = ACTIVE_DATA  # data to be sent over, emgDef.py
        buf[4] = 1  # streaming frequency divisor (1000/buf[4])

        if USB_PACKET_LENGTH == BUFFER_HALF_LENGTH:
            buf[5] = BUFFER_HALF
        elif USB_PACKET_LENGTH == BUFFER_FULL_LENGTH:
            buf[5] = BUFFER_FULL

        bytes_written = self.EMGPort.write(buf)
        assert bytes_written == bytesToSend, (
            "Wrong number of bytes send in _semg_stream_on"
        )
        # print('EMG stream: ON')

    def _semg_stream_aux_on(self):
        bytesToSend = 8
        buf = bytearray(bytesToSend)

        buf[0] = 0x00
        buf[1] = COMM_COMMAND_CONFIG_USB  # comm port type
        buf[2] = COMM_TX_DONE  # start/stop stream
        buf[3] = (
            DSP_DATA_AUX3  # data to be sent over, defined in seonghoEMG_definitions.h
        )
        buf[4] = 1  #  streaming frequency divisor (1000/buf[4])

        if USB_PACKET_LENGTH == BUFFER_HALF_LENGTH:
            buf[5] = BUFFER_HALF  #  Half buffer is 40 bytes
        elif USB_PACKET_LENGTH == BUFFER_FULL_LENGTH:
            buf[5] = BUFFER_FULL

        bytes_written = self.EMGPort.write(buf)
        assert bytes_written == bytesToSend, (
            "Wrong number of bytes send in _semg_stream_aux_on"
        )
        sleep(0.0005)

    def _semg_get_dataBufHalf(self):
        return self.mEMGData.dataBuf

    def _semg_get_dataBufFull(self):
        return self.mEMGDataFull.dataBuf

    def _semg_get_bufHalfButton(self, buttonNum):
        if buttonNum == 0:
            return self.mEMGData.sw1
        elif buttonNum == 1:
            return self.mEMGData.sw2

        return 128


class EMGDriver(EMGHardware):
    def __init__(self, port, baudrate):
        EMGHardware.__init__(self, port=port, baudrate=baudrate)
        self.port = port
        self.baudrate = baudrate
        self.eventTime = 0

    def __del__(self):
        self._semg_stream_off()

        self.EMGPort.close()

    def begin(self):
        self._semg_usb_init()

    def readData(self):
        if USB_PACKET_LENGTH == BUFFER_HALF_LENGTH:
            self._semg_usb_readHalf()
        elif USB_PACKET_LENGTH == BUFFER_FULL_LENGTH:
            self._semg_usb_readFull()

    def getDataBufHalf(self):
        return self._semg_get_dataBufHalf()

    def getDataBufFull(self):
        return self._semg_get_dataBufFull()

    def getHalfButton(self, button):
        return self._semg_get_bufHalfButton(button)


class EMGInterface:
    def __init__(self, port, baudrate):
        self.m_emg = EMGDriver(port, baudrate)

    def initialize(self):
        self.m_emg.begin()
        # self.m_emg._semg_stream_on()

        print("EMG Started.")

    def updateSensorState(self):
        self.m_emg.readData()
        # print(f'Bytes waiting: {self.m_emg.EMGPort.in_waiting}') # to check if serial port input buffer overflows

    def getChannelReading(self, channel):
        if USB_PACKET_LENGTH == BUFFER_HALF_LENGTH:
            emgs = self.m_emg.getDataBufHalf()
            return emgs[channel]
        elif USB_PACKET_LENGTH == BUFFER_FULL_LENGTH:
            emgs = self.m_emg.getDataBufFull()
            return emgs[channel]

        return 404

    def getButton(self, buttonNum):
        return self.m_emg.getHalfButton(buttonNum)

    def shutdown(self):
        print("Shutting down EMG")
        del self.m_emg

    @staticmethod
    def getMeasurement(unit):
        return 0.0


class EMGStreamer:
    def __init__(
        self, socketAddr="tcp://18.27.123.85:1236", port="COM9", baudrate=921600
    ):
        self.port = port
        self.baudrate = baudrate
        self.sensor = EMGInterface(port, baudrate)
        self.sensor.initialize()

        self.quitEvent = threading.Event()

        self.socketAddr = socketAddr
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.PUB)
        self.sock.bind(self.socketAddr)

        self.recordRate = 2000  # Hz
        self.printRate = 100

    def __del__(self):
        try:
            # close the socket
            self.sock.close()
            self.ctx.term()

            self.quitEvent.set()

        except Exception as e:
            print(f"__del__: Socket closing error {e}")

    def stream(self):
        a = 0
        while not self.quitEvent.is_set():
            start = time()
            self.sensor.updateSensorState()

            reading = self.sensor.m_emg.mEMGDataFull
            packedData = self.pack(reading)

            self.sock.send(packedData)

            if a == self.printRate:
                # self.sensor.m_emg.mEMGDataFull.emgDataFullPrint()
                a = 0

            a += 1

            if self.quitEvent.is_set():
                break

            endTime = time()
            sleep(max(1 / self.recordRate - endTime - start, 0))
            # print(f'Time since last send: {(endTime - start):07.5f}')

        print("Stopping stream.")

    def pack(self, data):
        packedData = struct.pack(
            "BBBBIHHffffffffffffffffBBBB",
            data.startCode,
            data.gpioState,
            data.dataType,
            data.freqScalar,
            data.osTime_us,
            data.osTime_ms,
            data.processingTime,
            data.dataBuf[0],
            data.dataBuf[1],
            data.dataBuf[2],
            data.dataBuf[3],
            data.dataBuf[4],
            data.dataBuf[5],
            data.dataBuf[6],
            data.dataBuf[7],
            data.dataBuf[8],
            data.dataBuf[9],
            data.dataBuf[10],
            data.dataBuf[11],
            data.dataBuf[12],
            data.dataBuf[13],
            data.dataBuf[14],
            data.dataBuf[15],
            data.gpioTimedState,
            data.sw1,
            data.sw2,
            data.newline,
        )

        return packedData

    def printEMG(self, data):
        print(f"""{len(data)} bytes sent to {self.socketAddr}\nTime: {data[5] / 1000:07.3f} s\n
                {data[7]:07.2f} {data[8]:07.2f} {data[9]:07.2f} {data[10]:07.2f}
                {data[11]:07.2f} {data[12]:07.2f} {data[13]:07.2f} {data[14]:07.2f}
                {data[15]:07.2f} {data[16]:07.2f} {data[17]:07.2f} {data[18]:07.2f}
                {data[19]:07.2f} {data[20]:07.2f} {data[21]:07.2f} {data[22]:07.2f}\n""")

    def startCommunication(self):
        self.streamThread = threading.Thread(target=self.stream, name="streamEMG")
        self.streamThread.daemon = False
        self.streamThread.start()


# Search for Serial Port to use
def setupSerial(passthrough=None):
    if passthrough:
        print(f"Using passthrough port {passthrough}")
        return passthrough

    print("Searching for serial ports...")
    com_ports_list = list(list_ports.comports())
    port = ""

    for p in com_ports_list:
        if p:
            if "COM" in p[0] and "USB" in p.description:
                print(f"Found EMG board at {p[0]}")
                print(p[0])
                return p[0]
    if not port:
        print("No port found")
        sys.exit()


if __name__ == "__main__":
    # run this file with no arguments and it will stream from the EMG board - otherwise (with precisely one additional integer argument), it will generate fake data to use with the GUI
    # if len(sys.argv) == 1:
    #     print('Starting EMG streamer...\n')
    #     port = setupSerial(passthrough='/dev/ttyUSB1')
    #     streamer = EMGStreamer(socketAddr='tcp://18.27.123.85:1236', port=port, baudrate=921600)

    # elif len(sys.argv) == 2:
    #     try:
    #         isNum = int(sys.argv[1])

    #         print('Starting fake EMG streamer...\n')
    #         streamer = FakeStreamer(socketAddr='tcp://18.27.123.85:1236')

    #     except Exception as exc:
    #         raise Exception(f'Wrong type (expected int, given {type(sys.argv[1])})') from exc

    # else:
    #     raise Exception(f'Wrong number of arguments ({len(sys.argv) - 1})')

    parser = argparse.ArgumentParser(description="EMG USB Interface")
    parser.add_argument(
        "-p", "--port", type=str, default="/dev/ttyUSB0", help="Serial port to use"
    )
    # parser.add_argument('-a', '--address', type=str, default='tcp://18.27.123.85:1236', help='Socket address to use')
    parser.add_argument(
        "-a",
        "--address",
        type=str,
        default="tcp://127.0.0.1:1236",
        help="Socket address to use",
    )
    parser.add_argument(
        "-b", "--baudrate", type=int, default=921600, help="Baudrate to use"
    )
    parser.add_argument("-f", "--fake", action="store_true", help="Use fake data")

    args = parser.parse_args()

    if args.fake:
        print("Starting fake EMG streamer...\n")
        streamer = FakeStreamer(socketAddr=args.address)
    else:
        print("Starting EMG streamer...\n")
        port = setupSerial(passthrough=args.port)
        streamer = EMGStreamer(
            socketAddr=args.address, port=port, baudrate=args.baudrate
        )

    streamer.startCommunication()

    dataReceiver = BCI_Data_Receiver("127.0.0.1", 1236, 1000)
    dataReceiver.asyncReceiveData(None)
    # app = QtWidgets.QApplication([])
    # ex = mainWindow()
    # ex.start()
    # exitCode = app.exec_()
    #
    # del streamer
    #
    # sys.exit(exitCode)
