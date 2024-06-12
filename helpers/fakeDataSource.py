'''
Mikey Fernandez modified 03/17/2021

# Change to 16 channels and a 76 bit packet, for testing with Haptix controller
Further modified - 92 bit packet 03/20/2021
'''

import numpy as np
import struct
import time
import zmq
import threading
from helpers.emgDef import *

class FakeStreamer():
    def __init__(self, socketAddr='tcp://18.27.123.85:1236'):
        self.quitEvent = threading.Event()

        self.socketAddr = socketAddr
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.PUB)
        self.sock.bind(self.socketAddr)

        self.recordRate = 2000 # Hz
        self.printRate = 100

        self.numChannels = 16
        self.numComponents = 10
        self.curTime = time.time()

        self.emgData = emgDataFull()

        self.offset = np.random.uniform(low=-1e4, high = 1e4, size=self.numChannels)
        self.artifactFreq = np.random.uniform(low=0.0025, high=0.01, size=self.numChannels)
        self.artifactAmp = np.random.uniform(low=10, high=100, size=self.numChannels)

    def __del__(self):
        try:
            # close the socket
            self.sock.close()
            self.ctx.term()

            self.quitEvent.set()

        except Exception as e:
            print(f'__del__: Socket closing error {e}')

    def generateEMG(self):
        thisTime = time.time()
        emgSignals = np.zeros(self.numChannels)
        for _ in range(self.numComponents):
            amplitude = np.random.uniform(low=0.1, high=1.0, size=self.numChannels)
            frequency = np.random.uniform(low=5, high=150, size=self.numChannels)  # Random frequency between 5 Hz and 150 Hz
            noise = np.random.normal(loc=0, scale=0.1, size=self.numChannels)

            signal = amplitude * np.sin(2*np.pi*frequency*thisTime) + noise
            emgSignals += signal

        motionArtifacts = self.artifactAmp*np.sin(2*np.pi*self.artifactFreq*thisTime)

        return emgSignals + motionArtifacts + self.offset
    
    def generatePacket(self, emg):
        self.emgData.startCode = 0
        self.emgData.gpioState = 0
        self.emgData.dataType = 0
        self.emgData.freqScalar = 1
        self.emgData.processingTime = 0

        self.emgData.dataBuf = emg
        
        self.emgData.gpioTimedState = 0
        self.emgData.sw1 = 0
        self.emgData.sw2 = 0
        self.emgData.newline = 0

        time.sleep(1/self.recordRate)

    def stream(self):
        a = 0
        while not self.quitEvent.is_set():
            self.emgData.osTime_us += int(1e6/self.recordRate)
            self.emgData.osTime_ms += int(1e3/self.recordRate)

            start = time.time()
            signals = self.generateEMG()
            self.generatePacket(signals)

            packedData = self.pack(self.emgData)

            self.sock.send(packedData)

            if a == self.printRate:
                # self.emgData.emgDataFullPrint()
                a = 0

            a += 1

            if self.quitEvent.is_set():
                break

            endTime = time.time()
            time.sleep(max(1/self.recordRate - endTime - start, 0))
        
        print('Stopping stream.')

    def pack(self, data):
        packedData = struct.pack('BBBBIHHffffffffffffffffBBBB', data.startCode, data.gpioState, data.dataType, 
            data.freqScalar, data.osTime_us, data.osTime_ms, data.processingTime, 
            data.dataBuf[0], data.dataBuf[1], data.dataBuf[2], data.dataBuf[3], data.dataBuf[4], data.dataBuf[5], data.dataBuf[6], data.dataBuf[7], 
            data.dataBuf[8], data.dataBuf[9], data.dataBuf[10], data.dataBuf[11], data.dataBuf[12], data.dataBuf[13], data.dataBuf[14], data.dataBuf[15], 
            data.gpioTimedState, data.sw1, data.sw2, data.newline)

        return packedData

    def startCommunication(self):
        self.streamThread = threading.Thread(target=self.stream, name='streamEMG')
        self.streamThread.daemon = False
        self.streamThread.start()

# ##################################
# class fakeDataSource():
#     def __init__(self, socketAddr='tcp://127.0.0.1:1235', frequency=100):
#         self.sendingHz = frequency
#         self.channelNum = 16.0
#         self.packetsPerSend = 1.0
#         self.fakeTime = 100*np.random.uniform()

#         self.sinFreq = 2000*np.pi/12
#         self.sinAmp = 10

#         self.highFreq = 1000 # we want some high frequency variation in the signal, this is what emg is - consider just multiplying whatever signal by this?

#         self.counter = 0

#         self.socketAddr = socketAddr
#         self.ctx = zmq.Context()
#         self.sock = self.ctx.socket(zmq.PUB)
#         self.sock.bind(self.socketAddr)

#     def __del__(self):
#         try:
#             self.sock.unbind(self.socketAddr)
#             self.sock.close()
#             self.ctx.term()

#         except Exception as e:
#             print(f'__del__: Socket closing error, {e}')        

#     def startService(self):
#         self.sendData()

#     def generateData(self):
#         # Generate fake data to send

#         # Get timestamps for this data
#         # if self.counter <= 5: # this sets the max value for each electrode
#         #     S = np.arange(3, 3 + 16*self.sinAmp, self.sinAmp)
#         # else: # this generates the actual oscillating activation data
#             # timeStamps = np.arange(self.fakeTime, self.fakeTime + 1/(self.sendingHz + 1), 1/self.sendingHz/self.packetsPerSend)
#             # print('timeStamps:', timeStamps)

#             # Make fake data based on the time - a sine curve and multiples of that sin curve
#         sin = self.sinAmp*np.sin(self.fakeTime*self.sinFreq)
#         multiples = np.arange(1, 17)
#         sinCosArray = np.array((sin, sin, sin, sin, sin, sin, sin, sin, sin, sin, sin, sin, sin, sin, sin, sin))
#         S = np.multiply(sinCosArray, multiples)
#         S = np.ndarray.flatten(S) + multiples*self.sinAmp

#         noise = np.random.normal(size=S.shape)
#         X = S + 0.25*noise  # Add noise
#         X = X.astype(float)
#         # X = np.abs(X)
#         X = np.clip(X, 0, None)

#         X = np.zeros_like(X)

#         return X

#     def sendData(self):
#         try:
#             while True:
#                 X = self.generateData()
#                 byte_data = struct.pack(f'{18}f{4}If', self.fakeTime, self.fakeTime, 
#                     X[0], X[1], X[2],  X[3],  X[4],  X[5],  X[6],  X[7], 
#                     X[8], X[9], X[10], X[11], X[12], X[13], X[14], X[15], 
#                     1, 0, 1, 0, self.sendingHz)

#                 # socket
#                 self.sock.send(byte_data)

#                 print(f'''{len(byte_data)} bytes sent to {self.socketAddr}\nTime: {self.fakeTime:07.3f} sec\nFrequency: {self.sendingHz:07.3f} Hz
#                     \t{X[0]:06.2f} {X[1]:06.2f} {X[2]:06.2f} {X[3]:06.2f}
#                     \t{X[4]:06.2f} {X[5]:06.2f} {X[6]:06.2f} {X[7]:06.2f}
#                     \t{X[8]:06.2f} {X[9]:06.2f} {X[10]:06.2f} {X[11]:06.2f}
#                     \t{X[12]:06.2f} {X[13]:06.2f} {X[14]:06.2f} {X[15]:06.2f}\n''')
#                 # Update the fake time
#                 self.fakeTime += 1/self.sendingHz
#                 self.counter += 1

#                 time.sleep(1/self.sendingHz)

#         except KeyboardInterrupt:
#             pass

#         print('\nFake data send complete.')

#     def startCommunication(self):
#         self.streamThread = threading.Thread(target=self.startService, name='streamEMG')
#         self.streamThread.daemon = False
#         self.streamThread.start()

if __name__ == '__main__':
    # The main program
    print("Starting fake data streamer.")
    streamer = FakeStreamer(socketAddr='tcp://18.27.123.85:1236')
    streamer.startCommunication()

    # dataSource = fakeDataSource(socketAddr='tcp://127.0.0.1:1235', frequency=1000)
    # dataSource.startCommunication()
