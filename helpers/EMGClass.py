# Mikey Fernandez, 11/09/2021
#
# EMGClass.py
# Define the EMG class for use with the LUKE arm
import matplotlib.pyplot as plt

import struct, os, sys, zmq, math
import numpy as np
import pandas as pd
from helpers.BesselFilter import BesselFilterArr
import threading
import platform


class EMG():
    def __init__(self, socketAddr='tcp://127.0.0.1:1235', numElectrodes=16, tauA=0.05, tauD=0.1, usedChannels=None, usingSynergies=False, samplingFreq=None, offlineData=None, maxVals=None, noiseLevel=None):
        self.numElectrodes = numElectrodes
        self.tauA = tauA
        self.tauD = tauD
        self.usingSynergies = usingSynergies
        self.exitEvent = threading.Event()

        if usedChannels is None:
            self.usedChannels = []
        else:
            self.usedChannels = usedChannels

        if offlineData is not None:
            self.offlineData = self.offline_data_gen(offlineData)
            self.maxVals = maxVals
            self.noiseLevel = noiseLevel
        else:
            self.offlineData = None

        self.emgHistory = np.empty((self.numElectrodes, 1))
        self.r_history = np.empty((self.numElectrodes, 1))
        self.f_history = np.empty((self.numElectrodes, 1))

        if platform.system() == 'Linux':
            self.boundsPath = '/home/haptix/haptix/haptix_controller/handsim/include/scaleFactors.txt'
            self.deltasPath = '/home/haptix/haptix/haptix_controller/handsim/include/deltas.txt'
            self.synergyPath = '/home/haptix/haptix/haptix_controller/handsim/include/synergyMat.csv'
        else:
            self.boundsPath = '/Users/jg/projects/biomech/UEA-AMI-Controller/handsim/include/scaleFactors.txt'
            self.deltasPath = '/Users/jg/projects/biomech/UEA-AMI-Controller/handsim/include/deltas.txt'
            self.synergyPath = '/Users/jg/projects/biomech/UEA-AMI-Controller/handsim/include/synergyMat.csv'

        self.socketAddr = socketAddr
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.connect(self.socketAddr)
        self.sock.subscribe('')

        # this is how the EMG package gets broken up
        self.OS_time = None
        self.OS_tick = None
        self.trigger = None
        self.switch1 = None
        self.switch2 = None
        self.end = None
        self.samplingFreq = samplingFreq # this SHOULD be 1 kHz - but don't assume that
        if self.offlineData is None:
            self.getBounds() # first 16: maximum values, second 16: minimum values
            self.getDeltas() # first 8: maximum deltas, second 8: minimum deltas
            if self.usingSynergies:
                self.getSynergyMat() # this is an [nSynergies x usedChannels] array
                self.numSynergies = self.synergyMat.shape[1]

        self.resetEMG()
        
        # initialize the EMG with the first signal (need the sampling frequency)
        self.readEMG()
        self.initFilters()

        # parameters for calculating iEMG
        self.int_window = .05 # sec - 50 ms integration window
        self.intFreq = 60 # group packets to this frequency in Hz
        self.numPackets = math.ceil(self.samplingFreq/self.intFreq)

        self.window_len = math.ceil(self.int_window*self.samplingFreq)
        self.rawHistory = np.zeros((self.numElectrodes, self.numPackets)) # switch to processing packets (apply filters only once on a number of packets)

    def __del__(self):
        try:
            # close the socket
            self.sock.close()
            self.ctx.term()

            self.exitEvent.set()

        except Exception as e:
            print(f'__del__: Socket closing error {e}')

    def offline_data_gen(self, data):
        padded = np.pad(data, ((0, 0), (self.numPackets*100, self.numPackets*100)), mode='edge')
        for i in range(padded.shape[1]):
            if i == padded.shape[1] - self.numPackets:
                self.exitEvent.set()
            yield padded[:, i]



    def resetEMG(self):
        self.rawEMG = [-1]*self.numElectrodes # this is RAW from the board - need to get iEMG
        self.iEMG = [-1]*self.numElectrodes # this is iEMG
        self.filtEMG = [-1]*self.numElectrodes # this is the high pass and motion artifcats EMG
        self.normedEMG = [-1]*self.numElectrodes # array of normalized EMG values
        self.muscleAct = [-1]*self.numElectrodes # array of muscle activation (through low pass muscle activation dynamics)
        self.prevAct = [-1]*self.numElectrodes # array of previous muscle activation values
        self.synergies = [-1]*self.numSynergies if self.usingSynergies else [-1]*self.numElectrodes

    def initFilters(self):
        self.powerLineFilterArray = BesselFilterArr(numChannels=self.numElectrodes, order=8, critFreqs=[58, 62], fs=self.samplingFreq, filtType='bandstop') # remove power line noise and multiples up to 600 Hz
        self.highPassFilters = BesselFilterArr(numChannels=self.numElectrodes, order=4, critFreqs=20, fs=self.samplingFreq, filtType='highpass') # high pass removes motion artifacts and drift
        self.lowPassFilters = BesselFilterArr(numChannels=self.numElectrodes, order=4, critFreqs=3, fs=self.samplingFreq, filtType='lowpass') # smooth the envelope, when not using 'actually' integrated EMG

    def startCommunication(self, raw=False):
        # set the emg thread up here
        # self.pipelineEMG()
        self.emgThread = threading.Thread(target=self.pipelineEMG, name='pipelineEMG', args=(raw,))
        self.emgThread.daemon = False
        self.emgThread.start()

    ##########################################################################
    # print functions
    def printNorms(self):
        print('EMG Bounds:')
        norms = self.bounds
        print(f'''\tMaxes:\n\t\t{norms[0]:07.2f} {norms[1]:07.2f} {norms[2]:07.2f} {norms[3]:07.2f}
            \t\t{norms[4]:07.2f} {norms[5]:07.2f} {norms[6]:07.2f} {norms[7]:07.2f}
            \t\t{norms[8]:07.2f} {norms[9]:07.2f} {norms[10]:07.2f} {norms[11]:07.2f}
            \t\t{norms[12]:07.2f} {norms[13]:07.2f} {norms[14]:07.2f} {norms[15]:07.2f}''')

        print(f'''\Mins:\n\t\t{norms[16]:07.2f} {norms[17]:07.2f} {norms[18]:07.2f} {norms[19]:07.2f}
            \t\t{norms[20]:07.2f} {norms[21]:07.2f} {norms[22]:07.2f} {norms[23]:07.2f}
            \t\t{norms[24]:07.2f} {norms[25]:07.2f} {norms[26]:07.2f} {norms[27]:07.2f}
            \t\t{norms[28]:07.2f} {norms[29]:07.2f} {norms[30]:07.2f} {norms[31]:07.2f}''')

    def printDeltas(self):
        print('EMG Deltas:')
        deltas = self.deltas

        print(f'''\tMaxes:\n\t\t{deltas[0]:07.2f} {deltas[1]:07.2f} {deltas[2]:07.2f} {deltas[3]:07.2f}\n\t\t{deltas[4]:07.2f} {deltas[5]:07.2f} {deltas[6]:07.2f} {deltas[7]:07.2f}''')
        print(f'''\tMins:\n\t\t{deltas[8]:07.2f} {deltas[9]:07.2f} {deltas[10]:07.2f} {deltas[11]:07.2f}\n\t\t{deltas[12]:07.2f} {deltas[13]:07.2f} {deltas[14]:07.2f} {deltas[15]:07.2f}''')

    def printSynergyMat(self):
        print(f'Synergy Matrix:\n{self.synergyMat}')

    def printSynergies(self):
        print(f'Synergies:\n{self.synergies}')

    def printRawEMG(self):
        print(f'Raw EMG at {self.OS_time}:')
        raw = self.rawEMG

        print(f'''\t{raw[0]:07.2f} {raw[1]:07.2f} {raw[2]:07.2f} {raw[3]:07.2f}\n\t{raw[4]:07.2f} {raw[5]:07.2f} {raw[6]:07.2f} {raw[7]:07.2f}\n\t{raw[8]:07.2f} {raw[9]:07.2f} {raw[10]:07.2f} {raw[11]:07.2f}\n\t{raw[12]:07.2f} {raw[13]:07.2f} {raw[14]:07.2f} {raw[15]:07.2f}\n''')

    def printiEMG(self):
        print(f'iEMG at {self.OS_time}:')
        iEMG = self.iEMG

        print(f'''\t{iEMG[0]:07.2f} {iEMG[1]:07.2f} {iEMG[2]:07.2f} {iEMG[3]:07.2f}\n\t{iEMG[4]:07.2f} {iEMG[5]:07.2f} {iEMG[6]:07.2f} {iEMG[7]:07.2f}\n\t{iEMG[8]:07.2f} {iEMG[9]:07.2f} {iEMG[10]:07.2f} {iEMG[11]:07.2f}\n\t{iEMG[12]:07.2f} {iEMG[13]:07.2f} {iEMG[14]:07.2f} {iEMG[15]:07.2f}\n''')

    def printiEMG(self):
        print(f'High Pass EMG at {self.OS_time}:')
        iEMG = self.filtEMG

        print(f'''\t{iEMG[0]:07.2f} {iEMG[1]:07.2f} {iEMG[2]:07.2f} {iEMG[3]:07.2f}\n\t{iEMG[4]:07.2f} {iEMG[5]:07.2f} {iEMG[6]:07.2f} {iEMG[7]:07.2f}\n\t{iEMG[8]:07.2f} {iEMG[9]:07.2f} {iEMG[10]:07.2f} {iEMG[11]:07.2f}\n\t{iEMG[12]:07.2f} {iEMG[13]:07.2f} {iEMG[14]:07.2f} {iEMG[15]:07.2f}\n''')

    def printNormedEMG(self):
        print(f'Normed EMG at {self.OS_time}:')
        emg = self.normedEMG

        print(f'''\t{emg[0]:07.2f} {emg[1]:07.2f} {emg[2]:07.2f} {emg[3]:07.2f}\n\t{emg[4]:07.2f} {emg[5]:07.2f} {emg[6]:07.2f} {emg[7]:07.2f}\n\t{emg[8]:07.2f} {emg[9]:07.2f} {emg[10]:07.2f} {emg[11]:07.2f}\n\t{emg[12]:07.2f} {emg[13]:07.2f} {emg[14]:07.2f} {emg[15]:07.2f}\n''')

    def printMuscleAct(self):
        print(f'Muscle Activation at {self.OS_time}:')
        mAct = self.muscleAct

        print(f'''\t{mAct[0]:07.2f} {mAct[1]:07.2f} {mAct[2]:07.2f} {mAct[3]:07.2f}\n\t{mAct[4]:07.2f} {mAct[5]:07.2f} {mAct[6]:07.2f} {mAct[7]:07.2f}\n\t{mAct[8]:07.2f} {mAct[9]:07.2f} {mAct[10]:07.2f} {mAct[11]:07.2f}\n\t{mAct[12]:07.2f} {mAct[13]:07.2f} {mAct[14]:07.2f} {mAct[15]:07.2f}\n''')

    ##########################################################################
    # get fields
    def getRawEMG(self, electrode):
        if electrode >= self.numElectrodes:
            raise ValueError(f'getRawEMG(): Asked for invalid electrode {electrode} of {self.numElectrodes}')
           
        return self.rawEMG[electrode]

    def getiEMG(self, electrode):
        if electrode >= self.numElectrodes:
            raise ValueError(f'getiEMG(): Asked for invalid electrode {electrode} of {self.numElectrodes}')
           
        return self.iEMG[electrode]

    def getFiltEMG(self, electrode):
        if electrode >= self.numElectrodes:
            raise ValueError(f'getFiltEMG(): Asked for invalid electrode {electrode} of {self.numElectrodes}')
           
        return self.filtEMG[electrode]
    
    def getNormedEMG(self, electrode):
        if electrode >= self.numElectrodes:
            raise ValueError(f'getNormedEMG(): Asked for invalid electrode {electrode} of {self.numElectrodes}')
           
        return self.normedEMG[electrode]

    def getFilteredEMG(self, electrode):
        if electrode >= self.numElectrodes:
            raise ValueError(f'getFilteredEMG(): Asked for invalid electrode {electrode} of {self.numElectrodes}')
           
        return self.muscleAct[electrode]

    def getSynergy(self, synergy):
        if synergy >= self.numSynergies:
            raise ValueError(f'getSynergy(): Asked for invalid synergy {synergy} of {self.numSynergies}')
           
        return self.synergies[synergy]

    def getBounds(self):
        if self.offlineData is None:
            try:
                with open(self.boundsPath, 'rb') as fifo:
                    normsPack = fifo.read()

                norms = struct.unpack('ffffffffffffffffffffffffffffffff', normsPack)
                self.bounds = list(norms)
                self.maxVals = np.asarray(self.bounds[:self.numElectrodes])
                self.noiseLevel = np.asarray(self.bounds[self.numElectrodes:])

            except OSError as e:
                print(f'getBounds(): Could not read bounds - {e}')

    def getDeltas(self):
        try:
            with open(self.deltasPath, 'rb') as fifo:
                deltasPack = fifo.read()

            deltas = struct.unpack('ffffffffffffffff', deltasPack)
            self.deltas = list(deltas)

        except OSError as e:
            print(f'getDeltas(): Could not read deltas - {e}')

    def getSynergyMat(self):
        try:
            df = pd.read_csv(self.synergyPath, sep=' ', header=None, engine='python')
            synergyMat = df.to_numpy()
            self.synergyMat = synergyMat

        except Exception as e:
            print(f'getSynergyMat(): Could not read matrix - {e}')

    ##########################################################################
    # actual calculations
    def readEMG(self):
        if self.offlineData is None:
            try:
                emgPack = self.sock.recv()

                emg = struct.unpack('ffffffffffffffffffIIIIf', emgPack)

                self.OS_time = emg[0]
                self.OS_tick = emg[1]
                self.rawEMG = emg[2:18]
                self.trigger = emg[18]
                self.switch1 = emg[19]
                self.switch2 = emg[20]
                self.end = emg[21]
                self.samplingFreq = emg[22]  # todo adapt sampling frequency

            except OSError as e:
                print(f'readEMG(): Could not read EMG - {e}')

    # read multiple EMG packets to save time and processing
    def readEMGPacket(self):
        for i in range(self.numPackets):
            if self.offlineData is None:
                emgPack = self.sock.recv()

                emg = struct.unpack('ffffffffffffffffffIIIIf', emgPack)

                self.OS_time = emg[0]
                self.OS_tick = emg[1]
                self.rawEMG = emg[2:18]
                self.trigger = emg[18]
                self.switch1 = emg[19]
                self.switch2 = emg[20]
                self.end = emg[21]
                self.samplingFreq = emg[22]

                self.rawHistory[:, i] = emg[2:18]
            else:
                self.rawHistory[:, i] = next(self.offlineData)


    # calculate integrated emg over multiple packets
    def intEMGPacket(self):
        emg = np.copy(self.rawHistory)

        emg = self.powerLineFilterArray.filter(emg)
        emg = self.highPassFilters.filter(emg)

        self.filtEMG = np.copy(np.asarray(emg)[:, -1]) # motion artificats and drift removed

        emg = np.abs(emg)
        emg = np.clip(emg - self.noiseLevel[:, None], 0, None)

        iEMG = np.clip(self.lowPassFilters.filter(emg), 0, None)


        if self.offlineData is None:
            self.iEMG = np.asarray(iEMG)[:, -1]
            # self.iEMG = np.mean(np.asarray(iEMG), axis=1)
        else:
            self.iEMG = np.asarray(iEMG)

        #######


    # normalize the EMG
    def normEMG(self):
        if self.offlineData is None:
            normed = self.iEMG/self.maxVals
        else:
            normed = self.iEMG/self.maxVals[:, None]
        self.normedEMG = np.clip(normed, 0, 1)


    # first order muscle activation dynamics
    def muscleDynamics(self):
        tauA = self.tauA
        tauD = self.tauD
        b = tauA/tauD
        Ts = 1/self.samplingFreq

        u = np.asarray(self.normedEMG)
        self.prevAct = self.muscleAct
        prevA = np.asarray(self.prevAct)

        # muscleAct = np.abs((u/tauA + prevA/Ts)/(1/Ts + (b + (1 - b)*u)/tauA))
        muscleAct = np.where(u > prevA, (u*(b + (1 - b)*u)/tauA + prevA/Ts)/(1/Ts + (b + (1 - b)*u)/tauA), (u/tauD + prevA/Ts)/(1/Ts + 1/tauD))
        muscleAct[muscleAct < 0] = 0
        muscleAct[muscleAct > 1] = 1

        self.muscleAct = muscleAct

    # get the synergies
    def synergyProd(self):
        if not self.numSynergies == len(self.usedChannels):
            raise ValueError(f'synergyProd(): Wrong number of channels ({len(self.usedChannels)} provided; require {self.numSynergies}')

        self.synergies = self.synergyMat @ self.normedEMG[self.usedChannels]
        return self.synergies

    # full EMG update pipeline
    def pipelineEMG(self, raw=False):
        while not self.exitEvent.is_set():
            self.readEMGPacket()
            if not raw:
                self.intEMGPacket()
                self.normEMG()

                if self.offlineData is None:
                    if self.usingSynergies: self.synergyProd()
                    self.muscleDynamics()
                    # self.emgHistory = np.concatenate((self.emgHistory, self.normedEMG[:, None]), axis=1)
                else:
                    # self.emgHistory = np.concatenate((self.emgHistory, self.iEMG[:, None]), axis=1)
                    self.emgHistory = np.concatenate((self.emgHistory, self.normedEMG), axis=1)  # integrated data for use
                    # self.r_history = np.concatenate((self.r_history, self.rawHistory), axis=1)  # raw data for figures
                    # self.f_history = np.concatenate((self.f_history, self.filtEMG), axis=1)  # filtered data for figures


if __name__ == '__main__':
    pass