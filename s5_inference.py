# Run $env:KMP_DUPLICATE_LIB_OK="TRUE" in terminal to solve OMP error


import os

import serial
from serial.tools import list_ports
import sys
import argparse
import zmq
import threading
import struct
import time
import pandas as pd
import platform
from pathlib import Path
import yaml
from os.path import join

sys.path.append('/home/haptix/haptix/haptix_controller/handsim/src')


from helpers.EMGClass import EMG
from helpers.BesselFilter import BesselFilterArr
from helpers.ExponentialFilter import ExponentialFilterArr
from helpers.psyonicControllers import psyonicControllers
from helpers.psyonicControllers import create_controllers
from helpers.predict_utils import Config
import numpy as np
from scipy import signal
from collections import deque

class SimplifiedForceProcessor:
    """
    Simplified force processing:
    - Zeroing of force sensors when entering free space
    - Baseline drift correction only during free space
    """
    
    def __init__(self, sampling_freq=60.0):
        self.sampling_freq = sampling_freq
        self.finger_names = ['index', 'middle', 'ring', 'pinky', 'thumb']
        
        # Simple drift tracking (moving average during free space)
        self.drift_window_size = 150  # 2.5 seconds at 60Hz
        self.drift_buffers = [deque(maxlen=self.drift_window_size) for _ in range(5)]
        self.current_drift = np.zeros(5)
        
        # Force normalization constants (same as training)
        self.max_forces_per_finger = np.array([12.0, 12.0, 12.0, 12.0, 12.0])
        
        print("SimplifiedForceProcessor initialized")
    
    def extract_finger_forces_from_sensors(self, sensors_dict, touchNames):
        """
        Extract per-finger forces from sensor dictionary using touchNames order
        """
        finger_forces = np.zeros(5)
        
        for finger_idx, finger in enumerate(touchNames):
            finger_total = 0.0
            sensors_found = 0
            
            for sensor_idx in range(6):
                sensor_name = f'{finger}{sensor_idx}_Force'
                if sensor_name in sensors_dict:
                    value = sensors_dict[sensor_name]
                    if value != -1:  # Valid sensor reading
                        finger_total += value
                        sensors_found += 1
            
            if sensors_found > 0:
                finger_forces[finger_idx] = finger_total
            # else: finger_forces[finger_idx] remains 0
        
        return finger_forces
	
    def reset_baseline_immediately(self, current_forces):
        """
        zeroing baseline when entering free space mode
        """
        self.current_drift = current_forces.copy()
        
        # Clear drift buffers and start fresh
        for buffer in self.drift_buffers:
            buffer.clear()
        
        print(f"Forces immediately zeroed - new baseline: {self.current_drift}")
    
    def update_drift_baseline(self, finger_forces):
        """Update drift baseline during free space mode"""
        for finger_idx in range(5):
            self.drift_buffers[finger_idx].append(finger_forces[finger_idx])
            
            # Update current drift estimate (median of recent values)
            if len(self.drift_buffers[finger_idx]) > 10:
                self.current_drift[finger_idx] = np.median(
                    list(self.drift_buffers[finger_idx])
                )
    
    def process_runtime_force(self, sensors_dict, touchNames, in_interaction_mode, mode_just_changed=False):
        """
        Process forces with immediate zeroing on free space entry
        
        Args:
            mode_just_changed: True if we just switched modes this cycle
        """
        # 1. Extract raw per-finger forces
        raw_forces = self.extract_finger_forces_from_sensors(sensors_dict, touchNames)
        
        # 2. Mode-based processing with immediate zeroing
        if in_interaction_mode:
            # INTERACTION MODE: Use RAW forces 
            processed_forces = np.maximum(raw_forces, 0.0)
            neural_network_forces = processed_forces  # Raw forces to neural network
            
        else:
            # FREE SPACE MODE
            if mode_just_changed:
                # JUST ENTERED FREE SPACE: Immediately zero forces
                self.reset_baseline_immediately(raw_forces)
            else:
                # CONTINUING IN FREE SPACE: Update drift correction
                self.update_drift_baseline(raw_forces)
            
            # Apply drift correction for mode switching
            corrected_forces = raw_forces - self.current_drift
            corrected_forces = np.maximum(corrected_forces, 0.0)
            processed_forces = corrected_forces
            neural_network_forces = np.zeros_like(corrected_forces)  # Neural network gets zeros
        
        # 3. Normalize for neural network
        normalized_forces = np.clip(neural_network_forces / self.max_forces_per_finger, 0.0, 1.0)
        
        # 4. Calculate aggregates for mode switching
        total_force = np.sum(processed_forces)
        max_finger_force = np.max(processed_forces)
        
        return {
            'raw': raw_forces,
            'processed': processed_forces,
            'normalized': normalized_forces,      # For NN
            'total_force': total_force,            
            'max_finger_force': max_finger_force,  # For mode switching
            'current_drift': self.current_drift.copy()
        }

class psyonicArm():
	def __init__(self, hand='right', usingEMG=False, stuffing=False, baud=460800, plotSocketAddr='tcp://127.0.0.1:1240', dummy=False):
		self.baud = baud
		self.hand = hand
		self.stuffing = stuffing
		self.usingEMG = usingEMG

		self.initialized = False
		self.impedanceMode = False

		# initialize hand state
		# define control mode format headers
		self.replyVariant = 0 # one of [0, 1, 2], which corresponds to index in the lists below
		self.controlMode = 'position'
		self.controlModeHeaders = {'position': [0x10, 0x11, 0x12], 'velocity': [0x20, 0x21, 0x22], 'torque': [0x30, 0x31, 0x32], 'voltage': [0x40, 0x41, 0x42], 'readOnly': [0xA0, 0xA1, 0xA2]}
		self.controlModeResponse = {1: 'position', 2: 'velocity', 3: 'torque', 4: 'voltage', 10: 'readOnly'} # achieved by rightshifting the format header by 4
		self.miscCommandHeaders = {'upsampleThumbEn': 0xC2, 'upsampleThumbDis': 0xC3, 'exitAPI': 0x7C}

		self.handAddr = 0x50 # don't change me
		self.replyModeHeader = self.controlModeHeaders[self.controlMode][self.replyVariant] # for setting the reply mode variant!
		self.responseBufferSize = lambda type: 39 if type == 2 else 72 # anonymous function handles both reply types
		self.posConversion = 32767/150
		self.velConversion = 32767/3000
		self.rotorConversion = 4
		self.currConversion = 620.606079
		self.volConversion = 3.3/4096
		self.resConversion = 33000
		self.resAdd = 10000
		self.forceResConversion = [121591.0, 0.878894] # I need to do this
		self.radToDeg = lambda rad: 180*rad/3.14159
		self.torqueConstant = 1.49 # mNm/A torque constant
		self.voltLimit = 3546
		self.supplyVoltage = 3.3 # volts

		# kinematic parameters
		self.fingerAngleFit = [1.0585, 41.4534]
		self.fingerLinkLengths = {'index': [0.03861, 0.03324], 'middle': [0.03861, 0.03324], 'ring': [0.03861, 0.03324], 'pinky': [0.03861, 0.03324], 'thumb': [0.03150, 0.05907]}

		# define arm information
		self.numMotors = 6
		self.numForce = 30
		initPos = 60
		self.curPos = initPos*np.ones(self.numMotors)
		self.handCom = self.curPos

		# setup communication with arm (serial port)
		# self.serialSet = self.setupSerial(passedPort='/dev/psyonicHand')
		self.serialSet = self.setupSerial(passedPort='COM7')
		if not self.serialSet:
			sys.exit('Error: Serial Port not found')

		# for sending to the plot
		self.plotSocketAddr = plotSocketAddr
		self.plotCTX = zmq.Context()
		self.plotSock = self.plotCTX.socket(zmq.PUB)
		self.plotSock.bind(self.plotSocketAddr)

		# setup message state
		self.startTimestamp = time.time()
		self.timestamp = self.startTimestamp

		# joint and sensor names and information
		self.jointNames = ['index', 'middle', 'ring', 'pinky', 'thumbFlex', 'thumbRot']
		self.sensorNames = ['index', 'middle', 'ring', 'pinky', 'thumbFlex', 'thumbRot']
		self.touchNames = ['index', 'middle', 'ring', 'pinky', 'thumb']
		self.sensorPos = [f'{name}_Pos' for name in self.sensorNames]
		self.sensorVel = [f'{name}_Vel' for name in self.sensorNames]
		self.sensorCur = [f'{name}_Cur' for name in self.sensorNames]
		self.sensorForce = [site for finger in [[f'{name}{i}_Force' for i in range(6)] for name in self.touchNames] for site in finger]
		self.sensorForceOffsets = dict.fromkeys(self.sensorForce, 0)
		self.sensorLimit = [f'{name}_Limit' for name in self.sensorNames]
		self.sensors = dict.fromkeys(self.sensorPos + self.sensorVel + self.sensorCur + self.sensorLimit + self.sensorForce, -1)

		self.gearRatios = dict(zip(self.jointNames, [649, 649, 649, 649, 649, 162.45]))

		self.jointRoM = {'index': [0, 120], 'middle': [0, 120], 'ring': [0, 120], 'pinky': [0, 120], 'thumbFlex': [0, 120], 'thumbRot': [-120, 0]}

		# for arm status
		self.recording = False
		self.isMoving = False
		self.movingEvent = threading.Event()
		self.exitEvent = threading.Event()
		self.replyChangedFlag = False

		# Force processing at 60Hz
		self.force_processing_counter = 0
		self.last_processed_forces = {}  # Cache last processed force values
		for finger in self.touchNames:
			for site in range(6):
				sensor_name = f'{finger}{site}_Force'
				self.last_processed_forces[sensor_name] = 0.0

		# neural net control loop rate
		self.Hz = 60
		self.loopRate = 10 # this is how much faster this should run than the neural net

		# lowpass filter joint commands
		self.lowpassCommands = BesselFilterArr(numChannels=self.numMotors, order=4, critFreqs=0.33, fs=self.Hz, filtType='lowpass')
		# self.lowpassCommands = ExponentialFilterArr(numChannels=self.numMotors, smoothingFactor=0.9, defaultValue=0)

		# exponential filter the force sensor readings
		self.filterForce = ExponentialFilterArr(numChannels=self.numForce, smoothingFactor=0.8, defaultValue=0) # todo

		self.simplified_force_processor = SimplifiedForceProcessor(sampling_freq=self.Hz)

		# store prior commands for some reason
		self.lastPosCom = -1*np.ones(self.numMotors)
		self.lastVelCom = -1*np.ones(self.numMotors)
		self.lastTorCom = -1*np.ones(self.numMotors)
		self.lastVolCom = -1*np.ones(self.numMotors)

		# 2 Mode Controller
		self.in_interaction_mode = False
		self.enter_timer = None
		self.exit_timer = None
		self.blend_start_time = None
		self.blend_duration = 0.20  # 200ms blending
		
		# Thresholds for mode switching
		self.ENTER_THRESHOLD = 0.8   # N
		self.EXIT_THRESHOLD = 0.6    # N  
		self.ENTER_DEBOUNCE = 0.10   # seconds
		self.EXIT_DEBOUNCE = 0.15    # seconds

		# for byte stuffing
		self.frameChar = 0x7E
		self.escChar = 0x7D
		self.escMask = 0x20

	def __del__(self):
		try:
			self.ser.close()
		except Exception as e:
			print(f'__del__: Serial port closing error {e}')

		try:
			# close the socket
			self.plotSock.close()
			self.plotCTX.term()
		except Exception as e:
			print(f'__del__: Plot socket closing error {e}')

	# Search for Serial Port to use
	def setupSerial(self, passedPort=None):
		if passedPort is not None:
			self.ser = serial.Serial(passedPort, self.baud, timeout=0.02, write_timeout=0.02)
			assert self.ser.is_open, 'Failed to open serial port'
			print(f'Connected to port {self.ser.name}')
			return True

		print('Searching for serial ports...')
		com_ports_list = list(list_ports.comports())
		port = ''

		for p in com_ports_list:
			if p:
				if platform.system() == 'Linux':
					if 'USB' in p[0] and 'FT232' in p.description:
						port = p
						break
				elif platform.system() == 'Windows':
					if 'COM' in p[0] and 'UART' in p.description:
						port = p
						break
				elif platform.system() == 'Darwin':
					if 'USB' in p.description and 'FT232' in p.description:
						port = p
						break
				else:
					raise Exception('Unsupported OS')

		if not port:
			print('No port found')
			sys.exit()

		try:
			self.ser = serial.Serial(port[0], self.baud, timeout=0.02, write_timeout=0.02)
			assert self.ser.is_open, 'Failed to open serial port'
			print(f'Connected to port {self.ser.name} ({port.description})')

			return True

		except Exception as e:
			print(f'Failed to Connect - {e}')
			return False

	####### PRINTING FUNCTIONS #######
	def printSensors(self):
		s = self.sensors # to save me from typing
		print(f'\nRunning time: {self.timestamp - self.startTimestamp:f}')
		print(f'\tCurrent mode: {self.controlMode} | current reply mode: {self.replyVariant}')
		print('\tJoint positions:')
		print(f'\t\t  Index pos: {s["index_Pos"]:8.3f} |     Middle pos: {s["middle_Pos"]:8.3f} |      Ring pos: {s["ring_Pos"]:8.3f}')
		print(f'\t\t  Pinky pos: {s["pinky_Pos"]:8.3f} | Thumb flex pos: {s["thumbFlex_Pos"]:8.3f} | Thumb rot pos: {s["thumbRot_Pos"]:8.3f}')
		print('\tJoint velocities:')
		print(f'\t\t  Index vel: {s["index_Vel"]:8.3f} |     Middle vel: {s["middle_Vel"]:8.3f} |      Ring vel: {s["ring_Vel"]:8.3f}')
		print(f'\t\t  Pinky vel: {s["pinky_Vel"]:8.3f} | Thumb flex vel: {s["thumbFlex_Vel"]:8.3f} | Thumb rot vel: {s["thumbRot_Vel"]:8.3f}')
		print('\tJoint current:')
		print(f'\t\t  Index cur: {s["index_Cur"]:8.3f} |     Middle cur: {s["middle_Cur"]:8.3f} |      Ring cur: {s["ring_Cur"]:8.3f}')
		print(f'\t\t  Pinky cur: {s["pinky_Cur"]:8.3f} | Thumb flex cur: {s["thumbFlex_Cur"]:8.3f} | Thumb rot cur: {s["thumbRot_Cur"]:8.3f}')
		print('\n\tForce sensors:')
		for joint in self.touchNames:
			touchReadings = [s[joint + f"{i}_Force"] for i in range(6)]
			print(f'\t\t   {joint:>8}: ', [f'{reading:8.3f}' for reading in touchReadings])
			# print(f'\t\t   {joint:>8}: ', [f'{reading:8d}' for reading in touchReadings])
		print(f'\n\tLimit sensors ({self.jointNames}):')
		print(f'\t\t  ', [s[joint + f"_Limit"] for joint in self.sensorNames])

	def printCom(self):
		printDict = {'position': 'Pos', 'velocity': 'Vel', 'torque': 'Tor', 'voltage': 'Vol'}
		if not self.controlMode in printDict.keys(): return # only print if you have a valid one!

		typ = printDict[self.controlMode]

		# print for any type, including some status
		print(f'\nRunning time: {self.timestamp - self.startTimestamp:f}')
		print(f'\tCurrent mode: {self.controlMode} | current reply mode: {self.replyVariant}')

		# print the command
		print(f'\t{self.controlMode} command:')
		print(f'\t\t  Index {typ}: {self.handCom[0]:8.3f} |     Middle {typ}: {self.handCom[1]:8.3f} |  Ring {typ}: {self.handCom[2]:8.3f}')
		print(f'\t\t  Pinky {typ}: {self.handCom[3]:8.3f} | Thumb flex {typ}: {self.handCom[4]:8.3f} | Thumb rot {typ}: {self.handCom[5]:8.3f}')

	####### LOGGING FUNCTIONS #######
	# set the first row of the recorded data - the title of each column
	def resetRecording(self):
		rawEMGNames = [f'raw{i}' for i in range(16)]
		iEMGNames = [f'iEMG{i}' for i in range(16)]
		posCom = [f'{name}_PosCom' for name in self.jointNames]
		velCom = [f'{name}_VelCom' for name in self.jointNames]
		torCom = [f'{name}_TorCom' for name in self.jointNames]
		volCom = [f'{name}_VolCom' for name in self.jointNames]
		titles = [['Timestamp', 'controlModeHeader'] + posCom + velCom + torCom + volCom + self.sensorPos + self.sensorVel + self.sensorCur + self.sensorLimit + self.sensorForce + rawEMGNames + iEMGNames + ['Trigger']]
		self.recordedData = None
		self.recordedData = titles

	def addLogEntry(self, emg=None):
		# newEntry = [time.time()]
		newEntry = [self.timestamp]
		newEntry.append(self.replyModeHeader)

		# add the command
		rawCom = [-1]*4*self.numMotors # raw commands only
		if self.controlMode == 'position': rawCom[:self.numMotors] = self.handCom
		elif self.controlMode == 'velocity': rawCom[self.numMotors:2*self.numMotors] = self.handCom
		elif self.controlMode == 'torque': rawCom[2*self.numMotors:3*self.numMotors] = self.handCom
		elif self.controlMode == 'voltage': rawCom[3*self.numMotors:4*self.numMotors] = self.handCom
		elif self.controlMode == 'readOnly': pass
		else: raise ValueError(f'Invalid control mode {self.controlMode}')

		newEntry.extend(rawCom)

		# add the sensor readings
		for joint in self.sensorPos:
			newEntry.extend([self.sensors[joint]])
		for joint in self.sensorVel:
			newEntry.extend([self.sensors[joint]])
		for joint in self.sensorCur:
			newEntry.extend([self.sensors[joint]])
		for joint in self.sensorLimit:
			newEntry.extend([self.sensors[joint]])
		for touchSite in self.sensorForce:
			newEntry.extend([self.sensors[touchSite]])

		if emg is None:
			newEntry.extend([0]*33) # hardcoded 33! 16 channels raw EMG, 16 normed iEMG, and 1 trigger
		else:
			newEntry.extend(emg.rawEMG)
			newEntry.extend(emg.normedEMG)
			newEntry.extend([emg.trigger])

		self.recordedData.append(newEntry)

	####### COMMUNICATION FUNCTIONS #######
	# Communicate with the hand
	def serialCom(self):
		while not self.stopEvent.is_set():
			if self.ser.is_open:
				startT = time.time()
				msg = self.generateTx(self.handCom)

				# print(self.replyVariant, self.controlMode, hex(self.replyModeHeader))

				msg = msg if not self.stuffing else self.byteStuff(msg)
				bytesWritten = self.ser.write(msg)

				assert bytesWritten == len(msg), f'Not all bytes sent - expected {len(msg)}, sent {bytesWritten}'

				# receive and unpack the response
				try:
					response = self.ser.read(self.responseBufferSize(self.replyVariant))
					response = response if not self.stuffing else self.byteUnstuff(response)

					if not self.replyChangedFlag:# and len(response) > 0:
						self.unpackResponse(response)
					# elif len(response) > 0:
					else:
						self.replyChangedFlag = False

				except serial.SerialTimeoutException:
					print('Timeout exception - no response received, skip')
					continue

				self.sendToPlots()
				time.sleep(max(0, 1/(self.loopRate*self.Hz) - (time.time() - startT)))

			else:
				print('Serial Port not open, trying to reconnect...')
				self.serialSet = self.setupSerial()
				if not self.serialSet:
					sys.exit('Error: Serial Port not found')

	def startComms(self):
		self.stopEvent = threading.Event()
		self.sendThread = threading.Thread(target=self.serialCom, name='serialCom')
		self.sendThread.daemon = True
		self.sendThread.start()

	def byteStuff(self, msg):
		# this should be applied to the packaged frame
		msgStuff = np.asarray(msg, dtype=np.uint8).copy()

		# find all instances of the ESC char, prepend the escape char, and xor them with 0x20
		inds = np.where(msgStuff == self.escChar)[0]
		msgStuff[inds] = np.bitwise_xor(msgStuff[inds], self.escMask)
		msgStuff = np.insert(msgStuff, inds, self.escChar)
		
		# find all frame chars in data, prepend the escape char, and xor them with 0x20
		inds = np.where(msgStuff == self.frameChar)[0]
		msgStuff[inds] = np.bitwise_xor(msgStuff[inds], self.escMask)
		msgStuff = np.insert(msgStuff, inds, self.escChar)

		# finally, prepend and postpend the frame characters
		msgStuff = np.insert(msgStuff, 0, self.frameChar)
		msgStuff = np.append(msgStuff, self.frameChar)

		return list(msgStuff)

	def byteUnstuff(self, response):
		return response

	def unpackResponse(self, responseRaw):
		responseRaw = responseRaw if not self.stuffing else self.byteUnstuff(responseRaw)

		# unpack the full response into a list of bytes?
		response = list(struct.unpack(f'<{len(responseRaw)}B', responseRaw))
		
		if len(response) == 0:
		# 	print('response length 0')
			return # sometimes this happens idk

		# validate the checksum
		if not self.calcChecksum(response[:-1]) == response[-1]:  return
		assert self.calcChecksum(response[:-1]) == response[-1], f'Checksum failed - expected {hex(self.calcChecksum(response[:-1]))}, got {hex(response[-1])}'

		# make sure the response is the right length
		if self.replyVariant in [0, 1] and not len(response) == 72:  assert len(response) == 72, f'Response is not the correct length for variant {self.replyVariant} - expected 72, got {len(response)}'
		elif self.replyVariant == 2 and not len(response) == 29:  assert len(response) == 39, f'Response is not the correct length for variant 2 - expected 39, got {len(response)}'

		print(f'Current replyVariant: {self.replyVariant}: response length:  {len(response)}')

		# get the format header
		if not (response[0] >> 4) in self.controlModeResponse.keys(): return
		self.controlMode = self.controlModeResponse[response[0] >> 4]

		# get the torque limitation status status
		limitStatus = response[-2]
		limitBitStatus = [(limitStatus >> (7 - i)) & 1 for i in range(self.numMotors)]

		# if np.any(limitBitStatus):
		# 	print(f'Limitation status: {limitBitStatus}')
		# 	raise Exception('Limitation status detected')

		process_forces_this_cycle = (self.force_processing_counter % self.loopRate == 0)
		self.force_processing_counter += 1

		# unpack the force sensor readings all at once, then store them properly
		# unpacking 12 bit values from 8 bit bytearray to store as 16 bit integers
		if self.replyVariant in [0, 1]:
			if process_forces_this_cycle:

				forceBytes = response[25:70]

				Ds = [0]*self.numForce
				
				for bitIdx in range(self.numForce * 12 - 4, -1, -4):
						dIdx = bitIdx // 12  # Calculate the index in the output list
						byteIdx = bitIdx // 8   # Calculate the byte index in the input array
						shiftVal = bitIdx % 8 # Calculate the shift value for bit extraction
						
						# Extract 4 bits, adjust them based on their position, and store in the output list
						Ds[dIdx] |= ((forceBytes[byteIdx] >> shiftVal) & 0x0F) << (bitIdx % 12)
			else:
				# Skip sensor processing and use cached values
				for finger in self.touchNames:
					for site in range(6):
						sensor_name = f'{finger}{site}_Force'
						self.sensors[sensor_name] = self.last_processed_forces[sensor_name]

			
		# unpack the bytes finger by finger
		for motor in range(self.numMotors):
			# apply the torque limitation status
			self.sensors[f'{self.sensorNames[motor]}_Limit'] = limitBitStatus[motor]

			# position
			posBytes = response[(4*motor) + 1:(4*motor) + 3]
			pDigital = int.from_bytes(posBytes, byteorder='little', signed=True)
			theta = pDigital/self.posConversion
			self.sensors[f'{self.sensorNames[motor]}_Pos'] = theta

			# velocity
			omega = -1
			if self.replyVariant in [1, 2]:
				velBytes = response[(4*motor) + 3:(4*motor) + 5] if self.replyVariant == 1 else response[(2*motor) + 25:(2*motor) + 27]
				velDigital = int.from_bytes(velBytes, byteorder='little', signed=True)
				omega = self.radToDeg(velDigital/self.rotorConversion/self.gearRatios[self.jointNames[motor]])

				self.sensors[f'{self.sensorNames[motor]}_Vel'] = omega

			# current
			amps = -1
			if self.replyVariant in [0, 2]:
				curBytes = response[(4*motor) + 3:(4*motor) + 5]
				curDigital = int.from_bytes(curBytes, byteorder='little', signed=True)
				amps = curDigital/self.currConversion

				self.sensors[f'{self.sensorNames[motor]}_Cur'] = amps

			# touch sensors - there are 6 sites for each of 5 fingers (the thumb would otherwise be double counted)
			# each site takes a byte and a half, so these need to be converted appropriately as 12 bit unsigned integers
			if self.replyVariant in [0, 1] and (motor < self.numMotors - 1) and process_forces_this_cycle:
				thisFingerTouch = Ds[motor*6:(motor + 1)*6]
				
				for site in range(6):
					# the below conversions are from the psyonic hand API
					D = thisFingerTouch[site] # raw FSR ADC value (already an integer from above)
					V = D*self.volConversion # voltage
					R = (self.resConversion/V) + self.resAdd if V > 0 else float('inf') # resistance
					F = (self.forceResConversion[0]/R) + self.forceResConversion[1] # force

					sensor_name = f'{self.touchNames[motor]}{site}_Force'
					processed_force = max(F - self.sensorForceOffsets[sensor_name], 0)

					# self.sensors[f'{self.touchNames[motor]}{site}_Force'] = max(F - self.sensorForceOffsets[f'{self.touchNames[motor]}{site}_Force'], 0)
					# self.sensors[f'{self.touchNames[motor]}{site}_Force'] = int(max([D - self.sensorForceOffsets[f'{self.touchNames[motor]}{site}_Force'], 0]))
					if self.sensorForceOffsets[sensor_name] != 0:
						# self.sensors[f'{self.touchNames[motor]}{site}_Force'] = int(self.filterForce.filterByIndex(self.sensors[f'{self.touchNames[motor]}{site}_Force'], motor*6 + site)[0])
						# self.sensors[f'{self.touchNames[motor]}{site}_Force'] = self.filterForce.filterByIndex(self.sensors[f'{self.touchNames[motor]}{site}_Force'], motor*6 + site)[0]
						processed_force = self.filterForce.filterByIndex(processed_force, motor*6 + site)[0]

						if D == 0:
							self.filterForce.resetFilterByIndex(motor*6 + site)

					
					self.sensors[sensor_name] = processed_force
					# Cache the processed value for next cycles
					self.last_processed_forces[sensor_name] = processed_force


	def sendToPlots(self):
		# send the commanded and actual arm position for plotting
		if self.initialized:
			if self.controlMode == 'position':
				sensorNames = [f'{name}_Pos' for name in self.sensorNames]
			elif self.controlMode == 'velocity' and self.replyVariant in [1, 2]:
				sensorNames = [f'{name}_Vel' for name in self.sensorNames]
			elif self.controlMode == 'torque' and self.replyVariant in [0, 2]:
				sensorNames = [f'{name}_Cur' for name in self.sensorNames]
			elif self.controlMode == 'voltage':
				sensorNames = [f'{name}_Pos' for name in self.sensorNames]
			else:
				sensorNames = [f'{name}_Pos' for name in self.sensorNames]

			s = self.sensors
			comSensor = [s.get(key) for key in sensorNames]
			forces = [s.get(key) for key in self.sensorForce]
			status = [s.get(key) for key in self.sensorLimit]

			newCom = np.append(self.handCom, comSensor)
			plotsPacked = [newCom, forces, status, self.controlMode, self.replyVariant]

			if self.handCom == [] or comSensor == []:
				raise Exception(f'Incorrect positions sent: {self.handCom} | {comSensor}')
			if forces == [] or status == []:
				raise Exception(f'Incorrect forces sent: {forces} | {status}')

			self.plotSock.send_pyobj(plotsPacked)

	####### COMMAND FUNCTIONS #######
	def calcChecksum(self, data):
		if not isinstance(data, list) or len(data) == 0: print(f'calcChecksum: not list - {type(data)}'); return
		
		return (-np.sum(data)) & 0xFF

	# Send Miscellanous Command to Ability Hand
	def createMiscCom(self, cmd):
		barr = [self.handAddr, cmd]
		chksum = self.calcChecksum(barr)
		barr.append(chksum)

		return barr

	# Generate Message to send to hand from array of positions (floating point)
	def generateTx(self, cmd):
		txBuf = [self.handAddr, self.replyModeHeader]

		# convert to integer
		if self.controlMode == 'position': convFactor = self.posConversion
		elif self.controlMode == 'velocity': convFactor = self.velConversion
		elif self.controlMode == 'torque': convFactor = self.velConversion
		elif self.controlMode == 'voltage': convFactor = self.velConversion
		else: convFactor = 0
		digitsData = [int(pos*convFactor) for pos in cmd]

		for i in range(self.numMotors):
			thisDigit = [digitsData[i] & 0xFF, (digitsData[i] >> 8) & 0xFF]
			txBuf += thisDigit

		# calculate checksum
		chksum = self.calcChecksum(txBuf)
		txBuf.append(chksum)

		return txBuf
	
	def setControlMode(self, mode):
		if mode == 'impedance':
			self.impedanceMode = True
			print('in impedance mode')
			mode = self.getCurControlMode()
			assert self.replyVariant in [0, 1], 'Impedance mode not supported for reply variant 2'
		else:
			self.impedanceMode = False

		assert mode in ['position', 'velocity', 'torque', 'voltage', 'readOnly'], f'Invalid control mode - {mode}'
		self.controlMode = mode
		self.replyModeHeader = self.controlModeHeaders[self.controlMode][self.replyVariant]
		self.replyChangedFlag = True

		# need to stop whatever movement is going on!
		if self.controlMode == 'position':
			self.handCom = self.getCurPos()
		elif self.controlMode == 'velocity':
			self.handCom = [0]*self.numMotors
		elif self.controlMode == 'torque':
			self.handCom = [0]*self.numMotors
		elif self.controlMode == 'voltage':
			self.handCom = [0]*self.numMotors

	def setReplyVariant(self, variant):
		assert variant in [0, 1, 2], f'Invalid reply variant {variant} ({type(variant)})'
		self.replyVariant = variant
		self.replyModeHeader = self.controlModeHeaders[self.controlMode][self.replyVariant]
		self.replyChangedFlag = True

	def isValidCommand(self, cmd):
		if not isinstance(cmd, list): print(f'not list - {type(cmd)}'); return False
		if len(cmd) != self.numMotors: print(f'not right number of commands - {len(cmd)}'); return False
		if not all([isinstance(val, (int, float)) for val in cmd]): print(f'not including ints, floats - ', [isinstance(val, (int, float)) for val in cmd]); return False

		if self.controlMode == 'position':
			for joint in self.jointNames:
				if not self.jointRoM[joint][0] <= cmd[self.jointNames.index(joint)] <= self.jointRoM[joint][1]: print(f'position out of range - {joint} (asking for {cmd[self.jointNames.index(joint)]})'); return False
		if self.controlMode == 'voltage':
			if not all([abs(val) <= self.voltLimit for val in cmd]): print('voltage out of range'); return False # the voltage commands are according to a PWM duty cycle

		return True

	####### CONTROL FUNCTIONS ######
	def initSensors(self):
		# this should send a command to the hand to get the initial reply back
		msg = self.createMiscCom(self.miscCommandHeaders['upsampleThumbEn'])
		msg = msg if not self.stuffing else self.byteStuff(msg)

		# self.ser.reset_input_buffer()
		bytesWritten = self.ser.write(msg)
		# self.ser.reset_output_buffer()

		assert bytesWritten == len(msg), f'initSensors(): Not all bytes sent - expected {len(msg)}, sent {bytesWritten}'

		# if you're byte stuffing, you can't just read a fixed data size, as the hand won't send it
		# response = bytearray()
		# # read the response
		# while len(response) < self.responseBufferSize(self.replyVariant):
		# 	response.extend(self.ser.read())
		response = self.ser.read(self.responseBufferSize(self.replyVariant))

		# if len(response) > 0: 
		self.unpackResponse(response)

		print('Zeroing force sensors...')
		self.zeroJoints()

		self.initialized = True

	def zeroJoints(self):
		# zero out the force sensors for the joints!
		curMode = self.controlMode
		curReply = self.replyVariant
		self.setControlMode('position') # don't move!
		self.setReplyVariant(0) # record force sensor readings

		# record on force sensors with no touch, then use the average as an offset
		# this is a hacky way to do it, but it works (supposedly)
		msg = self.generateTx(self.getCurPos())
		msg = msg if not self.stuffing else self.byteStuff(msg)

		start = time.time()
		forceSensorReadings = []
		
		# Store original counter and force processing during zeroing
		original_counter = self.force_processing_counter
		
		while (time.time() - start) < 2:
			# Force processing every cycle during zeroing
			self.force_processing_counter = 0  # This ensures forces are always processed
			
			# write the same message over and over again
			bytesWritten = self.ser.write(msg)
			assert bytesWritten == len(msg), f'zeroJoints(): Not all bytes sent - expected {len(msg)}, sent {bytesWritten}'

			response = self.ser.read(self.responseBufferSize(self.replyVariant))
			# if len(response) > 0:
			self.unpackResponse(response)

			# now save the force sensor readings
			if time.time() - start > 1:
				theseReadings = [self.sensors[site] for site in self.sensorForce]
				forceSensorReadings.append(theseReadings)
		
		# Restore original counter
		self.force_processing_counter = original_counter

		# now average the readings
		avgReadings = np.mean(forceSensorReadings, axis=0)
		self.sensorForceOffsets = dict(zip(self.sensorForce, avgReadings))

		per_finger_baselines = np.zeros(5)
		for finger_idx in range(5):
			start_idx = finger_idx * 6
			end_idx = start_idx + 6
			per_finger_baselines[finger_idx] = np.sum(avgReadings[start_idx:end_idx])
		
		# old:
		# Set per-finger offsets in the adaptive filter
		# if hasattr(self, 'adaptive_force_filter'):
		#     self.adaptive_force_filter.set_static_offsets(per_finger_baselines)
		#     print(f"Zero joints completed. Set {len(self.sensorForceOffsets)} sensor offsets.")
		#     print(f"Per-finger baselines: {per_finger_baselines}")
		# else:
		#     print("Warning: adaptive_force_filter not initialized")
		
		# new:
		# Set per-finger offsets in the simplified force processor
		if hasattr(self, 'simplified_force_processor'):
			self.simplified_force_processor.current_drift = per_finger_baselines.copy()
			print(f"Zero joints completed. Set {len(self.sensorForceOffsets)} sensor offsets.")
			print(f"Per-finger baselines: {per_finger_baselines}")
			print(f"SimplifiedForceProcessor baseline set: {self.simplified_force_processor.current_drift}")
		else:
			print("Warning: simplified_force_processor not initialized")

		# Force one more reading to test the zeroing
		self.force_processing_counter = 0  # Force processing
		bytesWritten = self.ser.write(msg)
		assert bytesWritten == len(msg), f'zeroJoints(): Not all bytes sent - expected {len(msg)}, sent {bytesWritten}'

		response = self.ser.read(self.responseBufferSize(self.replyVariant))
		# if len(response) > 0: 
		self.unpackResponse(response)
		
		# Test the zeroing effectiveness
		test_force_data = self.simplified_force_processor.process_runtime_force(
			self.sensors,
			self.touchNames,
			in_interaction_mode=False,  # Force free space mode
			mode_just_changed=True      # Force immediate zeroing
		)
		print(f"After zeroing - Total force: {test_force_data['total_force']:.3f}N")

		self.setControlMode(curMode)
		self.setReplyVariant(curReply)

	def get_all_raw_force_data(self):
		"""
		Get all 30 raw force sensor readings in the correct order
		Returns numpy array of shape (30,)
		"""
		raw_forces = np.zeros(self.numForce)
		
		for i, sensor_name in enumerate(self.sensorForce):
			if i < self.numForce:
				raw_forces[i] = self.sensors.get(sensor_name, 0.0)
		
		return raw_forces

	def sum_forces_per_finger(self, all_forces):
		"""
		Sum the 6 sensor readings per finger into 5 finger totals
		
		Args:
			all_forces: numpy array of shape (30,) with all sensor readings
		
		Returns:
			finger_forces: numpy array of shape (5,) with [index, middle, ring, pinky, thumb]
		"""
		finger_forces = np.zeros(5)
		
		# Each finger has 6 sensors (0-5, 6-11, 12-17, 18-23, 24-29)
		for finger_idx in range(5):
			start_idx = finger_idx * 6
			end_idx = start_idx + 6
			if end_idx <= len(all_forces):
				finger_forces[finger_idx] = np.sum(all_forces[start_idx:end_idx])
		
		return finger_forces

	def getCurControlMode(self):
		return self.controlMode
	
	def getCurReplyMode(self):
		return self.replyVariant

	def getCurPos(self):
		return [self.sensors[name] for name in self.sensorPos]
	
	def getCurVel(self):
		return [self.sensors[name] for name in self.sensorVel]
	
	def getCurCur(self):
		return [self.sensors[name] for name in self.sensorCur]
	
	def getDrivenAngles(self):
		return list(np.asarray(self.getCurPos())*self.fingerAngleFit[0] + self.fingerAngleFit[1])
	
	def calcJacobian(self, L, q):
		return np.array([L[0]*np.cos(q[0]) + L[1]*np.cos(q[1])*self.fingerAngleFit[0], -L[0]*np.sin(q[0]) - L[1]*np.sin(q[1])*self.fingerAngleFit[0]])
	
	def getAppliedTorque(self):
		# only force sensors 1 and 5 need to be considered for the digits, as the others are in the lateral direction
		# for the thumb, force sensors 1 and 5 are in the flexion direction and the others are in the rotation direction

		jointAngles = np.deg2rad(self.getCurPos())
		drivenAngles = np.deg2rad(self.getDrivenAngles())

		torques = np.zeros(self.numMotors)
		for joint in range(self.numMotors - 1):
			if joint < self.numMotors - 2:
				J5 = self.calcJacobian(self.fingerLinkLengths[self.touchNames[joint]], [jointAngles[joint], drivenAngles[joint]])
				J1 = self.calcJacobian([self.fingerLinkLengths[self.touchNames[joint]][0], self.fingerLinkLengths[self.touchNames[joint]][1]*0.8], [jointAngles[joint], drivenAngles[joint]])

				force1 = self.sensors[f'{self.touchNames[joint]}1_Force']/409.6
				force1_angle = np.pi - drivenAngles[joint] # this is perpendicular to the link

				force5 = self.sensors[f'{self.touchNames[joint]}5_Force']/409.6
				force5_angle = 5*np.pi/4 - drivenAngles[joint] # this is at a 45 degree angle to the link

				# get the forces component wise
				force1_comp = np.array([force1*np.cos(force1_angle), force1*np.sin(force1_angle)])
				force5_comp = np.array([force5*np.cos(force5_angle), force5*np.sin(force5_angle)])

				# get the torques
				thisTor = np.dot(J5, force5_comp) + np.dot(J1, force1_comp)
				thisTorMotor = thisTor*self.gearRatios[self.jointNames[joint]]
				torques[joint] = thisTorMotor

		return torques

	def mainControlLoop(self, emg=None, posDes=None, period=1):
		interpCount = 0; loopCount = 0
		T = time.time()
		startPos = np.asarray(self.getCurPos())
		thisLoopRate = self.loopRate*self.Hz
		self.movingEvent.clear(); self.isMoving = True

		# define the control parameters
		# impedance control
		int_cur = [0]*self.numMotors
		int_cur_max = 3000
		cur_err_prev = [0]*self.numMotors

		# position control
		int_f = [0]*self.numMotors
		int_f_max = 10000
		f_err_prev = [0]*self.numMotors

		try:
			while not self.movingEvent.is_set():
				# to run this loop at a consistent interval (LOOPRATEx faster than the neural net runs!)
				newT = time.time()
				time.sleep(max(1/thisLoopRate - (newT - T), 0))
				T = time.time()

				# move to specific position(s)
				if not posDes is None:
					if posDes.ndim > 1:
						if loopCount == posDes.shape[0]: break

						handCom = posDes[loopCount, :] # playback condition

					else:
						if loopCount > period*thisLoopRate: break

						handCom = loopCount/(period*thisLoopRate)*(posDes - startPos) + startPos # go to one position
						handCom = np.clip(handCom, [self.jointRoM[joint][0] for joint in self.jointNames], [self.jointRoM[joint][1] for joint in self.jointNames])

					loopCount += 1

				# EMG control
				elif emg is not None:
					# interpolate between outputs from the neural net model
					handCom = (self.NetCom - self.lastPosCom)/self.loopRate*interpCount + self.lastPosCom
					handCom = np.clip(handCom, [self.jointRoM[joint][0] for joint in self.jointNames], [self.jointRoM[joint][1] for joint in self.jointNames])

				# sinusoidal control
				else:
					handCom = [0]*self.numMotors
			
					if self.controlMode == 'position':
						index = max(0, self.sensors['index_Pos'])
						middle = max(0, self.sensors['middle_Pos'])
						ring = max(0, self.sensors['ring_Pos'])
						pinky = max(0, self.sensors['pinky_Pos'])
						thumbFlex = max(0, self.sensors['thumbFlex_Pos'])
						thumbRot = min(0, self.sensors['thumbRot_Pos'])

						index = self.genSinusoid(0.4, 'index', phase=0)
						middle = self.genSinusoid(0.4, 'middle', phase=1)
						ring = self.genSinusoid(0.4, 'ring', phase=2)
						pinky = self.genSinusoid(0.4, 'pinky', phase=3)
						# thumbFlex = self.genSinusoid(1, 'thumbFlex', phase=4)
						# thumbRot = self.genSinusoid(1, 'thumbRot', phase=5)

						# impedance control mode
						if self.impedanceMode:
							targetForce = 0
							targetJoints = [1, 1, 0, 0, 0, 0]
							targetPos = self.genSinusoid(5, 'index', phase=0, scale=0.5)
							# Kp_f = 0.5; Ki_f = 0.00; Kd_f = 0.1
							Kp_f = 10; Ki_f = 0.00; Kd_f = 2
							if self.replyVariant in [0, 1]:
								# fingerForce = np.asarray([self.sensors[f'{joint}5_Force'] + self.sensors[f'{joint}1_Force'] for joint in self.touchNames] + [self.sensors['thumb1_Force'] + self.sensors['thumb5_Force']]) # fingertip forces
								fingerForce = np.asarray([self.sensors[f'{joint}5_Force'] for joint in self.touchNames] + [self.sensors['thumb5_Force']]) # fingertip forces
								forceErr = targetForce - fingerForce
								posErr = targetPos - np.asarray(self.getCurPos())

								int_f += forceErr
								int_f = np.clip(int_f, -int_f_max, int_f_max) # anti-windup

								posUpdate = Kp_f*forceErr + Ki_f*int_f + Kd_f*(forceErr - f_err_prev)/thisLoopRate
								f_err_prev = forceErr

								comPos = [targetPos]*self.numMotors + np.multiply(posUpdate, targetJoints)
								comPos = np.clip(comPos, [self.jointRoM[joint][0] for joint in self.jointNames], [self.jointRoM[joint][1] for joint in self.jointNames])
								print(f'force: {fingerForce[0]:07.3f} | err: {forceErr[0]:07.3f} | posUpdate: {posUpdate[0]:07.3f} | indexDes: {comPos[0]:07.3f} | index: {index:07.3f}')
								
								[index, middle, ring, pinky, thumbFlex, thumbRot] = comPos

					elif self.controlMode == 'velocity':
						index = 0
						middle = 0
						ring = 0
						pinky = 0
						thumbFlex = 0
						thumbRot = 0

						index = self.genSinusoid(3, 'index', phase=0, scale=0.25)
						middle = self.genSinusoid(3, 'middle', phase=0, scale=0.25)
						ring = self.genSinusoid(3, 'ring', phase=0, scale=0.25)
						pinky = self.genSinusoid(3, 'pinky', phase=0, scale=0.25)
						thumbFlex = self.genSinusoid(3, 'thumbFlex', phase=0, scale=0.25)
						# thumbRot = self.genSinusoid(1, 'thumbRot', phase=5)

					elif self.controlMode == 'torque':
						index = 0
						middle = 0
						ring = 0
						pinky = 0
						thumbFlex = 0
						thumbRot = 0

						virtStiffness = 1
						virtDamping = 0.1
						targetPos = 30 + 15*np.sin(self.timestamp) # arbitrary target position
						curPos = np.asarray(self.getCurPos())
						curVel = np.asarray(self.getCurVel()) if self.replyVariant in [1, 2] else np.zeros(self.numMotors) # DANGEROUS - this depends on the control mode! We may not be updating the value regularly

						posError = targetPos - curPos
						velError = 0 - curVel

						torDes = virtStiffness*posError + virtDamping*velError
						curDes = torDes/self.torqueConstant

						index = curDes[0]
						middle = curDes[1]
						ring = curDes[2]
						pinky = curDes[3]
						thumbFlex = curDes[4]
						thumbRot = curDes[5]

					elif self.controlMode == 'voltage':
						index = 0
						middle = 0
						ring = 0
						pinky = 0
						thumbFlex = 0
						thumbRot = 0

						virtStiffness = 10
						virtDamping = 5
						torqueMul = 100
						targetPos = 30*(2 + np.sin(self.timestamp))*np.ones(self.numMotors) # arbitrary target position
						targetVel = 30*np.cos(self.timestamp)*np.ones(self.numMotors) # arbitrary target velocity
						targetPos[5] = 0 # thumb rotation
						curPos = np.asarray(self.getCurPos())
						curVel = np.asarray(self.getCurVel()) if self.replyVariant in [1, 2] else np.zeros(self.numMotors) # DANGEROUS - this depends on the control mode! We may not be updating the value regularly
						# curCur = np.asarray(self.getCurCur()) if self.replyVariant in [0, 2] else np.zeros(self.numMotors) # DANGEROUS - this depends on the control mode! We may not be updating the value regularly

						posError = targetPos - curPos
						velError = targetVel - curVel

						torDes = virtStiffness*posError + virtDamping*velError
						# print([f'{tor:08.2f}' for tor in torDes], [f'{pos:08.2f}' for pos in curPos], [f'{vel:08.2f}' for vel in curVel])

						# use the force sensors to modify the desired current?
						# get the torque from the force sensor readings
						# impedance control mode
						if self.impedanceMode:
							torquesApplied = self.getAppliedTorque()
							print([f'{tor:07.3f}' for tor in torquesApplied], ' | ', [f'{tor:07.3f}' for tor in torDes])
							torDes = torDes + torqueMul*torquesApplied
							# targetForce = 0
							# targetJoints = [1, 1, 1, 1, 0, 0]
							# Kp_f = 0.2; Ki_f = 0.0; Kd_f = 0.04
							# if self.replyVariant in [0, 1]:
							# 	fingerForce = np.asarray([self.sensors[f'{joint}5_Force'] for joint in self.touchNames] + [self.sensors['thumb5_Force']]) # fingertip forces
							# 	forceErr = targetForce - fingerForce

							# 	int_f += forceErr
							# 	int_f = np.clip(int_f, -int_f_max, int_f_max) # anti-windup

							# 	posUpdate = Kp_f*forceErr + Ki_f*int_f + Kd_f*(forceErr - f_err_prev)/thisLoopRate
							# 	f_err_prev = forceErr

							# 	comPos = [targetPos]*self.numMotors + np.multiply(posUpdate, targetJoints)
							# 	comPos = np.clip(comPos, [self.jointRoM[joint][0] for joint in self.jointNames], [self.jointRoM[joint][1] for joint in self.jointNames])
								
							# 	[index, middle, ring, pinky, thumbFlex, thumbRot] = comPos

						# close the current feedback loop now?
						# there is FOC control in the hand, so we don't need a feedback loop on the desired current
						# curErr = curDes - curCur
						# feedForward = np.zeros(self.numMotors) #self.torqueConstant*curVel

						# cur_Kp = 10; cur_Ki = 3; cur_Kd = 3

						# int_cur += curErr # integral error
						# int_cur = np.clip(int_cur, -int_cur_max, int_cur_max) # anti-windup

						# dam_cur = (curErr - cur_err_prev)/(1/self.loopRate) # derivative error
						# cur_err_prev = curErr

						# dutyCycle = (cur_Kp*curErr + cur_Ki*int_cur + cur_Kd*dam_cur + feedForward)/self.supplyVoltage
							
						curDes = torDes/self.torqueConstant
						dutyCycle = curDes/self.supplyVoltage
						dutyCycle = np.clip(dutyCycle, -self.voltLimit, self.voltLimit)

						index = dutyCycle[0]
						middle = dutyCycle[1]
						ring = dutyCycle[2]
						pinky = dutyCycle[3]
						thumbFlex = dutyCycle[4]
						thumbRot = dutyCycle[5]

					# [index, middle, ring, pinky, thumbFlex, thumbRot]
					handCom = [index, middle, ring, pinky, thumbFlex, thumbRot]						

				# self.printSensors()
				if not interpCount % self.loopRate:
					#### TAG PRINT
					# self.printSensors()
					if self.controlMode == 'position': print(f'{(time.time() - self.startTimestamp):07.3f}', [f'{com:07.3f}' for com in handCom]) # position
					# if self.controlMode == 'velocity': print(f'{(time.time() - self.startTimestamp):07.3f}', [f'{com:07.3f}' for com in handCom]) # velocity
					# if self.controlMode == 'torque': print(f'{(time.time() - self.startTimestamp):07.3f}', [f'{com:07.3f}' for com in torDes]) # torque
					# if self.controlMode == 'voltage': print(f'{(time.time() - self.startTimestamp):07.3f}', [f'{com:07.3f}' for com in curErr]) # voltage
					pass

				# self.handCom = self.getCurPos() # dont move arm
				handCom = list(handCom)
				if self.isValidCommand(handCom):
					self.handCom = handCom # don't move arm if the command is invalid

				self.timestamp = time.time()

				if self.recording: self.addLogEntry(emg)

				interpCount += 1; interpCount %= 4

				if not self.isMoving: break

		except KeyboardInterrupt:
			print('\nControl ended.')

		finally:
			self.isMoving = False
			self.movingEvent.set()

			# # need to stop whatever movement is going on!
			# if self.controlMode == 'position':
			# 	self.handCom = self.getCurPos()
			# elif self.controlMode == 'velocity':
			# 	self.handCom = [0]*self.numMotors
			# elif self.controlMode == 'torque':
			# 	self.handCom = [0]*self.numMotors
			# elif self.controlMode == 'voltage':
			# 	self.handCom = [0]*self.numMotors

	
	def get_processed_force_data(self, mode_just_changed=False):
		"""
		Get processed force data using simplified approach
		"""
		return self.simplified_force_processor.process_runtime_force(
			self.sensors, 
			self.touchNames, 
			self.in_interaction_mode,
			mode_just_changed=mode_just_changed
			)

	def update_interaction_mode(self, force_data, current_time):
		"""
		Handle mode switching with proper hysteresis and debouncing
		
		Args:
			force_data: dict from get_processed_force_data()
			current_time: current timestamp
			
		Returns:
			bool: True if mode changed
		"""
		total_force = force_data['total_force']
		mode_changed = False
		
		if not self.in_interaction_mode:
			# Try to enter interaction mode
			if total_force >= self.ENTER_THRESHOLD:
				if self.enter_timer is None:
					self.enter_timer = current_time
					print(f"Enter timer started: force={total_force:.3f}N")
				elif (current_time - self.enter_timer) >= self.ENTER_DEBOUNCE:
					# Transition to interaction mode
					self.in_interaction_mode = True
					self.enter_timer = None
					self.exit_timer = None
					self.blend_start_time = current_time
					mode_changed = True
					print(f"→ INTERACTION MODE (force: {total_force:.3f}N)")
			else:
				# Reset enter timer if force drops
				if self.enter_timer is not None:
					print(f"Enter timer reset: force={total_force:.3f}N")
				self.enter_timer = None
		
		else:
			# Try to exit interaction mode  
			if total_force <= self.EXIT_THRESHOLD:
				if self.exit_timer is None:
					self.exit_timer = current_time
					print(f"Exit timer started: force={total_force:.3f}N")
				elif (current_time - self.exit_timer) >= self.EXIT_DEBOUNCE:
					# Transition to free space mode
					self.in_interaction_mode = False
					self.exit_timer = None
					self.enter_timer = None
					self.blend_start_time = current_time
					mode_changed = True
					print(f"→ FREE SPACE MODE (force: {total_force:.3f}N)")
			else:
				# Reset exit timer if force rises
				if self.exit_timer is not None:
					print(f"Exit timer reset: force={total_force:.3f}N")
				self.exit_timer = None
		
		return mode_changed

	def blend_controller_commands(self, cmd_free, cmd_interaction, current_time):
		"""
		Blend between controller commands during transitions
		
		Args:
			cmd_free: free space controller command
			cmd_interaction: interaction controller command  
			current_time: current timestamp
			
		Returns:
			blended command
		"""
		if self.blend_start_time is None:
			# No active blend - use current mode
			return cmd_interaction if self.in_interaction_mode else cmd_free
		
		# Calculate blend factor
		blend_elapsed = current_time - self.blend_start_time
		if blend_elapsed >= self.blend_duration:
			# Blend complete
			self.blend_start_time = None
			return cmd_interaction if self.in_interaction_mode else cmd_free
		
		# Active blending
		alpha = np.clip(blend_elapsed / self.blend_duration, 0.0, 1.0)
		
		if self.in_interaction_mode:
			# Blending TO interaction mode
			cmd = (1 - alpha) * np.array(cmd_free) + alpha * np.array(cmd_interaction)
		else:
			# Blending TO free space mode  
			cmd = alpha * np.array(cmd_free) + (1 - alpha) * np.array(cmd_interaction)
		
		return cmd

	def reset_force_drift_baseline(self):
		"""
		Reset the drift baseline (call this when you want to recalibrate)
		"""
		for buffer in self.simplified_force_processor.drift_buffers:
			buffer.clear()
		self.simplified_force_processor.current_drift = np.zeros(5)
		print("Force drift baseline reset")

	# Run the neural net at self.Hz, allowing faster command interpolation to be sent to the arm
	def runNetForward(self, free_space_controller, interaction_controller):
		"""
		Simplified neural network control loop using the new force processor
		"""
		import time

		if self.replyVariant == 2:
			self.setReplyVariant(0)  # Force position mode for this controller
		
		# Initialize timing
		target_period = 1.0 / self.Hz  # 1/60 = 0.0167 seconds
		last_time = time.monotonic()
		
		print(f"Controller target: {self.Hz}Hz ({target_period*1000:.1f}ms)")
		print("Using simplified force processing approach")
		
		# Initialize state
		self.NetCom = np.array(self.getCurPos())
		previous_mode = False # Tracking previous mode
		
		# Cached commands for blending
		cached_fs_command = None
		cached_int_command = None
		
		# Debug counters
		loop_count = 0
		slow_loop_count = 0
		
		while not self.exitEvent.is_set():
			loop_start = time.monotonic()
			
			try:
				# 1. TIMING MANAGEMENT
				now = time.monotonic()
				dt = now - last_time
				dt = min(max(dt, 0.004), 0.020)  # Clamp to 50-250 Hz
				last_time = now
				
				# 2. Detecting mode changes
				mode_just_changed = (self.in_interaction_mode != previous_mode)
				if mode_just_changed:
					mode_str = "INTERACTION" if self.in_interaction_mode else "FREE_SPACE"
					print(f"Mode change: {mode_str}")

				# 3. Force data processing (with zeroing)
				force_data = self.simplified_force_processor.process_runtime_force(
					self.sensors,
					self.touchNames,
					self.in_interaction_mode,
					mode_just_changed=mode_just_changed
				)


				# 4. MODE SWITCHING
				if not mode_just_changed:
					mode_changed = self.update_interaction_mode(force_data, now)
				
				# 5. CONTROLLER EXECUTION
				try:
					if self.in_interaction_mode:
						# Use force data for interaction controller
						normalized_forces = force_data['normalized']
						cmd_interaction = interaction_controller.runModel(force_data=normalized_forces)
						cmd_free = cached_fs_command if cached_fs_command is not None else self.NetCom
					else:
						# Free space controller (EMG only) - forces are automatically 0
						cmd_free = free_space_controller.runModel()
						cmd_interaction = cached_int_command if cached_int_command is not None else self.NetCom
					
					# Cache commands for smooth transitions
					if self.in_interaction_mode:
						cached_int_command = cmd_interaction.copy() if hasattr(cmd_interaction, 'copy') else cmd_interaction
					else:
						cached_fs_command = cmd_free.copy() if hasattr(cmd_free, 'copy') else cmd_free
					
				except Exception as e:
					print(f"Controller error: {e}")
					# Safe fallback - hold current position
					cmd_free = self.NetCom
					cmd_interaction = self.NetCom
				
				# 6. COMMAND BLENDING
				blended_command = self.blend_controller_commands(cmd_free, cmd_interaction, now)
				
				# 6. SAFETY CHECKS AND CLAMPING
				posCom = np.asarray(blended_command, dtype=float)
				
				# Validate command
				if not np.isfinite(posCom).all():
					print("ERROR: Command contains NaN/inf - using safe fallback")
					posCom = self.NetCom  # Hold current position
				
				# Joint limit clamping
				mins = np.array([self.jointRoM[j][0] for j in self.jointNames], dtype=float)
				maxs = np.array([self.jointRoM[j][1] for j in self.jointNames], dtype=float)
				posCom = np.clip(posCom, mins, maxs)
				
				# Rate limiting 
				if hasattr(self, 'lastPosCom') and self.lastPosCom is not None:
					max_change_per_step = 2.0  # degrees per timestep (adjust as needed)
					change = posCom - self.lastPosCom
					change = np.clip(change, -max_change_per_step, max_change_per_step)
					posCom = self.lastPosCom + change
				
				# APPLY COMMAND FILTERING
				self.NetCom = np.asarray(self.lowpassCommands.filter(np.asarray([posCom]).T).T[0])
				self.lastPosCom = posCom

				# update mode tracking
				previous_mode = self.in_interaction_mode
				
				# DEBUG OUTPUT (every 2 seconds)
				loop_count += 1
				if loop_count % (2 * self.Hz) == 0:
					mode_str = "INTERACTION" if self.in_interaction_mode else "FREE_SPACE"
					total_force = force_data['total_force']
					drift_info = force_data['current_drift']
					print(f"Mode: {mode_str} | Force: {total_force:.3f}N | "
						f"Drift: [{drift_info[0]:.2f}, {drift_info[1]:.2f}, ...] | "
						f"Loops: {loop_count} | Slow: {slow_loop_count}")
				
				# TIMING CONTROL with monitoring
				processing_time = time.monotonic() - loop_start
				sleep_time = target_period - processing_time
				
				if sleep_time > 0:
					time.sleep(sleep_time)
				elif processing_time > target_period * 1.5:
					slow_loop_count += 1
					if slow_loop_count % 10 == 1:  # No spam
						print(f"SLOW LOOP: {processing_time*1000:.1f}ms > {target_period*1000:.1f}ms")
				
			except Exception as e:
				print(f"CRITICAL ERROR in control loop: {e}")
				# Emergency safe state
				self.NetCom = np.array(self.getCurPos())
				time.sleep(target_period)
			
			if self.exitEvent.is_set():
				break
		
		print("Neural network control loop exited")

# 	def runNetForward(self, free_space_controller, interaction_controller):
# 		T = time.time()
# 		self.NetCom = self.getCurPos()
# 
# 		ENTER_FORCE_THRESHOLD = 0.8
# 		EXIT_FORCE_THRESHOLD = 0.3
# 		INTERACTION_MODE_DEBOUNCE_TIME = 0.1  # seconds
# 
# 		self.in_interaction_mode = False
# 		exit_mode_start_time = None  # Track when force dropped below exit threshold
# 
# 		while not self.exitEvent.is_set():
# 			loop_start = time.time()
# 			# newT = time.time()
# 			target_period = 1.0 / self.Hz  # 1/60 = 0.0167 seconds
# 			sleep_time = target_period - (time.time() - loop_start)
# 			if sleep_time > 0:
# 				time.sleep(sleep_time)
# 			# time.sleep(max(1/(self.loopRate*self.Hz) - (newT - T), 0))
# 			T = time.time()
# 
# 			force_data = np.array([self.sensors[key] for key in self.sensorForce])
# 			max_finger_force = np.max(force_data)
# 			current_time = time.time()
# 
# 			if not self.in_interaction_mode:
# 				if max_finger_force > ENTER_FORCE_THRESHOLD:
# 					self.in_interaction_mode = True
# 					exit_mode_start_time = None
# 					print("Entered interaction mode")
# 			else:
# 				if max_finger_force < EXIT_FORCE_THRESHOLD:
# 					if exit_mode_start_time is None:
# 						exit_mode_start_time = current_time
# 					elif (current_time - exit_mode_start_time) > INTERACTION_MODE_DEBOUNCE_TIME:
# 						self.in_interaction_mode = False
# 						exit_mode_start_time = None
# 						print("Exited interaction mode")
# 				else:
# 					exit_mode_start_time = None  # Reset if force rises above exit threshold
# 
# 			self.lastposCom = self.NetCom
# 			if self.in_interaction_mode:
# 				posCom = interaction_controller.runModel()
# 			else:
# 				posCom = free_space_controller.runModel()
# 
# 			self.NetCom = np.asarray(self.lowpassCommands.filter(np.asarray([posCom]).T).T[0])
# 
# 			if self.exitEvent.is_set():
# 				break


	def runNetThread(self, free_space_controller, interaction_controller):
		self.netThread = threading.Thread(target=self.runNetForward, args=(free_space_controller, interaction_controller), name='runNetForward')
		self.netThread.daemon = True
		self.netThread.start()

	def manualPos(self, posDes):
		while True:
			print('Enter: [indexPos, middlePos, ringPos, pinkyPos, thumbFlex, thumbRot]')
			posRaw = input() # Take input
			if (posRaw == 'exit'): # Escape valve
				return

			try: # Turn input into array of floats
				posDes = [float(x) for x in posRaw.split()]
			except: # Uh-oh! Formatting wrong
				posDes = []

			if not self.isValidCommand(posDes): # repeat if not valid
				print('Input formatted incorrectly. Enter 6 valid joint positions.\n')
			else: # Otherwise exit
				break

		self.mainControlLoop(posDes=posDes)

		print('At desired position. Ending this movement.')

	def playbackRecording(self, loadedPositions):
		curPos = np.asarray(self.getCurPos())
		desStart = loadedPositions[0, :]
		toMove = (desStart - curPos).reshape(1, -1)

		period = 3
		rate = self.loopRate*self.Hz
		pts = np.arange(rate*period - 1).reshape(-1, 1)/(rate*period) # range along initial movement

		addlPos = toMove*pts + curPos # linear interpolation between current and start

		return np.append(addlPos, loadedPositions, axis=0)

	def genSinusoid(self, period, joint, phase=0, scale=1):
		if self.controlMode == 'position':
			rom = self.jointRoM[joint]
			return scale*(0.5*(rom[1] - rom[0])*(np.sin(2*np.pi*(self.startTimestamp - self.timestamp)/period + phase))) + 0.5*(rom[1] + rom[0])
			# this formulation looks bad, but it allows scaling the sinusoids about the midpoint of the range
		
		elif self.controlMode == 'velocity':
			rov = self.jointRoM[joint] # range of velocity
			return scale*(rov[1] - rov[0])*(np.sin(2*np.pi*(self.startTimestamp - self.timestamp)/period + phase))
	
		else:
			return 0
		
	def set_force_filtering_enabled(self, enabled):
		"""Enable/disable adaptive force filtering"""
		self.adaptive_force_filter.set_enabled(enabled)

	def get_force_filter_status(self):
		"""Get current filter status and debug info"""
		return self.adaptive_force_filter.get_debug_info()

	def reset_adaptive_baselines(self):
		"""Reset adaptive baselines (force recalibration)"""
		self.adaptive_force_filter.current_baselines = np.zeros(5)
		for buffer in self.adaptive_force_filter.baseline_buffers:
			buffer.clear()
		print("Adaptive baselines reset")


###################################################################
def callback():
	run = ''
	while run not in ['move', 'record', 'play', 'zero', 'manual', 'set', 'print', 'exit']:
		run = input("\nEnter 'move' to move the arm.\nEnter 'record' and then 'move' to record the arm's movement.\nEnter 'play' to replay recorded arm movements.\nEnter 'zero' to return the arm to joint positions of 0.\nEnter 'manual' to enter manual joint positions.\nEnter 'set' to change settings.\nEnter 'print' to pring sensor states.\nEnter 'exit' to quit:\n")

	return run

def saveThread(saveLocation, filename, data):
	os.mkdir(saveLocation) if not os.path.exists(saveLocation) else None
	dataToSave = np.array(data)
	# np.savetxt('/home/haptix/haptix/psyonic/logs/' + filename + '.csv', dataToSave, delimiter='\t', fmt='%s')
	np.savetxt(f'{saveLocation}/{filename}.csv', dataToSave, delimiter='\t', fmt='%s')

def main(arm, emg=None, saveLocation='', channels=None, free_space_controller=None, interaction_controller=None):
    """
    Updated main function to handle dual controllers
    """
    # Load calibration data
    calib_path = os.path.join('data', args.person_dir, 'recordings', 'Calibration', 'experiments', '1', 'scaling.yaml')
    with open(calib_path, 'r') as f:
        data = yaml.safe_load(f)
    noiseLevels = np.array(data['noiseLevels'], dtype=np.float32)
    maxVals = np.array(data['maxVals'], dtype=np.float32)

    if emg is not None:
        print('Connecting to EMG board...')
        emg = EMG(noiseLevel=noiseLevels, maxVals=maxVals, usedChannels=channels)

        print(f"Used channels: {emg.usedChannels}")
        print(f"YAML maxVals length: {len(maxVals)}")
        print(f"YAML noiseLevel length: {len(noiseLevels)}")
        print(f"EMG maxVals (first 8): {emg.maxVals[:8]}")
        print(f"EMG noiseLevel (first 8): {emg.noiseLevel[:8]}")

        emg.startCommunication()
        print('Connected.')

        # If controllers are provided, start the neural net thread
        if free_space_controller is not None and interaction_controller is not None:
            if not args.dummy:
                print('Starting dual controller neural net thread...')
                arm.runNetThread(free_space_controller, interaction_controller)

    # Set up case/switch for arm control
    try:
        while True:
            run = callback()
            if run == 'move':
                print(f'\n\nRunning arm...')
                if emg is not None:
                    arm.mainControlLoop(emg=emg)
                else:
                    arm.mainControlLoop()

                # Set recording to false, regardless of whether you have been recording
                if arm.recording:
                    filename = input('Enter a .csv filename: ')
                    if not filename == 'exit':
                        thread = threading.Thread(target=saveThread, args=[saveLocation, filename, arm.recordedData], name='saveThread')
                        thread.start()
                        arm.resetRecording()

                arm.recording = False

            elif run == 'record':
                print('\n\nRecording next arm movement...')
                arm.recording = True
                arm.resetRecording()

            elif run == 'play':
                validFilename = False
                while not validFilename:
                    filename = input('Enter a .csv filename to replay: ')
                    try:
                        loadedData = pd.read_csv(filename, delimiter='\t', header=0)
                        if filename == "exit": break
                        validFilename = True
                    except Exception as e:
                        print(f'Invalid filename with error {e}\n')

                if validFilename:
                    positionColTitles = ['index_PosCom', 'middle_PosCom', 'ring_PosCom', 'pinky_PosCom', 'thumbFlex_PosCom', 'thumbRot_PosCom']
                    loadedPositions = loadedData[positionColTitles].values
                    fullMove = arm.playbackRecording(loadedPositions)
                    arm.mainControlLoop(posDes=fullMove)

            elif run == 'zero':
                print('\n\nZeroing joints...')
                posDes = [0]*arm.numMotors
                curMode = arm.getCurControlMode()
                arm.setControlMode('position')
                arm.mainControlLoop(posDes=np.asarray(posDes))
                arm.setControlMode(curMode)

            elif run == 'manual':
                print('\n\nAccepting manual input...')
                while True:
                    print('Enter: [indexPos, middlePos, ringPos, pinkyPos, thumbFlex, thumbRot]')
                    posRaw = input()
                    if (posRaw == 'exit'):
                        break

                    try:
                        posDes = [float(x) for x in posRaw.split()]
                    except:
                        posDes = []

                    if not arm.isValidCommand(posDes):
                        print('Input formatted incorrectly. Enter 6 valid joint positions.\n')
                    else:
                        break

                arm.mainControlLoop(posDes=np.asarray(posDes))

            elif run == 'set':
                print('\n\nChanging settings...')
                while True:
                    print('Enter: [c (controlMode), r (replyVariant)]')
                    setRaw = input()
                    if (setRaw == 'exit'):
                        break

                    if setRaw in ['c', 'controlMode', 'control']:
                        while True:
                            print('Enter: [pos, vel, tor, vol, imp, read]')
                            setControl = input()
                            if setControl not in ['pos', 'vel', 'tor', 'vol', 'imp', 'read']:
                                print('Invalid setting. Enter a valid setting.\n')
                                continue
                            break

                        controlDict = {'pos': 'position', 'vel': 'velocity', 'tor': 'torque', 'vol': 'voltage', 'imp': 'impedance', 'read': 'readOnly'}
                        setControl = controlDict[setControl]
                        arm.setControlMode(setControl)
                        break

                    elif setRaw in ['r', 'replyVariant', 'reply']:
                        while True:
                            print('Enter: [0, 1, 2]')
                            setReply = input()
                            if setReply not in ['0', '1', '2']:
                                print('Invalid setting. Enter a valid setting.\n')
                                continue
                            break

                        arm.setReplyVariant(int(setReply))
                        break

                    else:
                        print('Invalid setting. Enter a valid setting.\n')
                        continue

            elif run == 'print':
                print('\n\nPrinting sensor states...')
                arm.printSensors()

            elif run == 'exit':
                print('Exiting.')
                break

            else:
                print(f'Invalid command {run}')

    except KeyboardInterrupt:
        pass

    print('Shutting Down.')
    if emg is not None:
        emg.exitEvent.set()
        if hasattr(arm, 'netThread') and arm.netThread.is_alive():
            arm.netThread.join()

    if hasattr(arm, 'movingEvent'):
        arm.movingEvent.set()


# Check the args and run
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Psyonic Ability Hand Command Line Interface')
    parser.add_argument('-t', '--tracker', help='Use MediaPipe hand tracker?', action='store_true')
    parser.add_argument('-e', '--emg', help='Use EMG control?', action='store_true')
    parser.add_argument('-l', '--laterality', type=str, help='Handedness (left or right)', default='left')
    parser.add_argument('-s', '--stuffing', help='Use byte stuffing?', action='store_true')
    parser.add_argument('--person_dir', type=str, required=True, help='Person directory')
    parser.add_argument('--free_space_model_name', type=str, required=True, help='Filename of free space model')
    parser.add_argument('--interaction_model_name', type=str, required=True, help='Filename of interaction model')
    parser.add_argument('--dummy', action='store_true', help='Run in dummy mode without connecting to real hand')

    args = parser.parse_args()

    # Construct full model paths
    fs_model_path = os.path.join('data', args.person_dir, 'models', args.free_space_model_name)
    inter_model_path = os.path.join('data', args.person_dir, 'models', args.interaction_model_name)

    emg = None

    if not args.dummy:
        print("Connecting to Psyonic Hand")
        arm = psyonicArm(hand=args.laterality, stuffing=args.stuffing, usingEMG=args.emg)
    else:
        print("Running in Dummy Mode")
        arm = None

    # EMG Setup
    if args.emg:
        # Load both configs to get features
        free_space_config_path = join('data', args.person_dir, 'configs', 'modular_fs.yaml')
        interaction_config_path = join('data', args.person_dir, 'configs', 'modular_inter.yaml')
        
        # Verify both config files exist
        if not os.path.exists(free_space_config_path):
            raise FileNotFoundError(f"Free space config not found: {free_space_config_path}")
        if not os.path.exists(interaction_config_path):
            raise FileNotFoundError(f"Interaction config not found: {interaction_config_path}")
        
        # Load free space config to extract EMG channels
        with open(free_space_config_path, 'r') as file:
            fs_wandb_config = yaml.safe_load(file)
            fs_config = Config(fs_wandb_config)
        
        # Load interaction config
        with open(interaction_config_path, 'r') as file:
            inter_wandb_config = yaml.safe_load(file)
            inter_config = Config(inter_wandb_config)
        
        # Extract only EMG channels (filter out force channels if present)
        channels = [int(feature[1]) for feature in fs_config.features if feature[0] == 'emg']

        print(f"Free space config - Features: {len(fs_config.features)} (EMG only)")
        print(f"Interaction config - Features: {len(inter_config.features)} (EMG + Force)")
        print(f"Output targets: {len(fs_config.targets)} DOF")
        print(f"EMG channels: {channels}")

        # Load calibration values
        calib_path = os.path.join('data', args.person_dir, 'recordings', 'Calibration', 'experiments', '1', 'scaling.yaml')
        with open(calib_path, 'r') as f:
            data = yaml.safe_load(f)
        noiseLevels = np.array(data['noiseLevels'], dtype=np.float32)
        maxVals = np.array(data['maxVals'], dtype=np.float32)
        
        emg = EMG(usedChannels=channels, noiseLevel=noiseLevels, maxVals=maxVals)

        # Measure actual stream rate
        print("Measuring EMG hardware rate...")
        actual_rate = emg.measure_actual_stream_rate(duration=5)
        print(f"Measured hardware rate: {actual_rate:.1f} Hz")
       		
        emg.samplingFreq = actual_rate
        print(f"Updated EMG sampling frequency to measured rate: {actual_rate} Hz")
        
        emg.startCommunication()
        print(f"EMG thread running: {emg.emgThread.is_alive()}")
        print(f"EMG numPackets: {emg.numPackets}")
        print(f"Expected output rate: {emg.samplingFreq / emg.numPackets} Hz")
        
        # Create both controllers with their respective configs
        free_space_controller, interaction_controller = create_controllers(
            emg=emg,
            arm=arm,
            config_free_space=fs_config,
            config_interaction=inter_config,
            free_space_model_path=fs_model_path,
            interaction_model_path=inter_model_path
        )

        if not args.dummy:
            print('Initializing sensor readings...')
            arm.initSensors()
            print('Sensors initialized.')
            arm.startComms()

        # Call main with both controllers
        main(arm, emg, saveLocation=f'data/{args.person_dir}/logs', channels=channels,
             free_space_controller=free_space_controller, interaction_controller=interaction_controller)

    elif args.tracker:
        if args.dummy:
            raise ValueError("Tracker mode requires real hardware connection. Cannot run in dummy mode.")

        trackerAddr = 'tcp://127.0.0.1:1239'
        print('Starting Psyonic Hand (tracker control)...')

        ctx = zmq.Context()
        trackerSock = ctx.socket(zmq.SUB)
        trackerSock.connect(trackerAddr)
        trackerSock.subscribe('')

        try:
            while True:
                arm.handCom = trackerSock.recv_pyobj()
        except KeyboardInterrupt:
            arm.stopEvent.set()
            sys.exit()

        print('Initializing sensor readings...')
        arm.initSensors()
        print('Sensors initialized.')
        arm.startComms()
        main(arm, emg, saveLocation=f'data/{args.person_dir}/logs')

    else:
        raise ValueError("Please specify either --emg or --tracker mode.")