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
from helpers.predict_utils import Config
import numpy as np

class psyonicArm():
	def __init__(self, hand='right', usingEMG=False, stuffing=False, baud=460800, plotSocketAddr='tcp://127.0.0.1:1240'):
		self.baud = baud
		self.hand = hand
		self.stuffing = stuffing
		self.usingEMG = usingEMG

		self.initialized = False
		self.impedanceMode = False

		# initialize hand state
		# define control mode format headers
		self.replyVariant = 2 # one of [0, 1, 2], which corresponds to index in the lists below
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
		self.forceResConversion = [121591.0, 0.878894] # These are not accurate
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
		self.serialSet = self.setupSerial(passedPort='/dev/psyonicHand')
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

		# neural net control loop rate
		self.Hz = 60
		self.loopRate = 10 # this is how much faster this should run than the neural net

		# lowpass filter joint commands
		self.lowpassCommands = BesselFilterArr(numChannels=self.numMotors, order=4, critFreqs=[1], fs=self.Hz, filtType='lowpass')
		# self.lowpassCommands = ExponentialFilterArr(numChannels=self.numMotors, smoothingFactor=0.9, defaultValue=0)

		# exponential filter the force sensor readings
		self.filterForce = ExponentialFilterArr(numChannels=self.numForce, smoothingFactor=0.8, defaultValue=0) # todo


		# store prior commands for some reason
		self.lastPosCom = -1*np.ones(self.numMotors)
		self.lastVelCom = -1*np.ones(self.numMotors)
		self.lastTorCom = -1*np.ones(self.numMotors)
		self.lastVolCom = -1*np.ones(self.numMotors)

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
		# if passedPort is not None:
		# 	self.ser = serial.Serial(str(Path(passedPort).resolve()), self.baud, timeout=0.02, write_timeout=0.02)
		# 	assert self.ser.is_open, 'Failed to open serial port'
		# 	print(f'Connected to port {self.ser.name}')
		# 	return True

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
			# print('response length 0')
			return # sometimes this happens idk

		# validate the checksum
		if not self.calcChecksum(response[:-1]) == response[-1]:  return
		assert self.calcChecksum(response[:-1]) == response[-1], f'Checksum failed - expected {hex(self.calcChecksum(response[:-1]))}, got {hex(response[-1])}'

		# make sure the response is the right length
		if self.replyVariant in [0, 1] and not len(response) == 72: return # assert len(response) == 72, f'Response is not the correct length for variant {self.replyVariant} - expected 72, got {len(response)}'
		elif self.replyVariant == 2 and not len(response) == 29: return # assert len(response) == 39, f'Response is not the correct length for variant 2 - expected 39, got {len(response)}'

		# get the format header
		if not (response[0] >> 4) in self.controlModeResponse.keys(): return
		self.controlMode = self.controlModeResponse[response[0] >> 4]

		# get the torque limitation status status
		limitStatus = response[-2]
		limitBitStatus = [(limitStatus >> (7 - i)) & 1 for i in range(self.numMotors)]

		# if np.any(limitBitStatus):
		# 	print(f'Limitation status: {limitBitStatus}')
		# 	raise Exception('Limitation status detected')

		# unpack the force sensor readings all at once, then store them properly
		# unpacking 12 bit values from 8 bit bytearray to store as 16 bit integers
		if self.replyVariant in [0, 1]:
			forceBytes = response[25:70]

			Ds = [0]*self.numForce
			
			for bitIdx in range(self.numForce * 12 - 4, -1, -4):
					dIdx = bitIdx // 12  # Calculate the index in the output list
					byteIdx = bitIdx // 8   # Calculate the byte index in the input array
					shiftVal = bitIdx % 8 # Calculate the shift value for bit extraction
					
					# Extract 4 bits, adjust them based on their position, and store in the output list
					Ds[dIdx] |= ((forceBytes[byteIdx] >> shiftVal) & 0x0F) << (bitIdx % 12)
			
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
			if self.replyVariant in [0, 1] and (motor < self.numMotors - 1):
				thisFingerTouch = Ds[motor*6:(motor + 1)*6]
				
				for site in range(6):
					# the below conversions are from the psyonic hand API
					D = thisFingerTouch[site] # raw FSR ADC value (already an integer from above)
					V = D*self.volConversion # voltage
					R = (self.resConversion/V) + self.resAdd if V > 0 else float('inf') # resistance
					F = (self.forceResConversion[0]/R) + self.forceResConversion[1] # force
					self.sensors[f'{self.touchNames[motor]}{site}_Force'] = max(F - self.sensorForceOffsets[f'{self.touchNames[motor]}{site}_Force'], 0)
					# self.sensors[f'{self.touchNames[motor]}{site}_Force'] = int(max([D - self.sensorForceOffsets[f'{self.touchNames[motor]}{site}_Force'], 0]))
					if self.sensorForceOffsets[f'{self.touchNames[motor]}{site}_Force'] != 0:
						# self.sensors[f'{self.touchNames[motor]}{site}_Force'] = int(self.filterForce.filterByIndex(self.sensors[f'{self.touchNames[motor]}{site}_Force'], motor*6 + site)[0])
						self.sensors[f'{self.touchNames[motor]}{site}_Force'] = self.filterForce.filterByIndex(self.sensors[f'{self.touchNames[motor]}{site}_Force'], motor*6 + site)[0]

						if D == 0:
							self.filterForce.resetFilterByIndex(motor*6 + site)

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
		while (time.time() - start) < 2:
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

		# now average the readings
		avgReadings = np.mean(forceSensorReadings, axis=0)
		self.sensorForceOffsets = dict(zip(self.sensorForce, avgReadings))

		bytesWritten = self.ser.write(msg)
		assert bytesWritten == len(msg), f'zeroJoints(): Not all bytes sent - expected {len(msg)}, sent {bytesWritten}'

		response = self.ser.read(self.responseBufferSize(self.replyVariant))
		# if len(response) > 0: 
		self.unpackResponse(response)

		self.setControlMode(curMode)
		self.setReplyVariant(curReply)

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
					handCom = (self.NetCom - self.lastposCom)/self.loopRate*interpCount + self.lastposCom
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

	# Run the neural net at self.Hz, allowing faster command interpolation to be sent to the arm
	def runNetForward(self, controller):
		T = time.time()
		self.NetCom = self.getCurPos()
		# controller.resetModel() # todo biophysical model
		while not self.exitEvent.is_set():
			newT = time.time()
			time.sleep(max(1/(self.loopRate*self.Hz) - (newT - T), 0))
			T = time.time()

			self.lastposCom = self.NetCom
			# posCom = controller.forwardDynamics() # todo biophysical model
			posCom = controller.runModel()
			self.NetCom = np.asarray(self.lowpassCommands.filter(np.asarray([posCom]).T).T[0])

			if self.exitEvent.is_set():
				break

	def runNetThread(self, controller):
		self.netThread = threading.Thread(target=self.runNetForward, args=[controller], name='runNetForward')
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


###################################################################
def callback():
	run = ''
	while run not in ['move', 'record', 'play', 'zero', 'manual', 'set', 'print', 'exit']:
		run = input("\nEnter 'move' to move the arm.\nEnter 'record' and then 'move' to record the arm's movement.\nEnter 'play' to replay recorded arm movements.\nEnter 'zero' to return the arm to joint positions of 0.\nEnter 'manual' to enter manual joint positions.\nEnter 'set' to change settings.\nEnter 'print' to pring sensor states.\nEnter 'exit' to quit:\n")

	return run

def saveThread(filename, data):
	dataToSave = np.array(data)
	np.savetxt('/home/haptix/haptix/psyonic/logs/' + filename + '.csv', dataToSave, delimiter='\t', fmt='%s')

def main(arm, emg=None):
	# connect to EMG board
	if emg is not None:
		print('Connecting to EMG board...')
		emg = EMG(usedChannels=[0, 1, 2, 4, 10, 11, 12, 13])
		emg.startCommunication()
		print('Connected.')

		# setup the controller class
		# controller = psyonicControllers(numMotors=arm.numMotors, arm=arm, freq_n=3, emg=emg)

		# start the neural net thread
		# arm.runNetThread(controller)

	# set up case/switch for arm using an input/callback structure -- this allows the behavior of the arm to be controlled without stopping this code
	try:
		while True:
			run = callback()
			if run == 'move':
				print(f'\n\nRunning arm...')
				if emg is not None:
					arm.mainControlLoop(emg=emg)
				else:
					arm.mainControlLoop()

				# set recording to false, regardless of whether you have been recording
				if arm.recording:
					filename = input('Enter a .csv filename: ')
					if not filename == 'exit':
						# dataToSave = np.array(arm.recordedData)
						# np.savetxt('/home/haptix/haptix/psyonic/logs/' + filename + '.csv', dataToSave, delimiter='\t', fmt='%s')
						thread = threading.Thread(target=saveThread, args=[filename, arm.recordedData], name='saveThread')
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
					posRaw = input() # Take input
					if (posRaw == 'exit'): # Escape valve
						return

					try: # Turn input into array of floats
						posDes = [float(x) for x in posRaw.split()]
					except: # Uh-oh! Formatting wrong
						posDes = []

					if not arm.isValidCommand(posDes): # repeat if not valid
						print('Input formatted incorrectly. Enter 6 valid joint positions.\n')
					else: # Otherwise exit
						break

				# curMode = arm.getCurControlMode()
				# arm.setControlMode('position')
				arm.mainControlLoop(posDes=np.asarray(posDes))
				# arm.setControlMode(curMode)

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
		arm.netThread.join()

	arm.movingEvent.set()

# Check the args and run
if __name__ == '__main__':
	# Define all arguments
	parser = argparse.ArgumentParser(description='Psyonic Ability Hand Command Line Interface')
	parser.add_argument('-t', '--tracker', help='Using the MediaPipe hand tracker?', action='store_true')
	parser.add_argument('-e', '--emg', help='Using EMG control?', action='store_true')
	parser.add_argument('-l', '--laterality', type=str, help='Handedness', default='left')
	parser.add_argument('-s', '--stuffing', help='Using byte stuffing?', action='store_true')
	parser.add_argument('--person_dir', type=str, required=True, help='Person directory')
	parser.add_argument('--config_name', type=str, required=True, help='Training configuration')
	parser.add_argument('--model_path', type=str, required=True, help='Model path')

	args = parser.parse_args()

	emg = None

	# instantiate arm class
	arm = psyonicArm(hand=args.laterality, stuffing=args.stuffing, usingEMG=args.emg)
	strInsert = ', byte stuffing' if args.stuffing else ', no byte stuffing'

	if args.emg:
		with open(join('data', args.person_dir, 'configs', args.config_name), 'r') as file:
			wandb_config = yaml.safe_load(file)
			config = Config(wandb_config)
		channels = [int(feature[1]) for feature in config.features]

		emg = EMG(usedChannels=channels)
		emg.startCommunication()
		print(f'Starting Psyonic Hand (EMG control{strInsert})...')

		# model_name = args.config_name.split('.')[0]
		# model_path = join('data', args.person_dir, 'models', f'{model_name}.pt')
		model_path = args.model_path

		controller = psyonicControllers(numMotors=arm.numMotors, arm=arm, freq_n=3, emg=emg, config=config, model_path=model_path)
		arm.runNetThread(controller)

	elif args.tracker:
		trackerAddr = 'tcp://127.0.0.1:1239'

		print(f'Starting Psyonic Hand (tracker control{strInsert})...')

		ctx = zmq.Context()
		trackerSock = ctx.socket(zmq.SUB)
		trackerSock.connect(trackerAddr)
		trackerSock.subscribe('') # Subscribe to all topics

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
	main(arm, emg)