# Mikey Fernandez 07/29/2022
# Make a GUI for the LUKE arm to plot the current and commanded positions

import os
os.environ['PYQTGRAPH_QT_LIB'] = 'PyQt5'
from PyQt5 import QtCore, QtWidgets, QtGui
import numpy as np
import pandas as pd
import sys
import zmq
import threading
from psyonicHand import psyonicArm
from EMGClass import EMG
# from psyonicControllers import psyonicControllers
from GUI.comPlotter import comPlotter
from GUI.emgPlotter import emgPlotter
from GUI.forcePlotter import forcePlotter
from GUI.manualPos import manualWindow
from GUI.settingsWindow import settingsWindow
import argparse
import time

class Psyonic_GUI(QtWidgets.QWidget):
	def __init__(self, arm=None, EMG=None, socketAddr='tcp://127.0.0.1:1240'):
		super(Psyonic_GUI, self).__init__()
		self.socketAddr = socketAddr

		self.title = "Psyonic Arm Controller"
		self.arm = arm
		self.emg = EMG
		self.numCom = self.arm.numMotors
		self.controlMode = self.arm.controlMode

		self.plotWidth = 600
		self.timer = QtCore.QTimer()
		self.timer.timeout.connect(self.updateGUI)
		self.plotBufs = np.zeros((2*self.numCom, self.plotWidth))
		self.sensorForces = None
		self.jointStatus = None

		self.initUI()

		# for communication with the Psyonic arm sender use a zmq socket
		self.socketAddr = socketAddr
		self.ctx = zmq.Context()
		self.sock = self.ctx.socket(zmq.SUB)
		self.sock.connect(self.socketAddr)
		self.sock.subscribe("") # Subscribe to all topics

		# for setting Psyonic arm status
		self.started = False
		self.firstMessage = True

		# for keyboard control
		self.keyboardCtrl = False
		self.jointCom = np.zeros(self.numCom)
		self.jointNames = ['index', 'middle', 'ring', 'pinky', 'thumbFlex', 'thumbRot']
		self.jointCtrld = 0 # will use modular arithmetic for this
		self.scale = 0.01
		self.setRoMs()

		self.startReceiving()

	def __del__(self):
		""" Garbage collection """
		try:
			self.sock.unbind(self.socketAddr)
			self.sock.close()
			self.ctx.term()

		except Exception as e:
			print(f'__del__: Socket closing error {e}')

	def initUI(self):
		self.setWindowTitle(self.title)
		self.ax = 100; self.ay = 100; self.aw = 1024; self.ah = 1024
		self.setGeometry(self.ax, self.ay, self.aw, self.ah)

		hbox = QtWidgets.QHBoxLayout()
		self.setLayout(hbox)

		# Add the plotter
		self.comPlotterWidget = comPlotter(channelNum=self.numCom, updateRate=self.arm.Hz, controlMode=self.controlMode, plotWidth=self.plotWidth)
		hbox.addWidget(self.comPlotterWidget)

		# Add the button panel
		self.button_panel = QtWidgets.QWidget(self)
		self.button_panel_layout = QtWidgets.QVBoxLayout()
		self.button_panel.setLayout(self.button_panel_layout)
		hbox.addWidget(self.button_panel)

		#Add all the buttons
		# ['move', 'record', 'play', 'zero', 'manual', 'set', 'print', 'exit']
		self.move_btn = QtWidgets.QPushButton("Move")
		self.record_btn = QtWidgets.QPushButton("Record")
		self.playback_btn = QtWidgets.QPushButton("Playback")
		self.zero_btn = QtWidgets.QPushButton("Zero")
		self.manual_btn = QtWidgets.QPushButton("Manual")
		self.settings_btn = QtWidgets.QPushButton("Settings")
		self.keyboard_btn = QtWidgets.QPushButton("Keyboard")
		self.emg_btn = QtWidgets.QPushButton("EMG Plotter")
		self.force_btn = QtWidgets.QPushButton("Force Plotter")

		self.button_panel_layout.addWidget(self.move_btn)
		self.button_panel_layout.addWidget(self.record_btn)
		self.button_panel_layout.addWidget(self.playback_btn)
		self.button_panel_layout.addWidget(self.zero_btn)
		self.button_panel_layout.addWidget(self.manual_btn)
		self.button_panel_layout.addWidget(self.settings_btn)
		self.button_panel_layout.addWidget(self.keyboard_btn)
		self.button_panel_layout.addWidget(self.emg_btn)
		self.button_panel_layout.addWidget(self.force_btn)

		if self.emg is None:
			self.emg_btn.setDisabled(True)

		self.keyboard_btn.setCheckable(True)

		# Connect all button functions
		self.linkBtnFunctions()

		self.show()

	def linkBtnFunctions(self):
		self.record_btn.clicked.connect(self.onRecord)
		self.move_btn.clicked.connect(self.onMove)
		self.playback_btn.clicked.connect(self.onPlayback)
		self.zero_btn.clicked.connect(self.onZero)
		self.manual_btn.clicked.connect(self.onManual)
		self.keyboard_btn.clicked.connect(self.onKeyboard)
		self.settings_btn.clicked.connect(self.onSet)
		self.emg_btn.clicked.connect(self.onEMG)
		self.force_btn.clicked.connect(self.onForce)

	def dataReadyCallback(self):
		while not self.arm.exitEvent.is_set():
			if self.arm.exitEvent.is_set():
				break

			plotsPacked = self.sock.recv_pyobj(flags=0)
			# this should contain the commanded and actual commands, as well as the command mode!
			# print(plotsPacked[0])

			newCom = plotsPacked[0]
			self.sensorForces = plotsPacked[1]
			self.jointStatus = plotsPacked[2]
			self.controlMode = plotsPacked[3]
			self.replyVariant = plotsPacked[4]

			self.plotBufs = np.roll(self.plotBufs, 1, axis=1)
			self.plotBufs[:, 0] = newCom
			self.jointCom = newCom[:self.numCom]

			if self.firstMessage:
				self.firstMessage = False
				self.started = True

	def start(self):
		self.timer.start(30) # how many milliseconds between screen updates

	def updateGUI(self):
		if self.started:
			self.comPlotterWidget.updateCurves(self.plotBufs, self.controlMode, self.jointStatus)

			try:
				if self.replyVariant in [0, 1]:
					self.forceWindow.updateForcePlots(self.sensorForces)
			except:
				pass

		# periodically check the arm flag
		if self.arm.movingEvent.is_set():
			self.enableButtons()

	def startReceiving(self):
		self.receiveThread = threading.Thread(target=self.dataReadyCallback, name='dataReadyCallback')
		self.receiveThread.daemon = True
		self.receiveThread.start()

	def armThreader(self, emg=None, posDes=None, period=1.5):
		self.moveThread = threading.Thread(target=self.arm.mainControlLoop, args=[emg, posDes, period], name='controlThread')
		self.moveThread.daemon = True
		self.moveThread.start()

	# key press events
	def keyPressEvent(self, e):
		if e.key() == QtCore.Qt.Key.Key_Q:
			self.arm.movingEvent.set()
			self.arm.exitEvent.set()
			self.close()
			return
		
		if e.key() == QtCore.Qt.Key.Key_S:
			self.onSet()

		if self.started:
			if e.key() == QtCore.Qt.Key.Key_M:
				self.onMove()

			if e.key() == QtCore.Qt.Key.Key_R:
				self.onRecord()

			if e.key() == QtCore.Qt.Key.Key_D:
				self.onMiddle()

			if e.key() == QtCore.Qt.Key.Key_K:
				self.keyboard_btn.toggle()
				self.onKeyboard()

			if not self.arm.isMoving:
				if e.key() == QtCore.Qt.Key.Key_Z:
					self.onZero()

			if self.keyboardCtrl:
				if e.key() == QtCore.Qt.Key.Key_8:
					self.incrementPosition(True)
				elif e.key() == QtCore.Qt.Key.Key_2:
					self.incrementPosition(False)
				elif e.key() == QtCore.Qt.Key.Key_4:
					self.changeJoint(False)
				elif e.key() == QtCore.Qt.Key.Key_6:
					self.changeJoint(True)

			if e.key() == QtCore.Qt.Key.Key_F:
				self.onForce()

		return super().keyReleaseEvent(e)

	def setRoMs(self):
		roms = []
		bounds = [[], []]
		for key in self.arm.jointRoM.keys():
			rom = self.arm.jointRoM[key]
			roms.append(abs(rom[1] - rom[0]))
			bounds[0].append(rom[0])
			bounds[1].append(rom[1])

		self.roms = roms
		self.bounds = bounds

	def clipJoints(self):
		if self.controlMode == 'position':
			self.jointCom = np.clip(self.jointCom, self.bounds[0], self.bounds[1])

	def incrementPosition(self, direction):
		# dir represents the direction of increment - True means to increase the position, False means to decrease
		mult = 1 if direction else -1
		mult *= -1 if self.jointCtrld == 5 else 1 # thumb rotation is inverted
		
		self.jointCom[self.jointCtrld] = self.jointCom[self.jointCtrld] + mult*self.scale*self.roms[self.jointCtrld]
		self.clipJoints()
		# self.armThreader(posDes=np.array(self.jointCom), period=1/self.arm.Hz)
		self.arm.handCom = np.array(self.jointCom)

	def changeJoint(self, direction):
		# dir represents the order of joint switching (with wrap-around) - True means to increase the joint index, False means to decrease
		# according to [index, middle, ring, pinky, thumbFlex, thumbRot]
		inc = 1 if direction else -1

		self.jointCtrld = (self.jointCtrld + inc) % self.numCom

		self.keyboard_btn.setText(self.jointNames[self.jointCtrld])

	def onRecord(self):
		if not self.arm.recording:
			self.arm.recording = True
			self.arm.resetRecording()
			self.record_btn.setText("Stop")
			self.record_btn.setStyleSheet("background-color: red")

			self.move_btn.setDisabled(True)
			self.zero_btn.setDisabled(True)
			self.manual_btn.setDisabled(True)
			self.keyboard_btn.setDisabled(True)
			self.playback_btn.setDisabled(True)
			self.settings_btn.setDisabled(True)

			if self.arm.usingEMG:
				self.armThreader(emg=self.emg)
			else:
				self.armThreader()

		else:
			self.arm.recording = False
			saveData = np.copy(self.arm.recordedData)
			self.arm.movingEvent.set()
			self.arm.isMoving = False
			self.record_btn.setText("Record")
			self.record_btn.setStyleSheet("background-color: white")

			# Save recorded data
			save_dialog = QtWidgets.QFileDialog(self)
			save_dialog.setDefaultSuffix('.csv')
			save_dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)

			saveDir = '/home/haptix/haptix/psyonic/logs'
			if not os.path.exists(saveDir): os.mkdir(saveDir)

			ok = False
			while not ok:
				filename, ok = save_dialog.getSaveFileName(caption='Enter record name', directory=saveDir, filter='CSV files (*.csv)')

				if filename == 'exit':
					break

			if ok:
				saveThread = threading.Thread(target=self.saveDataThread, name='saveThread', args=[filename, saveData])
				saveThread.start()

	def onMove(self):
		if not self.arm.isMoving:
			self.zero_btn.setDisabled(True)
			self.manual_btn.setDisabled(True)
			self.keyboard_btn.setDisabled(True)
			self.playback_btn.setDisabled(True)
			self.record_btn.setDisabled(True)
			self.settings_btn.setDisabled(True)

			if self.arm.usingEMG:
				self.armThreader(emg=self.emg)
			else:
				self.armThreader()

			self.move_btn.setText("Stop")
			self.move_btn.setStyleSheet("background-color: red")

		else:
			self.arm.movingEvent.set()
			self.arm.isMoving = False
			self.move_btn.setText("Move")
			self.move_btn.setStyleSheet("background-color: white")

	def onZero(self):
		self.disableButtons()
		self.armThreader(posDes=np.array([0, 0, 0, 0, 0, 0]))

	def onMiddle(self):
		self.disableButtons()
		self.armThreader(posDes=np.array([60, 60, 60, 60, 60, -60]))

	def onManual(self):
		self.disableButtons()

		manWindow = manualWindow(parent=self)
		manWindow.exec_()

		posDes = manWindow.posDes
	
		self.armThreader(posDes=np.asarray(posDes))

	def onSet(self):
		controlMode = self.arm.getCurControlMode() if not self.arm.impedanceMode else 'impedance'
		replyVariant = str(self.arm.getCurReplyMode())
		setWindow = settingsWindow(parent=self, controlMode=controlMode, replyVariant=replyVariant)
		setWindow.exec_()

		self.arm.setControlMode(setWindow.controlMode)
		self.arm.setReplyVariant(setWindow.replyVariant)

	def onKeyboard(self):
		if self.keyboard_btn.isChecked():
			self.keyboard_btn.setText(self.jointNames[self.jointCtrld])
			self.keyboardCtrl = True

			self.zero_btn.setDisabled(True)
			self.manual_btn.setDisabled(True)
			self.playback_btn.setDisabled(True)
		else:
			self.keyboard_btn.setText("Keyboard")
			self.keyboardCtrl = False

	def onPlayback(self):
		self.disableButtons()

		validFilename = False
		while not validFilename:
			# Load recorded data
			load_dialog = QtWidgets.QFileDialog(self)
			load_dialog.setDefaultSuffix('.csv')
			load_dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptOpen)
			filename, ok = load_dialog.getOpenFileName(caption='Enter record name', directory='/home/haptix/haptix/psyonic/logs', filter='Text files (*.csv)')

			if ok:
				try:
					if not filename.endswith('.csv'): filename += '.csv'

					loadedData = pd.read_csv(filename, delimiter='\t', header=0)
					validFilename = True
				except: pass
			else:
				self.enableButtons()
				break

		if validFilename:
			positionColTitles = ['index_PosCom', 'middle_PosCom', 'ring_PosCom', 'pinky_PosCom', 'thumbFlex_PosCom', 'thumbRot_PosCom']
			self.controlMode = 'position'
			loadedPositions = loadedData[positionColTitles].values
			fullMove = self.arm.playbackRecording(loadedPositions)

			self.armThreader(posDes=fullMove)

	def onEMG(self):
		emgWindow = emgPlotter(emg=self.emg, parent=self)
		emgWindow.show()
		emgWindow.start()

	def onForce(self):
		self.forceWindow = forcePlotter(hand=self.arm.hand)
		self.forceWindow.show()

	def enableButtons(self):
		self.record_btn.setDisabled(False)
		self.manual_btn.setDisabled(False)
		self.move_btn.setDisabled(False)
		self.keyboard_btn.setDisabled(False)
		self.zero_btn.setDisabled(False)
		self.playback_btn.setDisabled(False)
		self.settings_btn.setDisabled(False)

		if not self.emg is None:
			self.emg_btn.setDisabled(False)

	def disableButtons(self):
		self.record_btn.setDisabled(True)
		self.manual_btn.setDisabled(True)
		self.move_btn.setDisabled(True)
		self.keyboard_btn.setDisabled(True)
		self.zero_btn.setDisabled(True)
		self.playback_btn.setDisabled(True)
		self.emg_btn.setDisabled(True)
		self.settings_btn.setDisabled(True)

	def saveDataThread(self, filename, saveData):
		if not filename.endswith('.csv'): filename += '.csv'
		dataToSave = np.array(saveData)
		np.savetxt(filename, dataToSave, delimiter='\t', fmt='%s')

	def closeEvent(self, event:QtGui.QCloseEvent):
		if self.started:
			self.arm.exitEvent.set()
			self.arm.movingEvent.set()
			self.receiveThread.join()

		self.timer.stop()
		event.accept()

		return super().closeEvent(event)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Psyonic Ability Hand GUI Interface')
	parser.add_argument('-e', '--emg', help='Using EMG control?', action='store_true')
	parser.add_argument('-l', '--laterality', type=str, help='Handedness', default='left')
	parser.add_argument('-s', '--stuffing', help='Using byte stuffing?', action='store_true')
	args = parser.parse_args()

	emg = None

	assert args.laterality in ['left', 'right'], 'Invalid laterality. Please choose either left or right.'

	# instantiate arm class
	arm = psyonicArm(hand=args.laterality, stuffing=args.stuffing, usingEMG=args.emg)

	strInsert = ', byte stuffing' if args.stuffing else ', no byte stuffing'
	if args.emg:
		print(f'Starting Psyonic Hand (EMG control{strInsert})...')
		usedChannels = [0, 1, 2, 4, 5, 8, 10, 12]
		emg = EMG(usedChannels=usedChannels)
		emg.startCommunication()

		controller = psyonicControllers(numMotors=arm.numMotors, arm=arm, freq_n=3, emg=emg)
		arm.runNetThread(controller)

	print('Initializing sensor readings...')
	arm.initSensors()
	print('Sensors initialized.')

	arm.startComms()

	app = QtWidgets.QApplication([])
	gui = Psyonic_GUI(arm=arm, EMG=emg)
	gui.start()
	exitCode = app.exec_()

	# close threads
	if args.emg:
		emg.exitEvent.set()
		arm.netThread.join()

	sys.exit(exitCode)
