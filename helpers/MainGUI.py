import os
os.environ['PYQTGRAPH_QT_LIB'] = 'PyQt5'
from PyQt5 import QtCore, QtWidgets
import pandas as pd
import numpy as np
import sys
import time
import threading
import zmq
import struct

import helpers.floatingCurves as fc
from helpers.BesselFilter import BesselFilterArr

class mainWindow(QtWidgets.QWidget):
    def __init__(self):
        super(mainWindow, self).__init__()
        self.title = 'BIOMECH EMG VISUALIZER'
        self.ax, self.ay, self.aw, self.ah = 100, -1500, 2000, 1200

        # EMG / data parameters
        self.emgChannels = 16
        self.nonEMGChannels = 4
        self.channelNum = self.emgChannels + self.nonEMGChannels
        self.packetNum = 15
        self.sampleRate = 1000
        self.plotWndSize = 8000
        self.scaleFactorsPath = '../UEA-AMI-Controller/handsim/include/scaleFactors.txt'

        # read normalization bounds
        self.getBounds()
        # setup UI
        self.initUI()

        # ─── Direct ZMQ subscriber to raw-EMG PUB ──────────────────────────
        self.ip = '127.0.0.1'
        self.port = 1236
        self.ctx = zmq.Context()
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.connect(f"tcp://{self.ip}:{self.port}")
        self.sub.subscribe(b"")

        # start background thread to recv packets
        threading.Thread(target=self._recv_loop, daemon=True).start()

        # timer for plot updates
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.updateGUI)

        # buffers for plotting
        self.plotBufs = [np.zeros(self.plotWndSize) for _ in range(self.channelNum)]

        # filtering flags and filters
        self.doesFilter   = False
        self.doesiEMGFilter = False
        self.powerLineFilters = BesselFilterArr(numChannels=self.emgChannels, order=4,
                                                critFreqs=[58,62], fs=self.sampleRate, filtType='bandstop')
        self.highpassFilters   = BesselFilterArr(numChannels=self.emgChannels, order=4,
                                                critFreqs=8, fs=self.sampleRate, filtType='highpass')
        self.lowpassFilters    = BesselFilterArr(numChannels=self.emgChannels, order=4,
                                                critFreqs=3, fs=self.sampleRate, filtType='lowpass')

        # recording state
        self.isRecording = False
        self.recordingBuf = []

        # ultrasound support (unchanged)
        self.ultrasoundPort = 1237
        self.ctx2 = zmq.Context()
        self.sock = self.ctx2.socket(zmq.PULL)
        self.sock.bind(f'tcp://*:{self.ultrasoundPort}')
        self.quitEvent = threading.Event()
        self.ultrasoundThread = threading.Thread(
            target=self.ultrasoundSignalRecv, name='ultrasoundThread')
        self.ultrasoundThread.start()

    def __del__(self):
        try:
            self.sock.close()
            self.ctx2.term()
            self.quitEvent.set()
            self.ultrasoundThread.join()
        except Exception as e:
            print(f'__del__: Socket closing error {e}')

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.ax, self.ay, self.aw, self.ah)
        layout = QtWidgets.QHBoxLayout(self)
        self.dataPlottingWidget = fc.floatingCurves(self)
        layout.addWidget(self.dataPlottingWidget)

        # button panel
        self.buttonPanel = QtWidgets.QWidget(self)
        self.buttonPanelLayout = QtWidgets.QVBoxLayout(self.buttonPanel)
        layout.addWidget(self.buttonPanel)
        self.recordBtn = QtWidgets.QPushButton('Record')
        self.filterBtn = QtWidgets.QPushButton('Filters')
        self.iEMGBtn   = QtWidgets.QPushButton('iEMG')
        self.normsBtn  = QtWidgets.QPushButton('Manual')
        self.buttonPanelLayout.addWidget(self.recordBtn)
        self.buttonPanelLayout.addWidget(self.filterBtn)
        self.buttonPanelLayout.addWidget(self.iEMGBtn)
        self.buttonPanelLayout.addWidget(self.normsBtn)
        self.iEMGBtn.setDisabled(True)
        self.linkBtnFunctions()
        self.show()

    def getBounds(self):
        try:
            with open(self.scaleFactorsPath, 'rb') as f:
                normsPack = f.read()
            norms = struct.unpack('32f', normsPack)
            self.maxVals     = np.asarray(norms[:self.emgChannels])
            self.noiseLevel  = np.asarray(norms[self.emgChannels:])
        except OSError as e:
            print(f'getBounds(): Could not read bounds - {e}')

    def linkBtnFunctions(self):
        self.recordBtn.clicked.connect(self.onRecordBtnClicked)
        self.filterBtn.clicked.connect(self.onFilterBtnClicked)
        self.iEMGBtn.clicked.connect(self.oniEMGBtnClicked)
        self.normsBtn.clicked.connect(self.onNormsBtnClicked)

    def _recv_loop(self):
        fmt = "BBBBIHH" + "f"*self.emgChannels + "BBBB"
        pkt_size = struct.calcsize(fmt)
        while True:
            msg = self.sub.recv()
            if len(msg) != pkt_size:
                continue
            pkt = struct.unpack(fmt, msg)
            ch    = list(pkt[7:7+self.emgChannels])
            extra = [pkt[24], pkt[25], pkt[26], pkt[4]]
            data  = ch + extra
            QtCore.QMetaObject.invokeMethod(
                self, "_invoke_callback",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(object, data)
            )

    @QtCore.pyqtSlot(object)
    def _invoke_callback(self, newData):
        self.dataReadyCallback(newData)

    def dataReadyCallback(self, newData):
        # newData is a flat list: [ch0, ch1, … ch15, switch1, switch2, newline, timestamp]
        d = np.asarray(newData[:self.channelNum])

        # apply filters only if requested
        if self.doesFilter:
            # extract the raw EMG channels (shape (16,))
            raw = d[:self.emgChannels]

            # reshape to (channels, samples) = (16, 1)
            sig = raw.reshape(self.emgChannels, 1)

            # high-pass then notch
            sig = self.highpassFilters.filter(sig)
            sig = self.powerLineFilters.filter(sig)

            if self.doesiEMGFilter:
                # envelope: abs → noise clip → low-pass → normalize
                sig = np.abs(sig)
                sig = np.clip(sig - self.noiseLevel[:, None], 0, None)
                sig = self.lowpassFilters.filter(sig)
                sig = np.clip(sig / self.maxVals[:, None], 0, 1)

            # write back the single filtered sample for each channel
            d[:self.emgChannels] = sig[:, 0]

        # shift and append to each plot buffer
        for i in range(self.channelNum):
            self.plotBufs[i] = np.roll(self.plotBufs[i], -self.packetNum)
            self.plotBufs[i][-self.packetNum:] = d[i]

        # if recording, store the raw newData packet packetNum times
        if self.isRecording:
            for _ in range(self.packetNum):
                self.recordingBuf.append(newData)

    def start(self):
        self.timer.start(100)

    def updateGUI(self):
        for i in range(self.channelNum):
            self.dataPlottingWidget.updateCurve(i, self.plotBufs[i])

    def keyPressEvent(self, e):
        if e.isAutoRepeat(): return
        if e.key() == QtCore.Qt.Key_R: self.onRecordBtnClicked()
        if e.key() == QtCore.Qt.Key_F: self.onFilterBtnClicked()
        if e.key() == QtCore.Qt.Key_I and self.doesFilter: self.oniEMGBtnClicked()
        return super().keyPressEvent(e)

    def saveThread(self, saveName, saveData):
        pd.DataFrame(saveData).to_csv(saveName, sep='\t', index=False)

    def onRecordBtnClicked(self):
        self.isRecording = not self.isRecording
        if self.isRecording:
            self.recordBtn.setText('Stop')
            self.recordBtn.setStyleSheet('background-color: red')
        else:
            self.recordBtn.setText('Record')
            self.recordBtn.setStyleSheet('')
            fname, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, 'Save File', f'EMG_Recording_{time.time()}.csv', 'CSV Files (*.csv)')
            if fname:
                threading.Thread(target=self.saveThread, args=(fname, self.recordingBuf), daemon=True).start()
            self.recordingBuf.clear()

    def onFilterBtnClicked(self):
        self.doesFilter = not self.doesFilter
        if self.doesFilter:
            self.filterBtn.setStyleSheet('background-color: gray')
            self.powerLineFilters.resetFilters()
            self.highpassFilters.resetFilters()
            self.iEMGBtn.setDisabled(False)
        else:
            self.filterBtn.setStyleSheet('')
            self.iEMGBtn.setDisabled(True)
            self.doesiEMGFilter = True
            self.oniEMGBtnClicked()

    def oniEMGBtnClicked(self):
        self.doesiEMGFilter = not self.doesiEMGFilter
        self.iEMGBtn.setStyleSheet('background-color: gray' if self.doesiEMGFilter else '')

    def onNormsBtnClicked(self):
        normsGUI = normsWindow(parent=self, rows=4, cols=4)
        normsGUI.show()

    def ultrasoundSignalRecv(self):
        while not self.quitEvent.is_set():
            packed = self.sock.recv()
            saveName = struct.unpack(f'{len(packed)}s', packed)[0].decode()
            if not self.isRecording:
                self.onRecordBtnClicked()
            else:
                self.onRecordBtnClicked()
                fname = saveName.replace('ultra','emg') + '.csv'
                threading.Thread(target=self.saveThread, args=(fname,self.recordingBuf), daemon=True).start()

class normsWindow(QtWidgets.QWidget):
    class gridWidget(QtWidgets.QWidget):
        def __init__(self, index, parent=None):
            super().__init__()
            self._p = parent
            self.grid = QtWidgets.QGridLayout(self)
            for r in range(self._p.rows):
                for c in range(self._p.cols):
                    tb = QtWidgets.QLineEdit(self)
                    val = self._p.vals[index*self._p.cols*self._p.rows + c*self._p.rows + r]
                    tb.setText(str(val))
                    self.grid.addWidget(tb, r, c)
    def __init__(self, parent=None, rows=4, cols=4):
        super().__init__(parent, QtCore.Qt.Window)
        self._parent = parent
        self.rows, self.cols = rows, cols
        self.vals = np.hstack((self._parent.maxVals, self._parent.noiseLevel))
        self.setWindowTitle('EMG Norms')
        self.setGeometry(parent.ax+50, parent.ay+50, parent.aw//3, parent.ah//3)
        self.tabs = QtWidgets.QTabWidget(self)
        self.tabs.addTab(normsWindow.gridWidget(0,self), 'Maxes')
        self.tabs.addTab(normsWindow.gridWidget(1,self), 'Noise')
        self.saveBtn = QtWidgets.QPushButton('Save', self)
        self.cancelBtn = QtWidgets.QPushButton('Cancel', self)
        self.saveBtn.clicked.connect(self.saveChanges)
        self.cancelBtn.clicked.connect(self.cancelChanges)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.tabs)
        layout.addWidget(self.saveBtn)
        layout.addWidget(self.cancelBtn)

    def saveChanges(self):
        newVals = []
        for t in range(self.tabs.count()):
            w = self.tabs.widget(t)
            for r in range(self.rows):
                for c in range(self.cols):
                    newVals.append(float(w.grid.itemAtPosition(r,c).widget().text()))
        try:
            normsBytes = struct.pack('32f', *newVals)
            with open(self._parent.scaleFactorsPath, 'wb') as f:
                f.write(normsBytes)
            self._parent.maxVals    = np.asarray(newVals[:16])
            self._parent.noiseLevel = np.asarray(newVals[16:])
            QtWidgets.QMessageBox.information(self, 'Saved', 'Norms saved successfully')
            self.close()
        except Exception:
            QtWidgets.QMessageBox.critical(self, 'Error', 'Failed to save norms')

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = mainWindow()
    w.start()
    sys.exit(app.exec_())
