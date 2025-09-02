# BUGMAN 12/20/2018 for MIT Fluid Interface

# BUGMAN Nov 2 2020 Modified for MIT Biomech
# Mikey Fernandez 1/15/2022 Modified for MIT Biomech - adjustments to plotting speed for smoothness

import os

os.environ["PYQTGRAPH_QT_LIB"] = "PyQt5"
import struct
import sys
import threading
import time

import helpers.floatingCurves as fc
import numpy as np
import pandas as pd
import zmq
from helpers.BCI_Data_Receiver import BCI_Data_Receiver
from helpers.BesselFilter import BesselFilterArr
from PyQt5 import QtCore, QtWidgets


class mainWindow(QtWidgets.QWidget):
    def __init__(self):
        super(mainWindow, self).__init__()
        self.title = "BIOMECH EMG VISUALIZER"
        self.ax = 100
        self.ay = -1500
        self.aw = 2000
        self.ah = 1200

        self.emgChannels = 16
        self.nonEMGChannels = 4
        self.channelNum = self.emgChannels + self.nonEMGChannels
        self.packetNum = 15
        self.sampleRate = 1000
        self.plotWndSize = 8000
        self.scaleFactorsPath = "../UEA-AMI-Controller/handsim/include/scaleFactors.txt"

        self.getBounds()
        self.initUI()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.updateGUI)

        # Init Data structures
        self.plotBufs = list()

        # For filters
        self.doesFilter = False
        self.doesiEMGFilter = False
        self.powerLineFilters = BesselFilterArr(
            numChannels=self.emgChannels,
            order=4,
            critFreqs=[58, 62],
            fs=self.sampleRate,
            filtType="bandstop",
        )
        self.highpassFilters = BesselFilterArr(
            numChannels=self.emgChannels,
            order=4,
            critFreqs=8,
            fs=self.sampleRate,
            filtType="highpass",
        )
        self.lowpassFilters = BesselFilterArr(
            numChannels=self.emgChannels,
            order=4,
            critFreqs=3,
            fs=self.sampleRate,
            filtType="lowpass",
        )

        # The ip of user's machine
        # self.ip = '18.27.123.85' # Mikey's computer ethernet
        self.ip = "127.0.0.1"
        self.port = 1236
        self.dataReceiver = BCI_Data_Receiver(self.ip, self.port, self.sampleRate)
        self.dataReceiver.asyncReceiveData(self.dataReadyCallback)

        for _ in range(self.channelNum):
            self.plotBufs.append(np.zeros(self.plotWndSize))

        # For data Recording
        self.isRecording = False
        self.recordingBuf = list()

        # channel labells
        self.channelLabels = ["" for _ in range(self.emgChannels)]

        # ip address for receiving ultrasound record signal
        self.ultrasoundPort = 1237
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.PULL)
        self.sock.bind(f"tcp://*:{self.ultrasoundPort}")
        self.quitEvent = threading.Event()

        # make ultrasound thread
        self.ultrasoundThread = threading.Thread(
            target=self.ultrasoundSignalRecv, name="ultrasoundThread"
        )
        self.ultrasoundThread.start()

    def __del__(self):
        try:
            # close the socket
            self.sock.close()
            self.ctx.term()

            self.quitEvent.set()
            self.ultrasoundThread.join()

        except Exception as e:
            print(f"__del__: Socket closing error {e}")

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.ax, self.ay, self.aw, self.ah)

        hbox = QtWidgets.QHBoxLayout()
        self.setLayout(hbox)

        # Perpare the array
        self.dataPlottingWidget = fc.floatingCurves(self)

        hbox.addWidget(self.dataPlottingWidget)

        # Add the button panel
        self.buttonPanel = QtWidgets.QWidget(self)
        self.buttonPanelLayout = QtWidgets.QVBoxLayout()
        self.buttonPanel.setLayout(self.buttonPanelLayout)
        hbox.addWidget(self.buttonPanel)

        # Add all the buttons
        self.recordBtn = QtWidgets.QPushButton("Record")
        self.filterBtn = QtWidgets.QPushButton("Filters")
        self.iEMGBtn = QtWidgets.QPushButton("iEMG")
        self.normsBtn = QtWidgets.QPushButton("Manual")

        self.buttonPanelLayout.addWidget(self.recordBtn)
        self.buttonPanelLayout.addWidget(self.filterBtn)
        self.buttonPanelLayout.addWidget(self.iEMGBtn)
        self.buttonPanelLayout.addWidget(self.normsBtn)
        self.iEMGBtn.setDisabled(True)

        # Connect all button functions
        self.linkBtnFunctions()

        self.show()

    def getBounds(self):
        try:
            with open(self.scaleFactorsPath, "rb") as fifo:
                normsPack = fifo.read()

            norms = struct.unpack("32f", normsPack)
            bounds = list(norms)
            self.maxVals = np.asarray(bounds[: self.emgChannels])
            self.noiseLevel = np.asarray(bounds[self.emgChannels :])

        except OSError as e:
            print(f"getBounds(): Could not read bounds - {e}")

    def linkBtnFunctions(self):
        self.recordBtn.clicked.connect(self.onRecordBtnClicked)
        self.filterBtn.clicked.connect(self.onFilterBtnClicked)
        self.iEMGBtn.clicked.connect(self.oniEMGBtnClicked)
        self.normsBtn.clicked.connect(self.onNormsBtnClicked)

    def dataReadyCallback(self, newData):
        d = np.asarray(newData[: self.channelNum])

        if self.doesFilter:
            d[: self.emgChannels] = self.highpassFilters.filter(d[: self.emgChannels])
            d[: self.emgChannels] = self.powerLineFilters.filter(d[: self.emgChannels])

            if self.doesiEMGFilter:
                d[: self.emgChannels] = np.clip(
                    np.abs(d[: self.emgChannels]) - self.noiseLevel[:, None], 0, None
                )
                d[: self.emgChannels] = np.clip(
                    self.lowpassFilters.filter(d[: self.emgChannels]), 0, None
                )
                d[: self.emgChannels] = np.divide(
                    d[: self.emgChannels], self.maxVals[:, None]
                )

        # update plotting
        for i in range(self.channelNum):
            self.plotBufs[i] = self.plotBufs[i][self.packetNum :]
            self.plotBufs[i] = np.append(self.plotBufs[i], d[i])

        if self.isRecording:
            # save new data to the recordingBuf
            newData = np.array(newData)
            # toSave = np.concatenate((newData, time.time()*np.ones((1, newData.shape[1]))), axis=0) # only needed this column to time sync the ultrasound and EMG for DB
            toSave = newData
            for data_index in range(self.packetNum):
                self.recordingBuf.append(toSave[:, data_index])

    def start(self):
        self.timer.start(100)  # how many milliseconds between screen updates

    def updateGUI(self):
        for i in range(self.channelNum):
            self.dataPlottingWidget.updateCurve(i, self.plotBufs[i])

    def keyPressEvent(self, e):
        """press keys to interface with the GUI"""
        if e.isAutoRepeat():
            return
        if e.key() == QtCore.Qt.Key_R:
            self.onRecordBtnClicked()
        if e.key() == QtCore.Qt.Key_F:
            self.onFilterBtnClicked()
        if e.key() == QtCore.Qt.Key_I and self.doesFilter:
            self.oniEMGBtnClicked()

        return super().keyPressEvent(e)

    def saveThread(self, saveName, saveData):
        df = pd.DataFrame(saveData)
        df.to_csv(saveName, encoding="utf-8", sep="\t", index=False)

    def onRecordBtnClicked(self):
        """The callback function for the click event of record button"""
        if not self.isRecording:  # Start recording
            self.isRecording = not self.isRecording
            self.recordBtn.setText("Stop")
            self.recordBtn.setStyleSheet("background-color: red")
        else:  # stop recording
            self.recordBtn.setText("Record")
            self.isRecording = not self.isRecording
            self.recordBtn.setStyleSheet("background-color: white")

            name = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save File",
                f"EMG_Recording_{time.time()}.csv",
                "CSV Files (*.csv)",
            )
            if not name[0] == "":
                saveData = np.asarray(self.recordingBuf)
                savingThread = threading.Thread(
                    target=self.saveThread, name="saveThread", args=[name[0], saveData]
                )
                savingThread.start()

            # Clear the recording buffer
            self.recordingBuf.clear()

    def onFilterBtnClicked(self):
        self.doesFilter = not self.doesFilter

        if self.doesFilter:  # turn on filters
            self.filterBtn.setStyleSheet("background-color: gray")
            self.powerLineFilters.resetFilters()
            self.highpassFilters.resetFilters()
            self.iEMGBtn.setDisabled(False)
        else:  # turn off filters
            self.filterBtn.setStyleSheet("background-color: white")
            self.filterBtn.setStyleSheet("background-color: white")
            self.iEMGBtn.setDisabled(True)
            self.doesiEMGFilter = True
            self.oniEMGBtnClicked()

    def oniEMGBtnClicked(self):
        self.doesiEMGFilter = not self.doesiEMGFilter

        if self.doesiEMGFilter:
            self.iEMGBtn.setStyleSheet("background-color: gray")
            self.lowpassFilters.resetFilters()
        else:
            self.iEMGBtn.setStyleSheet("background-color: white")

    def onNormsBtnClicked(self):
        # when the norms button is clicked we want to open a new window with the grids reading in the current maxes and noise values, allowing them to be changed and saved
        normsGUI = normsWindow(parent=self, rows=4, cols=4)
        normsGUI.show()

    ##############
    def ultrasoundSignalRecv(self):
        while not self.quitEvent.is_set():
            packed = self.sock.recv()
            saveName = struct.unpack(f"{len(packed)}s", packed)
            saveStr = saveName[0].decode("utf-8")

            if not self.isRecording:
                self.isRecording = not self.isRecording
                self.recordBtn.setText("Stop")
                self.recordBtn.setStyleSheet("background-color: red")

            else:
                self.recordBtn.setText("Record")
                self.isRecording = not self.isRecording
                self.recordBtn.setStyleSheet("background-color: white")

                if not saveStr == "":
                    saveStr = saveStr.replace("ultra", "emg")
                    saveData = np.asarray(self.recordingBuf)
                    savingThread = threading.Thread(
                        target=self.saveThread,
                        name="saveThread",
                        args=[f"{saveStr}.csv", saveData],
                    )
                    savingThread.start()

                self.recordingBuf.clear()


class normsWindow(QtWidgets.QWidget):
    class gridWidget(QtWidgets.QWidget):
        def __init__(self, index, parent=None):
            super().__init__()

            self._p = parent

            self.grid = QtWidgets.QGridLayout()

            for row in range(self._p.rows):
                for col in range(self._p.cols):
                    textbox = QtWidgets.QLineEdit(self)
                    # textbox.setText(str(self._p.vals[index*self._p.cols*self._p.rows + row*self._p.cols + col]))  # Set default value to 0
                    textbox.setText(
                        str(
                            self._p.vals[
                                index * self._p.cols * self._p.rows
                                + col * self._p.rows
                                + row
                            ]
                        )
                    )  # Set default value to 0
                    self.grid.addWidget(textbox, row, col)

            self.setLayout(self.grid)

    def __init__(self, parent=None, rows=4, cols=4):
        super(normsWindow, self).__init__(parent, QtCore.Qt.WindowType.Window)

        self._parent = parent

        self.setWindowTitle("EMG Norms")
        self.setGeometry(
            self._parent.ax + 50,
            self._parent.ay + 50,
            self._parent.aw // 3,
            self._parent.ah // 3,
        )  # open with a constant appropriate offset

        self.vals = np.hstack((self._parent.maxVals, self._parent.noiseLevel))

        self.tabs = QtWidgets.QTabWidget()
        self.rows = rows
        self.cols = cols

        self.tab1 = normsWindow.gridWidget(index=0, parent=self)
        self.tab2 = normsWindow.gridWidget(index=1, parent=self)

        self.tabs.addTab(self.tab1, "Maxes")
        self.tabs.addTab(self.tab2, "Noise")
        # tab 1 is the maxes, tab 2 is the noise values

        self.saveButton = QtWidgets.QPushButton("Save")
        self.cancelButton = QtWidgets.QPushButton("Cancel")

        self.saveButton.clicked.connect(self.saveChanges)
        self.cancelButton.clicked.connect(self.cancelChanges)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.tabs)
        layout.addWidget(self.saveButton)
        layout.addWidget(self.cancelButton)
        self.setLayout(layout)

    def saveChanges(self):
        failedFlag = False  # tracks if something invalid is set
        newVals = np.ones_like(self.vals)
        for tab in range(self.tabs.count()):
            current_widget = self.tabs.widget(tab)
            for row in range(self.rows):
                for col in range(self.cols):
                    thisVal = (
                        current_widget.grid.itemAtPosition(row, col).widget().text()
                    )

                    try:
                        if tab == 0 and float(thisVal) <= 0:
                            thisVal = "1"  # don't let you set the max to 0 or negative! It'll crash!

                        # newVals[tab*self.cols*self.rows + row*self.cols + col] = thisVal
                        newVals[tab * self.cols * self.rows + col * self.rows + row] = (
                            thisVal
                        )

                    except Exception as e:
                        print(e)
                        failedFlag = True

        msg = QtWidgets.QMessageBox(parent=self)
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok)

        try:
            if failedFlag:
                raise ValueError

            normsBytes = struct.pack("32f", *newVals)

            with open(self._parent.scaleFactorsPath, "wb") as output:
                output.write(normsBytes)

            self._parent.maxVals = newVals[: self.rows * self.cols]
            self._parent.noiseLevel = newVals[self.rows * self.cols :]

            msg.setText("Saved Successfully")
            msg.exec()
            self.close()

        except Exception as e:
            print(e)
            msg.setText("Saving failed!")
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.exec()

    def cancelChanges(self):
        for tab in range(self.tabs.count()):
            current_widget = self.tabs.widget(tab)
            for row in range(self.rows):
                for col in range(self.cols):
                    textbox = current_widget.grid.itemAtPosition(row, col).widget()
                    # textbox.setText(str(self.vals[tab*self.cols*self.rows + row*self.cols + col]))  # reset to default
                    textbox.setText(
                        str(
                            self.vals[
                                tab * self.cols * self.rows + col * self.rows + row
                            ]
                        )
                    )  # reset to default


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    ex = mainWindow()
    ex.start()
    sys.exit(app.exec_())
