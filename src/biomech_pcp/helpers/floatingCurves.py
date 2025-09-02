"""
This class is a widget it allows displaying multiple graphs and float each as a window if needed on real time.
Updating the graph will be done outside of this class.
"""

import pyqtgraph as pg
from PyQt5 import QtWidgets


class floatingCurves_Max(QtWidgets.QMainWindow):
    def __init__(self, curve: pg.PlotDataItem, oldWidget: pg.PlotWidget, parent=None):
        super(floatingCurves_Max, self).__init__(parent)
        self.oldWidget = oldWidget
        self.curve = curve
        plotWidget = pg.PlotWidget()
        plotWidget.addItem(curve)
        _ = QtWidgets.QWidget(self)
        self.setCentralWidget(plotWidget)

    def closeEvent(self, a0):
        self.oldWidget.addItem(self.curve)

        return super().closeEvent(a0)


class floatingCurves(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(floatingCurves, self).__init__()
        self._p = parent

        self.channelNum = self._p.channelNum
        self.curveList = []
        self.plotWidgetList = []
        self.labelList = []
        self.checkList = []

        # Perpare the layout
        self.layout = QtWidgets.QGridLayout()

        self.setLayout(self.layout)

        self.generateGraphsArray()

    def addCurve(self, newCurve: pg.PlotDataItem, x, y):
        self.curveList.append(newCurve)

        plotWidget = pg.PlotWidget()
        plotWidget.addItem(newCurve)

        self.plotWidgetList.append(plotWidget)
        self.layout.addWidget(plotWidget, x, y)

        # Add the button
        button = QtWidgets.QPushButton("+", plotWidget)
        button.resize(20, 20)
        button.move(20, 0)
        button.clicked.connect(self.make_btn_floatWnd(len(self.curveList) - 1))

        # add the checkbox
        check = QtWidgets.QCheckBox(plotWidget)
        check.setStyleSheet("QCheckBox::indicator{width: 20px;height: 20px;}")
        check.setCheckState(2)
        check.stateChanged.connect(
            lambda state, index=len(self.checkList): self.disableChannel(state, index)
        )
        self.checkList.append(check)

        # add the label
        label = QtWidgets.QLineEdit(plotWidget)
        label.setFixedWidth(50)
        label.setFixedHeight(20)
        label.move(320, 0)
        self.labelList.append(label)

    def generateGraphsArray(self):
        for i in range(self.channelNum):
            newCurve = pg.PlotDataItem()
            y = i // 4
            x = i % 4
            self.addCurve(newCurve, x, y)

    def make_btn_floatWnd(self, index):
        def btn_floatWnd():
            newWnd = floatingCurves_Max(
                self.curveList[index], self.plotWidgetList[index], self
            )
            newWnd.show()

        return btn_floatWnd

    def disableChannel(self, state, index):
        self.checkList[index] = state

    def updateCurve(self, index, data: list()):
        if not self.checkList[index]:
            data = [0] * len(data)
        self.curveList[index].setData(data)


# Used for testing

# win = pg.GraphicsWindow()
# win.addPlot()
