import sys
import os
import numpy as np
import pandas as pd
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QLineEdit, QFileDialog, QComboBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import time

class LivePlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=6, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = fig.add_subplot(111)
        super(LivePlotCanvas, self).__init__(fig)
        self.setParent(parent)
        fig.tight_layout()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Prosthetic Hand Force Control GUI")
        self.setGeometry(100, 100, 1000, 600)

        # Initialize data buffers for plotting
        self.start_time = time.time()
        self.time_data = []
        self.live_force_data = []
        self.target_force_data = []

        # Default mode is Program Mode
        self.current_mode = "Program"

        # Preloaded program and demo data (if any)
        self.program_data = None
        self.demo_data = None

        self.initUI()
        self.timer = QtCore.QTimer()
        self.timer.setInterval(100)  # update every 100ms
        self.timer.timeout.connect(self.update_plot)

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main horizontal layout: Plot on left, Error info on right
        main_h_layout = QHBoxLayout(central_widget)
        
        # Left side: Vertical layout with controls and plot
        left_layout = QVBoxLayout()
        
        # Controls Layout
        controls_layout = QHBoxLayout()
        # Mode selection
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Program", "Manual", "Demo"])
        self.mode_combo.currentTextChanged.connect(self.mode_changed)
        controls_layout.addWidget(QLabel("Mode:"))
        controls_layout.addWidget(self.mode_combo)
        
        # File load button for program mode
        self.load_program_button = QPushButton("Load Program CSV")
        self.load_program_button.clicked.connect(self.load_program_csv)
        controls_layout.addWidget(self.load_program_button)
        
        # Manual force input (initially hidden)
        self.manual_input = QLineEdit()
        self.manual_input.setPlaceholderText("Enter constant force")
        font = self.manual_input.font()
        font.setPointSize(14)
        self.manual_input.setFont(font)
        controls_layout.addWidget(QLabel("Manual Force:"))
        controls_layout.addWidget(self.manual_input)
        self.manual_input.hide()
        
        # Demo file load button (optional)
        self.load_demo_button = QPushButton("Load Demo CSV (optional)")
        self.load_demo_button.clicked.connect(self.load_demo_csv)
        controls_layout.addWidget(self.load_demo_button)
        
        # Start/Stop buttons
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_simulation)
        controls_layout.addWidget(self.start_button)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_simulation)
        controls_layout.addWidget(self.stop_button)
        
        left_layout.addLayout(controls_layout)
        
        # Plot Canvas
        self.canvas = LivePlotCanvas(self, width=8, height=5)
        left_layout.addWidget(self.canvas)
        
        # Initialize plot lines and set larger fonts for labels and ticks
        self.live_line, = self.canvas.ax.plot([], [], label="Live Force", color='blue', linewidth=2)
        self.target_line, = self.canvas.ax.plot([], [], label="Target Force", color='red', linewidth=2)
        self.canvas.ax.legend(fontsize=14)
        self.canvas.ax.set_xlabel("Time (s)", fontsize=16)
        self.canvas.ax.set_ylabel("Force", fontsize=16)
        self.canvas.ax.set_title("Force Sensor vs. Target Force", fontsize=18)
        self.canvas.ax.grid(True)
        for label in (self.canvas.ax.get_xticklabels() + self.canvas.ax.get_yticklabels()):
            label.set_fontsize(14)
        
        # Right side: Error info panel
        self.error_info = QLabel("Error Info:\nCurrent Error: N/A\nMean Error: N/A")
        self.error_info.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        error_font = self.error_info.font()
        error_font.setPointSize(9)
        self.error_info.setFont(error_font)
        self.error_info.setMinimumWidth(200)
        
        # Add left layout and error info to main horizontal layout with stretch factors (5:1)
        main_h_layout.addLayout(left_layout, stretch=5)
        main_h_layout.addWidget(self.error_info, stretch=1)

    def mode_changed(self, mode):
        self.current_mode = mode
        self.reset_buffers()
        if mode == "Manual":
            self.manual_input.show()
        else:
            self.manual_input.hide()
        print(f"Mode changed to {mode}")

    def load_program_csv(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(self, "Load Program CSV", "",
                                                  "CSV Files (*.csv);;All Files (*)", options=options)
        if filename:
            try:
                self.program_data = pd.read_csv(filename)
                if not {'timestamp', 'force'}.issubset(self.program_data.columns):
                    QtWidgets.QMessageBox.warning(self, "Error", "CSV must have 'timestamp' and 'force' columns.")
                    self.program_data = None
                else:
                    print("Program CSV loaded successfully.")
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Error", f"Failed to load CSV: {e}")

    def load_demo_csv(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(self, "Load Demo CSV", "",
                                                  "CSV Files (*.csv);;All Files (*)", options=options)
        if filename:
            try:
                self.demo_data = pd.read_csv(filename)
                if not {'timestamp', 'force'}.issubset(self.demo_data.columns):
                    QtWidgets.QMessageBox.warning(self, "Error", "CSV must have 'timestamp' and 'force' columns.")
                    self.demo_data = None
                else:
                    print("Demo CSV loaded successfully.")
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Error", f"Failed to load CSV: {e}")

    def start_simulation(self):
        self.reset_buffers()
        self.start_time = time.time()
        self.timer.start()
        print("Simulation started.")

    def stop_simulation(self):
        self.timer.stop()
        print("Simulation stopped.")

    def reset_buffers(self):
        self.time_data = []
        self.live_force_data = []
        self.target_force_data = []
        self.canvas.ax.cla()
        self.live_line, = self.canvas.ax.plot([], [], label="Live Force", color='blue', linewidth=2)
        self.target_line, = self.canvas.ax.plot([], [], label="Target Force", color='red', linewidth=2)
        self.canvas.ax.set_xlabel("Time (s)", fontsize=16)
        self.canvas.ax.set_ylabel("Force", fontsize=16)
        self.canvas.ax.set_title("Force Sensor vs. Target Force", fontsize=18)
        self.canvas.ax.legend(fontsize=14)
        self.canvas.ax.grid(True)
        for label in (self.canvas.ax.get_xticklabels() + self.canvas.ax.get_yticklabels()):
            label.set_fontsize(14)
        self.error_info.setText("Error Info:\nCurrent Error: N/A\nMean Error: N/A")
        self.canvas.draw()

    def update_plot(self):
        current_time = time.time() - self.start_time
        self.time_data.append(current_time)
        live_force = self.get_live_force(current_time)
        self.live_force_data.append(live_force)
        target_force = self.get_target_force(current_time)
        self.target_force_data.append(target_force)
        
        # Calculate error metrics
        current_error = abs(live_force - target_force)
        mean_error = np.mean(np.abs(np.array(self.live_force_data) - np.array(self.target_force_data)))
        error_text = f"Error Info:\nCurrent Error: {current_error:.2f}\nMean Error: {mean_error:.2f}"
        self.error_info.setText(error_text)
        
        self.live_line.set_data(self.time_data, self.live_force_data)
        self.target_line.set_data(self.time_data, self.target_force_data)
        
        # Set sliding window: if more than 15 seconds, show last 15 seconds
        if current_time > 15:
            self.canvas.ax.set_xlim(current_time - 15, current_time)
        else:
            self.canvas.ax.set_xlim(0, 15)
        
        self.canvas.ax.relim()
        self.canvas.ax.autoscale_view(scalex=False)  # only update y-axis limits
        self.canvas.draw()

    def get_live_force(self, t):
        if self.current_mode == "Demo":
            if self.demo_data is not None:
                timestamps = self.demo_data['timestamp'].values
                forces = self.demo_data['force'].values
                return float(np.interp(t, timestamps, forces))
            else:
                return 5 * np.sin(2 * np.pi * 0.2 * t) + np.random.normal(0, 0.5)
        else:
            return 5 * np.sin(2 * np.pi * 0.2 * t) + np.random.normal(0, 0.5)

    def get_target_force(self, t):
        if self.current_mode == "Program":
            if self.program_data is not None:
                timestamps = self.program_data['timestamp'].values
                forces = self.program_data['force'].values
                return float(np.interp(t, timestamps, forces))
            else:
                return 0
        elif self.current_mode == "Manual":
            try:
                value = float(self.manual_input.text())
                return value
            except ValueError:
                return 0
        elif self.current_mode == "Demo":
            return 3.0
        else:
            return 0

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
