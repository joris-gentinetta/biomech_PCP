import argparse
import tkinter as tk

import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

parser = argparse.ArgumentParser(description="Show a video.")
parser.add_argument("--file", type=str, required=True, help="Video file")
args = parser.parse_args()

# Create a Tkinter window
window = tk.Tk()
window.title("Video")

# Open the video file
cap = cv2.VideoCapture(args.file)

# Get the total number of frames
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = 1200
height = 700

# Create a Figure and Axes with matplotlib
fig = Figure(figsize=(width / 80, height / 80))  # Convert pixels to inches for figsize
ax = fig.add_subplot(111)

# Create a Canvas widget to display the video frames
canvas = FigureCanvasTkAgg(fig, master=window)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack()

# Create a Scale widget to act as the slider
slider = tk.Scale(
    window, from_=0, to=total_frames - 1, length=600, orient=tk.HORIZONTAL
)
slider.pack()
current_frame = None


def save_frame():
    cv2.imwrite("frame.jpg", cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR))
    print("frame saved")


button = tk.Button(window, text="Save Frame", command=save_frame)
button.pack()


def update_image(value):
    global current_frame
    # Set the video position to the slider value
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(value))

    # Read the frame at the current position
    ret, frame = cap.read()

    # Convert the frame from BGR to RGB color space
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    current_frame = frame

    # Clear the axes for the new frame
    ax.clear()

    # Display the frame on the axes
    ax.imshow(frame)

    # Update the canvas with the new frame
    canvas.draw()


# Bind the Scale widget's command to the update function
slider.configure(command=update_image)

# Start the Tkinter main loop
window.mainloop()
