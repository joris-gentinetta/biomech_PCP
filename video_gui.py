import cv2
import tkinter as tk
from PIL import Image, ImageTk
import argparse


parser = argparse.ArgumentParser(description='Capture a video.')
parser.add_argument('--file', type=str, required=True, help='Video file')
args = parser.parse_args()


# Create a Tkinter window
window = tk.Tk()

# Open the video file
cap = cv2.VideoCapture(args.file)

# Get the total number of frames
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = 1200
height = 700
# Create a Canvas widget to display the video frames
canvas = tk.Canvas(window, width=width, height=height)
canvas.pack()

# Create a Scale widget to act as the slider
slider = tk.Scale(window, from_=0, to=total_frames, length=600, orient=tk.HORIZONTAL)
slider.pack()
from PIL import Image

def update_image(value):
    # Set the video position to the slider value
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(value))

    # Read the frame at the current position
    ret, frame = cap.read()

    # Convert the frame from BGR to RGB color space
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize the frame to match the size of the canvas
    original_height, original_width = frame.shape[:2]

    # Define the new dimensions you want


    # Calculate the ratios of the new dimensions to the old dimensions
    width_ratio = float(original_width) / width
    height_ratio = float(original_height) / height

    # Use the maximum of these ratios to maintain the aspect ratio
    max_ratio = max(width_ratio, height_ratio)

    # Calculate the new dimensions while maintaining the aspect ratio
    new_width = int(original_width / max_ratio)
    new_height = int(original_height / max_ratio)

    # Resize the frame
    frame = cv2.resize(frame, (new_width, new_height))

    # Convert the frame to a PhotoImage object
    image = ImageTk.PhotoImage(image=Image.fromarray(frame))

    # Display the PhotoImage on the Canvas
    canvas.create_image(0, 0, anchor=tk.NW, image=image)

    # Keep a reference to the image object to prevent it from being garbage collected
    canvas.image = image


# Bind the Scale widget's command to the update function
slider.configure(command=update_image)

# Start the Tkinter main loop
window.mainloop()