# Upper Extremity Prosthesis Control Data Generation

## Installation

Follow the steps below to set up your environment:

1. Create a new conda environment with Python 3.8:

```bash
conda create -n datagen python=3.8
```

2. Activate the newly created environment:

```bash
conda activate datagen
```

3. Install PyTorch based on your system: [PyTorch website](https://pytorch.org/get-started/locally/)

4. Install the remaining dependencies from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Scripts

### 0.1. EMG Board Setup
- Find the port of the emg board:
```bash
# device not plugged in
ls /dev/ > temp/notPluggedIn.txt

# device plugged in
ls /dev/ > temp/pluggedIn.txt

# <port> is the entry that contains tty e.g.: tty.usbserial-DO02GBUB
diff notPluggedIn.txt pluggedIn.txt
```

- Connect to the EMG board:
```bash
python emgInterface.py -p /dev/<port>
```


### 1. Data Collection with `1_collect_data.py`

The `1_collect_data.py` script is used to capture a video and EMG data. It has several command line arguments that you can
use to customize its behavior. To use it first connect the EMG board (see the 0.1 EMG Board Setup)

- `--data_dir`: This argument is required. It specifies the output directory where the video and EMG data will be saved.

- `--dummy_emg`: This argument is optional. If used, the script will generate dummy EMG data for testing purposes
  without the need for an EMG board.


Here is an example of how to run the script:

```bash
python 1_collect_data.py --data_dir data/joris/test --dummy_emg 
```

### 2. Data Preprocessing with `preprocess_data.py`

The `preprocess_data.py` script is used to first display a frame of the video and then crop it in width/height and time.
EMG data is adapted accordingly.

- `--data_dir`: This argument is required. It specifies the directory where the collected data is stored.

- `--experiment_name`: This argument is required. It specifies the name of the experiment
  to be used for saving the processed data.

- `--x_start`, `--x_end`, `--y_start`, `--y_end`: These arguments are optional with a default value of 0 and -1. They specify
  the coordinates for cropping the video when the `--process` argument is used.

- `--start_frame`, `--end_frame`: These arguments are optional with default values of 0 and -1 respectively. They
  specify the start and end frames for cropping the video when the `--process` argument is used.

- `--trigger_channel`: This argument is required. It specifies the channel number of the trigger signal in the EMG data.

- `--trigger_value`: This argument is required. It specifies the trigger value in the EMG data. Trigger is detected when
  the absolute value signal crosses the trigger_value.

- `--process`: This argument is optional. If used, the script will process the video. If not used, the script will show the video and the EMG trigger channel

Here is an example of how to run the script for visualization:

```bash
python preprocess_data.py --data_dir data/joris/trigger_2 --experiment_name 1 --trigger_channel 7 
```

And here is an example of how to run the script for processing:

```bash
python preprocess_data.py --data_dir data/joris/trigger_2 --experiment_name 1 --start_frame 544 --end_frame 14246 --trigger_channel 7 --trigger_value 600 --process
```

### 3. Video Processing with `process_video.py`

The `process_video.py` script is used to process the video data. It first gets the 2D poses, then raises them to 3D, and
finally computes the angles and saves them to a dataframe. It has several command line arguments that you can use to
customize its behavior:


- `--data_dir`: This argument is required. It specifies the directory where the video data is stored.

- `--experiment_name`: This argument is required. It specifies the name of the experiment to be used for saving the
  processed data.

- `--visualize`: This argument is optional with a default value of True. If used, the script will visualize the output.

- `--intact_hand`: This argument is optional, either Right or Left. It specifies the hand that is intact. If not used, the script will run for both hands.

- `--hand_roi_size`: This argument is optional with a default value of 800. It specifies the size of the region of interest
  around the hand.

- `--plane_frames_start`, `--plane_frames_end`: These arguments are optional with default values of 0 and 20 respectively. They define the range of frames that are used to get the arm lengths in the 0 plane.

- `--video_start`, `--video_end`: This argument is optional with a default value of -1. It specifies the end frame of the video, used to produce "cropped_" files

- `--process`: This argument is optional. If used, the script will process the video. If not used, the script will only
  display the video and a sample of hand ROIs.


First you can run the script without the `--process` argument to determine the hand ROI size and the plane frames:
```bash
python process_video.py --data_dir data/joris/test --experiment_name 1
```

Then you can run the script with the `--process` argument to process the video:
```bash
python process_video.py --data_dir data/joris/test --experiment_name 1 --visualize --intact_hand Right --hand_roi_size 400 --plane_frames_start 0 --plane_frames_end 40 --process
```

### 4. Training with `train.py`

### 5. Predict with `predict.py`
- Connect to the EMG board (see the 0.1. EMG Board Setup)
- Connect to the hand:
```bash
python psyonicHand.py -e
```
- Type `move` to start the hand.







## Data Output

- aligned_emg.npy: EMG data aligned with the video data.
- angles.parquet: Angles computed from the video data. To access individual angles:

```python
import pandas as pd
from os.path import join
data_dir = 'data/joris/test'
angles = pd.read_parquet(join(data_dir, 'angles.parquet'))
# To get the index angel for the left hand in the first frame:
left_hand_index_angle = angles.loc[0, ('Left', 'indexAng')]
```

### Angles Explanation
Available angles: `['indexAng', 'midAng', 'ringAng', 'pinkyAng', 'thumbInPlaneAng', 'thumbOutPlaneAng', 'elbowAngle', 'wristRot', 'wristFlex']`

- `indexAng`: This is the maximum angle between the Index Finger's Proximal Interphalangeal (PIP) joint and Metacarpophalangeal (MCP) joint, and the line between the MCP joint and the wrist. The angle is 0 when the finger is fully extended and increases as the finger is flexed.

- `midAng`: This is the maximum angle between the Middle Finger's PIP joint and MCP joint, and the line between the MCP joint and the wrist. The angle is 0 when the finger is fully extended and increases as the finger is flexed.

- `ringAng`: This is the maximum angle between the Ring Finger's PIP joint and MCP joint, and the line between the MCP joint and the wrist. The angle is 0 when the finger is fully extended and increases as the finger is flexed.

- `pinkyAng`: This is the maximum angle between the Pinky Finger's PIP joint and MCP joint, and the line between the MCP joint and the wrist. The angle is 0 when the finger is fully extended and increases as the finger is flexed.

- `thumbInPlaneAng`: This is the angle between the thumb and the line between the pinky and index fingers in the plane of the palm. The angle is 0 when the thumb is parallel to the line between the pinky and index fingers, and increases when the thumb is pulled towards the pinky in the palm plane.

- `thumbOutPlaneAng`: This is the angle between the thumb and the line between the pinky and index fingers in the plane orthogonal to the palm. The angle is 0 when the thumb is parallel to the line between the pinky and index fingers, and increases when the thumb is pulled towards the pinky in the plane orthogonal to the palm.

- `elbowAngle`: This is the angle between the lower arm and the upper arm at the elbow joint. The angle is 0 when the elbow is fully flexed and increases as the elbow is extended.

- `wristRot`: This is the angle of rotation of the wrist. For the right hand, the angle is 0 when the palm points inwards and increases when the hand turns counterclockwise when viewed from outside. For the left hand, the angle is 0 when the palm points inwards and increases when the hand turns clockwise when viewed from outside.

- `wristFlex`: This is the angle of flexion of the wrist. The angle is 0 when the palm is parallel to the lower arm, increases when the wrist is extended, and decreases when it is flexed.