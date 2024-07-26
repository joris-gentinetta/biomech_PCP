# Upper Extremity Prosthesis Control Pipeline

## Installation

Follow the steps below to set up your environment:

1. Create a new conda environment with Python 3.8:

```bash
conda create -n PCP python=3.8
```

2. Activate the newly created environment:

```bash
conda activate PCP
```

3. Install PyTorch based on your system: [PyTorch website](https://pytorch.org/get-started/locally/)

4. Install the remaining dependencies from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

5. Install the local 'dynamics' package:

```bash
pip install -e dynamics
```

## Scripts

### 0.1. EMG Board Setup
- Find the port of the emg board:
```bash
# create a temp directory
mkdir temp

# emg board not plugged in:
ls /dev/ > temp/notPluggedIn.txt

# emg board plugged in:
ls /dev/ > temp/pluggedIn.txt

# <port> is the entry that contains tty e.g.: tty.usbserial-DO02GBUB
diff temp/notPluggedIn.txt temp/pluggedIn.txt

# remove the temp directory
rm -r temp
```

- Connect to the EMG board:
```bash
python 0_emgInterface.py -p /dev/<port>
```
e.g.:
```bash
python 0_emgInterface.py -p /dev/tty.usbserial-DO02GBUB
```


### 1. Data Collection with `1_collect_data.py`

The `1_collect_data.py` script is used to capture EMG data. To use it first connect the EMG board (see 0.1 EMG Board Setup)

- `--data_dir`: This argument is required. It specifies the output directory where the EMG data will be saved.

- `--dummy_emg`: This argument is optional. If used, the script will generate dummy EMG data for testing purposes
  without the need for an EMG board.


Here is an example of how to run the script:

```bash
python 1_collect_data.py --data_dir data/test_person/test_data --dummy_emg 
```

Put the recorded video in the data directory that you specified. The file name of the video is not important, but it should be in the .mp4 format.

### 2. Data Preprocessing with `2_preprocess_data.py`

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

First inspect the Trigger channel and the video to determine the parameters:

```bash
python 2_preprocess_data.py --data_dir data/test_person/test_recording --experiment_name 1 --trigger_channel 15
```

Then run the script with the `--process` argument to process the data:

```bash
python 2_preprocess_data.py --data_dir data/test_person/test_recording --experiment_name 1 --start_frame 400 --end_frame 1000 --trigger_channel 15 --trigger_value 600 --process
```

### 3. Video Processing with `3_process_video.py`

The `3_process_video.py` script is used to process the video data. It first gets the 2D poses, then raises them to 3D, and
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
python 3_process_video.py --data_dir data/test_person/test_recording --experiment_name 1
```

Then you can run the script with the `--process` argument to process the video:
```bash
python 3_process_video.py --data_dir data/test_person/test_recording --experiment_name 1 --visualize --intact_hand Right --hand_roi_size 400 --plane_frames_start 0 --plane_frames_end 40 --process
```

### 4. Training with `4_train.py`

The `4_train.py` script is used for training a time-series model on EMG data, predicting joint angles. It supports data exploration, hyperparameter search, testing, and saving the trained model. Below are the command-line arguments and usage examples.

#### Command Line Arguments

- `--person_dir`: (Required) Specifies the directory of the person's data.
- `--intact_hand`: (Required) Specifies the intact hand (Right/Left).
- `--config_name`: (Required) The name of the training configuration file (YAML format).
- `-v, --visualize`: (Optional) If used, plots data exploration results.
- `-hs, --hyperparameter_search`: (Optional) If used, performs a hyperparameter search.
- `-t, --test`: (Optional) If used, tests the model.
- `-s, --save_model`: (Optional) If used, saves the trained model.

#### Usage

1. **Data Exploration**

   To visualize the data for exploration, use the `--visualize` argument. This will plot the features and targets for the specified person.

   ```bash
   python 4_train.py --person_dir A_1 --intact_hand Left --config_name simple_GRU --visualize
   ```

2. **Hyperparameter Search**

   To perform a hyperparameter search, use the `--hyperparameter_search` argument. This will run a sweep using the configuration specified in the YAML file.
   ```bash
   python 4_train.py --person_dir A_1 --intact_hand Left --config_name simple_GRU --hyperparameter_search
   ```

3. **Testing the Model**

   To test the model, use the --test argument. This will train the model on the training set and evaluate it on the test set.   
   ```bash
   python 4_train.py --person_dir A_1 --intact_hand Left --config_name simple_GRU --test
   ```
   
4. **Saving the Model**
   To save the trained model, use the --save_model argument. This will save the model to the specified directory.
   ```bash
   python 4_train.py --person_dir A_1 --intact_hand Left --config_name simple_GRU --save_model
   ```


### 5. Control the Prosthetic Hand
- Connect to the EMG board (see the 0.1. EMG Board Setup)
- Connect to the hand:
```bash
python 5_inference.py -e --person_dir A_1 --config_name simple_GRU
```
- Type `move` to start the hand.


### 6. Visualize the model output on the test data with `6_visualize_results.py`
To visualize the model output on the test data, use the `6_visualize_results.py` script. It will start a pybullet simulation and show the model output on the test data.
These are the command line arguments that you can use to customize the behavior of the script:

- `--data_dir`: This argument is required. It specifies the directory where the video data is stored.
- `--experiment_name`: This argument is required. It specifies the name of the experiment to be used for saving the processed data.
- `--intact_hand`: This argument is optional, either Right or Left. It specifies the hand that is intact (Left/Right).
- `--model_name`: This argument is optional. It specifies the model name to be used for inference.
- `--video`: This argument is optional. If used, the script will display the video.

Example:
```bash
python 6_visualize_results.py --data_dir data/A_1/recordings/minJerk/pinchCloseOpen --experiment_name 1 --intact_hand Left --model_name simple_GRU
```

## Data Output

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