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

3. Install PyTorch based on your system: [PyTorch website](https://pytorch.org/)

4. Install the remaining dependencies from the requirements.txt file:

```bash
pip install -r requirements.txt
```

5. Install the local 'dynamics' package:

```bash
pip install -e dynamics
```

## Usage Overview

The pipeline enables EMG- and force-aware control of the PSYONIC Ability Hand prosthesis.

The workflow consists of:
- Patient folder setup
- Connecting the EMG board
- Data acquisition (calibration, free-space, interaction)
- Preprocessing
- Model training
- Inference / real-time control

## 1. Patient Folder Setup

1. Copy an existing `<person_id>` folder from `data/` as a template.
2. This folder contains a `configs/` directory with:
   - `modular_fs.yaml` (free-space controller config)
   - `modular_inter.yaml` (interaction controller config)
3. Rename the folder to the patient's name, pseudonym, or ID.
4. These config files will be automatically updated during preprocessing.

## 2. Connecting the EMG Board

Two options exist:

**With GUI (Linux/macOS only):** Use the external Biomech_EMG_USB tool (not included in this repo). This shows live EMG readings and helps verify connection quality.

**Without GUI (cross-platform, including Windows):** Run:

```bash
python s0_emgInterface.py
```

Check that the printed timestamps are chronological. If they jump around, the connection is unstable and the EMG board must be restarted.

In the GUI, misaligned timestamps also indicate lost connection.

## 3. Data Acquisition

### Calibration Data

1. Run the calibration script:

```bash
python s1.5_collect_calib_data.py --person_id <person_id> --movement Calibration --hand_side <Left/Right> --calibrate_emg
```

- **Step 1:** Relaxed baseline EMG for 10s (noise calibration).
- **Step 2:** Maximal voluntary contraction (MVC) for 10s (muscle activation range).
- Results are stored in `recordings/calibration/`.

2. Next, record free-space hand poses:

```bash
python s1.5_collect_calib_data.py --person_id <person_id> --movement <hand_pose> --hand_side <Left/Right>
```

**`<hand_pose>` options:**
- `indexFlEx`
- `mrpFlEx`
- `fingersFlEx`
- `handClOp`
- `thumbAbAd`
- `thumbFlEx`
- `indexDigitsFlEx`
- `pinchClOp`

During the sync iterations (not stored), the patient aligns with the prosthesis trajectory.
During rec iterations, data is recorded and stored.

### Interaction Data

Run:

```bash
python s1.5_collect_force_data.py --person_id <person_id> --grip <grip_type> --hand <Left/Right> --gui
```

**`<grip_type>` options:**
- `hook`
- `power_grip`
- `tripod`
- `pinch`

**Default:** 60s trial, PID update frequency 100–200 Hz.

**Workflow:**
1. Hand slowly closes into grip (approach phase).
2. Place rigid/soft object into hand.
3. When contact is detected, the PID controller follows the target force trajectory.
4. Patient modulates grip force via muscle co-contraction, following the moving "tail" in the GUI.

Interaction trials are saved with `_interaction` suffix in the movement name.

## 4. Preprocessing

After acquisition, process all recordings:

```bash
python s2.5_process_all_data.py --person_id <person_id> --hand_side <Left/Right>
```

Optionally restrict to one movement:

```bash
python s2.5_process_all_data.py --person_id <person_id> --hand_side <Left/Right> --movement indexFlEx
```

This step:
- Filters and aligns EMG, force, and kinematics
- Updates `modular_fs.yaml` and `modular_inter.yaml` with used EMG channels, movements, etc.
- Config files can be edited manually to adjust model architecture.

## 5. Training

Train models using:

```bash
python s4_train.py --person_dir <person_id> --intact_hand <Left/Right> --controller_mode <free_space|interaction|both> -t -s
```

Use `-v` instead of `-t -s` to only visualize data.

**Modes:**
- `--controller_mode free_space` → train free-space only
- `--controller_mode interaction` → train interaction only
- `--controller_mode both` → train both models

Models are saved under `data/<person_id>/models/`.

## 6. Inference (Real-Time Control)

Run the online controller:

```bash
python s5_inference.py -e --person_dir <person_id> \
    --free_space_model_name <free_space_model_name> \
    --interaction_model_name <interaction_model_name>
```

- Requires trained models from step 5.
- Uses EMG + force feedback in real-time to control the PSYONIC Ability Hand.

## Scripts

- `s0_emgInterface.py` — connect to EMG board (with/without GUI).
- `s0_emgFilter.py` — filter, rectify, normalize EMG stream.
- `s1.5_collect_calib_data.py` — collect calibration + free-space trajectories.
- `s1.5_collect_force_data.py` — collect interaction force trials with PID + GUI.
- `s2.5_process_all_data.py` — preprocess and align all data, update configs.
- `s4_train.py` — train free-space/interaction/both models.
- `s5_inference.py` — run trained models for real-time control.
- `psyonicControllers.py` — controller class for running trained models on the prosthesis.
- `EMGClass.py / BCI_Data_Receiver.py` — EMG acquisition and streaming utilities.
- `predict_utils.py / models.py` — training, datasets, neural network architectures.
