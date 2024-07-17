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

5. Install the local `dynamics` package:

```bash
pip install -e dynamics
```

## Scripts

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
   python 4_train.py --person_dir A_1 --intact_hand Left --config_name denseNet --visualize
   ```

2. **Hyperparameter Search**

   To perform a hyperparameter search, use the `--hyperparameter_search` argument. This will run a sweep using the configuration specified in the YAML file.
   ```bash
   python 4_train.py --person_dir A_1 --intact_hand Left --config_name denseNet --hyperparameter_search
   ```

3. **Testing the Model**

   To test the model, use the --test argument. This will train the model on the training set and evaluate it on the test set.   
   ```bash
   python 4_train.py --person_dir A_1 --intact_hand Left --config_name denseNet --test
   ```
   
4. **Saving the Model**
   To save the trained model, use the --save_model argument. This will save the model to the specified directory.
   ```bash
   python 4_train.py --person_dir A_1 --intact_hand Left --config_name denseNet --save_model
   ```

### 6. Visualize the model output on the test data with `6_visualize_results.py`
To visualize the model output on the test data, use the `6_visualize_results.py` script. It will start a pybullet simulation and show the model output on the test data.
These are the command line arguments that you can use to customize the behavior of the script:

- `--data_dir`: This argument is required. It specifies the directory where the video data is stored.
- `--experiment_name`: This argument is required. It specifies the name of the experiment to be used for saving the processed data.
- `--intact_hand`: This argument is optional, either Right or Left. It specifies the hand that is intact (Left/Right).
- `--config_name`: This argument is optional. It specifies the model name to be used for inference.
- `--video`: This argument is optional. If used, the script will display the video.

Example:
```bash
python 6_visualize_results.py --data_dir data/A_1/recordings/minJerk/pinchCloseOpen --experiment_name 1 --intact_hand Left --config_name denseNet
```

