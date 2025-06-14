import math
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# save as parquet file
import pandas as pd
import torch
import wget
import yaml
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from helpers.BesselFilter import BesselFilterArr
from helpers.EMGClass import EMG

DATA_DIR = Path(__file__).resolve().parent / "data"
SUBSAMPLERATE_1 = 2  # downsample from 2000 Hz to 1000 Hz
SUBSAMPLERATE_2 = 17  # downsample from 1000 Hz to 60 Hz
EMG_SCALER = 0.0004


def download_datasets(subject: int, time: int) -> None:
    """
    Download the NINAPRO DB8 dataset.

    Args:
        subject (int): Subject number (0-11).
        time (int): Time point (0-2).
    """
    name = f"S{subject + 1:.0f}_E1_A{time + 1:.0f}.mat"
    url = "http://ninapro.hevs.ch/files/DB8/" + name
    save_dir = DATA_DIR / f"subject_{subject}" / "mat_files"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_file = save_dir / f"time_{time}.mat"
    if not save_file.exists():
        wget.download(url, out=str(save_file))


def transform_data(save_path: Path) -> None:
    """Transform the raw data from the NINAPRO DB8 dataset into a usable format.
    Args:
        save_path (Path): Path to the directory where the transformed data will be saved.
    """
    if (save_path / "emg.npy").exists() and (save_path / "angles.npy").exists():
        return
    print(f"Processing subject {subject}, time {time}...")
    signal = torch.HalfTensor(
        loadmat(DATA_DIR / f"subject_{subject}" / "mat_files" / f"time_{time}.mat")[
            "emg"
        ]
    )  # input
    dof = torch.Tensor(
        loadmat(DATA_DIR / f"subject_{subject}" / "mat_files" / f"time_{time}.mat")[
            "glove"
        ]
    )  # target
    doa = dof_to_doa(dof)

    signal = np.array(signal)
    doa = np.array(doa)

    save_path.mkdir(parents=True, exist_ok=True)
    np.save(save_path / "emg.npy", signal)
    np.save(save_path / "angles.npy", doa)


def dof_to_doa(x: torch.Tensor) -> torch.Tensor:
    """Convert Degrees of Freedom (DoF) to Degrees of Articulation (DoA)
    Args:
        x (torch.Tensor): Input tensor of shape (Any, 18).

    Returns:
        torch.Tensor: Transformed tensor of shape (Any, 5).
    """
    linear_transformation_matrix = torch.Tensor(
        [
            [0.639, 0.000, 0.000, 0.000, 0.000],
            [0.383, 0.000, 0.000, 0.000, 0.000],
            [0.000, 1.000, 0.000, 0.000, 0.000],
            [-0.639, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.400, 0.000, 0.000],
            [0.000, 0.000, 0.600, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.400, 0.000],
            [0.000, 0.000, 0.000, 0.600, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.1667],
            [0.000, 0.000, 0.000, 0.000, 0.3333],
            [0.000, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.1667],
            [0.000, 0.000, 0.000, 0.000, 0.3333],
            [0.000, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.000],
            [-0.19, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.000],
        ]
    )
    """
    Matrix found in
    "Agamemnon Krasoulis, Sethu Vijayakumar, and Kianoush Nazarpour. Effect of user practice
    on prosthetic finger control with an intuitive myoelectric decoder. 13. ISSN 1662-453X. URL
    https://www.frontiersin.org/articles/10.3389/fnins.2019.00891."
    """
    linear_transformation_matrix = linear_transformation_matrix[0:-1].transpose(0, 1)
    linear_transformation_matrix = linear_transformation_matrix.unsqueeze(0)
    x = x.unsqueeze(2).float()
    y = linear_transformation_matrix @ x
    return y[:, :, 0].half()


def filter_emg_like_real(data_dir: Path) -> None:
    """Filter the EMG data as if it was recorded in real-time
    Args:
        data_dir (Path): Path to the directory containing the EMG data.
    """
    emg_data = np.load(data_dir / "emg.npy").T

    maxVals = np.array([0.001 for i in range(16)])  # TODO(JG): determine scalers
    noiseLevel = np.array([0 for i in range(16)])

    emg = EMG(
        samplingFreq=2000, offlineData=emg_data, maxVals=maxVals, noiseLevel=noiseLevel
    )
    emg.startCommunication()
    emg.emgThread.join()
    normalized_emg = emg.emgHistory[:, emg.numPackets * 100 + 1 :]
    normalized_emg = normalized_emg[:, : emg_data.shape[1]]

    np.save(data_dir / "cropped_emg.npy", normalized_emg.T)


def filter_emg(data_dir: Path) -> None:
    """Filter the EMG data
    Args:
        data_dir (Path): Path to the directory containing the EMG data.
    """
    num_electrodes = 16
    sampling_freq = 1000  # (downsampled from 2000 Hz)
    powerLineFilterArray = BesselFilterArr(
        numChannels=num_electrodes,
        order=8,
        critFreqs=[58, 62],
        fs=sampling_freq,
        filtType="bandstop",
    )  # remove power line noise and multiples up to 600 Hz
    highPassFilters = BesselFilterArr(
        numChannels=num_electrodes,
        order=4,
        critFreqs=20,
        fs=sampling_freq,
        filtType="highpass",
    )  # high pass removes motion artifacts and drift
    lowPassFilters = BesselFilterArr(
        numChannels=num_electrodes,
        order=4,
        critFreqs=3,
        fs=sampling_freq,
        filtType="lowpass",
    )  # smooth the envelope, when not using 'actually' integrated EMG

    emg = np.load(data_dir / "emg.npy").astype(
        np.float32
    )  # shape: (num_samples, num_channels)
    emg = emg[::SUBSAMPLERATE_1]  # downsample to 1000 Hz
    # zero pad the first 3000 samples to remove the initial noise
    emg = np.pad(emg, ((3000, 0), (0, 0)), mode="constant", constant_values=0)
    emg = emg.T  # shape: (num_channels, num_samples)

    maxVals = np.array([0.001 for i in range(16)])  # TODO(JG): determine scalers
    noiseLevel = np.array([0 for i in range(16)])

    emg = powerLineFilterArray.filter(emg)
    emg = highPassFilters.filter(emg)

    emg = np.abs(emg)
    # emg = np.clip(emg - noiseLevel[:, None], 0, None)
    emg = np.clip(lowPassFilters.filter(emg), 0, None)
    # emg = emg / maxVals[:, None]  # normalize to 1
    # emg = np.clip(emg, 0, 1)
    emg = emg.T  # shape: (num_samples, num_channels)
    # remove the first 3000 samples again
    emg = emg[3000:, :]
    # downsample to 60 Hz
    emg = emg[
        ::SUBSAMPLERATE_2, :
    ]  # 1000 Hz / 60 Hz = 16.67, so we take every 17th sample
    # from matplotlib import pyplot as plt

    # plt.plot(emg)
    # plt.show()
    emg = emg / EMG_SCALER
    np.save(data_dir / "cropped_emg.npy", emg)


def process_angles(data_dir: Path) -> None:
    """Process the angles data
    Args:
        data_dir (Path): Path to the directory containing the angles data.
    """
    angles = np.load(data_dir / "angles.npy").astype(
        np.float32
    )  # shape: (num_samples, 18)

    # downsample to 60 Hz
    angles = angles[::SUBSAMPLERATE_1]  # downsample from 1000 Hz to 60 Hz
    angles = angles[::SUBSAMPLERATE_2]

    df = pd.DataFrame(angles)
    # y1, thumb rotation; y2, thumb flexion; y3, index flexion; y4, middle flexion; y5, ring/little flexion.
    df.columns = pd.MultiIndex.from_tuples(
        [
            ("Left", "thumb_rot"),
            ("Left", "thumb_flex"),
            ("Left", "index_flex"),
            ("Left", "middle_flex"),
            ("Left", "ring_little_flex"),
        ]
    )
    # convert to radians
    df = df * np.pi / 180.0
    df.to_parquet(data_dir / "cropped_angles.parquet")

    smoothed_df = df.copy()
    for column in df.columns:
        smoothed_df[column] = gaussian_filter1d(df[column], sigma=1.5, radius=2)

    smoothed_df.to_parquet(data_dir / "cropped_smooth_angles.parquet")


if __name__ == "__main__":
    for subject in tqdm(range(12), desc="Subjects"):
        for time in tqdm(range(3), desc=f"Subject {subject} Times", leave=False):
            save_path = (
                DATA_DIR
                / f"subject_{subject}"
                / "recordings"
                / f"time_{time}"
                / "experiments"
                / "1"
            )
            download_datasets(subject, time)
            transform_data(save_path)
            filter_emg(save_path)
            process_angles(save_path)
