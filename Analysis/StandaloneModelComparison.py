"""
compare_emg_vs_force.py

Standalone comparison script for EMG-only vs EMG+Force models.
Trains two models with identical architectures and hyperparameters,
differing only in input features (EMG vs EMG+Force).

NEW: Added --test_trial option with "all" for leave-one-out cross-validation

Usage:
python compare_emg_vs_force.py \
  --data_root data/Person01 \
  --recordings_yaml configs/recordings_p01.yaml \
  --out_dir results/p01_emg_vs_force \
  --intact_hand Left \
  --test_trial all \
  --model_type DenseNet \
  --hidden_size 128 \
  --n_layers 3 \
  --seq_len 20 \
  --batch_size 64 \
  --n_epochs 80 \
  --learning_rate 1e-3 \
  --early_stopping_patience 10 \
  --early_stopping_delta 1e-4 \
  --wandb_mode disabled \
  --seed 42
"""

import argparse
import json
import math
import os
import random
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add project root to path 
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from helpers.predict_utils import get_data, rescale_data, train_model, Config, TSDataset, TSDataLoader
from helpers.models import TimeSeriesRegressorWrapper


# ---- hardcoded_config.py (inline at top of your script) ----
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class HardConfig:
    # ---- data / run ----
    data_root: str = "data/P5_869_interaction"
    out_dir: str   = "results/P5_869_interaction_emg_vs_force"
    intact_hand: str = "Left"
    device: str = "auto"
    seed: int = 42
    make_plots: bool = True
    test_trial: str = "pinch_interaction#2"  # NEW: which trial(s) to test on

    # Movement sets (names per your actual folder structure)
    free_space_movements: list = field(default_factory=lambda: [
        "indexFlEx", "mrpFlEx", "fingersFlEx", "handClOp", "thumbAbAd", "thumbFlEx", "pinchClOp"
    ])
    interaction_movements: list = field(default_factory=lambda: [
        "pinch_interaction", "tripod_interaction", "hook_interaction", "power_grip_interaction"
    ])

    # ---- model / training (will be overwritten by YAML load below) ----
    model_type: str = "ModularModel"
    batch_size: int = 8
    seq_len: int = 64
    warmup_steps: int = 1
    weight_decay: float = 1e-4
    early_stopping_patience: int = 20
    early_stopping_delta: float = 5e-3
    n_epochs: int = 14           # choose what you used in FS model
    learning_rate: float = 1e-3  # choose what you used in FS model
    wandb_mode: str = "disabled"
    wandb_project: str = "emg_vs_force_comparison"

    # ModularModel submodules (from your FS spec)
    activation_model: dict = field(default_factory=lambda: {
        "model_type": "GRU", "hidden_size": 32, "n_layers": 2, "tanh": True, "n_freeze_epochs": 0
    })
    muscle_model: dict = field(default_factory=lambda: {
        "model_type": "PhysMuscleModel", "n_freeze_epochs": 0
    })
    joint_model: dict = field(default_factory=lambda: {
        "model_type": "PhysJointModel", "n_freeze_epochs": 0, "speed_mode": False
    })

    # ---- features / targets (from FS spec) ----
    emg_features: list = field(default_factory=lambda: [
        ['emg','0'], ['emg','1'], ['emg','2'], ['emg','11'], ['emg','12'], ['emg','13'], ['emg','14'], ['emg','15']
    ])
    force_features: list = field(default_factory=list)  # filled by finalize()
    targets: list = field(default_factory=lambda: [
        ['Left','index_Pos'], ['Left','middle_Pos'], ['Left','ring_Pos'],
        ['Left','pinky_Pos'], ['Left','thumbFlex_Pos'], ['Left','thumbRot_Pos']
    ])

    def finalize(self):
        H = self.intact_hand.capitalize()
        if not self.force_features:
            self.force_features = [
                [H, 'index_Force'], [H, 'middle_Force'], [H, 'ring_Force'],
                [H, 'pinky_Force'], [H, 'thumb_Force'],
            ]
        return self

# --- Compatibility shim so all model.forward calls tolerate "states=..." ---


class _StatesKeywordShim(torch.nn.Module):
    """Wrap a model so forward(x, states=...) works even if the base
    forward doesn't accept the 'states' keyword. Also proxy attrs like .device.
    """
    def __init__(self, base):
        super().__init__()
        self.base = base  # register as submodule so .to(), .eval(), etc. work

    @property
    def device(self):
        # Prefer base.device; otherwise infer from parameters
        if hasattr(self.base, "device"):
            return self.base.device
        try:
            return next(self.base.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def forward(self, x, *args, **kwargs):
        # Accept states=... as keyword; pass positionally to be universal
        states = kwargs.pop("states", None)
        try:
            return self.base(x, states)
        except TypeError:
            # Base may actually accept 'states' as kwarg; try that
            return self.base(x, *args, states=states, **kwargs)
        except Exception:
            # Last resort: ignore 'states' entirely
            return self.base(x, *args, **kwargs)

    def __getattr__(self, name):
        # Delegate everything not found on this shim to the base model
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base, name)

# Monkey-patch the wrapper so every created model is wrapped with the shim
__orig_tsrw_init = TimeSeriesRegressorWrapper.__init__

def __patched_tsrw_init(self, *args, **kwargs):
    __orig_tsrw_init(self, *args, **kwargs)
    # Wrap underlying model once
    try:
        self.model = _StatesKeywordShim(self.model)
    except Exception:
        pass

TimeSeriesRegressorWrapper.__init__ = __patched_tsrw_init




def parse_args():
    parser = argparse.ArgumentParser(description='Compare EMG-only vs EMG+Force models')
    
    # Data and setup
    # parser.add_argument('--data_root', type=str, default='data/patient1', required=True, help='Root data directory (e.g., data/Person01)')
    # parser.add_argument('--recordings_yaml', type=str, required=False, help='YAML file listing recordings to use')
    # parser.add_argument('--out_dir', type=str, required=True, help='Output directory for results')
    # parser.add_argument('--intact_hand', type=str, default='Right', required=True, choices=['Left', 'Right'], help='Intact hand side')
    parser.add_argument('--test_trial', type=str, default='pinch_interaction#2', help='Trial to test on, or "all" for leave-one-out')
    
    # Model architecture
    parser.add_argument('--model_type', type=str, default='GRU', help='Model type (DenseNet, GRU, etc.)')
    parser.add_argument('--hidden_size', type=int, default=32, help='Hidden layer size')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--seq_len', type=int, default=64, help='Sequence length')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--n_epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=3, help='Warmup steps')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--early_stopping_delta', type=float, default=1e-4, help='Early stopping delta')
    
    # Misc
    parser.add_argument('--wandb_mode', type=str, default='disabled', choices=['online', 'offline', 'disabled'], help='WandB mode')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--make_plots', choices=['true', 'false'], default='true', help='Generate plots')
    parser.add_argument('--device', type=str, default='auto', help='Device (auto, cpu, cuda)')
    
    return parser.parse_args()

def _as_tuple_str(x):
    """Return the canonical string form of a 2-tuple, e.g. ('Left','index_Force')."""
    if isinstance(x, (list, tuple)) and len(x) == 2:
        return f"('{x[0]}', '{x[1]}')"
    return str(x)

def normalize_cols_to_df(cols, df_cols):
    """
    Given a list of feature/target names (tuples/lists/strings) and the actual df columns,
    return the same names converted to the exact type/representation used by df.columns.
    """
    df_cols_set = set(df_cols)
    # Case 1: df uses real tuples
    if all(isinstance(c, tuple) for c in df_cols):
        out = []
        for c in cols:
            if isinstance(c, tuple):
                out.append(c)
            elif isinstance(c, list):
                out.append(tuple(c))
            else:  # string -> try to parse minimal ('Left','index_Force') form
                # Very conservative fallback: leave as-is; user data shouldn't need this path
                out.append(c)
        return out
    # Case 2: df uses strings (should be rare in your case)
    df_cols_str_set = set(map(str, df_cols))
    out = []
    for c in cols:
        s = _as_tuple_str(c)
        if s in df_cols_str_set:
            out.append(s)
        else:
            # Also try plain str(c) as a fallback
            s2 = str(c)
            out.append(s2 if s2 in df_cols_str_set else s)
    return out

def col_in_df(col, df_cols):
    """Robust membership check for tuple-or-string column labels."""
    if col in df_cols:
        return True
    col_str = _as_tuple_str(col)
    return col_str in set(map(str, df_cols))

def _to_hashable(cols):
    out = []
    for c in cols:
        if isinstance(c, list):
            out.append(tuple(c))
        else:
            out.append(c)
    return out

def set_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"✅ Set all seeds to {seed}")


def infer_emg_channels(trial_dir: str) -> int:
    """Infer number of EMG channels from data file."""
    emg_path = os.path.join(trial_dir, 'aligned_filtered_emg.npy')
    if not os.path.exists(emg_path):
        raise FileNotFoundError(f"EMG data not found: {emg_path}")
    
    emg = np.load(emg_path)
    n_channels = emg.shape[1] if emg.ndim > 1 else 1
    print(f"✅ Detected {n_channels} EMG channels from {trial_dir}")
    return n_channels


def verify_trial_data(trial_dir: str) -> bool:
    """Verify that a trial directory has all required data files."""
    required_files = ['aligned_angles.parquet', 'aligned_filtered_emg.npy', 'aligned_timestamps.npy']
    missing = []
    
    for filename in required_files:
        filepath = os.path.join(trial_dir, filename)
        if not os.path.exists(filepath):
            missing.append(filename)
    
    if missing:
        print(f"❌ Missing files in {trial_dir}: {missing}")
        return False
    
    return True


def drop_constant_columns(df: pd.DataFrame, cols: List) -> List:
    """Drop constant/zero-variance columns from feature list."""
    keep = []
    for c in cols:
        if c not in df.columns:
            print(f"ℹ️  Column not found, skipping: {c}")
            continue
            
        values = df[c].values
        if np.nanstd(values) > 1e-8:
            keep.append(c)
        else:
            print(f"ℹ️  Dropping constant feature: {c} (std={np.nanstd(values):.2e})")
    
    return keep

def discover_features_and_targets(sample_parquet_path: str, intact_hand: str, n_emg_channels: int) -> Tuple[List, List, List]:
    """
    Inspect a sample parquet file to discover available features and targets.
    
    Returns:
        emg_features: List of EMG feature column names
        force_features: List of force feature column names  
        targets: List of target column names
    """
    df = pd.read_parquet(sample_parquet_path)
    
    print(f"Sample parquet columns ({len(df.columns)} total):")
    for i, col in enumerate(df.columns):
        print(f"  {i:2d}: {col}")
    
    hand_cap = intact_hand.capitalize()
    
    # EMG features - use actual channel count
    emg_features = [('emg', str(i)) for i in range(n_emg_channels)]
    
    # Force features
    force_features = [
        (hand_cap, 'index_Force'),
        (hand_cap, 'middle_Force'), 
        (hand_cap, 'ring_Force'),
        (hand_cap, 'pinky_Force'),
        (hand_cap, 'thumb_Force'),
    ]
    
    # Position targets
    targets = [
        (hand_cap, 'index_Pos'),
        (hand_cap, 'middle_Pos'),
        (hand_cap, 'ring_Pos'), 
        (hand_cap, 'pinky_Pos'),
        (hand_cap, 'thumbFlex_Pos'),
        (hand_cap, 'thumbRot_Pos'),
    ]
    
    # Verify targets exist (required)
    missing_targets = [t for t in targets if t not in df.columns]
    if missing_targets:
        print(f"❌ Missing target features: {missing_targets}")
        raise ValueError("Required targets not found in data")
    
    # Check force features (optional - filter out missing ones)
    missing_force = [f for f in force_features if f not in df.columns]
    if missing_force:
        print(f"⚠️  Missing force features: {missing_force}")
        print("→ Proceeding without missing force columns for EMG+Force model.")
        force_features = [f for f in force_features if f not in missing_force]
        
        if len(force_features) == 0:
            print("⚠️  No force features available - EMG+Force model will be identical to EMG-only")
    
    print(f"✅ EMG features: {len(emg_features)} channels")
    print(f"✅ Force features: {len(force_features)} fingers")
    print(f"✅ Targets: {len(targets)} joints")
    
    return emg_features, force_features, targets

def build_config(cfg: HardConfig, features: list, targets: list, name: str, num_train_dirs: int = 1, num_test_dirs: int = 1) -> Config:
    # Create enough dummy recording names to match val_losses + test_losses length
    # The evaluate_model function returns val_losses + test_losses, so we need enough names
    train_recordings = [f"fold_train_{i}" for i in range(num_train_dirs)]
    test_recordings = [f"fold_test_{i}" for i in range(num_test_dirs)]
    
    d = {
        "name": name,
        "parameters": {
            "model": {
                "features": features,
                "targets": targets,
                "model_type": cfg.model_type,
                "seq_len": cfg.seq_len,
                "batch_size": cfg.batch_size,
                "n_epochs": cfg.n_epochs,
                "learning_rate": cfg.learning_rate,
                "weight_decay": cfg.weight_decay,
                "warmup_steps": cfg.warmup_steps,
                "early_stopping_patience": cfg.early_stopping_patience,
                "early_stopping_delta": cfg.early_stopping_delta,
                "wandb_mode": cfg.wandb_mode,
                "wandb_project": cfg.wandb_project,
                "recordings": train_recordings,
                "test_recordings": test_recordings,

                # ModularModel specifics — this keeps your FS architecture
                "activation_model": cfg.activation_model,
                "muscle_model": cfg.muscle_model,
                "joint_model": cfg.joint_model,
            }
        }
    }
    config = Config(d)
    
    # CRITICAL FIX: Add features and targets as direct attributes
    config.features = features
    config.targets = targets
    
    # Also add other commonly accessed attributes at the top level
    config.seq_len = cfg.seq_len
    config.batch_size = cfg.batch_size
    config.n_epochs = cfg.n_epochs
    config.learning_rate = cfg.learning_rate
    config.weight_decay = cfg.weight_decay
    config.warmup_steps = cfg.warmup_steps
    config.early_stopping_patience = cfg.early_stopping_patience
    config.early_stopping_delta = cfg.early_stopping_delta
    config.wandb_mode = cfg.wandb_mode
    config.wandb_project = cfg.wandb_project
    config.model_type = cfg.model_type
    config.recordings = train_recordings
    config.test_recordings = test_recordings
    
    # ModularModel attributes
    config.activation_model = cfg.activation_model
    config.muscle_model = cfg.muscle_model
    config.joint_model = cfg.joint_model
    
    return config

def enumerate_trial_units(cfg: HardConfig):
    """
    Returns a list of trial units:
      [{"trial_id": "<movement>#<trial>", "dir": "<abs_path>", "group":"free_space|interaction", "movement":"<movement>"}]
    """
    units = []

    # Free-space: exactly trial "1"
    for mv in cfg.free_space_movements:
        tdir = os.path.join(cfg.data_root, "recordings", mv, "experiments", "1")
        if verify_trial_data(tdir):
            units.append({"trial_id": f"{mv}#1", "dir": tdir, "group": "free_space", "movement": mv})
        else:
            print(f"⚠️  Missing free-space trial: {tdir}")

    # Interaction: multiple numeric trials under experiments/
    for mv in cfg.interaction_movements:
        base = os.path.join(cfg.data_root, "recordings", mv, "experiments")
        if not os.path.isdir(base):
            print(f"⚠️  No experiments dir for {mv}: {base}")
            continue
        for name in sorted(os.listdir(base), key=lambda s: (len(s), s)):
            if not name.isdigit(): 
                continue
            tdir = os.path.join(base, name)
            if verify_trial_data(tdir):
                units.append({"trial_id": f"{mv}#{name}", "dir": tdir, "group": "interaction", "movement": mv})
            else:
                print(f"⚠️  Missing interaction trial: {tdir}")

    return units

def build_groups_from_units(units):
    groups = {"free_space": [], "interaction": []}
    for u in units:
        groups[u["group"]].append(u["trial_id"])
    return groups


def train_one_fold(
    model_name: str,
    config: Config,
    trainsets: List,
    valsets: List, 
    testsets: List,
    out_dir_models: Path,
    device: torch.device,
    fold_id: int
) -> str:
    """Train one model for one fold and return saved model path."""
    print(f"\n🚀 Training {model_name} for fold {fold_id}...")
    
    # Update config name with fold info
    config.name = f"{model_name}_fold_{fold_id}"
    
    # Train model
    model = train_model(
        trainsets, valsets, testsets, device,
        config.wandb_mode, config.wandb_project, config.name,
        config, person_dir=str(out_dir_models.parent)
    )
    
    # Save model
    model_dir = out_dir_models / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"fold_{fold_id}_best.pt"
    
    model.save(str(model_path))
    
    # Save config
    config_path = model_dir / f"fold_{fold_id}_config.json"
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    
    print(f"✅ Saved {model_name} model to {model_path}")
    return str(model_path)


def load_model_for_prediction(model_path: str, config: Config, device: torch.device) -> TimeSeriesRegressorWrapper:
    """Load a trained model for prediction."""
    model = TimeSeriesRegressorWrapper(
        device=device,
        input_size=len(config.features),
        output_size=len(config.targets),
        **config.to_dict()  # Use consistent config loading
    )
    model.load(model_path)
    model.to(device)
    model.eval()
    return model


def predict_on_trial(
    model: TimeSeriesRegressorWrapper,
    test_df: pd.DataFrame,
    features: List,
    targets: List,
    intact_hand: str,
    device: torch.device
) -> pd.DataFrame:
    """
    Run inference on a test trial and return predictions in degrees.
    
    Returns DataFrame with columns: timestamp, *_gt_deg, *_pred_deg
    """
    # Use the model's predict method (from your existing code)
    pred_scaled = model.predict(test_df, features, targets)
    
    # Robust shape handling
    if isinstance(pred_scaled, torch.Tensor):
        pred_scaled = pred_scaled.detach().cpu().numpy()
    if pred_scaled.ndim == 3:  # [B, T, J] -> [T, J]
        pred_scaled = pred_scaled[0]
    elif pred_scaled.ndim == 1 and len(pred_scaled) == len(targets):  # [J] -> [1, J]
        pred_scaled = pred_scaled.reshape(1, -1)
    
    # Get ground truth (scaled)
    gt_scaled = test_df[targets].values
    
    # Convert both to degrees using your rescale_data function
    pred_df = pd.DataFrame(pred_scaled, columns=targets)
    pred_deg = rescale_data(pred_df, intact_hand)
    
    gt_df = pd.DataFrame(gt_scaled, columns=targets) 
    gt_deg = rescale_data(gt_df, intact_hand)
    
    # Build result DataFrame
    result_df = pd.DataFrame()
    result_df['timestamp'] = test_df.index  # Assuming timestamp is the index
    
    # Add ground truth and predictions in degrees
    joint_names = ['index', 'middle', 'ring', 'pinky', 'thumbFlex', 'thumbRot']
    for i, joint in enumerate(joint_names):
        result_df[f'{joint}_gt_deg'] = gt_deg.iloc[:, i]
        result_df[f'{joint}_pred_deg'] = pred_deg.iloc[:, i]
    
    return result_df


def compute_metrics(
    merged_df: pd.DataFrame,
    joint_names: List[str],
    model_tag: str,
    trial_id: str,
    pred_suffix: str = "",
    force_col: Optional[str] = None
) -> List[Dict]:
    """
    Compute metrics for one model on one trial.
    
    Returns list of metric dictionaries (one per joint).
    """
    from scipy import stats
    
    metrics = []
    
    for joint in joint_names:
        gt_col = f'{joint}_gt_deg'
        pred_col = f'{joint}_pred_deg{pred_suffix}'
        
        if gt_col not in merged_df.columns or pred_col not in merged_df.columns:
            print(f"⚠️  Skipping {joint} - missing columns ({gt_col}, {pred_col})")
            continue
        
        gt = merged_df[gt_col].values
        pred = merged_df[pred_col].values
        
        # Remove any NaN values
        mask = ~(np.isnan(gt) | np.isnan(pred))
        gt_clean = gt[mask]
        pred_clean = pred[mask]
        
        if len(gt_clean) == 0:
            print(f"⚠️  No valid data for {joint}")
            continue
        
        # Compute metrics
        mae = np.mean(np.abs(pred_clean - gt_clean))
        rmse = np.sqrt(np.mean((pred_clean - gt_clean) ** 2))
        
        # Add R² and correlation
        if len(gt_clean) > 1 and np.var(gt_clean) > 1e-8:
            r_squared = 1 - np.sum((gt_clean - pred_clean)**2) / np.sum((gt_clean - np.mean(gt_clean))**2)
            r_pearson, p_pearson = stats.pearsonr(gt_clean, pred_clean)
        else:
            r_squared = 0.0
            r_pearson = 0.0
            p_pearson = 1.0
        
        metrics.append({
            'trial_id': trial_id,
            'model': model_tag,
            'joint': joint,
            'mae': mae,
            'rmse': rmse,
            'r_squared': r_squared,
            'r_pearson': r_pearson,
            'p_pearson': p_pearson,
            'n_samples': len(gt_clean)
        })
    
    return metrics


def align_predictions(pred_a: pd.DataFrame, pred_b: pd.DataFrame) -> pd.DataFrame:
    """Align predictions from two models by timestamp."""
    # Merge on timestamp with suffixes
    merged = pd.merge(pred_a, pred_b, on='timestamp', suffixes=('_emg', '_force'), how='inner')
    
    # Rename ground truth columns (they should be identical)
    joint_names = ['index', 'middle', 'ring', 'pinky', 'thumbFlex', 'thumbRot']
    for joint in joint_names:
        gt_emg = f'{joint}_gt_deg_emg'
        gt_force = f'{joint}_gt_deg_force'
        if gt_emg in merged.columns and gt_force in merged.columns:
            # Use EMG version as canonical ground truth and drop force version
            merged[f'{joint}_gt_deg'] = merged[gt_emg]
            merged = merged.drop([gt_emg, gt_force], axis=1)
    
    return merged


def compute_paired_significance(merged_df: pd.DataFrame, joint_names: List[str], trial_id: str) -> List[Dict]:
    """
    Compute paired significance tests between EMG-only and EMG+Force models.
    
    Returns list of significance test results.
    """
    from scipy import stats
    
    results = []
    
    for joint in joint_names:
        gt_col = f'{joint}_gt_deg'
        pred_emg_col = f'{joint}_pred_deg_emg'
        pred_force_col = f'{joint}_pred_deg_force'
        
        # Check if all required columns exist
        required_cols = [gt_col, pred_emg_col, pred_force_col]
        if not all(col in merged_df.columns for col in required_cols):
            print(f"⚠️  Skipping significance test for {joint} - missing columns")
            continue
        
        gt = merged_df[gt_col].values
        pred_emg = merged_df[pred_emg_col].values
        pred_force = merged_df[pred_force_col].values
        
        # Remove NaN values
        mask = ~(np.isnan(gt) | np.isnan(pred_emg) | np.isnan(pred_force))
        gt_clean = gt[mask]
        pred_emg_clean = pred_emg[mask]
        pred_force_clean = pred_force[mask]
        
        if len(gt_clean) < 3:  # Need minimum samples for test
            print(f"⚠️  Too few samples for significance test: {joint}")
            continue
        
        # Compute absolute errors
        error_emg = np.abs(pred_emg_clean - gt_clean)
        error_force = np.abs(pred_force_clean - gt_clean)
        
        # Paired Wilcoxon test (non-parametric)
        try:
            wilcoxon_stat, wilcoxon_p = stats.wilcoxon(error_emg, error_force, alternative='two-sided')
        except ValueError:
            # Handle case where differences are all zero
            wilcoxon_stat, wilcoxon_p = 0.0, 1.0
        
        # Effect size (mean difference in errors)
        mean_diff = np.mean(error_emg) - np.mean(error_force)  # Positive = EMG worse
        
        results.append({
            'trial_id': trial_id,
            'joint': joint,
            'wilcoxon_stat': wilcoxon_stat,
            'wilcoxon_p': wilcoxon_p,
            'mean_error_diff': mean_diff,  # EMG_error - Force_error
            'n_samples': len(gt_clean)
        })
    
    return results

def aggregate_metrics(per_trial_metrics: pd.DataFrame, groups: Dict[str, List[str]]) -> pd.DataFrame:
    """Aggregate per-trial metrics into summary statistics."""
    if len(per_trial_metrics) == 0:
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=['group', 'model', 'joint', 'mae_mean', 'mae_std', 'rmse_mean', 'rmse_std', 'n_samples_sum'])
    
    summary_rows = []
    
    # Overall summary
    overall = per_trial_metrics.groupby(['model', 'joint']).agg({
        'mae': ['mean', 'std'],
        'rmse': ['mean', 'std'],
        'r_squared': ['mean', 'std'],
        'r_pearson': ['mean', 'std'],
        'n_samples': 'sum'
    }).round(3)
    
    # Flatten column names
    overall.columns = ['_'.join(col).strip() for col in overall.columns]
    overall = overall.reset_index()
    overall['group'] = 'overall'
    summary_rows.append(overall)
    
    # Group-wise summaries
    for group_name, trial_list in groups.items():
        group_metrics = per_trial_metrics[per_trial_metrics['trial_id'].isin(trial_list)]
        if len(group_metrics) == 0:
            continue
            
        group_summary = group_metrics.groupby(['model', 'joint']).agg({
            'mae': ['mean', 'std'],
            'rmse': ['mean', 'std'],
            'r_squared': ['mean', 'std'],
            'r_pearson': ['mean', 'std'],
            'n_samples': 'sum'
        }).round(3)
        
        group_summary.columns = ['_'.join(col).strip() for col in group_summary.columns]
        group_summary = group_summary.reset_index()
        group_summary['group'] = group_name
        summary_rows.append(group_summary)
    
    return pd.concat(summary_rows, ignore_index=True)


def plot_quicklook(merged_df: pd.DataFrame, out_dir: Path):
    """Generate quick visualization plots."""
    joint_names = ['index', 'middle', 'ring', 'pinky', 'thumbFlex', 'thumbRot']
    
    # Check if we have data to plot
    if len(merged_df) == 0:
        print("⚠️  No data to plot")
        return
    
    # Remove NaN values for plotting
    plot_df = merged_df.dropna()
    if len(plot_df) == 0:
        print("⚠️  No valid data after dropping NaN values")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, joint in enumerate(joint_names):
        ax = axes[i]
        
        gt_col = f'{joint}_gt_deg'
        pred_emg_col = f'{joint}_pred_deg_emg' 
        pred_force_col = f'{joint}_pred_deg_force'
        
        # Check if columns exist
        if not all(col in plot_df.columns for col in [gt_col, pred_emg_col, pred_force_col]):
            ax.text(0.5, 0.5, f'Missing data\nfor {joint}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{joint.title()} Position')
            continue
        
        # Plot ground truth and both predictions
        ax.plot(plot_df['timestamp'], plot_df[gt_col], 
                label='Ground Truth', color='black', linewidth=2)
        ax.plot(plot_df['timestamp'], plot_df[pred_emg_col], 
                label='EMG Only', color='blue', alpha=0.7)
        ax.plot(plot_df['timestamp'], plot_df[pred_force_col], 
                label='EMG + Force', color='red', alpha=0.7)
        
        ax.set_title(f'{joint.title()} Position')
        ax.set_ylabel('Angle (degrees)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'quicklook_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved quicklook plot to {out_dir / 'quicklook_comparison.png'}")


def plot_error_vs_force(merged_df: pd.DataFrame, out_dir: Path):
    """Plot error vs force magnitude (if force data available)."""
    joint_names = ['index', 'middle', 'ring', 'pinky', 'thumbFlex', 'thumbRot']
    
    # Check if we have data to plot
    if len(merged_df) == 0:
        print("⚠️  No data for error plot")
        return
    
    # Remove NaN values for plotting
    plot_df = merged_df.dropna()
    if len(plot_df) == 0:
        print("⚠️  No valid data after dropping NaN values")
        return
    
    error_data = []
    for joint in joint_names:
        gt_col = f'{joint}_gt_deg'
        pred_emg_col = f'{joint}_pred_deg_emg'
        pred_force_col = f'{joint}_pred_deg_force'
        
        # Check if columns exist
        if not all(col in plot_df.columns for col in [gt_col, pred_emg_col, pred_force_col]):
            continue
        
        gt = plot_df[gt_col]
        pred_emg = plot_df[pred_emg_col]
        pred_force = plot_df[pred_force_col]
        
        error_emg = np.abs(pred_emg - gt)
        error_force = np.abs(pred_force - gt)
        
        error_data.extend([
            {'joint': joint, 'model': 'EMG Only', 'error': e} for e in error_emg
        ])
        error_data.extend([
            {'joint': joint, 'model': 'EMG + Force', 'error': e} for e in error_force
        ])
    
    if not error_data:
        print("⚠️  No error data to plot")
        return
    
    error_df = pd.DataFrame(error_data)
    
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=error_df, x='joint', y='error', hue='model')
    plt.title('Prediction Error by Joint and Model')
    plt.ylabel('Absolute Error (degrees)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_dir / 'error_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved error comparison plot to {out_dir / 'error_comparison.png'}")


def plot_active_fingers_only(merged_df: pd.DataFrame, out_dir: Path, movement: str):
    """Generate plots focusing only on fingers active in the specific interaction."""
    
    # Define active fingers for each interaction type
    active_fingers_map = {
        'pinch_interaction': ['index', 'thumbFlex'],  # index finger + thumb
        'tripod_interaction': ['index', 'middle', 'thumbFlex'],  # index + middle + thumb  
        'hook_interaction': ['index', 'middle', 'ring', 'pinky'],  # all fingers except thumb
        'power_grip_interaction': ['index', 'middle', 'ring', 'pinky', 'thumbFlex']  # all fingers + thumb
    }
    
    if movement not in active_fingers_map:
        print(f"⚠️  Unknown movement type: {movement}")
        return
    
    active_joints = active_fingers_map[movement]
    print(f"📊 Creating active finger plots for {movement}: {active_joints}")
    
    # Check if we have data to plot
    if len(merged_df) == 0:
        print("⚠️  No data to plot")
        return
    
    # Remove NaN values for plotting
    plot_df = merged_df.dropna()
    if len(plot_df) == 0:
        print("⚠️  No valid data after dropping NaN values")
        return
    
    # === ACTIVE FINGERS QUICKLOOK PLOT ===
    n_joints = len(active_joints)
    if n_joints <= 3:
        fig, axes = plt.subplots(1, n_joints, figsize=(5*n_joints, 4))
    else:
        fig, axes = plt.subplots(2, (n_joints+1)//2, figsize=(5*((n_joints+1)//2), 8))
    
    if n_joints == 1:
        axes = [axes]
    elif n_joints > 1:
        axes = axes.flatten()
    
    for i, joint in enumerate(active_joints):
        ax = axes[i] if n_joints > 1 else axes[0]
        
        gt_col = f'{joint}_gt_deg'
        pred_emg_col = f'{joint}_pred_deg_emg' 
        pred_force_col = f'{joint}_pred_deg_force'
        
        # Check if columns exist
        if not all(col in plot_df.columns for col in [gt_col, pred_emg_col, pred_force_col]):
            ax.text(0.5, 0.5, f'Missing data\nfor {joint}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{joint.title()} Position (Active)')
            continue
        
        # Plot ground truth and both predictions
        ax.plot(plot_df['timestamp'], plot_df[gt_col], 
                label='Ground Truth', color='black', linewidth=2)
        ax.plot(plot_df['timestamp'], plot_df[pred_emg_col], 
                label='EMG Only', color='blue', alpha=0.7)
        ax.plot(plot_df['timestamp'], plot_df[pred_force_col], 
                label='EMG + Force', color='red', alpha=0.7)
        
        ax.set_title(f'{joint.title()} Position (Active)')
        ax.set_ylabel('Angle (degrees)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    if n_joints > 1:
        for j in range(n_joints, len(axes)):
            axes[j].set_visible(False)
    
    plt.suptitle(f'{movement.replace("_", " ").title()} - Active Fingers Only')
    plt.tight_layout()
    plt.savefig(out_dir / 'quicklook_active_fingers.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved active fingers quicklook to {out_dir / 'quicklook_active_fingers.png'}")
    
    # === ACTIVE FINGERS ERROR COMPARISON PLOT ===
    error_data = []
    for joint in active_joints:
        gt_col = f'{joint}_gt_deg'
        pred_emg_col = f'{joint}_pred_deg_emg'
        pred_force_col = f'{joint}_pred_deg_force'
        
        # Check if columns exist
        if not all(col in plot_df.columns for col in [gt_col, pred_emg_col, pred_force_col]):
            continue
        
        gt = plot_df[gt_col]
        pred_emg = plot_df[pred_emg_col]
        pred_force = plot_df[pred_force_col]
        
        error_emg = np.abs(pred_emg - gt)
        error_force = np.abs(pred_force - gt)
        
        error_data.extend([
            {'joint': joint, 'model': 'EMG Only', 'error': e} for e in error_emg
        ])
        error_data.extend([
            {'joint': joint, 'model': 'EMG + Force', 'error': e} for e in error_force
        ])
    
    if not error_data:
        print("⚠️  No error data to plot for active fingers")
        return
    
    error_df = pd.DataFrame(error_data)
    
    plt.figure(figsize=(max(6, len(active_joints)*1.5), 6))
    sns.boxplot(data=error_df, x='joint', y='error', hue='model')
    plt.title(f'{movement.replace("_", " ").title()} - Active Fingers Error Comparison')
    plt.ylabel('Absolute Error (degrees)')
    plt.xlabel('Active Joints')
    if len(active_joints) > 3:
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_dir / 'error_active_fingers.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved active fingers error plot to {out_dir / 'error_active_fingers.png'}")

def plot_cross_validation_summary(per_trial_metrics: pd.DataFrame, out_dir: Path):
    """Generate cross-validation summary plots when using leave-one-out."""
    if len(per_trial_metrics) == 0:
        print("⚠️  No metrics to plot for CV summary")
        return
    
    # Plot MAE across all test trials
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: MAE by trial and model
    plt.subplot(2, 2, 1)
    mae_pivot = per_trial_metrics.pivot_table(index='trial_id', columns='model', values='mae', aggfunc='mean')
    mae_pivot.plot(kind='bar', ax=plt.gca(), width=0.8)
    plt.title('Mean Absolute Error by Test Trial')
    plt.ylabel('MAE (degrees)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: RMSE by trial and model
    plt.subplot(2, 2, 2)
    rmse_pivot = per_trial_metrics.pivot_table(index='trial_id', columns='model', values='rmse', aggfunc='mean')
    rmse_pivot.plot(kind='bar', ax=plt.gca(), width=0.8)
    plt.title('Root Mean Square Error by Test Trial')
    plt.ylabel('RMSE (degrees)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: R² by trial and model
    plt.subplot(2, 2, 3)
    r2_pivot = per_trial_metrics.pivot_table(index='trial_id', columns='model', values='r_squared', aggfunc='mean')
    r2_pivot.plot(kind='bar', ax=plt.gca(), width=0.8)
    plt.title('R² Score by Test Trial')
    plt.ylabel('R² Score')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Overall comparison boxplot
    plt.subplot(2, 2, 4)
    sns.boxplot(data=per_trial_metrics, x='model', y='mae')
    plt.title('Overall MAE Distribution')
    plt.ylabel('MAE (degrees)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'cross_validation_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved cross-validation summary to {out_dir / 'cross_validation_summary.png'}")


def write_manifest(out_path: Path, args, recordings: List[str], groups: Dict):
    """Write run manifest for reproducibility."""
    manifest = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'command_line': ' '.join(sys.argv),
        'args': vars(args),
        'python_version': sys.version,
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'recordings': recordings,
        'groups': groups,
        'git_commit': None,  # Could add git info if needed
    }
    
    with open(out_path, 'w') as f:
        yaml.dump(manifest, f, indent=2)


def run_single_fold_comparison(
    cfg: HardConfig,
    emg_features: List,
    force_features: List,
    targets: List,
    train_units: List,
    test_units: List,
    fold_idx: int,
    device: torch.device,
    out_dir: Path
) -> Tuple[List[Dict], List[Dict]]:
    """
    Run comparison for a single fold (train/test split).
    
    Returns:
        fold_metrics: List of metric dictionaries for this fold
        fold_significance: List of significance test results for this fold
    """
    print(f"\n{'='*60}")
    print(f"FOLD {fold_idx}: Testing on {[u['trial_id'] for u in test_units]}")
    print(f"Training on {len(train_units)} trials")
    print(f"{'='*60}")
    
    train_dirs = [u["dir"] for u in train_units]
    test_dirs = [u["dir"] for u in test_units]
    
    num_train_dirs = len(train_dirs)
    num_test_dirs = len(test_dirs)
    
    # Train EMG+Force model
    print(f"\n🚀 Training EMG+Force model for fold {fold_idx}...")
    cfg_force = build_config(cfg, emg_features + force_features, targets, f"fold_{fold_idx}_emg_force", num_train_dirs, num_test_dirs)
    tr_force, va_force, _, te_force_all = get_data(cfg_force, train_dirs, cfg.intact_hand, visualize=False, test_dirs=test_dirs)
    force_model = train_model(
        tr_force, va_force, te_force_all, device,
        cfg_force.wandb_mode, cfg_force.wandb_project, cfg_force.name,
        cfg_force, person_dir=str(out_dir)
    )
    
    # Train EMG-only model
    print(f"\n🚀 Training EMG-only model for fold {fold_idx}...")
    cfg_emg = build_config(cfg, emg_features, targets, f"fold_{fold_idx}_emg_only", num_train_dirs, num_test_dirs)
    tr_emg, va_emg, _, te_emg_all = get_data(cfg_emg, train_dirs, cfg.intact_hand, visualize=False, test_dirs=test_dirs)
    emg_model = train_model(
        tr_emg, va_emg, te_emg_all, device,
        cfg_emg.wandb_mode, cfg_emg.wandb_project, cfg_emg.name,
        cfg_emg, person_dir=str(out_dir)
    )
    
    # Save models for this fold
    fold_models_dir = out_dir / 'models' / f'fold_{fold_idx}'
    fold_models_dir.mkdir(parents=True, exist_ok=True)
    
    emg_model_path = fold_models_dir / 'emg_only_best.pt'
    force_model_path = fold_models_dir / 'emg_force_best.pt'
    
    emg_model.save(str(emg_model_path))
    force_model.save(str(force_model_path))
    
    # Save configs
    with open(fold_models_dir / 'emg_only_config.json', 'w') as f:
        json.dump(cfg_emg.to_dict(), f, indent=2)
    with open(fold_models_dir / 'emg_force_config.json', 'w') as f:
        json.dump(cfg_force.to_dict(), f, indent=2)
    
    print(f"✅ Saved fold {fold_idx} models to {fold_models_dir}")
    
    # Test both models
    emg_model.eval()
    force_model.eval()
    
    fold_metrics = []
    fold_significance = []
    
    for i, test_unit in enumerate(test_units):
        print(f"\nTesting on {test_unit['trial_id']} (fold {fold_idx}, test {i+1}/{len(test_units)})")
        
        te_emg_single = [te_emg_all[i]]
        te_force_single = [te_force_all[i]]
        
        pred_emg = predict_on_trial(emg_model, te_emg_single[0], cfg_emg.features, cfg_emg.targets, cfg.intact_hand, device)
        pred_force = predict_on_trial(force_model, te_force_single[0], cfg_force.features, cfg_force.targets, cfg.intact_hand, device)
        
        # Save predictions
        preds_dir = out_dir / 'predictions' / f"fold_{fold_idx}" / f"{test_unit['trial_id'].replace('#','_')}"
        preds_dir.mkdir(parents=True, exist_ok=True)
        pred_emg.to_csv(preds_dir / 'preds_emg_only.csv', index=False)
        pred_force.to_csv(preds_dir / 'preds_emg_plus_force.csv', index=False)
        
        merged = align_predictions(pred_emg, pred_force)
        merged.to_csv(preds_dir / 'merged_comparison.csv', index=False)
        print(f"  Aligned {len(merged)} prediction points")
        
        # Compute metrics and significance tests
        joints = ['index','middle','ring','pinky','thumbFlex','thumbRot']
        trial_metrics_emg = compute_metrics(merged, joints, 'EMG Only', test_unit['trial_id'], pred_suffix="_emg")
        trial_metrics_force = compute_metrics(merged, joints, 'EMG + Force', test_unit['trial_id'], pred_suffix="_force")
        trial_significance = compute_paired_significance(merged, joints, test_unit['trial_id'])
        
        # Add fold information to metrics
        for metric in trial_metrics_emg + trial_metrics_force:
            metric['fold'] = fold_idx
        for sig in trial_significance:
            sig['fold'] = fold_idx
        
        fold_metrics.extend(trial_metrics_emg + trial_metrics_force)
        fold_significance.extend(trial_significance)
        
        # Generate plots for this trial
        if cfg.make_plots:
            plot_quicklook(merged, preds_dir)
            plot_error_vs_force(merged, preds_dir)
            plot_active_fingers_only(merged, preds_dir, test_unit['movement'])
    
    return fold_metrics, fold_significance


def main():
    import ast  # needed to parse stringified tuple column names
    
    # Parse command line arguments
    args = parse_args()
    
    # Update config with command line arguments
    CFG = HardConfig()
    # CFG.data_root = args.data_root
    # CFG.out_dir = args.out_dir
    # CFG.intact_hand = args.intact_hand
    CFG.test_trial = args.test_trial
    CFG.model_type = args.model_type
    CFG.seq_len = args.seq_len
    CFG.batch_size = args.batch_size
    CFG.n_epochs = args.n_epochs
    CFG.learning_rate = args.learning_rate
    CFG.weight_decay = args.weight_decay
    CFG.warmup_steps = args.warmup_steps
    CFG.early_stopping_patience = args.early_stopping_patience
    CFG.early_stopping_delta = args.early_stopping_delta
    CFG.wandb_mode = args.wandb_mode
    CFG.seed = args.seed
    CFG.make_plots = args.make_plots == 'true'
    CFG.device = args.device
    CFG = CFG.finalize()

    def _to_hashable(cols):
        out = []
        for c in cols:
            if isinstance(c, list):
                out.append(tuple(c))
            elif isinstance(c, str) and c.startswith("("):
                # parse "('Left', 'index_Force')" -> ('Left','index_Force')
                try:
                    parsed = ast.literal_eval(c)
                    out.append(tuple(parsed) if isinstance(parsed, (list, tuple)) else c)
                except Exception:
                    out.append(c)
            else:
                out.append(c)
        return out

    set_seeds(CFG.seed)
    out_dir = Path(CFG.out_dir)
    (out_dir / 'models').mkdir(parents=True, exist_ok=True)
    (out_dir / 'predictions').mkdir(exist_ok=True)
    (out_dir / 'metrics').mkdir(exist_ok=True)
    (out_dir / 'logs').mkdir(exist_ok=True)

    device = torch.device('cuda' if (CFG.device == 'auto' and torch.cuda.is_available()) else (CFG.device if CFG.device != 'auto' else 'cpu'))
    print(f"✅ Using device: {device}")

    # 1) Enumerate all trial units per your folder layout
    units = enumerate_trial_units(CFG)
    if not units:
        print("❌ No valid trials found.")
        return 1

    groups = build_groups_from_units(units)
    print(f"✅ Trials found: {len(units)} (free_space={len(groups['free_space'])}, interaction={len(groups['interaction'])})")

    # 2) Separate units by type
    free_space_units = [u for u in units if u['group'] == 'free_space']
    interaction_units = [u for u in units if u['group'] == 'interaction']
    
    print(f"Free-space units: {len(free_space_units)}")
    print(f"Interaction units: {len(interaction_units)}")

    if not interaction_units:
        print("❌ No interaction units found for comparison")
        return 1

    # 3) Check force availability using first interaction trial
    print("\n--- Checking Force Feature Availability ---")
    force_final = []
    try:
        sample_df = pd.read_parquet(os.path.join(interaction_units[0]["dir"], 'aligned_angles.parquet'))

        print(f"🔍 Available columns in interaction data:")
        for i, col in enumerate(sample_df.columns):
            print(f"  {i:2d}: {col!r}")  # repr shows if they are strings

        # Normalize CFG.force_features to whatever sample_df actually uses
        candidate_force = normalize_cols_to_df(CFG.force_features, sample_df.columns)

        for raw, norm in zip(CFG.force_features, candidate_force):
            if col_in_df(norm, sample_df.columns):
                force_final.append(norm)
                print(f"✅ Found force feature: {norm}")
            else:
                print(f"❌ Missing force feature: {norm}")

        if force_final:
            print(f"🔍 Checking variance for force features...")
            keep = []
            for f in force_final:
                try:
                    vals = sample_df[f]
                except KeyError:
                    vals = sample_df[str(f)]
                if np.nanstd(vals.values) > 1e-8:
                    keep.append(f)
                    print(f"✅ Keeping force feature: {f} (std={np.nanstd(vals.values):.2e})")
                else:
                    print(f"ℹ️  Dropping constant force feature: {f} (std={np.nanstd(vals.values):.2e})")
            force_final = keep

            print(f"🔍 DEBUG: Final force features:")
            for i, f in enumerate(force_final):
                print(f"  {i}: {f!r} (type: {type(f)})")

            print(f"✅ Using force features: {force_final}")
        else:
            print("⚠️  No force features found in interaction data")
    except Exception as e:
        print(f"⚠️  Error checking force features: {e}")
        traceback.print_exc()
        force_final = []

    # === OPTION A CRITICAL STEP: convert features/targets to HASHABLE TUPLES ===
    emg_features   = _to_hashable(CFG.emg_features)
    targets        = _to_hashable(CFG.targets)
    force_features = _to_hashable(force_final)  # handles list->tuple and "('L','x')" -> ('L','x')

    # 4) Determine test strategy based on test_trial argument
    all_metric_rows = []
    all_significance_rows = []
    
    if CFG.test_trial.lower() == "all":
        print(f"\n🔄 LEAVE-ONE-OUT CROSS-VALIDATION MODE")
        print(f"Will test each of {len(interaction_units)} interaction trials separately")
        
        # Leave-one-out: each interaction trial becomes test set once
        for fold_idx, test_unit in enumerate(interaction_units):
            train_units = [u for u in interaction_units if u['trial_id'] != test_unit['trial_id']]
            test_units = [test_unit]
            
            if len(train_units) == 0:
                print(f"⚠️  Skipping fold {fold_idx} - no training data")
                continue
            
            fold_metrics, fold_significance = run_single_fold_comparison(
                CFG, emg_features, force_features, targets,
                train_units, test_units, fold_idx, device, out_dir
            )
            
            all_metric_rows.extend(fold_metrics)
            all_significance_rows.extend(fold_significance)
        
        test_strategy = "leave_one_out"
        
    else:
        print(f"\n🎯 SINGLE TEST TRIAL MODE")
        print(f"Testing specifically on: {CFG.test_trial}")
        
        # Single test trial mode (original behavior)
        test_units = [u for u in interaction_units if u['trial_id'] == CFG.test_trial]
        train_units = [u for u in interaction_units if u['trial_id'] != CFG.test_trial]
        
        if not test_units:
            print(f"❌ Test trial '{CFG.test_trial}' not found in interaction units")
            available_trials = [u['trial_id'] for u in interaction_units]
            print(f"Available trials: {available_trials}")
            return 1
        
        if len(train_units) == 0:
            print("❌ No training data available")
            return 1
        
        fold_metrics, fold_significance = run_single_fold_comparison(
            CFG, emg_features, force_features, targets,
            train_units, test_units, 0, device, out_dir
        )
        
        all_metric_rows.extend(fold_metrics)
        all_significance_rows.extend(fold_significance)
        
        test_strategy = "single_trial"

    # 5) Save and analyze results
    print(f"\n📊 Saving results for {test_strategy} strategy...")
    
    per_trial = pd.DataFrame(all_metric_rows)
    per_trial.to_csv(out_dir / 'metrics' / 'per_trial_metrics.csv', index=False)

    if all_significance_rows:
        significance_df = pd.DataFrame(all_significance_rows)
        significance_df.to_csv(out_dir / 'metrics' / 'significance_tests.csv', index=False)

    # Updated groups for aggregation
    if test_strategy == "leave_one_out":
        updated_groups = {
            'interaction': [u['trial_id'] for u in interaction_units]
        }
    else:
        updated_groups = {
            'interaction': [u['trial_id'] for u in test_units]
        }
    
    summary = aggregate_metrics(per_trial, updated_groups)
    summary.to_csv(out_dir / 'metrics' / 'summary_metrics.csv', index=False)
    with open(out_dir / 'metrics' / 'summary.json','w') as f:
        json.dump(summary.to_dict(orient='records'), f, indent=2)

    # Generate cross-validation summary plots if using leave-one-out
    if CFG.make_plots and test_strategy == "leave_one_out":
        plot_cross_validation_summary(per_trial, out_dir)

    # Write manifest with test strategy info
    manifest_data = {
        'test_strategy': test_strategy,
        'test_trial': CFG.test_trial,
        'n_folds': len(interaction_units) if test_strategy == "leave_one_out" else 1,
        'recordings': [u['trial_id'] for u in units],
        'groups': updated_groups
    }
    write_manifest(out_dir / 'run_manifest.yaml', args, [u['trial_id'] for u in units], manifest_data)

    print("\n✅ Comparison complete!")
    print(f"\nSTRATEGY: {test_strategy}")
    
    if test_strategy == "leave_one_out":
        print(f"- Performed leave-one-out cross-validation on {len(interaction_units)} interaction trials")
        print(f"- Each trial was used as test set exactly once")
        print(f"- Total models trained: {2 * len(interaction_units)} (EMG-only + EMG+Force for each fold)")
    else:
        print("- Trained 2 models on the same interaction training data")
        print("- EMG-only model: uses only EMG features")
        print("- EMG+Force model: uses EMG + force features")
        print(f"- Tested both models on 1 held-out interaction trial: {CFG.test_trial}")

    if len(per_trial) > 0:
        print("\n📈 QUICK RESULTS PREVIEW:")
        if test_strategy == "leave_one_out":
            # Show cross-validation results
            cv_metrics = per_trial.groupby('model').agg({
                'mae': ['mean', 'std'], 
                'rmse': ['mean', 'std'], 
                'r_squared': ['mean', 'std']
            }).round(3)
            print("Cross-validation results (mean ± std):")
            print(cv_metrics)
            
            # Show per-trial breakdown
            print(f"\nPer-trial results:")
            trial_summary = per_trial.groupby(['trial_id', 'model']).agg({
                'mae': 'mean', 
                'rmse': 'mean', 
                'r_squared': 'mean'
            }).round(3)
            print(trial_summary)
            
        else:
            # Single trial results
            avg_metrics = per_trial.groupby('model').agg({'mae': 'mean', 'rmse': 'mean', 'r_squared': 'mean'}).round(3)
            print(avg_metrics)

    return 0


if __name__ == "__main__":
    sys.exit(main())