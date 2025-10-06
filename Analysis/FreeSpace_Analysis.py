import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

def find_minima_with_window(signal, window_size=10, min_distance=20):
    """
    Find local minima in a signal using a sliding window approach
    to ensure they are true minima within a neighborhood.
    
    Args:
        signal: 1D array of signal values
        window_size: size of window to check for local minimum
        min_distance: minimum distance between consecutive minima
    
    Returns:
        indices of minima
    """
    # Find local minima using scipy
    minima_indices, _ = find_peaks(-signal, distance=min_distance)
    
    # Refine minima using sliding window
    refined_minima = []
    
    for idx in minima_indices:
        # Define window around the candidate minimum
        start = max(0, idx - window_size // 2)
        end = min(len(signal), idx + window_size // 2 + 1)
        window = signal[start:end]
        
        # Check if this point is indeed the minimum in the window
        local_min_idx = np.argmin(window)
        global_min_idx = start + local_min_idx
        
        # Only keep if it's close to the original detection
        if abs(global_min_idx - idx) <= window_size // 2:
            refined_minima.append(global_min_idx)
    
    return np.array(refined_minima)

def extract_iterations(ground_truth, predictions, minima_indices):
    """
    Extract iterations between consecutive minima.
    
    Args:
        ground_truth: ground truth signal
        predictions: predicted signal
        minima_indices: indices of minima in ground_truth
    
    Returns:
        list of dictionaries containing iteration data
    """
    iterations = []
    
    for i in range(len(minima_indices) - 1):
        start_idx = minima_indices[i]
        end_idx = minima_indices[i + 1] + 1  # Include the next minimum
        
        iteration_data = {
            'gt': ground_truth[start_idx:end_idx],
            'pred': predictions[start_idx:end_idx],
            'start_idx': start_idx,
            'end_idx': end_idx,
            'length': end_idx - start_idx
        }
        iterations.append(iteration_data)
    
    return iterations

def normalize_iteration_length(iterations, target_length=None):
    """
    Normalize all iterations to the same length using interpolation.
    
    Args:
        iterations: list of iteration dictionaries
        target_length: desired length (if None, use the first iteration's length)
    
    Returns:
        normalized iterations with consistent length
    """
    if len(iterations) == 0:
        return []
    
    if target_length is None:
        target_length = iterations[0]['length']
    
    normalized_iterations = []
    
    for iteration in iterations:
        original_length = iteration['length']
        
        # Create interpolation functions
        x_old = np.linspace(0, 1, original_length)
        x_new = np.linspace(0, 1, target_length)
        
        # Interpolate ground truth and predictions
        gt_interp = interp1d(x_old, iteration['gt'], kind='cubic')
        pred_interp = interp1d(x_old, iteration['pred'], kind='cubic')
        
        normalized_iteration = {
            'gt': gt_interp(x_new),
            'pred': pred_interp(x_new),
            'original_length': original_length,
            'normalized_length': target_length
        }
        normalized_iterations.append(normalized_iteration)
    
    return normalized_iterations

def create_range_band_visualization(normalized_iterations, finger_name="Thumb"):
    """
    Create a range band visualization showing ground truth with prediction ranges.
    
    Args:
        normalized_iterations: list of normalized iteration dictionaries
        finger_name: name of the finger for the plot title
    """
    # Use the first iteration's ground truth as the representative
    representative_gt = normalized_iterations[0]['gt']
    
    # Collect all predictions
    all_predictions = np.array([iteration['pred'] for iteration in normalized_iterations])
    
    # Calculate statistics for the range band
    pred_mean = np.mean(all_predictions, axis=0)
    pred_std = np.std(all_predictions, axis=0)
    pred_min = np.min(all_predictions, axis=0)
    pred_max = np.max(all_predictions, axis=0)
    pred_25 = np.percentile(all_predictions, 25, axis=0)
    pred_75 = np.percentile(all_predictions, 75, axis=0)
    
    # Create the plot with much more compressed x-axis for steeper curves
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))  # Narrower figure
    
    # Plot 1: Range band visualization
    x = np.arange(len(representative_gt)) / 10  # Scale x-axis to 1/10 length for very steep curves
    
    # Fill between percentiles for range band
    ax1.fill_between(x, pred_min, pred_max, alpha=0.2, color='orange', label='Min-Max Range')
    ax1.fill_between(x, pred_25, pred_75, alpha=0.4, color='orange', label='25th-75th Percentile')
    
    # Plot mean prediction and ground truth
    ax1.plot(x, pred_mean, 'orange', linewidth=2, label=f'Mean Prediction ({len(normalized_iterations)} iterations)')
    ax1.plot(x, representative_gt, 'blue', linewidth=2, label='Representative Ground Truth')
    
    ax1.set_xlabel('Normalized Sample Index (scaled)')
    ax1.set_ylabel(f'{finger_name} Position (deg)')
    ax1.set_title(f'{finger_name} Position: Ground Truth vs Prediction Range Band')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Individual iterations overlay
    for i, iteration in enumerate(normalized_iterations):
        alpha = 0.3 if i > 0 else 0.8  # Make first iteration more visible
        ax2.plot(x, iteration['pred'], 'orange', alpha=alpha, linewidth=1)
    
    ax2.plot(x, representative_gt, 'blue', linewidth=2, label='Representative Ground Truth')
    ax2.plot(x, normalized_iterations[0]['pred'], 'red', linewidth=2, alpha=0.8, 
             label='First Iteration Prediction')
    
    ax2.set_xlabel('Normalized Sample Index (scaled)')
    ax2.set_ylabel(f'{finger_name} Position (deg)')
    ax2.set_title(f'All {len(normalized_iterations)} Prediction Iterations Overlaid')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, (pred_mean, pred_std, pred_min, pred_max)

def analyze_prediction_quality(normalized_iterations):
    """
    Calculate metrics to assess prediction quality.
    
    Args:
        normalized_iterations: list of normalized iteration dictionaries
    
    Returns:
        dictionary of quality metrics
    """
    if len(normalized_iterations) == 0:
        return {
            'mse_per_iteration': [],
            'mae_per_iteration': [],
            'overall_mse': float('nan'),
            'overall_mae': float('nan'),
            'avg_prediction_std': float('nan'),
            'max_prediction_std': float('nan'),
            'num_iterations': 0
        }
    
    representative_gt = normalized_iterations[0]['gt']
    all_predictions = np.array([iteration['pred'] for iteration in normalized_iterations])
    
    # Calculate metrics for each iteration
    mse_per_iteration = []
    mae_per_iteration = []
    
    for iteration in normalized_iterations:
        mse = np.mean((iteration['gt'] - iteration['pred'])**2)
        mae = np.mean(np.abs(iteration['gt'] - iteration['pred']))
        mse_per_iteration.append(mse)
        mae_per_iteration.append(mae)
    
    # Calculate overall metrics using representative ground truth
    pred_mean = np.mean(all_predictions, axis=0)
    overall_mse = np.mean((representative_gt - pred_mean)**2)
    overall_mae = np.mean(np.abs(representative_gt - pred_mean))
    
    # Calculate consistency metrics
    pred_std_across_iterations = np.std(all_predictions, axis=0)
    avg_std = np.mean(pred_std_across_iterations)
    max_std = np.max(pred_std_across_iterations)
    
    metrics = {
        'mse_per_iteration': mse_per_iteration,
        'mae_per_iteration': mae_per_iteration,
        'overall_mse': overall_mse,
        'overall_mae': overall_mae,
        'avg_prediction_std': avg_std,
        'max_prediction_std': max_std,
        'num_iterations': len(normalized_iterations)
    }
    
    return metrics

def analyze_single_finger(df, dfo, col, finger_name, split_idx, gt_shift=0):
    """
    Analyze a single finger and return normalized iterations and metrics.
    
    Args:
        df: predictions dataframe
        dfo: ground truth dataframe
        col: column name to analyze
        finger_name: name of the finger
        split_idx: index to split train/test
        gt_shift: manual shift for ground truth alignment (positive = shift right, negative = shift left)
    """
    # Get predictions and ground truth for test set
    pred_test = df[col].iloc[:].reset_index(drop=True)
    gt_test = dfo[col].iloc[split_idx:].reset_index(drop=True)
    
    # Apply manual shift to ground truth
    if gt_shift != 0:
        if gt_shift > 0:
            # Shift right: pad with first value at beginning, truncate end
            gt_shifted = pd.concat([
                pd.Series([gt_test.iloc[0]] * gt_shift),
                gt_test.iloc[:-gt_shift]
            ]).reset_index(drop=True)
        else:
            # Shift left: truncate beginning, pad with last value at end
            shift_amount = abs(gt_shift)
            gt_shifted = pd.concat([
                gt_test.iloc[shift_amount:],
                pd.Series([gt_test.iloc[-1]] * shift_amount)
            ]).reset_index(drop=True)
        
        gt_test = gt_shifted
        print(f"Applied shift of {gt_shift} samples to {finger_name} ground truth")
    
    # Ensure both arrays have the same length
    min_length = min(len(pred_test), len(gt_test))
    pred_test = pred_test[:min_length]
    gt_test = gt_test[:min_length]
    
    print(f"Analyzing {finger_name} with {len(gt_test)} samples (shift: {gt_shift})")
    
    # Find minima in ground truth
    minima_indices = find_minima_with_window(gt_test.values, window_size=20, min_distance=50)
    print(f"Found {len(minima_indices)} minima for {finger_name}")
    
    if len(minima_indices) < 2:
        print(f"Warning: Not enough minima found for {finger_name} (need at least 2)")
        return [], {
            'mse_per_iteration': [],
            'mae_per_iteration': [],
            'overall_mse': float('nan'),
            'overall_mae': float('nan'),
            'avg_prediction_std': float('nan'),
            'max_prediction_std': float('nan'),
            'num_iterations': 0
        }
    
    # Extract iterations between minima
    iterations = extract_iterations(gt_test.values, pred_test.values, minima_indices)
    print(f"Extracted {len(iterations)} iterations for {finger_name}")
    
    if len(iterations) == 0:
        print(f"Warning: No iterations extracted for {finger_name}")
        return [], {
            'mse_per_iteration': [],
            'mae_per_iteration': [],
            'overall_mse': float('nan'),
            'overall_mae': float('nan'),
            'avg_prediction_std': float('nan'),
            'max_prediction_std': float('nan'),
            'num_iterations': 0
        }
    
    # Normalize iteration lengths
    normalized_iterations = normalize_iteration_length(iterations)
    
    # Analyze prediction quality
    metrics = analyze_prediction_quality(normalized_iterations)
    
    return normalized_iterations, metrics

def get_active_fingers_for_movement(file_path):
    """
    Determine which fingers are active based on the movement type in the file path.
    
    Args:
        file_path: path to the data file containing movement type
    
    Returns:
        list of active finger names for the detected movement
    """
    # Movement patterns and their active fingers
    movement_patterns = {
        'indexFlEx': ['Index'],
        'mrpFlEx': ['Middle', 'Ring', 'Pinky'],
        'fingersFlEx': ['Index', 'Middle', 'Ring', 'Pinky'],
        'handClOp': ['ThumbRot', 'ThumbFlex', 'Index', 'Middle', 'Ring', 'Pinky'],
        'thumbFlEx': ['ThumbRot'],
        'thumbAbAd': ['ThumbFlex'],  
        'pinchClOp': ['Index', 'ThumbRot', 'ThumbFlex']
    }
    
    # Extract movement type from file path
    for movement_type in movement_patterns.keys():
        if movement_type in file_path:
            active_fingers = movement_patterns[movement_type]
            print(f"Detected movement: {movement_type}")
            print(f"Active fingers: {active_fingers}")
            return active_fingers
    
    # Default to all fingers if movement type not detected
    print("Movement type not detected, analyzing all fingers")
    return ['Index', 'Middle', 'Ring', 'Pinky', 'ThumbFlex', 'ThumbRot']

def create_active_fingers_visualization(all_finger_data, active_fingers):
    """
    Create a visualization for only the active fingers, with consistent 3-column layout.
    Always uses 3 plots per row, with placeholders when needed.
    
    Args:
        all_finger_data: dictionary with finger names as keys and (normalized_iterations, metrics) as values
        active_fingers: list of active finger names to display
    """
    num_active = len(active_fingers)
    
    # Always use 3 columns, calculate rows needed
    num_cols = 3
    num_rows = (num_active + num_cols - 1) // num_cols  # Ceiling division
    
    # Create subplot layout
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    
    # Always ensure axes is a flat list for consistent indexing
    if num_rows == 1 and num_cols == 1:
        axes = [axes]  # Single subplot case
    elif num_rows == 1:
        axes = list(axes)  # Single row, multiple columns
    else:
        axes = axes.flatten()  # Multiple rows, flatten to 1D
    
    # Plot active fingers
    for i, finger_name in enumerate(active_fingers):
        ax = axes[i]
        
        if finger_name in all_finger_data:
            normalized_iterations, metrics = all_finger_data[finger_name]
            
            if len(normalized_iterations) == 0:
                # Handle case with no valid iterations
                ax.text(0.5, 0.5, f'{finger_name}\nNo valid iterations\n(insufficient minima)', 
                        ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.set_title(f'{finger_name}', fontsize=11)
                ax.axis('off')
                continue
            
            # Use the first iteration's ground truth as the representative
            representative_gt = normalized_iterations[0]['gt']
            
            # Collect all predictions
            all_predictions = np.array([iteration['pred'] for iteration in normalized_iterations])
            
            # Calculate statistics for the range band
            pred_mean = np.mean(all_predictions, axis=0)
            pred_min = np.min(all_predictions, axis=0)
            pred_max = np.max(all_predictions, axis=0)
            pred_25 = np.percentile(all_predictions, 25, axis=0)
            pred_75 = np.percentile(all_predictions, 75, axis=0)
            
            # Create x-axis
            x = np.arange(len(representative_gt))
            
            # Fill between percentiles for range band
            ax.fill_between(x, pred_min, pred_max, alpha=0.2, color='orange', label='Min-Max')
            ax.fill_between(x, pred_25, pred_75, alpha=0.4, color='orange', label='25th-75th')
            
            # Plot mean prediction and ground truth
            ax.plot(x, pred_mean, 'orange', linewidth=2, label=f'Mean Pred')
            ax.plot(x, representative_gt, 'blue', linewidth=2, label='Ground Truth')
            
            # *** FIX: Set y-axis limits to include full range band with padding ***
            # Find the overall min and max values including all data
            all_values = np.concatenate([representative_gt, pred_min, pred_max])
            y_min = np.min(all_values)
            y_max = np.max(all_values)
            
            # Add 5% padding to ensure nothing is cut off
            y_range = y_max - y_min
            padding = y_range * 0.05
            ax.set_ylim(y_min - padding, y_max + padding)
            
            # Formatting
            ax.set_xlabel('Sample Index')
            if i % num_cols == 0:  # Label y-axis for leftmost plots in each row
                ax.set_ylabel('Position (deg)')
            ax.set_title(f'{finger_name}\n({len(normalized_iterations)} iter)', fontsize=11)
            ax.grid(True, alpha=0.3)
            
            # Add metrics as text
            
            mae = metrics['overall_mae']
            if not np.isnan(mae):
                ax.text(0.02, 0.98, f'MAE: {mae:.1f}', 
                        transform=ax.transAxes, verticalalignment='top', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            # Handle missing finger data
            ax.text(0.5, 0.5, f'{finger_name}\nNo Data', ha='center', va='center', 
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{finger_name}', fontsize=11)
            ax.axis('off')
    
    # Hide unused subplots (placeholders)
    for j in range(num_active, num_rows * num_cols):
        if j < len(axes):
            axes[j].axis('off')
    
    # Add a single legend for all subplots
    if len(active_fingers) > 0 and active_fingers[0] in all_finger_data:
        # Find the first subplot with actual data for legend
        for finger_name in active_fingers:
            if finger_name in all_finger_data and len(all_finger_data[finger_name][0]) > 0:
                idx = active_fingers.index(finger_name)
                if idx < len(axes):
                    handles, labels = axes[idx].get_legend_handles_labels()
                    if handles:  # Only add legend if there are handles
                        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=4)
                    break
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12, hspace=0.4)  # Make room for legend
    return fig


# Main execution function
def main():
    # ========== MANUAL SHIFT SETTINGS ==========
    # Adjust these values to fine-tune ground truth alignment
    # Positive values shift ground truth to the right (delay)
    # Negative values shift ground truth to the left (advance)
    manual_shifts = {
        'Index': 6,      # Shift for index finger
        'Middle': 3,     # Shift for middle finger  
        'Ring': 3,       # Shift for ring finger
        'Pinky': 3,      # Shift for pinky finger
        'ThumbFlex': 0,  # Shift for thumb flexion
        'ThumbRot': 32    # Shift for thumb rotation
    }
    
    # Example usage:
    # manual_shifts = {
    #     'Index': 7,      # Shift index ground truth 7 samples to the right
    #     'Middle': -4,    # Shift middle ground truth 4 samples to the left
    #     'Ring': 0,       # No shift for ring
    #     'Pinky': 2,      # Shift pinky ground truth 2 samples to the right
    #     'ThumbFlex': 0,  # No shift for thumb flexion
    #     'ThumbRot': 0    # No shift for thumb rotation
    # }
    # ============================================
    
    # Load data (same as your original script)
    df_path = 'data/EmanuelFull/recordings/indexFlEx/experiments/1/pred_angles-Emanuel.parquet'
    dfo_path = 'data/EmanuelFull/recordings/indexFlEx/experiments/1/aligned_angles.parquet'

    df = pd.read_parquet(df_path)
    dfo = pd.read_parquet(dfo_path)
    
    # Determine active fingers based on file path
    active_fingers = get_active_fingers_for_movement(df_path)
    
    # Define all possible fingers
    finger_configs = {
        'Index': "('Right', 'index_Pos')",
        'Middle': "('Right', 'middle_Pos')",
        'Ring': "('Right', 'ring_Pos')",
        'Pinky': "('Right', 'pinky_Pos')",
        'ThumbFlex': "('Right', 'thumbFlex_Pos')",
        'ThumbRot': "('Right', 'thumbRot_Pos')"
    }
    
    # Calculate split index (for 4/5 train, 1/5 test)
    n = len(dfo)
    split_idx = int(n / 5 * 4)
    
    # Print shift settings
    print("=== Manual Shift Settings ===")
    for finger in active_fingers:
        shift = manual_shifts.get(finger, 0)
        print(f"{finger}: {shift:+d} samples")
    print("=" * 30)
    
    # Analyze only active fingers
    all_finger_data = {}
    
    for finger_name in active_fingers:
        if finger_name in finger_configs:
            col = finger_configs[finger_name]
            shift = manual_shifts.get(finger_name, 0)  # Get shift for this finger, default to 0
            
            if col in df.columns and col in dfo.columns:
                normalized_iterations, metrics = analyze_single_finger(df, dfo, col, finger_name, split_idx, shift)
                
                # Only add to all_finger_data if we have valid iterations
                if len(normalized_iterations) > 0:
                    all_finger_data[finger_name] = (normalized_iterations, metrics)
                    
                    # Print quality metrics for this finger
                    print(f"\n=== {finger_name} Finger Metrics (shift: {shift:+d}) ===")
                    print(f"Overall MSE: {metrics['overall_mse']:.3f}")
                    print(f"Overall MAE: {metrics['overall_mae']:.3f}")
                    print(f"Average prediction std: {metrics['avg_prediction_std']:.3f}")
                else:
                    # Still add to maintain the layout, but with empty data
                    all_finger_data[finger_name] = ([], metrics)
                    print(f"\n=== {finger_name} Finger (shift: {shift:+d}) ===")
                    print("No valid iterations found (insufficient minima)")
            else:
                print(f"Warning: Column {col} not found for {finger_name}")
        else:
            print(f"Warning: Unknown finger name {finger_name}")
    
    # Create the visualization for active fingers only
    if all_finger_data:
        fig = create_active_fingers_visualization(all_finger_data, active_fingers)
        plt.show()
    else:
        print("No valid finger data found for active fingers!")

if __name__ == "__main__":
    main()