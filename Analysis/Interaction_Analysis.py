import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.signal import correlate
from scipy.stats import pearsonr
import os
from pathlib import Path
import warnings
import json
warnings.filterwarnings('ignore')

class FullPatientAnalyzer:
    def __init__(self, patient_name, intact_hand='Right'):
        """
        Comprehensive analyzer for all patient data across all interactions and experiments.
        
        Args:
            patient_name (str): Patient name (e.g., 'patient1')
            intact_hand (str): Hand type ('Right' or 'Left')
        """
        self.patient_name = patient_name
        self.patient_folder = Path('data') / patient_name
        self.intact_hand = intact_hand
        
        # Analysis results will be stored here
        self.analysis_results_folder = self.patient_folder / 'analysis_results'
        self.analysis_results_folder.mkdir(exist_ok=True)
        
        # Focus on the most important DoFs for analysis
        self.position_dofs = ['index_Pos', 'middle_Pos', 'ring_Pos', 'pinky_Pos', 'thumbFlex_Pos']
        self.position_labels = ['Index', 'Middle', 'Ring', 'Pinky', 'Thumb Flex']
        
        # Corresponding force inputs
        self.force_inputs = ['index_Force', 'middle_Force', 'ring_Force', 'pinky_Force', 'thumb_Force']
        self.force_labels = ['Index Force', 'Middle Force', 'Ring Force', 'Pinky Force', 'Thumb Force']
        
        # Define interaction-specific active fingers
        self.interaction_configs = {
            'pinch_interaction': {
                'active_positions': ['index_Pos', 'thumbFlex_Pos', 'thumbRot_Pos'],
                'active_forces': ['index_Force', 'thumb_Force'],
                'description': 'Pinch Grasp (Thumb + Index)'
            },
            'tripod_interaction': {
                'active_positions': ['index_Pos', 'middle_Pos', 'thumbFlex_Pos', 'thumbRot_Pos'],
                'active_forces': ['index_Force', 'middle_Force', 'thumb_Force'],
                'description': 'Tripod Grasp (Thumb + Index + Middle)'
            },
            'hook_interaction': {
                'active_positions': ['index_Pos', 'middle_Pos', 'ring_Pos', 'pinky_Pos'],
                'active_forces': ['index_Force', 'middle_Force', 'ring_Force', 'pinky_Force'],
                'description': 'Hook Grasp (All fingers except thumb)'
            },
            'power_grip_interaction': {
                'active_positions': ['index_Pos', 'middle_Pos', 'ring_Pos', 'pinky_Pos', 'thumbFlex_Pos', 'thumbRot_Pos'],
                'active_forces': ['index_Force', 'middle_Force', 'ring_Force', 'pinky_Force', 'thumb_Force'],
                'description': 'Power Grip (All fingers)'
            }
        }
        
        # Store all results for cross-analysis
        self.all_results = {}
        
    def get_active_dofs_for_interaction(self, interaction_name):
        """Get the active DoFs for a specific interaction type."""
        if interaction_name in self.interaction_configs:
            config = self.interaction_configs[interaction_name]
            return config['active_positions'], config['active_forces'], config['description']
        else:
            # Default to all if interaction not recognized
            print(f"Warning: Unknown interaction '{interaction_name}', using all DoFs")
            return self.position_dofs, self.force_inputs, "Unknown Interaction (All DoFs)"
        
    def discover_data_structure(self):
        """Discover all interaction folders and experiments for the patient."""
        recordings_folder = self.patient_folder / 'recordings'
        
        if not recordings_folder.exists():
            raise FileNotFoundError(f"Recordings folder not found: {recordings_folder}")
        
        self.data_structure = {}
        
        # Find all *_interaction folders
        interaction_folders = [f for f in recordings_folder.iterdir() 
                             if f.is_dir() and f.name.endswith('_interaction')]
        
        if not interaction_folders:
            raise FileNotFoundError(f"No *_interaction folders found in {recordings_folder}")
        
        print(f"Found {len(interaction_folders)} interaction types:")
        
        for interaction_folder in interaction_folders:
            interaction_name = interaction_folder.name
            experiments_folder = interaction_folder / 'experiments'
            
            if experiments_folder.exists():
                # Find all numbered experiment folders
                experiment_folders = [f for f in experiments_folder.iterdir() 
                                    if f.is_dir() and f.name.isdigit()]
                
                if experiment_folders:
                    experiment_numbers = sorted([int(f.name) for f in experiment_folders])
                    self.data_structure[interaction_name] = experiment_numbers
                    print(f"  • {interaction_name}: Experiments {experiment_numbers}")
                else:
                    print(f"  • {interaction_name}: No numbered experiment folders found")
            else:
                print(f"  • {interaction_name}: No experiments folder found")
        
        if not self.data_structure:
            raise FileNotFoundError("No valid data structure found")
        
        total_experiments = sum(len(experiments) for experiments in self.data_structure.values())
        print(f"\nTotal experiments to analyze: {total_experiments}")
        
    def load_single_experiment_data(self, interaction_name, experiment_num):
        """Load data for a single experiment."""
        exp_path = (self.patient_folder / 'recordings' / interaction_name / 
                   'experiments' / str(experiment_num))
        
        pred_file = exp_path / f'pred_angles-{self.patient_name}.parquet'
        gt_file = exp_path / 'aligned_angles.parquet'
        
        if not pred_file.exists():
            raise FileNotFoundError(f"Prediction file not found: {pred_file}")
        if not gt_file.exists():
            raise FileNotFoundError(f"Ground truth file not found: {gt_file}")
        
        pred_df = pd.read_parquet(pred_file)
        gt_df = pd.read_parquet(gt_file)
        
        # Use last N alignment (worked best in diagnostic)
        total_pred = len(pred_df)
        gt_test = gt_df.iloc[-total_pred:].reset_index(drop=True)
        pred_test = pred_df.reset_index(drop=True)
        
        return gt_test, pred_test
    
    def extract_analysis_data(self, gt_test, pred_test):
        """Extract clean data for analysis from a single experiment."""
        analysis_data = {}
        
        for i, (dof, force_input) in enumerate(zip(self.position_dofs, self.force_inputs)):
            pos_col = f"('{self.intact_hand}', '{dof}')"
            force_col = f"('{self.intact_hand}', '{force_input}')"
            
            if (pos_col in gt_test.columns and pos_col in pred_test.columns and 
                force_col in gt_test.columns):
                
                # Extract data
                gt_pos = gt_test[pos_col].values
                pred_pos = pred_test[pos_col].values
                force = gt_test[force_col].values
                
                # Remove NaN values
                mask = ~(np.isnan(gt_pos) | np.isnan(pred_pos) | np.isnan(force))
                
                if np.sum(mask) > 10:  # Need sufficient data points
                    analysis_data[dof] = {
                        'gt_position': gt_pos[mask],
                        'pred_position': pred_pos[mask],
                        'force': force[mask],
                        'label': self.position_labels[i],
                        'force_label': self.force_labels[i]
                    }
        
        return analysis_data
    
    def extract_active_analysis_data(self, gt_test, pred_test, interaction_name):
        """Extract analysis data for only active fingers in this interaction."""
        analysis_data = {}
        
        # Get active DoFs for this interaction
        active_positions, active_forces, _ = self.get_active_dofs_for_interaction(interaction_name)
        
        # Only analyze active DoFs
        for dof in active_positions:
            # Find corresponding force input
            force_input = None
            if dof in ['index_Pos']:
                force_input = 'index_Force'
            elif dof in ['middle_Pos']:
                force_input = 'middle_Force'
            elif dof in ['ring_Pos']:
                force_input = 'ring_Force'
            elif dof in ['pinky_Pos']:
                force_input = 'pinky_Force'
            elif dof in ['thumbFlex_Pos', 'thumbRot_Pos']:
                force_input = 'thumb_Force'
            
            # Skip if this force is not active for this interaction
            if force_input not in active_forces:
                continue
            
            pos_col = f"('{self.intact_hand}', '{dof}')"
            force_col = f"('{self.intact_hand}', '{force_input}')"
            
            if (pos_col in gt_test.columns and pos_col in pred_test.columns and 
                force_col in gt_test.columns):
                
                # Extract data
                gt_pos = gt_test[pos_col].values
                pred_pos = pred_test[pos_col].values
                force = gt_test[force_col].values
                
                # Remove NaN values
                mask = ~(np.isnan(gt_pos) | np.isnan(pred_pos) | np.isnan(force))
                
                if np.sum(mask) > 10:  # Need sufficient data points
                    # Get proper labels
                    try:
                        dof_idx = self.position_dofs.index(dof)
                        dof_label = self.position_labels[dof_idx]
                    except ValueError:
                        dof_label = dof.replace('_Pos', '')
                    
                    try:
                        force_idx = self.force_inputs.index(force_input)
                        force_label = self.force_labels[force_idx]
                    except ValueError:
                        force_label = force_input.replace('_Force', '')
                    
                    analysis_data[dof] = {
                        'gt_position': gt_pos[mask],
                        'pred_position': pred_pos[mask],
                        'force': force[mask],
                        'label': dof_label,
                        'force_label': force_label,
                        'is_active': True
                    }
        
        return analysis_data
    
    def calculate_curve_correlation(self, signal1, signal2, max_lag=50):
        """Calculate cross-correlation to find how similar curve patterns are."""
        if len(signal1) != len(signal2) or len(signal1) < max_lag * 2:
            return 0, 0
            
        # Normalize signals to focus on pattern, not magnitude
        s1_norm = (signal1 - np.mean(signal1)) / (np.std(signal1) + 1e-8)
        s2_norm = (signal2 - np.mean(signal2)) / (np.std(signal2) + 1e-8)
        
        # Calculate cross-correlation
        cross_corr = correlate(s1_norm, s2_norm, mode='full')
        
        # Find the lag with maximum correlation
        lags = np.arange(-len(s1_norm) + 1, len(s1_norm))
        
        # Focus on small lags (real-time relationship)
        lag_mask = np.abs(lags) <= max_lag
        if np.sum(lag_mask) > 0:
            best_idx = np.argmax(np.abs(cross_corr[lag_mask]))
            valid_lags = lags[lag_mask]
            valid_corrs = cross_corr[lag_mask]
            
            best_lag = valid_lags[best_idx]
            best_correlation = valid_corrs[best_idx] / len(s1_norm)  # Normalize
        else:
            best_lag = 0
            best_correlation = np.corrcoef(s1_norm, s2_norm)[0, 1]
            
        return best_correlation, best_lag
    
    def analyze_single_experiment(self, analysis_data):
        """Analyze relationships for a single experiment."""
        relationships = {}
        
        for dof in analysis_data:
            data = analysis_data[dof]
            
            try:
                # Basic correlations
                force_gt_corr = np.corrcoef(data['force'], data['gt_position'])[0, 1]
                force_pred_corr = np.corrcoef(data['force'], data['pred_position'])[0, 1]
                gt_pred_corr = np.corrcoef(data['gt_position'], data['pred_position'])[0, 1]
                
                # Curve pattern correlations
                force_gt_curve_corr, force_gt_lag = self.calculate_curve_correlation(
                    data['force'], data['gt_position'])
                force_pred_curve_corr, force_pred_lag = self.calculate_curve_correlation(
                    data['force'], data['pred_position'])
                
                # Performance metrics
                rmse = np.sqrt(mean_squared_error(data['gt_position'], data['pred_position']))
                r2 = r2_score(data['gt_position'], data['pred_position'])
                
                relationships[dof] = {
                    'force_gt_corr': force_gt_corr,
                    'force_pred_corr': force_pred_corr,
                    'gt_pred_corr': gt_pred_corr,
                    'force_gt_curve_corr': force_gt_curve_corr,
                    'force_pred_curve_corr': force_pred_curve_corr,
                    'force_gt_lag': force_gt_lag,
                    'force_pred_lag': force_pred_lag,
                    'rmse': rmse,
                    'r2': max(r2, -10),  # Clamp for display
                    'data_std': np.std(data['gt_position']),
                    'n_samples': len(data['gt_position'])
                }
                
            except Exception as e:
                print(f"    Warning: Analysis failed for {dof}: {e}")
                relationships[dof] = None
        
        return relationships
    
    def create_interaction_specific_timeline(self, analysis_data, relationships, interaction_name, experiment_num):
        """Create timeline showing only active fingers for this interaction type."""
        # Get interaction configuration
        active_positions, active_forces, interaction_description = self.get_active_dofs_for_interaction(interaction_name)
        
        # Extract only the active analysis data
        gt_test, pred_test = self.load_single_experiment_data(interaction_name, experiment_num)
        active_data = self.extract_active_analysis_data(gt_test, pred_test, interaction_name)
        
        if not active_data:
            print(f"    No active finger data for {interaction_name}")
            return None
        
        n_active_dofs = len(active_data)
        
        # Create figure with appropriate size
        fig_height = max(8, 2.5 * n_active_dofs)
        fig, axes = plt.subplots(n_active_dofs, 1, figsize=(16, fig_height))
        
        # Handle single subplot case
        if n_active_dofs == 1:
            axes = [axes]
        
        # Plot each active DoF
        plot_idx = 0
        for dof in active_data.keys():
            if plot_idx >= len(axes):
                break
                
            ax = axes[plot_idx]
            data = active_data[dof]
            
            # Downsample for plotting
            max_points = 600
            n_samples = len(data['force'])
            if n_samples > max_points:
                indices = np.linspace(0, n_samples-1, max_points, dtype=int)
                force_plot = data['force'][indices]
                gt_plot = data['gt_position'][indices]
                pred_plot = data['pred_position'][indices]
                time_idx = indices
            else:
                force_plot = data['force']
                gt_plot = data['gt_position']
                pred_plot = data['pred_position']
                time_idx = np.arange(n_samples)
            
            # Create twin axis for force
            ax2 = ax.twinx()
            
            # Plot positions on left axis (enhanced styling)
            line1 = ax.plot(time_idx, gt_plot, 'r-', linewidth=2.5, alpha=0.9, label='Actual Position')
            line2 = ax.plot(time_idx, pred_plot, 'b-', linewidth=2.5, alpha=0.9, label='Predicted Position')
            
            # Plot force on right axis
            line3 = ax2.plot(time_idx, force_plot, 'g-', linewidth=2, alpha=0.8, label='Force Input')
            
            # Enhanced formatting
            ax.set_ylabel('Position (degrees)', color='black', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Force (N)', color='green', fontsize=12, fontweight='bold')
            ax2.tick_params(axis='y', labelcolor='green')
            
            # Title with active status and metrics
            ax.set_title(f'🎯 {data["label"]} (ACTIVE in {interaction_description})', 
                       fontsize=12, fontweight='bold', pad=15)
            
            # Enhanced legend
            lines = line1 + line2 + line3
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper left', fontsize=11, framealpha=0.9)
            
            # Enhanced grid
            ax.grid(True, alpha=0.4, linewidth=0.8)
            
            # Only show x-label on bottom plot
            if plot_idx == n_active_dofs - 1:
                ax.set_xlabel('Sample Index', fontsize=12, fontweight='bold')
            else:
                ax.tick_params(axis='x', labelbottom=False)
            
            plot_idx += 1
        
        plt.tight_layout()
        
        # Enhanced title with interaction info
        plt.suptitle(f'{self.patient_name} - {interaction_name.replace("_", " ").title()} - Experiment {experiment_num}\n'
                     f'🎯 {interaction_description} - Active Fingers Only',
                     y=0.98, fontsize=14, fontweight='bold')
        
        # Add interaction info box
        info_text = f"Active Positions: {', '.join([pos.replace('_Pos', '') for pos in active_positions])}\n"
        info_text += f"Active Forces: {', '.join([force.replace('_Force', '') for force in active_forces])}"
        
        fig.text(0.02, 0.02, info_text, 
                fontsize=10, style='italic', 
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightblue", alpha=0.8))
        
        return fig
    
    def create_force_position_timeline(self, analysis_data, relationships, interaction_name, experiment_num):
        """Create timeline showing force, actual position, and predicted position."""
        n_joints = len(analysis_data)
        if n_joints == 0:
            return None
            
        fig, axes = plt.subplots(n_joints, 1, figsize=(16, 3 * n_joints))
        if n_joints == 1:
            axes = [axes]
            
        for i, dof in enumerate(analysis_data.keys()):
            ax = axes[i]
            data = analysis_data[dof]
            
            # Downsample for plotting
            max_points = 500
            n_samples = len(data['force'])
            if n_samples > max_points:
                indices = np.linspace(0, n_samples-1, max_points, dtype=int)
                force_plot = data['force'][indices]
                gt_plot = data['gt_position'][indices]
                pred_plot = data['pred_position'][indices]
                time_idx = indices
            else:
                force_plot = data['force']
                gt_plot = data['gt_position']
                pred_plot = data['pred_position']
                time_idx = np.arange(n_samples)
            
            # Create twin axis for force
            ax2 = ax.twinx()
            
            # Plot positions on left axis
            line1 = ax.plot(time_idx, gt_plot, 'r-', linewidth=2, alpha=0.8, label='Actual Position')
            line2 = ax.plot(time_idx, pred_plot, 'b-', linewidth=2, alpha=0.8, label='Predicted Position')
            
            # Plot force on right axis
            line3 = ax2.plot(time_idx, force_plot, 'g-', linewidth=1.5, alpha=0.7, label='Force Input')
            
            # Formatting
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Position (degrees)', color='black')
            ax2.set_ylabel('Force (N)', color='green')
            ax2.tick_params(axis='y', labelcolor='green')
            
            # Title with key metrics
            if dof in relationships and relationships[dof]:
                rel = relationships[dof]
                ax.set_title(f'{data["label"]} - Multi-Modal Control\n'
                            f'Position R²: {rel["r2"]:.3f} | '
                            f'Force-Position Corr: {rel["force_gt_corr"]:.3f} | '
                            f'Curve Pattern Similarity: {rel["force_gt_curve_corr"]:.3f}',
                            fontsize=11, fontweight='bold')
            else:
                ax.set_title(f'{data["label"]} - Multi-Modal Control', 
                           fontsize=11, fontweight='bold')
            
            # Combined legend
            lines = line1 + line2 + line3
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper left')
            
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle(f'{self.patient_name} - {interaction_name} - Experiment {experiment_num}\n'
                     f'Force-Position Control Analysis',
                     y=0.98, fontsize=14, fontweight='bold')
        return fig
    
    def create_correlation_analysis(self, analysis_data, relationships, interaction_name, experiment_num):
        """Create correlation analysis plots."""
        n_joints = len(analysis_data)
        if n_joints == 0:
            return None
            
        fig, axes = plt.subplots(2, n_joints, figsize=(4 * n_joints, 8))
        if n_joints == 1:
            axes = axes.reshape(-1, 1)
        
        for i, dof in enumerate(analysis_data.keys()):
            data = analysis_data[dof]
            
            # Top row: Force vs Position correlation
            ax1 = axes[0, i]
            
            # Sample for scatter plot
            n_samples = len(data['force'])
            if n_samples > 300:
                indices = np.random.choice(n_samples, 300, replace=False)
                force_scatter = data['force'][indices]
                gt_scatter = data['gt_position'][indices]
                pred_scatter = data['pred_position'][indices]
            else:
                force_scatter = data['force']
                gt_scatter = data['gt_position']
                pred_scatter = data['pred_position']
            
            # Force vs Position
            ax1.scatter(force_scatter, gt_scatter, alpha=0.6, s=20, color='red', label='Actual')
            ax1.scatter(force_scatter, pred_scatter, alpha=0.6, s=20, color='blue', label='Predicted')
            
            ax1.set_xlabel('Force (N)')
            ax1.set_ylabel('Position (degrees)')
            ax1.set_title(f'{data["label"]}\nForce → Position Relationship')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Add correlation text
            if dof in relationships and relationships[dof]:
                rel = relationships[dof]
                ax1.text(0.05, 0.95, f'Corr (Force→Actual): {rel["force_gt_corr"]:.3f}\n'
                                     f'Corr (Force→Pred): {rel["force_pred_corr"]:.3f}',
                        transform=ax1.transAxes, fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Bottom row: Predicted vs Actual Position
            ax2 = axes[1, i]
            
            ax2.scatter(gt_scatter, pred_scatter, alpha=0.6, s=20, color='purple')
            
            # Perfect prediction line
            min_pos = min(np.min(gt_scatter), np.min(pred_scatter))
            max_pos = max(np.max(gt_scatter), np.max(pred_scatter))
            ax2.plot([min_pos, max_pos], [min_pos, max_pos], 'k--', alpha=0.5, label='Perfect')
            
            ax2.set_xlabel('Actual Position (degrees)')
            ax2.set_ylabel('Predicted Position (degrees)')
            if dof in relationships and relationships[dof]:
                ax2.set_title(f'Position Control Accuracy\nR² = {relationships[dof]["r2"]:.3f}')
            else:
                ax2.set_title(f'Position Control Accuracy')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        plt.suptitle(f'{self.patient_name} - {interaction_name} - Experiment {experiment_num}\n'
                     f'Correlation Analysis',
                     y=0.98, fontsize=14, fontweight='bold')
        return fig
    
    def create_pattern_similarity_analysis(self, analysis_data, relationships, interaction_name, experiment_num):
        """Analyze and visualize curve pattern similarities."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Collect data for analysis
        joint_names = []
        force_gt_corrs = []
        force_pred_corrs = []
        position_r2s = []
        curve_similarities = []
        
        for dof in analysis_data:
            if dof in relationships and relationships[dof]:
                rel = relationships[dof]
                joint_names.append(analysis_data[dof]['label'])
                force_gt_corrs.append(rel['force_gt_corr'])
                force_pred_corrs.append(rel['force_pred_corr'])
                position_r2s.append(rel['r2'])
                curve_similarities.append(rel['force_gt_curve_corr'])
        
        if not joint_names:
            plt.close(fig)
            return None
        
        # Plot 1: Force-Position Correlations
        ax1 = axes[0, 0]
        x = np.arange(len(joint_names))
        width = 0.35
        
        ax1.bar(x - width/2, force_gt_corrs, width, label='Force ↔ Actual Pos', alpha=0.8, color='red')
        ax1.bar(x + width/2, force_pred_corrs, width, label='Force ↔ Predicted Pos', alpha=0.8, color='blue')
        
        ax1.set_xlabel('Joint')
        ax1.set_ylabel('Correlation')
        ax1.set_title('Force-Position Correlations')
        ax1.set_xticks(x)
        ax1.set_xticklabels(joint_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-1, 1)
        
        # Plot 2: Position Control Performance
        ax2 = axes[0, 1]
        bars = ax2.bar(joint_names, position_r2s, alpha=0.8, color='green')
        ax2.set_xlabel('Joint')
        ax2.set_ylabel('R² Score')
        ax2.set_title('Position Control Performance')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add R² values on bars
        for bar, r2 in zip(bars, position_r2s):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{r2:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 3: Curve Pattern Similarity
        ax3 = axes[1, 0]
        ax3.bar(joint_names, curve_similarities, alpha=0.8, color='orange')
        ax3.set_xlabel('Joint')
        ax3.set_ylabel('Curve Pattern Similarity')
        ax3.set_title('Force-Position Curve Pattern Matching')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(-1, 1)
        
        # Plot 4: Summary Table
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Create summary table
        summary_data = []
        for i, joint in enumerate(joint_names):
            summary_data.append([
                joint,
                f"{force_gt_corrs[i]:.3f}",
                f"{position_r2s[i]:.3f}",
                f"{curve_similarities[i]:.3f}"
            ])
        
        # Add averages
        if joint_names:
            summary_data.append([
                "AVERAGE",
                f"{np.mean(force_gt_corrs):.3f}",
                f"{np.mean(position_r2s):.3f}",
                f"{np.mean(curve_similarities):.3f}"
            ])
        
        table = ax4.table(cellText=summary_data,
                         colLabels=['Joint', 'Force-Pos\nCorrelation', 'Position\nR²', 'Curve\nSimilarity'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        ax4.set_title('Performance Summary', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.suptitle(f'{self.patient_name} - {interaction_name} - Experiment {experiment_num}\n'
                     f'Pattern Analysis',
                     y=0.98, fontsize=14, fontweight='bold')
        return fig
    
    def create_experiment_plots(self, analysis_data, relationships, interaction_name, experiment_num):
        """Create plots for a single experiment."""
        if not analysis_data:
            return []
        
        plots = []
        
        # Create output directory for this specific experiment
        output_dir = (self.analysis_results_folder / interaction_name / 
                     'experiments' / str(experiment_num))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 1. NEW: Interaction-specific timeline (only active fingers)
            fig0 = self.create_interaction_specific_timeline(analysis_data, relationships, 
                                                           interaction_name, experiment_num)
            if fig0:
                filename0 = output_dir / 'interaction_specific_timeline.png'
                fig0.savefig(filename0, dpi=150, bbox_inches='tight', facecolor='white')
                plots.append(('interaction_specific_timeline', filename0))
                plt.close(fig0)
            
            # 2. Force-Position Timeline (all analyzed fingers)
            fig1 = self.create_force_position_timeline(analysis_data, relationships, 
                                                     interaction_name, experiment_num)
            if fig1:
                filename1 = output_dir / 'force_position_timeline.png'
                fig1.savefig(filename1, dpi=150, bbox_inches='tight', facecolor='white')
                plots.append(('force_position_timeline', filename1))
                plt.close(fig1)
            
            # 3. Correlation Analysis
            fig2 = self.create_correlation_analysis(analysis_data, relationships, 
                                                   interaction_name, experiment_num)
            if fig2:
                filename2 = output_dir / 'correlation_analysis.png'
                fig2.savefig(filename2, dpi=150, bbox_inches='tight', facecolor='white')
                plots.append(('correlation_analysis', filename2))
                plt.close(fig2)
            
            # 4. Pattern Analysis
            fig3 = self.create_pattern_similarity_analysis(analysis_data, relationships, 
                                                         interaction_name, experiment_num)
            if fig3:
                filename3 = output_dir / 'pattern_analysis.png'
                fig3.savefig(filename3, dpi=150, bbox_inches='tight', facecolor='white')
                plots.append(('pattern_analysis', filename3))
                plt.close(fig3)
                
        except Exception as e:
            print(f"    Warning: Plot creation failed: {e}")
        
        return plots
    
    def save_experiment_summary(self, analysis_data, relationships, interaction_name, experiment_num):
        """Save numerical results as JSON for later cross-analysis."""
        output_dir = (self.analysis_results_folder / interaction_name / 
                     'experiments' / str(experiment_num))
        
        # Prepare summary data
        summary = {
            'patient': self.patient_name,
            'interaction': interaction_name,
            'experiment': experiment_num,
            'n_joints_analyzed': len(analysis_data),
            'joints': {}
        }
        
        for dof in analysis_data:
            if dof in relationships and relationships[dof]:
                rel = relationships[dof]
                summary['joints'][dof] = {
                    'label': analysis_data[dof]['label'],
                    'force_gt_correlation': float(rel['force_gt_corr']),
                    'force_pred_correlation': float(rel['force_pred_corr']),
                    'gt_pred_correlation': float(rel['gt_pred_corr']),
                    'force_gt_curve_correlation': float(rel['force_gt_curve_corr']),
                    'position_r2': float(rel['r2']),
                    'position_rmse': float(rel['rmse']),
                    'n_samples': int(rel['n_samples'])
                }
        
        # Save as JSON
        summary_file = output_dir / 'analysis_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary_file
    
    def analyze_single_experiment_complete(self, interaction_name, experiment_num):
        """Complete analysis for a single experiment."""
        print(f"  Analyzing {interaction_name} - Experiment {experiment_num}...")
        
        try:
            # Load data
            gt_test, pred_test = self.load_single_experiment_data(interaction_name, experiment_num)
            
            # Extract analysis data
            analysis_data = self.extract_analysis_data(gt_test, pred_test)
            
            if not analysis_data:
                print(f"    No valid analysis data found")
                return None
            
            print(f"    Found {len(analysis_data)} joints with valid data")
            
            # Analyze relationships
            relationships = self.analyze_single_experiment(analysis_data)
            
            # Create and save plots
            plots = self.create_experiment_plots(analysis_data, relationships, 
                                               interaction_name, experiment_num)
            
            # Save numerical summary
            summary_file = self.save_experiment_summary(analysis_data, relationships, 
                                                      interaction_name, experiment_num)
            
            print(f"    ✓ Created {len(plots)} plots and saved summary")
            
            # Store for cross-analysis
            experiment_key = f"{interaction_name}_exp{experiment_num}"
            self.all_results[experiment_key] = {
                'interaction': interaction_name,
                'experiment': experiment_num,
                'analysis_data': analysis_data,
                'relationships': relationships,
                'plots': plots,
                'summary_file': summary_file
            }
            
            return self.all_results[experiment_key]
            
        except Exception as e:
            print(f"    ⚠️ Failed: {e}")
            return None
    
    def create_cross_experiment_analysis(self):
        """Create analysis comparing across all experiments and interactions."""
        if not self.all_results:
            print("No results available for cross-analysis")
            return
        
        print(f"\nCreating cross-experiment analysis...")
        
        # Collect all data for comparison
        comparison_data = {
            'interactions': [],
            'experiments': [],
            'avg_r2': [],
            'avg_force_corr': [],
            'avg_curve_sim': [],
            'n_joints': []
        }
        
        for key, result in self.all_results.items():
            if result and result['relationships']:
                # Calculate averages for this experiment
                valid_relationships = [rel for rel in result['relationships'].values() if rel]
                
                if valid_relationships:
                    avg_r2 = np.mean([rel['r2'] for rel in valid_relationships])
                    avg_force_corr = np.mean([rel['force_gt_corr'] for rel in valid_relationships])
                    avg_curve_sim = np.mean([rel['force_gt_curve_corr'] for rel in valid_relationships])
                    
                    comparison_data['interactions'].append(result['interaction'])
                    comparison_data['experiments'].append(result['experiment'])
                    comparison_data['avg_r2'].append(avg_r2)
                    comparison_data['avg_force_corr'].append(avg_force_corr)
                    comparison_data['avg_curve_sim'].append(avg_curve_sim)
                    comparison_data['n_joints'].append(len(valid_relationships))
        
        if not comparison_data['interactions']:
            print("No valid data for cross-analysis")
            return
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: R² by interaction type
        ax1 = axes[0, 0]
        interactions = comparison_data['interactions']
        r2_values = comparison_data['avg_r2']
        
        # Group by interaction type
        unique_interactions = list(set(interactions))
        interaction_r2s = {interaction: [] for interaction in unique_interactions}
        
        for interaction, r2 in zip(interactions, r2_values):
            interaction_r2s[interaction].append(r2)
        
        # Box plot
        ax1.boxplot([interaction_r2s[interaction] for interaction in unique_interactions],
                   labels=unique_interactions)
        ax1.set_title('Position Control Performance by Interaction Type')
        ax1.set_ylabel('Average R² Score')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Force correlation by interaction type
        ax2 = axes[0, 1]
        force_corrs = comparison_data['avg_force_corr']
        interaction_force_corrs = {interaction: [] for interaction in unique_interactions}
        
        for interaction, force_corr in zip(interactions, force_corrs):
            interaction_force_corrs[interaction].append(force_corr)
        
        ax2.boxplot([interaction_force_corrs[interaction] for interaction in unique_interactions],
                   labels=unique_interactions)
        ax2.set_title('Force-Position Correlation by Interaction Type')
        ax2.set_ylabel('Average Force-Position Correlation')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Performance over experiments
        ax3 = axes[1, 0]
        experiments = comparison_data['experiments']
        ax3.scatter(experiments, r2_values, c=[unique_interactions.index(i) for i in interactions], 
                   alpha=0.7, s=60)
        ax3.set_xlabel('Experiment Number')
        ax3.set_ylabel('Average R² Score')
        ax3.set_title('Performance Trend Across Experiments')
        ax3.grid(True, alpha=0.3)
        
        # Add legend for interactions
        for i, interaction in enumerate(unique_interactions):
            ax3.scatter([], [], c=f'C{i}', label=interaction, s=60)
        ax3.legend()
        
        # Plot 4: Summary statistics table
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Calculate summary statistics
        summary_stats = []
        for interaction in unique_interactions:
            interaction_data = [(r2, fc, cs) for r2, fc, cs, inter in 
                              zip(r2_values, force_corrs, comparison_data['avg_curve_sim'], interactions)
                              if inter == interaction]
            
            if interaction_data:
                r2s, fcs, css = zip(*interaction_data)
                summary_stats.append([
                    interaction,
                    f"{np.mean(r2s):.3f}",
                    f"{np.mean(fcs):.3f}",
                    f"{np.mean(css):.3f}",
                    f"{len(interaction_data)}"
                ])
        
        # Overall summary
        summary_stats.append([
            "OVERALL",
            f"{np.mean(r2_values):.3f}",
            f"{np.mean(force_corrs):.3f}",
            f"{np.mean(comparison_data['avg_curve_sim']):.3f}",
            f"{len(r2_values)}"
        ])
        
        table = ax4.table(cellText=summary_stats,
                         colLabels=['Interaction', 'Avg R²', 'Avg Force\nCorrelation', 'Avg Curve\nSimilarity', 'N Exp'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.3, 2)
        ax4.set_title('Cross-Experiment Summary', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.suptitle(f'{self.patient_name} - Cross-Experiment Analysis\n'
                     f'Performance Comparison Across All Interactions',
                     y=0.98, fontsize=16, fontweight='bold')
        
        # Save cross-analysis plot
        cross_analysis_file = self.analysis_results_folder / 'cross_experiment_analysis.png'
        fig.savefig(cross_analysis_file, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        print(f"✓ Cross-experiment analysis saved: {cross_analysis_file}")
        
        return cross_analysis_file
    
    def create_comprehensive_patient_report(self):
        """Create a comprehensive report for the entire patient analysis."""
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE PATIENT ANALYSIS REPORT")
        print(f"{'='*80}")
        print(f"Patient: {self.patient_name}")
        print(f"Total experiments analyzed: {len(self.all_results)}")
        print(f"{'='*80}")
        
        if not self.all_results:
            print("No results available for reporting")
            return
        
        # Organize results by interaction type
        interaction_summary = {}
        all_r2s = []
        all_force_corrs = []
        all_curve_sims = []
        
        for key, result in self.all_results.items():
            if result and result['relationships']:
                interaction = result['interaction']
                
                if interaction not in interaction_summary:
                    interaction_summary[interaction] = {
                        'experiments': [],
                        'r2_scores': [],
                        'force_correlations': [],
                        'curve_similarities': [],
                        'n_joints_total': 0
                    }
                
                # Calculate averages for this experiment
                valid_relationships = [rel for rel in result['relationships'].values() if rel]
                
                if valid_relationships:
                    avg_r2 = np.mean([rel['r2'] for rel in valid_relationships])
                    avg_force_corr = np.mean([rel['force_gt_corr'] for rel in valid_relationships])
                    avg_curve_sim = np.mean([rel['force_gt_curve_corr'] for rel in valid_relationships])
                    
                    interaction_summary[interaction]['experiments'].append(result['experiment'])
                    interaction_summary[interaction]['r2_scores'].append(avg_r2)
                    interaction_summary[interaction]['force_correlations'].append(avg_force_corr)
                    interaction_summary[interaction]['curve_similarities'].append(avg_curve_sim)
                    interaction_summary[interaction]['n_joints_total'] += len(valid_relationships)
                    
                    all_r2s.append(avg_r2)
                    all_force_corrs.append(avg_force_corr)
                    all_curve_sims.append(avg_curve_sim)
        
        # Report by interaction type
        print(f"\nPERFORMANCE BY INTERACTION TYPE:")
        print(f"{'-'*70}")
        print(f"{'Interaction':<20} {'Exp':<4} {'Avg R²':<8} {'Force Corr':<10} {'Curve Sim':<10}")
        print(f"{'-'*70}")
        
        for interaction, summary in interaction_summary.items():
            if summary['r2_scores']:
                avg_r2 = np.mean(summary['r2_scores'])
                avg_force_corr = np.mean(summary['force_correlations'])
                avg_curve_sim = np.mean(summary['curve_similarities'])
                n_exp = len(summary['experiments'])
                
                print(f"{interaction:<20} {n_exp:<4} {avg_r2:<8.3f} {avg_force_corr:<10.3f} {avg_curve_sim:<10.3f}")
        
        # Overall summary
        if all_r2s:
            print(f"{'-'*70}")
            print(f"{'OVERALL AVERAGE':<20} {len(all_r2s):<4} {np.mean(all_r2s):<8.3f} {np.mean(all_force_corrs):<10.3f} {np.mean(all_curve_sims):<10.3f}")
        
        # Key insights
        print(f"\nKEY INSIGHTS:")
        print(f"{'-'*50}")
        
        if all_r2s:
            best_interaction = max(interaction_summary.keys(), 
                                 key=lambda x: np.mean(interaction_summary[x]['r2_scores']) if interaction_summary[x]['r2_scores'] else -999)
            worst_interaction = min(interaction_summary.keys(), 
                                  key=lambda x: np.mean(interaction_summary[x]['r2_scores']) if interaction_summary[x]['r2_scores'] else 999)
            
            print(f"• Best performing interaction: {best_interaction} "
                  f"(Avg R² = {np.mean(interaction_summary[best_interaction]['r2_scores']):.3f})")
            print(f"• Most challenging interaction: {worst_interaction} "
                  f"(Avg R² = {np.mean(interaction_summary[worst_interaction]['r2_scores']):.3f})")
            print(f"• Overall control performance: R² = {np.mean(all_r2s):.3f}")
            print(f"• Force feedback integration: Correlation = {np.mean(all_force_corrs):.3f}")
            print(f"• Pattern synchronization: Similarity = {np.mean(all_curve_sims):.3f}")
            
            # Performance assessment
            excellent_exp = sum(1 for r2 in all_r2s if r2 > 0.7)
            good_exp = sum(1 for r2 in all_r2s if 0.4 < r2 <= 0.7)
            moderate_exp = sum(1 for r2 in all_r2s if 0.2 < r2 <= 0.4)
            poor_exp = sum(1 for r2 in all_r2s if r2 <= 0.2)
            
            print(f"• Excellent performance (R² > 0.7): {excellent_exp}/{len(all_r2s)} experiments")
            print(f"• Good performance (0.4 < R² ≤ 0.7): {good_exp}/{len(all_r2s)} experiments")
            print(f"• Moderate performance (0.2 < R² ≤ 0.4): {moderate_exp}/{len(all_r2s)} experiments")
            print(f"• Needs improvement (R² ≤ 0.2): {poor_exp}/{len(all_r2s)} experiments")
        
        print(f"\nDATA STRUCTURE ANALYZED:")
        print(f"{'-'*50}")
        for interaction, experiments in self.data_structure.items():
            analyzed_count = sum(1 for key in self.all_results.keys() 
                               if key.startswith(interaction.replace('_interaction', '_interaction')))
            print(f"• {interaction}: {analyzed_count}/{len(experiments)} experiments analyzed")
        
        print(f"\nOUTPUT FILES GENERATED:")
        print(f"{'-'*50}")
        total_plots = sum(len(result['plots']) for result in self.all_results.values() if result)
        print(f"• Interaction-specific timeline plots: {len(self.all_results)} (focused on active fingers)")
        print(f"• Individual experiment plots: {total_plots}")
        print(f"• Cross-experiment analysis: 1")
        print(f"• JSON summaries: {len(self.all_results)}")
        print(f"• Analysis results folder: {self.analysis_results_folder}")
    
    def run_full_patient_analysis(self):
        """Run the complete analysis for the entire patient."""
        print(f"Starting comprehensive analysis for patient: {self.patient_name}")
        print(f"{'='*80}")
        
        try:
            # Discover data structure
            self.discover_data_structure()
            
            # Analyze each experiment
            successful_analyses = 0
            failed_analyses = 0
            
            for interaction_name, experiment_numbers in self.data_structure.items():
                print(f"\nAnalyzing {interaction_name}:")
                
                for experiment_num in experiment_numbers:
                    result = self.analyze_single_experiment_complete(interaction_name, experiment_num)
                    
                    if result:
                        successful_analyses += 1
                    else:
                        failed_analyses += 1
            
            print(f"\n{'='*60}")
            print(f"INDIVIDUAL EXPERIMENT ANALYSIS COMPLETE")
            print(f"{'='*60}")
            print(f"Successful analyses: {successful_analyses}")
            print(f"Failed analyses: {failed_analyses}")
            
            # Create cross-experiment analysis
            if self.all_results:
                self.create_cross_experiment_analysis()
            
            # Generate comprehensive report
            self.create_comprehensive_patient_report()
            
            print(f"\n{'='*80}")
            print(f"FULL PATIENT ANALYSIS COMPLETED SUCCESSFULLY!")
            print(f"{'='*80}")
            print(f"Results saved to: {self.analysis_results_folder}")
            print(f"Total experiments analyzed: {len(self.all_results)}")
            
            return self.all_results
            
        except Exception as e:
            print(f"Error in full patient analysis: {e}")
            import traceback
            traceback.print_exc()
            return None

# Example usage
if __name__ == "__main__":
    # Set patient name here
    PATIENT_NAME = "P5_869_interaction"  # Change this to analyze different patients

    analyzer = FullPatientAnalyzer(
        patient_name=PATIENT_NAME,
        intact_hand='Left'
    )
    
    try:
        results = analyzer.run_full_patient_analysis()
        
        if results:
            print(f"\n🎉 Analysis complete! Check the results in:")
            print(f"📁 {analyzer.analysis_results_folder}")
            print(f"\nGenerated files:")
            print(f"• interaction_specific_timeline.png - NEW: Shows only active fingers per movement")
            print(f"• force_position_timeline.png - All analyzed fingers timeline")
            print(f"• correlation_analysis.png - Correlation analysis")
            print(f"• pattern_analysis.png - Pattern similarity analysis")
            print(f"• cross_experiment_analysis.png - Overall comparison")
            print(f"• analysis_summary.json files - Numerical data")
            print(f"\nMovement-specific analysis:")
            print(f"• Pinch: Shows only thumb + index")
            print(f"• Tripod: Shows only thumb + index + middle") 
            print(f"• Hook: Shows only fingers (no thumb)")
            print(f"• Power grip: Shows all fingers")
        else:
            print("❌ Analysis failed. Check the error messages above.")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()