#!/usr/bin/env python3
"""
Standalone script to debug EMG noise level calculation
Visualizes each step of the filtering pipeline to identify issues
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import argparse
import os
from helpers.BesselFilter import BesselFilterArr

def plot_signal_comparison(signals, titles, time_axes, noise_levels=None, 
                          figure_title="Signal Comparison", channels_to_plot=None):
    """Plot multiple signals for comparison with optional noise level lines"""
    n_signals = len(signals)
    n_channels = signals[0].shape[0]
    
    if channels_to_plot is None:
        channels_to_plot = range(n_channels)
    
    n_plot_channels = len(channels_to_plot)
    
    fig, axes = plt.subplots(n_plot_channels, n_signals, figsize=(5*n_signals, 3*n_plot_channels))
    if n_plot_channels == 1:
        axes = axes.reshape(1, -1)
    if n_signals == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle(figure_title, fontsize=16)
    
    for i, ch in enumerate(channels_to_plot):
        for j, (signal, title) in enumerate(zip(signals, titles)):
            ax = axes[i, j]
            
            # Use the appropriate time axis for each signal
            time_axis = time_axes[j] if isinstance(time_axes, list) else time_axes
            
            # Plot signal
            ax.plot(time_axis, signal[ch, :], 'b-', linewidth=0.5, alpha=0.8)
            
            # Add noise level line if provided
            if noise_levels is not None and j == len(signals) - 1:  # Only on last column
                ax.axhline(y=noise_levels[ch], color='r', linestyle='--', 
                          linewidth=2, label=f'Noise: {noise_levels[ch]:.2f}')
                ax.legend()
            
            # Calculate and display statistics
            mean_val = np.mean(signal[ch, :])
            std_val = np.std(signal[ch, :])
            max_val = np.max(signal[ch, :])
            min_val = np.min(signal[ch, :])
            
            ax.set_title(f'{title} - Ch {ch}')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
            ax.grid(True, alpha=0.3)
            
            # Add statistics text
            stats_text = f'μ={mean_val:.1f}, σ={std_val:.1f}\nmin={min_val:.1f}, max={max_val:.1f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=8, 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    return fig

def plot_histogram_analysis(signal, noise_level, channel, title="Histogram Analysis"):
    """Plot histogram with noise level visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'{title} - Channel {channel}', fontsize=14)
    
    # Histogram
    hist, bins = np.histogram(signal, bins=100)
    ax1.hist(signal, bins=100, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(x=noise_level, color='red', linestyle='--', linewidth=2, 
                label=f'Noise Level: {noise_level:.2f}')
    ax1.axvline(x=np.mean(signal), color='green', linestyle='-', linewidth=2, 
                label=f'Mean: {np.mean(signal):.2f}')
    ax1.axvline(x=np.mean(signal) + np.std(signal), color='orange', linestyle=':', 
                linewidth=2, label=f'Mean + σ: {np.mean(signal) + np.std(signal):.2f}')
    ax1.set_xlabel('Amplitude')
    ax1.set_ylabel('Count')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Cumulative distribution
    cumsum = np.cumsum(hist)
    cumsum_norm = cumsum / cumsum[-1]
    ax2.plot(bins[:-1], cumsum_norm, 'b-', linewidth=2)
    ax2.axvline(x=noise_level, color='red', linestyle='--', linewidth=2)
    ax2.axhline(y=0.88, color='gray', linestyle=':', alpha=0.5, label='88% threshold')
    
    # Find where noise level intersects CDF
    idx = np.searchsorted(bins[:-1], noise_level)
    if idx < len(cumsum_norm):
        zero_percentage = cumsum_norm[idx] * 100
        ax2.plot(noise_level, cumsum_norm[idx], 'ro', markersize=8)
        ax2.text(noise_level, cumsum_norm[idx] + 0.05, f'{zero_percentage:.1f}%', 
                ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow'))
    
    ax2.set_xlabel('Amplitude')
    ax2.set_ylabel('Cumulative Probability')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    return fig

def analyze_emg_pipeline(rest_data, mvc_data, sf_rest, sf_mvc, artifact_cut=400, 
                        channels_to_analyze=None, save_plots=False, output_dir=None):
    """
    Debug EMG pipeline step by step
    
    rest_data, mvc_data: [samples, channels]
    """
    print("=== EMG Pipeline Debug Analysis ===\n")
    
    # Transpose to [channels, samples] for processing
    rest_data = rest_data.T
    mvc_data = mvc_data.T
    num_channels = rest_data.shape[0]
    
    if channels_to_analyze is None:
        channels_to_analyze = list(range(min(4, num_channels)))  # Default to first 4 channels
    
    # Create time axes for each signal
    time_rest = np.arange(rest_data.shape[1]) / sf_rest
    time_mvc = np.arange(mvc_data.shape[1]) / sf_mvc
    
    # Cut artifacts
    rest_cut = rest_data[:, artifact_cut:]
    mvc_cut = mvc_data[:, artifact_cut:]
    time_rest_cut = time_rest[artifact_cut:]
    time_mvc_cut = time_mvc[artifact_cut:]
    
    print(f"Rest data shape: {rest_data.shape}, after cut: {rest_cut.shape}")
    print(f"MVC data shape: {mvc_data.shape}, after cut: {mvc_cut.shape}")
    print(f"Sampling frequencies - Rest: {sf_rest:.1f} Hz, MVC: {sf_mvc:.1f} Hz\n")
    
    # === STEP 1: Raw Signal Analysis ===
    print("STEP 1: Analyzing raw signals...")
    
    # Plot raw signals with separate time axes
    fig1 = plot_signal_comparison(
        [rest_cut, mvc_cut],
        ["Rest (Raw)", "MVC (Raw)"],
        [time_rest_cut, time_mvc_cut],  # Pass list of time axes
        figure_title="Step 1: Raw EMG Signals",
        channels_to_plot=channels_to_analyze
    )
    
    # === STEP 2: Bandstop Filter (Powerline) ===
    print("\nSTEP 2: Applying bandstop filter (58-62 Hz)...")
    
    notch_rest = BesselFilterArr(numChannels=num_channels, order=8, 
                                critFreqs=[58,62], fs=sf_rest, filtType='bandstop')
    rest_notched = notch_rest.filter(rest_cut)
    
    notch_mvc = BesselFilterArr(numChannels=num_channels, order=8, 
                               critFreqs=[58,62], fs=sf_mvc, filtType='bandstop')
    mvc_notched = notch_mvc.filter(mvc_cut)
    
    # Cut filter transients (2 seconds worth of samples)
    transient_cut = int(2 * sf_rest)
    rest_notched = rest_notched[:, transient_cut:]
    mvc_notched = mvc_notched[:, transient_cut:]
    time_rest_cut = time_rest_cut[transient_cut:]
    time_mvc_cut = time_mvc_cut[transient_cut:]
    
    fig2 = plot_signal_comparison(
        [rest_notched, mvc_notched],
        ["Rest (After Notch)", "MVC (After Notch)"],
        [time_rest_cut, time_mvc_cut],
        figure_title="Step 2: After Bandstop Filter",
        channels_to_plot=channels_to_analyze
    )
    
    # === STEP 3: Highpass Filter ===
    print("\nSTEP 3: Applying highpass filter (20 Hz)...")
    
    hp_rest = BesselFilterArr(numChannels=num_channels, order=4, 
                             critFreqs=20, fs=sf_rest, filtType='highpass')
    rest_hp = hp_rest.filter(rest_notched)
    
    hp_mvc = BesselFilterArr(numChannels=num_channels, order=4, 
                            critFreqs=20, fs=sf_mvc, filtType='highpass')
    mvc_hp = hp_mvc.filter(mvc_notched)
    
    # Cut additional transients after highpass
    transient_cut_hp = int(1 * sf_rest)  # 1 second for highpass
    rest_hp = rest_hp[:, transient_cut_hp:]
    mvc_hp = mvc_hp[:, transient_cut_hp:]
    time_rest_cut = time_rest_cut[transient_cut_hp:]
    time_mvc_cut = time_mvc_cut[transient_cut_hp:]
    
    fig3 = plot_signal_comparison(
        [rest_hp, mvc_hp],
        ["Rest (After HP)", "MVC (After HP)"],
        [time_rest_cut, time_mvc_cut],
        figure_title="Step 3: After Highpass Filter",
        channels_to_plot=channels_to_analyze
    )
    
    # === STEP 4: Rectification ===
    print("\nSTEP 4: Rectifying signals...")
    
    rest_rect = np.abs(rest_hp)
    mvc_rect = np.abs(mvc_hp)
    
    fig4 = plot_signal_comparison(
        [rest_rect, mvc_rect],
        ["Rest (Rectified)", "MVC (Rectified)"],
        [time_rest_cut, time_mvc_cut],
        figure_title="Step 4: After Rectification",
        channels_to_plot=channels_to_analyze
    )
    
    # === STEP 5: Noise Level Calculation ===
    print("\nSTEP 5: Calculating noise levels...")
    
    noise_levels = []
    noise_methods = []
    
    for ch in range(num_channels):
        ch_signal = rest_rect[ch, :]
        mvc_ref = np.percentile(mvc_rect[ch, :], 95)
        
        # Method 1: Percentile-based
        sorted_signal = np.sort(ch_signal)
        noise_percentile = sorted_signal[int(len(sorted_signal) * 0.88)]
        
        # Method 2: Statistical
        mean_val = np.mean(ch_signal)
        std_val = np.std(ch_signal)
        noise_statistical = mean_val + 2 * std_val  # Using 2*std as in simple method
        
        # Method 3: MVC-based cap
        max_allowed = mvc_ref * 0.10
        
        methods = {
            'percentile': noise_percentile,
            'statistical': noise_statistical,
            'mvc_cap': max_allowed
        }
        noise_methods.append(methods)
        
        # Choose method (simple uses statistical)
        final_noise = min(noise_statistical, max_allowed)
        noise_levels.append(final_noise)
        
        if ch in channels_to_analyze:
            print(f"\n  Channel {ch}:")
            print(f"    Rest signal: mean={mean_val:.2f}, std={std_val:.2f}")
            print(f"    MVC reference (95th): {mvc_ref:.1f}")
            print(f"    Noise estimates:")
            print(f"      Percentile (88th): {noise_percentile:.2f}")
            print(f"      Statistical (μ+2σ): {noise_statistical:.2f}")
            print(f"      MVC cap (10%): {max_allowed:.2f}")
            print(f"    Final noise: {final_noise:.2f}")
    
    noise_levels = np.array(noise_levels)
    
    # Plot histograms for analyzed channels
    for ch in channels_to_analyze:
        fig_hist = plot_histogram_analysis(
            rest_rect[ch, :], 
            noise_levels[ch], 
            ch,
            "Rest Signal Distribution"
        )
        if save_plots and output_dir:
            fig_hist.savefig(os.path.join(output_dir, f'histogram_ch{ch}.png'), dpi=150)
    
    # === STEP 6: Noise Subtraction ===
    print("\nSTEP 6: Applying noise subtraction...")
    
    rest_denoised = np.clip(rest_rect - noise_levels[:, None], 0, None)
    mvc_denoised = np.clip(mvc_rect - noise_levels[:, None], 0, None)
    
    # Calculate zero percentages
    print("\n  Zero percentages after noise subtraction:")
    for ch in channels_to_analyze:
        zero_pct = np.sum(rest_denoised[ch, :] == 0) / len(rest_denoised[ch, :]) * 100
        print(f"    Channel {ch}: {zero_pct:.1f}%")
    
    fig5 = plot_signal_comparison(
        [rest_denoised, mvc_denoised],
        ["Rest (Denoised)", "MVC (Denoised)"],
        [time_rest_cut, time_mvc_cut],
        noise_levels=noise_levels,
        figure_title="Step 5: After Noise Subtraction",
        channels_to_plot=channels_to_analyze
    )
    
    # === STEP 7: Lowpass Filter (Envelope) ===
    print("\nSTEP 7: Applying lowpass filter (3 Hz envelope)...")
    
    lp_rest = BesselFilterArr(numChannels=num_channels, order=4, 
                             critFreqs=3, fs=sf_rest, filtType='lowpass')
    rest_final = lp_rest.filter(rest_denoised)
    
    lp_mvc = BesselFilterArr(numChannels=num_channels, order=4, 
                            critFreqs=3, fs=sf_mvc, filtType='lowpass')
    mvc_final = lp_mvc.filter(mvc_denoised)
    
    fig6 = plot_signal_comparison(
        [rest_final, mvc_final],
        ["Rest (Final)", "MVC (Final)"],
        [time_rest_cut, time_mvc_cut],
        figure_title="Step 6: Final Filtered EMG",
        channels_to_plot=channels_to_analyze
    )
    
    # === Final Analysis ===
    print("\n=== FINAL ANALYSIS ===")
    
    # Calculate SNR
    print("\nSignal-to-Noise Ratios:")
    print(f"{'Channel':<10} {'MVC Max':<12} {'Noise Level':<12} {'SNR (dB)':<12} {'Quality':<12}")
    print("-" * 60)
    
    for ch in range(num_channels):
        mvc_max = np.max(mvc_final[ch, :])
        if noise_levels[ch] > 0:
            snr = 20 * np.log10(mvc_max / noise_levels[ch])
            quality = "EXCELLENT" if snr > 20 else "GOOD" if snr > 15 else "ACCEPTABLE" if snr > 10 else "POOR"
        else:
            snr = float('inf')
            quality = "PERFECT"
        
        if ch in channels_to_analyze:
            print(f"{ch:<10} {mvc_max:<12.1f} {noise_levels[ch]:<12.2f} {snr:<12.1f} {quality:<12}")
    
    # Save all plots if requested
    if save_plots and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig1.savefig(os.path.join(output_dir, 'step1_raw.png'), dpi=150)
        fig2.savefig(os.path.join(output_dir, 'step2_notch.png'), dpi=150)
        fig3.savefig(os.path.join(output_dir, 'step3_highpass.png'), dpi=150)
        fig4.savefig(os.path.join(output_dir, 'step4_rectified.png'), dpi=150)
        fig5.savefig(os.path.join(output_dir, 'step5_denoised.png'), dpi=150)
        fig6.savefig(os.path.join(output_dir, 'step6_final.png'), dpi=150)
        print(f"\nPlots saved to: {output_dir}")
    
    plt.show()
    
    return noise_levels, noise_methods

def main():
    parser = argparse.ArgumentParser(description="Debug EMG noise level calculation")
    # parser.add_argument("--rest_file", "-r", required=True, 
    #                    help="Path to rest EMG data (calib_rest_emg.npy)")
    # parser.add_argument("--mvc_file", "-m", required=True, 
    #                    help="Path to MVC EMG data (calib_mvc_emg.npy)")
    # parser.add_argument("--rest_timestamps", "-rt", required=True,
    #                    help="Path to rest timestamps (calib_rest_timestamps.npy)")
    # parser.add_argument("--mvc_timestamps", "-mt", required=True,
    #                    help="Path to MVC timestamps (calib_mvc_timestamps.npy)")
    parser.add_argument("--channels", "-c", nargs='+', type=int,
                       help="Specific channels to analyze (default: first 4)")
    parser.add_argument("--artifact_cut", "-a", type=int, default=400,
                       help="Number of samples to cut from beginning (default: 400)")
    parser.add_argument("--save_plots", "-s", action="store_true",
                       help="Save plots to output directory")
    parser.add_argument("--output_dir", "-o", default="emg_debug_plots",
                       help="Directory to save plots (default: emg_debug_plots)")
    
    args = parser.parse_args()

    base_dir = "data/Emanuell/recordings/Calibration/experiments/1"
    rest_data = np.load(os.path.join(base_dir, 'calib_rest_emg.npy'))
    mvc_data = np.load(os.path.join(base_dir, 'calib_mvc_emg.npy'))
    rest_timestamps = np.load(os.path.join(base_dir, 'calib_rest_timestamps.npy'))
    mvc_timestamps = np.load(os.path.join(base_dir, 'calib_mvc_timestamps.npy'))
    
    # Load data
    # print(f"Loading data from:")
    # print(f"  Rest: {args.rest_file}")
    # print(f"  MVC: {args.mvc_file}")
    # 
    # rest_data = np.load(args.rest_file)
    # mvc_data = np.load(args.mvc_file)
    # rest_timestamps = np.load(args.rest_timestamps)
    # mvc_timestamps = np.load(args.mvc_timestamps)
    
    # Calculate sampling frequencies
    sf_rest = (len(rest_timestamps) - 1) / (rest_timestamps[-1] - rest_timestamps[0])
    sf_mvc = (len(mvc_timestamps) - 1) / (mvc_timestamps[-1] - mvc_timestamps[0])
    
    channels = [0,1,2,4,12,13,14,15]

    # Run analysis
    noise_levels, noise_methods = analyze_emg_pipeline(
        rest_data, mvc_data, sf_rest, sf_mvc,
        artifact_cut=args.artifact_cut,
        channels_to_analyze=channels,
        save_plots=args.save_plots,
        output_dir=args.output_dir
    )
    
    # Save results
    if args.save_plots:
        results = {
            'noise_levels': noise_levels.tolist(),
            'noise_methods': noise_methods,
            'sampling_frequencies': {
                'rest': sf_rest,
                'mvc': sf_mvc
            }
        }
        
        import json
        with open(os.path.join(args.output_dir, 'noise_analysis_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {args.output_dir}/noise_analysis_results.json")

if __name__ == "__main__":
    main()