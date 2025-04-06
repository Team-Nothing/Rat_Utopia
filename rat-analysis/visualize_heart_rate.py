import os
import json
import numpy as np
import matplotlib
matplotlib.use('TkAgg') # Explicitly set an interactive backend
import matplotlib.pyplot as plt
import torch
from scipy.interpolate import interp1d
from scipy.signal import medfilt, butter, sosfilt, find_peaks, filtfilt
from scipy import signal
import matplotlib.gridspec as gridspec
from pathlib import Path
import sys
import argparse # Use argparse for command-line argument handling
# import time # Import time for pausing
from scipy.fft import rfft, rfftfreq # Import FFT functions

# Set matplotlib to use the 'Agg' backend for non-interactive environments
# plt.switch_backend('Agg') # Commented out to allow interactive plotting

def load_heart_rate_data(file_path):
    """Load heart rate data from text file"""
    times = []
    pressures = []
    timestamps = []
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                try:
                    data = json.loads(line)
                    times.append(data['t'])
                    pressures.append(data['p'])
                    timestamps.append(data['T'])
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse line: {line}")
                    continue
    
    return np.array(times), np.array(pressures), np.array(timestamps)

def normalize_signal(signal):
    """Normalizes the signal using z-score normalization."""
    if signal is None or len(signal) == 0: return None
    mean = np.mean(signal)
    std_dev = np.std(signal)
    if std_dev == 0:
        print("Warning: Standard deviation is zero. Cannot normalize.")
        # Return the original signal (or signal of zeros) if std dev is zero
        return signal - mean
    normalized = (signal - mean) / std_dev
    return normalized

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Apply bandpass filter to isolate frequencies of interest"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    sos = butter(order, [low, high], btype='band', output='sos')
    filtered_data = sosfilt(sos, data)
    return filtered_data

def preprocess_data(times, pressures, timestamps):
    """Preprocess and normalize the data"""
    # Convert timestamps to seconds from the start
    relative_time = (timestamps - timestamps[0]) / 1e6  # microseconds to seconds
    
    # Sort data by time (if not already sorted)
    sort_idx = np.argsort(relative_time)
    relative_time = relative_time[sort_idx]
    pressures = pressures[sort_idx]
    
    # Apply median filter to remove noise
    pressures_filtered = medfilt(pressures, kernel_size=5)
    
    # Remove any duplicate time points for interpolation
    unique_times, unique_indices = np.unique(relative_time, return_index=True)
    unique_pressures = pressures_filtered[unique_indices]
    
    # Create interpolation function
    f_interp = interp1d(unique_times, unique_pressures, kind='cubic', bounds_error=False, fill_value="extrapolate")
    
    # Create evenly spaced time points for interpolation
    time_interp = np.linspace(unique_times.min(), unique_times.max(), num=len(unique_times))
    pressure_interp = f_interp(time_interp)
    
    # Normalize the pressure data
    pressure_norm = (pressure_interp - np.mean(pressure_interp)) / np.std(pressure_interp)
    
    return time_interp, pressure_norm

def plot_time_domain(time, pressure, output_dir, sample_name):
    """Plot time domain data"""
    plt.figure(figsize=(12, 6))
    plt.plot(time, pressure)
    plt.title(f'Heart Rate Pressure Data - {sample_name} (Time Domain)')
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Pressure')
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{sample_name}_time_domain.png')
    plt.savefig(output_path)
    print(f"Saved time domain plot to: {output_path}")
    plt.close()

def perform_fft_analysis(time, pressure):
    """Perform FFT analysis using PyTorch with windowing"""
    sample_rate = 1 / (time[1] - time[0])  # Sampling rate in Hz
    print(f"Sample rate: {sample_rate:.2f} Hz")
    
    # If data is longer than 30 seconds, use windowing
    if len(pressure) > 30 * sample_rate:
        window_size = int(15 * sample_rate)  # 15-second window
        step_size = int(window_size / 2)  # 50% overlap
        
        windows = []
        for i in range(0, len(pressure) - window_size, step_size):
            segment = pressure[i:i + window_size]
            
            # Apply Hann window to reduce spectral leakage
            windowed_segment = segment * np.hanning(len(segment))
            windows.append(windowed_segment)
        
        # Process each window
        fft_results = []
        for windowed_segment in windows:
            # Convert to PyTorch tensor
            segment_tensor = torch.from_numpy(windowed_segment).float()
            
            # Apply FFT
            fft_result = torch.fft.rfft(segment_tensor)
            fft_magnitude = torch.abs(fft_result)
            fft_results.append(fft_magnitude.numpy())
        
        # Average the FFT results
        fft_magnitude_np = np.mean(fft_results, axis=0)
        
        # Calculate frequency bins based on window size
        freq_bins = np.fft.rfftfreq(window_size, 1/sample_rate)
    else:
        # For shorter data, apply Hann window to the entire signal
        windowed_pressure = pressure * np.hanning(len(pressure))
        
        # Convert to PyTorch tensor
        pressure_tensor = torch.from_numpy(windowed_pressure).float()
        
        # Perform FFT
        fft_result = torch.fft.rfft(pressure_tensor)
        fft_magnitude = torch.abs(fft_result)
        
        # Convert to numpy for plotting
        fft_magnitude_np = fft_magnitude.numpy()
        
        # Calculate frequency bins
        freq_bins = np.fft.rfftfreq(len(pressure), 1/sample_rate)
    
    # Calculate power spectrum (squared magnitude)
    power_spectrum = fft_magnitude_np**2
    
    # Smooth the power spectrum using convolution
    window_length = 5  # Length of the smoothing window
    smoothing_window = np.ones(window_length) / window_length
    smoothed_spectrum = np.convolve(power_spectrum, smoothing_window, mode='same')
    
    print(f"Frequency resolution: {freq_bins[1]:.5f} Hz")
    print(f"Maximum detectable frequency: {freq_bins[-1]:.2f} Hz")
    
    return freq_bins, smoothed_spectrum, power_spectrum, fft_magnitude_np

def plot_frequency_domain(freq_bins, power_spectrum, fft_magnitude, output_dir, sample_name, heart_rate_freq=None, breath_rate_freq=None):
    """Plot frequency domain data with enhanced visualizations"""
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(14, 12))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1])
    
    # 1. Full spectrum plot with log scale
    ax1 = plt.subplot(gs[0])
    ax1.plot(freq_bins, power_spectrum)
    ax1.set_title(f'Heart Rate Frequency Spectrum - {sample_name} (Full Range, Log Scale)')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Power')
    ax1.set_yscale('log')  # Log scale to better see the peaks
    ax1.grid(True)
    
    # 2. Zoomed heart rate range
    ax2 = plt.subplot(gs[1])
    heart_rate_mask = (freq_bins >= 4.0) & (freq_bins <= 6.0)
    heart_rate_freqs = freq_bins[heart_rate_mask]
    heart_rate_power = power_spectrum[heart_rate_mask]
    
    ax2.plot(heart_rate_freqs, heart_rate_power)
    ax2.fill_between(heart_rate_freqs, 0, heart_rate_power, alpha=0.2)
    ax2.set_title(f'Heart Rate Range - {sample_name} (260-350 BPM)')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Power')
    
    # Highlight heart rate range
    ax2.axvspan(4.33, 5.83, alpha=0.2, color='red', label='Heart Rate Range (260-350 BPM)')
    
    # Mark detected heart rate
    if heart_rate_freq is not None:
        ax2.axvline(x=heart_rate_freq, color='r', linestyle='--', 
                   label=f'Heart Rate: {heart_rate_freq:.2f} Hz ({heart_rate_freq*60:.1f} BPM)')
    
    ax2.grid(True)
    ax2.legend()
    
    # 3. Zoomed breathing rate range
    ax3 = plt.subplot(gs[2])
    breath_rate_mask = (freq_bins >= 0.5) & (freq_bins <= 1.2)
    breath_rate_freqs = freq_bins[breath_rate_mask]
    breath_rate_power = power_spectrum[breath_rate_mask]
    
    ax3.plot(breath_rate_freqs, breath_rate_power)
    ax3.fill_between(breath_rate_freqs, 0, breath_rate_power, alpha=0.2)
    ax3.set_title(f'Breathing Rate Range - {sample_name} (40-60 BPM)')
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Power')
    
    # Highlight breathing rate range
    ax3.axvspan(0.67, 1.0, alpha=0.2, color='blue', label='Breathing Rate Range (40-60 BPM)')
    
    # Mark detected breathing rate
    if breath_rate_freq is not None:
        ax3.axvline(x=breath_rate_freq, color='b', linestyle='--',
                   label=f'Breathing Rate: {breath_rate_freq:.2f} Hz ({breath_rate_freq*60:.1f} BPM)')
    
    ax3.grid(True)
    ax3.legend()
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{sample_name}_frequency_analysis.png')
    plt.savefig(output_path)
    print(f"Saved enhanced frequency analysis plot to: {output_path}")
    plt.close()
    
    # Also save a linear scale version of the main plot
    plt.figure(figsize=(12, 6))
    plt.plot(freq_bins, fft_magnitude)
    plt.title(f'Heart Rate Frequency Spectrum - {sample_name} (Linear Scale)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.xlim(0, 7)
    
    # Highlight ranges
    plt.axvspan(4.33, 5.83, alpha=0.2, color='red', label='Heart Rate Range (260-350 BPM)')
    plt.axvspan(0.67, 1.0, alpha=0.2, color='blue', label='Breathing Rate Range (40-60 BPM)')
    
    plt.legend()
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'{sample_name}_frequency_linear.png')
    plt.savefig(output_path)
    plt.close()

def analyze_peaks(freq_bins, power_spectrum):
    """Analyze and identify heart rate and breathing rate from FFT peaks with enhanced detection"""
    # Heart rate range for rats (260-350 BPM = 4.33-5.83 Hz)
    heart_rate_mask = (freq_bins >= 4.33) & (freq_bins <= 5.83)
    heart_rate_freqs = freq_bins[heart_rate_mask]
    heart_rate_power = power_spectrum[heart_rate_mask]
    
    # Breathing rate range for rats under anesthesia (40-60 BPM = 0.67-1.0 Hz)
    breath_rate_mask = (freq_bins >= 0.67) & (freq_bins <= 1.0)
    breath_rate_freqs = freq_bins[breath_rate_mask]
    breath_rate_power = power_spectrum[breath_rate_mask]
    
    heart_rate_freq = None
    breath_rate_freq = None
    
    # Enhanced peak detection for heart rate
    if len(heart_rate_freqs) > 0 and len(heart_rate_power) > 0:
        # Find multiple peaks and select the most prominent one
        min_height = 0.1 * np.max(heart_rate_power) if np.max(heart_rate_power) > 0 else 0
        peaks, _ = find_peaks(heart_rate_power, height=min_height, distance=5)
        
        if len(peaks) > 0:
            # Select the highest peak
            max_peak_idx = np.argmax(heart_rate_power[peaks])
            peak_idx = peaks[max_peak_idx]
            heart_rate_freq = heart_rate_freqs[peak_idx]
            heart_rate_bpm = heart_rate_freq * 60
            print(f"Detected heart rate: {heart_rate_freq:.2f} Hz ({heart_rate_bpm:.1f} BPM)")
        else:
            # Fall back to simple max if peak detection fails
            heart_peak_idx = np.argmax(heart_rate_power)
            heart_rate_freq = heart_rate_freqs[heart_peak_idx]
            heart_rate_bpm = heart_rate_freq * 60
            print(f"Detected heart rate (using max): {heart_rate_freq:.2f} Hz ({heart_rate_bpm:.1f} BPM)")
    else:
        print("No clear heart rate peak detected in expected range")
    
    # Enhanced peak detection for breathing rate
    if len(breath_rate_freqs) > 0 and len(breath_rate_power) > 0:
        # Find multiple peaks and select the most prominent one
        min_height = 0.1 * np.max(breath_rate_power) if np.max(breath_rate_power) > 0 else 0
        peaks, _ = find_peaks(breath_rate_power, height=min_height, distance=5)
        
        if len(peaks) > 0:
            # Select the highest peak
            max_peak_idx = np.argmax(breath_rate_power[peaks])
            peak_idx = peaks[max_peak_idx]
            breath_rate_freq = breath_rate_freqs[peak_idx]
            breath_rate_bpm = breath_rate_freq * 60
            print(f"Detected breathing rate: {breath_rate_freq:.2f} Hz ({breath_rate_bpm:.1f} BPM)")
        else:
            # Fall back to simple max if peak detection fails
            breath_peak_idx = np.argmax(breath_rate_power)
            breath_rate_freq = breath_rate_freqs[breath_peak_idx]
            breath_rate_bpm = breath_rate_freq * 60
            print(f"Detected breathing rate (using max): {breath_rate_freq:.2f} Hz ({breath_rate_bpm:.1f} BPM)")
    else:
        print("No clear breathing rate peak detected in expected range")
    
    return heart_rate_freq, breath_rate_freq

def process_shorter_segment(time, pressure, output_dir, sample_name):
    """Process a shorter segment of the data to reduce noise and focus on clearer signals"""
    # Use the middle third of the data to avoid boundary artifacts
    total_length = len(pressure)
    
    # If data is very long, take a 60-second segment from the middle
    if total_length > 6000:  # Approximately 60 seconds at 100 Hz
        segment_size = 6000
        start_idx = (total_length - segment_size) // 2
        end_idx = start_idx + segment_size
    else:
        # Just use the middle third
        start_idx = total_length // 3
        end_idx = total_length - start_idx
    
    time_segment = time[start_idx:end_idx]
    pressure_segment = pressure[start_idx:end_idx]
    
    # Apply additional bandpass filtering for clearer signal
    # Focus on 0.5-10 Hz range to capture both breathing and heart rates
    sample_rate = 1 / (time[1] - time[0])
    filtered_pressure = bandpass_filter(pressure_segment, 0.5, 10, sample_rate)
    
    # Plot the filtered time domain segment
    plt.figure(figsize=(12, 6))
    plt.plot(time_segment, filtered_pressure)
    plt.title(f'Filtered Pressure Data Segment - {sample_name}')
    plt.xlabel('Time (s)')
    plt.ylabel('Filtered Pressure')
    plt.grid(True)
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{sample_name}_filtered_segment.png')
    plt.savefig(output_path)
    print(f"Saved filtered segment plot to: {output_path}")
    plt.close()
    
    return time_segment, filtered_pressure

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Applies a Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    # Check if cutoff frequencies are valid
    if not (0 < low < 1) or not (0 < high < 1) or low >= high:
        print(f"Error: Invalid cutoff frequencies for bandpass filter ({lowcut} Hz, {highcut} Hz) with fs={fs} Hz.")
        print(f"Calculated normalized frequencies (must be between 0 and 1, low < high): low={low}, high={high}")
        return data # Return unfiltered data or handle error appropriately
    try:
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, data)
        return y
    except Exception as e:
        print(f"Error during filtering: {e}")
        return data # Return unfiltered data in case of error

def calculate_hr(times_s, pressures, fs):
    """Calculates heart rate from the pressure signal."""
    if fs is None or pressures is None or times_s is None or fs <= 0:
         print("Error: Invalid input for HR calculation.")
         return None, None, None

    # --- Parameters ---
    lowcut = 3.0  # Lower cutoff frequency (Hz)
    highcut = 15.0 # Upper cutoff frequency (Hz)
    # Adjust peak detection parameters based on signal characteristics
    # Height might need tuning based on filtered signal amplitude
    # Distance should correspond to minimum plausible heart period (e.g., 60 / 600 BPM = 0.1s)
    min_peak_distance_s = 60.0 / 600.0 # Max plausible HR = 600 BPM
    min_peak_distance_samples = int(min_peak_distance_s * fs)

    # 1. Filter the signal
    print(f"Applying bandpass filter ({lowcut}-{highcut} Hz)...")
    filtered_pressure = butter_bandpass_filter(pressures, lowcut, highcut, fs)
    if np.array_equal(filtered_pressure, pressures): # Check if filtering failed
        print("Warning: Filtering may have failed, proceeding with unfiltered data for peak detection.")
        # Decide if you want to stop or proceed with unfiltered data
        # return None, None, filtered_pressure # Option: stop if filtering fails

    # 2. Find peaks
    # Normalize filtered signal for consistent height threshold? Optional.
    # filtered_norm = filtered_pressure / np.max(np.abs(filtered_pressure))
    # peaks, properties = find_peaks(filtered_norm, distance=min_peak_distance_samples, height=0.3) # Height threshold on normalized signal
    
    # Use non-normalized signal - requires adjusting 'height' based on typical signal amplitude
    # Start with a relative height threshold (e.g., 30% of max peak)
    height_threshold = 0.3 * np.max(filtered_pressure) if np.max(filtered_pressure) > 0 else 0
    print(f"Finding peaks with min distance {min_peak_distance_samples} samples and height > {height_threshold:.2f}...")
    peaks, properties = find_peaks(filtered_pressure, distance=min_peak_distance_samples, height=height_threshold)

    if len(peaks) < 2:
        print("Error: Not enough peaks detected to calculate heart rate.")
        return None, None, filtered_pressure

    print(f"Detected {len(peaks)} peaks.")

    # 3. Calculate RR intervals and Heart Rate (BPM)
    peak_times_s = times_s[peaks]
    rr_intervals_s = np.diff(peak_times_s)

    # Calculate heart rate (BPM)
    heart_rates_bpm = 60.0 / rr_intervals_s

    # Calculate time points for HR (midpoint of RR interval)
    hr_times_s = peak_times_s[:-1] + rr_intervals_s / 2.0

    # 4. Filter unrealistic HR values
    min_hr = 100  # Min plausible HR (BPM)
    max_hr = 600  # Max plausible HR (BPM)
    valid_hr_indices = np.where((heart_rates_bpm >= min_hr) & (heart_rates_bpm <= max_hr))[0]

    if len(valid_hr_indices) == 0:
        print("Warning: No valid heart rate values found within the plausible range.")
        return None, None, filtered_pressure # Or return empty arrays: np.array([]), np.array([])

    filtered_hr_bpm = heart_rates_bpm[valid_hr_indices]
    filtered_hr_times_s = hr_times_s[valid_hr_indices]

    print(f"Calculated {len(filtered_hr_bpm)} valid HR points.")
    print(f"Mean HR: {np.mean(filtered_hr_bpm):.2f} BPM, Median HR: {np.median(filtered_hr_bpm):.2f} BPM")


    return filtered_hr_times_s, filtered_hr_bpm, filtered_pressure

def main():
    # Get list of available data directories
    data_dir = 'data/raw'
    dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    # Print available directories
    print("Available data directories:")
    for i, d in enumerate(dirs):
        print(f"{i+1}. {d}")
    
    # Get user input for which directory to analyze
    selected = 0
    while selected < 1 or selected > len(dirs):
        try:
            selected = int(input(f"Select a directory to analyze (1-{len(dirs)}): "))
        except ValueError:
            print("Please enter a valid number")
    
    selected_dir = dirs[selected-1]
    heart_rate_file = os.path.join(data_dir, selected_dir, 'heart_rate.txt')
    output_dir = os.path.join('plots', 'heart_rate', selected_dir)
    
    print(f"Analyzing: {heart_rate_file}")
    
    # Load and process data
    raw_times, pressures, timestamps = load_heart_rate_data(heart_rate_file)
    print(f"Loaded {len(raw_times)} data points")
    
    # Calculate relative time in seconds from timestamps
    if len(timestamps) > 0:
        times_s = (timestamps - timestamps[0]) / 1_000_000.0
    else:
        print("Error: No timestamps loaded.")
        sys.exit(1)

    # Preprocess and normalize
    time_interp, pressure_norm = preprocess_data(raw_times, pressures, timestamps)
    print("Data preprocessed and normalized")
    
    # Plot full time domain
    plot_time_domain(time_interp, pressure_norm, output_dir, selected_dir)
    
    # Process a shorter segment for better signal quality
    print("Processing shorter segment for clearer signal...")
    time_segment, filtered_pressure = process_shorter_segment(time_interp, pressure_norm, output_dir, selected_dir)
    
    # Perform FFT analysis on the filtered segment
    print("Performing FFT analysis on filtered segment...")
    freq_bins, power_spectrum, raw_power, fft_magnitude = perform_fft_analysis(time_segment, filtered_pressure)
    
    # Analyze peaks
    heart_rate_freq, breath_rate_freq = analyze_peaks(freq_bins, power_spectrum)
    
    # Plot frequency domain with enhanced visualization
    plot_frequency_domain(freq_bins, power_spectrum, fft_magnitude, output_dir, selected_dir, 
                          heart_rate_freq, breath_rate_freq)
    
    print(f"Analysis complete. Results saved to: {output_dir}")

    # 2. Calculate Heart Rate
    print("Calculating heart rate...")
    hr_times_s, hr_bpm, filtered_signal = calculate_hr(times_s, pressures, 1 / (time_interp[1] - time_interp[0]))

    # 3. Plotting
    print("Generating plot...")
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True) # Increased figure height

    # Plot 1: Raw Pressure Signal
    axes[0].plot(time_interp, pressures, label='Raw Pressure Signal', color='blue', linewidth=0.8)
    axes[0].set_ylabel('Pressure (arb. units)')
    axes[0].set_title(f'Raw Signal - {selected_dir}')
    axes[0].grid(True)
    axes[0].legend()

    # Plot 2: Filtered Pressure Signal with Detected Peaks
    if filtered_signal is not None:
        axes[1].plot(time_interp, filtered_signal, label='Filtered Signal (3-15 Hz)', color='green', linewidth=0.8)
        # Find peaks again on the *same* filtered signal to plot them
        # Use the same parameters as in calculate_hr
        min_peak_distance_s = 60.0 / 600.0 # Max plausible HR = 600 BPM
        min_peak_distance_samples = int(min_peak_distance_s * (1 / (time_interp[1] - time_interp[0]))) if (1 / (time_interp[1] - time_interp[0])) else 0
        height_threshold = 0.3 * np.max(filtered_signal) if np.max(filtered_signal) > 0 else 0
        peaks, _ = find_peaks(filtered_signal, distance=min_peak_distance_samples, height=height_threshold)
        
        if len(peaks) > 0:
             axes[1].plot(time_interp[peaks], filtered_signal[peaks], "x", label="Detected Peaks", color='red', markersize=5)

        axes[1].set_ylabel('Filtered Pressure')
        axes[1].set_title('Filtered Signal and Detected Peaks')
        axes[1].grid(True)
        axes[1].legend()
    else:
         axes[1].set_title('Filtered Signal (Processing Error)')
         axes[1].text(0.5, 0.5, 'Could not generate filtered signal', horizontalalignment='center', verticalalignment='center', transform=axes[1].transAxes)


    # Plot 3: Heart Rate
    if hr_times_s is not None and hr_bpm is not None and len(hr_times_s) > 0:
        axes[2].plot(hr_times_s, hr_bpm, 'o-', label='Heart Rate (BPM)', color='red', markersize=3, linewidth=1)
        axes[2].set_ylabel('Heart Rate (BPM)')
        axes[2].set_xlabel('Time (s)')
         # Set Y-axis limits based on expected range + buffer
        axes[2].set_ylim(100, 600) # Typical rat HR range is 250-500, allow buffer
        axes[2].set_title('Calculated Heart Rate')
        axes[2].grid(True)
        axes[2].legend()
        
        # Add mean/median HR text
        mean_hr = np.mean(hr_bpm)
        median_hr = np.median(hr_bpm)
        axes[2].text(0.02, 0.95, f'Mean: {mean_hr:.1f} BPM\nMedian: {median_hr:.1f} BPM',
                     transform=axes[2].transAxes, fontsize=9, verticalalignment='top',
                     bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

    else:
        axes[2].set_title('Calculated Heart Rate (No Valid Data)')
        axes[2].text(0.5, 0.5, 'Could not calculate heart rate', horizontalalignment='center', verticalalignment='center', transform=axes[2].transAxes)
        axes[2].set_ylim(100, 600)


    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout to prevent title overlap
    plt.suptitle(f'Heart Rate Analysis: {heart_rate_file}', fontsize=14, y=0.99) # Add overall title

    # Show the plot (required for PyCharm execution without exiting)
    print("Displaying plot. Close the plot window to exit.")
    plt.show()

    print("Script finished.")

if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Load, normalize, and visualize raw heart rate pressure data in segments with sine wave overlay.")
    parser.add_argument("data_folder", nargs='?', type=str,
                        default="data/raw/20250329_130718",
                        help="Path to the data folder containing heart_rate.txt. Defaults to data/raw/20241123_113139.")
    parser.add_argument("--segment_duration", type=float, default=2.0,
                        help="Duration of each data segment to display in seconds. Default: 2.0")
    parser.add_argument("--pause_duration", type=float, default=5.0,
                        help="Duration each segment is displayed before updating in seconds. Default: 5.0")

    args = parser.parse_args()

    # --- Configuration ---
    data_folder_path = Path(args.data_folder)
    heart_rate_filename = "heart_rate.txt"
    filepath = data_folder_path / heart_rate_filename
    segment_duration = args.segment_duration
    pause_duration = args.pause_duration

    # --- Workflow ---
    # 1. Load Data
    raw_times, pressures, timestamps = load_heart_rate_data(filepath)
    if pressures is None or timestamps is None: sys.exit(1)

    # Calculate relative time in seconds
    if len(timestamps) > 1:
        times_s = (timestamps - timestamps[0]) / 1_000_000.0
        # Estimate sampling frequency
        fs = 1.0 / np.median(np.diff(times_s))
        print(f"Estimated Sampling Frequency: {fs:.2f} Hz")
    else:
        print("Error: Not enough data points to estimate sampling frequency.")
        sys.exit(1)

    # 2. Normalize Data
    print("\nNormalizing pressure signal...")
    normalized_pressures = normalize_signal(pressures)
    if normalized_pressures is None: sys.exit(1)

    # 3. Perform FFT analysis on the entire signal to find dominant HR frequency
    print("\nPerforming FFT to find dominant heart rate frequency...")
    dominant_hr_freq = None
    try:
        n_points = len(normalized_pressures)
        if n_points > 1 and fs > 0:
            yf = rfft(normalized_pressures)
            xf = rfftfreq(n_points, 1 / fs)
            power = np.abs(yf)**2

            # Define HR range in Hz (250-350 BPM)
            min_hr_hz = 250 / 60.0  # 4.17 Hz
            max_hr_hz = 350 / 60.0  # 5.83 Hz

            hr_indices = np.where((xf >= min_hr_hz) & (xf <= max_hr_hz))[0]

            if len(hr_indices) > 0:
                peak_index_in_hr_range = hr_indices[np.argmax(power[hr_indices])]
                dominant_hr_freq = xf[peak_index_in_hr_range]
                print(f"--> Dominant frequency in {min_hr_hz:.2f}-{max_hr_hz:.2f} Hz range: {dominant_hr_freq:.2f} Hz ({dominant_hr_freq * 60:.1f} BPM)")
            else:
                print(f"--> Warning: No frequency components found in the {min_hr_hz:.2f}-{max_hr_hz:.2f} Hz range.")
        else:
            print("--> Warning: Not enough data or invalid sampling frequency for FFT.")
    except Exception as e:
        print(f"--> Error during FFT analysis: {e}")

    # 4. Plotting Segments with Sine Wave Overlay
    print(f"\nGenerating plot with {segment_duration}s segments...")
    plt.ion() # Turn on interactive mode
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    total_duration = times_s[-1]
    start_time = 0.0
    global_min = np.min(normalized_pressures)
    global_max = np.max(normalized_pressures)
    y_padding = (global_max - global_min) * 0.1

    while start_time < total_duration:
        end_time = start_time + segment_duration
        indices = np.where((times_s >= start_time) & (times_s < end_time))[0]

        if len(indices) > 0:
            current_times = times_s[indices]
            current_pressures = normalized_pressures[indices]

            ax.clear()
            # Plot normalized pressure data
            ax.plot(current_times, current_pressures, label='Norm. Pressure (Z-score)', color='purple', linewidth=0.8)

            # Plot sine wave overlay if frequency was found
            if dominant_hr_freq is not None:
                amplitude = np.std(current_pressures) if len(current_pressures) > 1 else 1
                sine_wave = amplitude * np.sin(2 * np.pi * dominant_hr_freq * current_times)
                bpm = dominant_hr_freq * 60 # Calculate BPM
                ax.plot(current_times, sine_wave, label=f'Sine ({dominant_hr_freq:.2f} Hz / {bpm:.1f} BPM)', color='red', linestyle='--', linewidth=1)

            ax.set_ylabel('Normalized Pressure (Z-score)')
            ax.set_xlabel('Time (s)')
            ax.set_title(f'Normalized Signal: {start_time:.2f}s - {end_time:.2f}s ({data_folder_path.name})')
            ax.set_xlim(start_time, end_time)
            ax.set_ylim(global_min - y_padding, global_max + y_padding)
            ax.grid(True, linestyle=':')
            ax.legend(loc='upper right') # Add legend

            try:
                display_path = filepath.relative_to(Path.cwd())
            except ValueError:
                display_path = filepath
            fig.suptitle(f'Heart Rate Data Visualization: {display_path}', fontsize=14)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            plt.draw()
            plt.pause(pause_duration)

        start_time = end_time

    print("\nFinished displaying all segments.")
    plt.ioff()
    ax.set_title(f'Normalized Signal: Final Segment ({data_folder_path.name})')
    print("Close the plot window to exit.")
    plt.show()

    print("\nScript finished.") 