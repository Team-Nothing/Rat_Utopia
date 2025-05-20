import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import argparse
from pathlib import Path
import shutil

# Target sampling rate
TARGET_RATE = 120 # Hz

def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def copy_raw_files(raw_dir, processed_dir):
    """Copy heart_rate.txt and hr_annotations.json from raw to processed directory."""
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    hr_file = raw_dir / 'heart_rate.txt'
    annotations_file = raw_dir / 'hr_annotations.json'
    
    if hr_file.exists():
        shutil.copy(hr_file, processed_dir / 'heart_rate.txt')
        print(f"Copied {hr_file} to {processed_dir / 'heart_rate.txt'}")
    else:
        raise FileNotFoundError(f"Heart rate file not found at: {hr_file}")
    
    if annotations_file.exists():
        shutil.copy(annotations_file, processed_dir / 'hr_annotations.json')
        print(f"Copied {annotations_file} to {processed_dir / 'hr_annotations.json'}")
    else:
        raise FileNotFoundError(f"Annotations file not found at: {annotations_file}")

def load_heart_rate_data(data_dir, original_rate, target_rate):
    """Load heart rate data (pressure), normalize, and resample to target_rate."""
    data_path = Path(data_dir) / 'heart_rate.txt'
    if not data_path.exists():
        raise FileNotFoundError(f"Heart rate data file not found at: {data_path}")
    
    hr_values = []
    try:
        with open(data_path, 'r') as f:
            for line in f:
                try:
                    data_point = json.loads(line.strip())
                    if 'p' in data_point:
                        hr_values.append(float(data_point['p']))
                    else:
                        print(f"Warning: 'p' key not found in line: {line.strip()}")
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse line as JSON: {line.strip()}")
                except ValueError:
                    print(f"Warning: Could not convert 'p' value to float in line: {line.strip()}")
                    
    except Exception as e:
         raise ValueError(f"Error reading or processing {data_path}: {e}")

    if not hr_values:
        raise ValueError(f"No valid heart rate data found in {data_path}")

    hr_data_original = np.array(hr_values)
    num_original_points = len(hr_data_original)
    duration = num_original_points / original_rate
    print(f"Original data: {num_original_points} points at {original_rate} Hz, duration: {duration:.2f}s")

    # Normalize original data
    if np.std(hr_data_original) > 0:
        hr_data_normalized = (hr_data_original - np.mean(hr_data_original)) / np.std(hr_data_original)
    else:
        hr_data_normalized = hr_data_original - np.mean(hr_data_original)

    # Create time vectors
    t_original = np.linspace(0, duration, num_original_points)
    num_target_points = int(duration * target_rate)
    t_target = np.linspace(0, duration, num_target_points)

    # Interpolate
    interp_func = interp1d(t_original, hr_data_normalized, kind='linear', bounds_error=False, fill_value='extrapolate')
    resampled_data = interp_func(t_target)

    print(f"Resampled data to {target_rate} Hz: {len(resampled_data)} points.")
    return resampled_data, duration

def load_annotations(data_dir, duration, target_rate):
    """Load annotations, process peak/valley to 1/0, and interpolate to target_rate."""
    annotations_path = Path(data_dir) / 'hr_annotations.json'
    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotations file not found at: {annotations_path}")
    
    try:
        with open(annotations_path, 'r') as f:
            raw_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON from {annotations_path}: {e}")
    except Exception as e:
        raise ValueError(f"Error reading {annotations_path}: {e}")

    if isinstance(raw_data, dict) and 'annotations' in raw_data:
        annotation_list = raw_data['annotations']
    elif isinstance(raw_data, list):
        annotation_list = raw_data
    else:
        raise ValueError(f"Unexpected JSON structure in {annotations_path}. Expected a list or {{'annotations': [...]}}.")

    times = []
    values = []
    for item in annotation_list:
        try:
            if isinstance(item, str):
                ann = json.loads(item)
            elif isinstance(item, dict):
                ann = item
            else:
                print(f"Warning: Skipping unrecognized annotation format: {item}")
                continue

            if 'time' in ann and 'type' in ann:
                times.append(float(ann['time']))
                values.append(1 if ann['type'] == 'peak' else 0)
            else:
                print(f"Warning: Missing 'time' or 'type' key in annotation: {ann}")
        except json.JSONDecodeError:
            print(f"Warning: Could not parse annotation item as JSON: {item}")
        except TypeError:
            print(f"Warning: Annotation item has unexpected type: {item}")
        except ValueError:
            print(f"Warning: Could not convert 'time' to float in annotation: {ann}")

    if not times:
        print(f"Warning: No valid annotations found or processed in {annotations_path}.")
        num_target_points = int(duration * target_rate)
        return np.zeros(num_target_points)

    # --- Interpolation based on loaded times and 0/1 values --- 
    num_target_points = int(duration * target_rate)
    x_interp = np.linspace(0, duration, num_target_points)
    
    interpolated_values = np.zeros(num_target_points) # Default to zeros
    
    if len(times) > 1:
        sorted_indices = np.argsort(times)
        times_sorted = np.array(times)[sorted_indices]
        values_sorted = np.array(values)[sorted_indices]
        
        unique_times, unique_indices = np.unique(times_sorted, return_index=True)
        unique_values = values_sorted[unique_indices]
        
        if len(unique_times) > 1:
            # Use cubic interpolation for the sparse 0/1 annotation signal
            interp_func_ann = interp1d(unique_times, unique_values, kind='cubic', bounds_error=False, fill_value=(unique_values[0], unique_values[-1]))
            interpolated_values = interp_func_ann(x_interp)
        elif len(unique_times) == 1:
            print("Warning: Only one unique annotation time point found. Interpolating as constant.")
            interpolated_values = np.full(num_target_points, unique_values[0])
            
    elif len(times) == 1:
        print("Warning: Only one annotation point found. Interpolating as constant.")
        # Simple approach: find nearest point in target time and set it
        # Or just return constant value if that's acceptable
        interpolated_values = np.full(num_target_points, values[0])

    print(f"Interpolated annotations to {target_rate} Hz: {len(interpolated_values)} points.")
    return interpolated_values

def plot_data_window(data, annotations, start_idx, end_idx, sample_name, target_rate, no_use_ranges=None):
    """Plot a window of resampled heart rate data with annotations and no-use ranges (using point index)."""
    plt.close('all')
    plt.figure(figsize=(12, 6))
    window_data = data[start_idx:end_idx]
    window_annotations = annotations[start_idx:end_idx]
    # Use point index for the x-axis
    x_axis = range(start_idx, end_idx)
    
    plt.plot(x_axis, window_data, label=f'Normalized Heart Rate ({target_rate} Hz)', color='blue')
    plt.plot(x_axis, window_annotations, label='Annotations (Interpolated)', color='green', linestyle='--')
    
    if no_use_ranges:
        # Plot no-use ranges using point indices
        for start, end in no_use_ranges:
            # Clip the annotation span to the current window's index range
            plot_start = max(start, start_idx)
            plot_end = min(end, end_idx)
            
            if plot_start < plot_end: # Ensure there is an overlapping range to plot
                plt.axvspan(plot_start, plot_end, alpha=0.3, color='gray', 
                           label='No-Use' if start == no_use_ranges[0][0] else "")
    
    plt.title(f'Heart Rate Data Annotation - {sample_name} (Points {start_idx} to {end_idx-1})')
    plt.xlabel(f'Data Point Index ({target_rate} Hz sampling)') # Updated x-axis label
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)

def save_annotations(config_path, no_use_ranges):
    """Save the annotated no-use ranges (indices refer to resampled data) to the config file."""
    config = load_config(config_path)
    if 'train-heart-rate' not in config:
        config['train-heart-rate'] = {}
    # Add sampling rate info to saved annotations for clarity
    config['train-heart-rate']['sampling_rate_hz'] = TARGET_RATE 
    config['train-heart-rate']['no-use'] = no_use_ranges
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Updated config file with annotations (at {TARGET_RATE} Hz): {config_path}")

def main():
    parser = argparse.ArgumentParser(description="Annotate no-use data points for heart rate data.")
    parser.add_argument('data_dir', type=str, help="Directory containing the processed data and config.json")
    parser.add_argument('--window_duration', type=float, default=2.0, help="Duration of each window in seconds (default: 2.0)")
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    config_path = data_path / 'config.json'

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found at: {data_path}")

    record_id = data_path.name
    raw_dir = Path('data/raw') / record_id
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found at: {raw_dir}")
    print(f"Using raw directory: {raw_dir}")

    # Ensure raw files exist before copying (copy_raw_files raises error if not found)
    copy_raw_files(raw_dir, args.data_dir)

    config = load_config(str(config_path))
    print(f"Loaded config from: {config_path}")
    original_rate = config.get('data_rate_hz', 100) # Get original rate, default 100Hz
    print(f"Original data rate from config: {original_rate} Hz (defaulted if not found)")

    # Load and resample heart rate data
    hr_data_resampled, duration = load_heart_rate_data(str(data_path), original_rate, TARGET_RATE)
    
    # Load and interpolate annotations to target rate
    annotations_resampled = load_annotations(str(data_path), duration, TARGET_RATE)

    # --- Use resampled data from here on --- 
    total_points = len(hr_data_resampled)
    print(f"Total points after resampling to {TARGET_RATE} Hz: {total_points}")
    
    # Load existing no-use ranges (Note: these indices refer to TARGET_RATE data if saved previously)
    train_data = config.get('train-heart-rate', {})
    # Check if existing annotations match the target rate
    existing_no_use = []
    if 'no-use' in train_data:
        saved_rate = train_data.get('sampling_rate_hz', None)
        if saved_rate == TARGET_RATE:
            existing_no_use = train_data.get('no-use', [])
            print(f"Loaded existing no-use ranges saved at {TARGET_RATE} Hz.")
        elif saved_rate is not None:
            print(f"Warning: Existing no-use ranges were saved at {saved_rate} Hz, but target rate is {TARGET_RATE} Hz. Discarding old ranges.")
        else:
             print(f"Warning: Existing no-use ranges found but saved without sampling rate info. Assuming they are for {TARGET_RATE} Hz.")
             existing_no_use = train_data.get('no-use', [])
             
    if existing_no_use:
        print("Existing no-use ranges loaded:")
        for start, end in existing_no_use:
            print(f"  - Points {start} to {end}")

    # Calculate window size based on TARGET_RATE
    window_size = int(args.window_duration * TARGET_RATE)
    print(f"Window size: {window_size} points ({args.window_duration}s at {TARGET_RATE} Hz)")
    current_start = 0
    no_use_ranges = existing_no_use.copy() # Start with loaded ranges (if any)
    changes_made = False # Track if new annotations are added

    history = []
    history_index = -1

    def save_state(action, data_before):
        nonlocal history, history_index, changes_made
        history = history[:history_index + 1]
        history.append({'action': action, 'data_before': data_before})
        history_index += 1
        changes_made = True # Mark changes when state is saved

    def undo():
        nonlocal history_index, no_use_ranges, changes_made
        if history_index < 0:
            print("No actions to undo.")
            return
        state = history[history_index]
        action = state['action']
        data_before = state['data_before']
        if action == 'annotate':
            no_use_ranges = data_before['no_use_ranges']
            print("Undid annotating a no-use range.")
        history_index -= 1
        # If history becomes empty after undo, reset changes_made
        if history_index == -1 and no_use_ranges == existing_no_use:
             changes_made = False

    def redo(): # Limited redo for simplicity
        nonlocal history_index, changes_made
        if history_index >= len(history) - 1:
            print("No actions to redo.")
            return
        history_index += 1
        state = history[history_index]
        action = state['action']
        # Re-apply the action - simplistic redo, only for annotate here
        if action == 'annotate':
            # This assumes the 'annotate' action always appends
            # A full redo would need 'data_after' state
            if 'range_to_add' in state['data_before']:
                 no_use_ranges.append(state['data_before']['range_to_add'])
                 print("Redid annotating a no-use range.")
                 changes_made = True
            else:
                 print("Redo information incomplete.")
                 history_index -= 1 # Revert index if cannot redo
        else:
            print("Redo not implemented for this action.")
            history_index -= 1 # Revert index

    plt.ion()

    while True:
        current_end = min(current_start + window_size, total_points)
        if current_start >= total_points:
             print("Reached end of data.")
             current_start = max(0, total_points - window_size) # Go back one window
             current_end = total_points
             
        print(f"Displaying window: Points {current_start} to {current_end-1}")
        # Pass TARGET_RATE to plotting function
        plot_data_window(hr_data_resampled, annotations_resampled, current_start, current_end, data_path.name, TARGET_RATE, no_use_ranges)

        print("Commands:")
        print("  Navigation: 'n' or 'next' (next window), 'l' or 'last' (previous window)")
        print("  Annotation: 'ann' or 'annotate start,end' (mark no-use range - indices are for resampled data)")
        print("  Other: 'undo' (revert last action), 'redo' (reapply undone action), 'done' (finish and save), 'exit' (exit without saving)")
        user_input = input("Enter command: ").strip().lower()

        if user_input == 'done':
            plt.ioff()
            if changes_made:
                save_annotations(str(config_path), no_use_ranges)
            else:
                print("No changes were made to annotations.")
            break
        elif user_input == 'exit':
            plt.ioff()
            print("Exiting without saving changes.")
            break
        elif user_input in ['n', 'next']:
            if current_end >= total_points:
                print("Already at the last window.")
            else:
                current_start = current_end
        elif user_input in ['l', 'last']:
            if current_start <= 0:
                print("Already at the first window.")
            else:
                current_start = max(0, current_start - window_size)
        elif user_input == 'undo':
            undo()
        elif user_input == 'redo':
            redo()
        elif user_input.startswith('ann ') or user_input.startswith('annotate '):
            try:
                range_str = user_input.split(' ', 1)[1]
                start, end = map(int, range_str.split(','))
                # Ensure indices are within the bounds of the resampled data
                start = max(0, min(start, total_points - 1))
                end = max(0, min(end, total_points - 1))
                if start > end:
                    print(f"Invalid range. Start index must be <= end index.")
                    continue
                range_to_add = [start, end]
                # Pass the range being added for potential redo
                save_state('annotate', {'no_use_ranges': no_use_ranges.copy(), 'range_to_add': range_to_add})
                no_use_ranges.append(range_to_add)
                print(f"Added no-use range: Points {start} to {end} (at {TARGET_RATE} Hz)")
            except (ValueError, IndexError):
                print("Invalid input. Please enter in the format 'ann start,end' or 'annotate start,end'.")
        else:
            print("Unknown command. Use 'n'/'next', 'l'/'last', 'ann'/'annotate start,end', 'undo', 'redo', 'done', or 'exit'.")

if __name__ == '__main__':
    main() 