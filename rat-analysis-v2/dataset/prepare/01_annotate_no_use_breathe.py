import os
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.signal import find_peaks
import argparse
from pathlib import Path


def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def extract_rectangle_data_from_split_videos(data_dir, config):
    """Extract mean intensity of R channel from specified rectangle in each frame of split videos."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise ValueError(f"Data directory not found: {data_dir}")

    video_files = sorted([f for f in data_dir.glob('video_*.mp4') if f.is_file()], 
                         key=lambda x: int(x.stem.split('_')[-1]) if x.stem.split('_')[-1].isdigit() else 0)
    if not video_files:
        raise ValueError(f"No video files found in {data_dir}")

    expected_frames_per_segment = config.get('target_frames_per_segment', 600)
    total_expected_frames = config.get('total_frames', 0)
    number_of_segments = config.get('number_of_segments', len(video_files))
    r_values = []
    frame_count = 0

    # Limit processing to the number of segments specified in config
    video_files = video_files[:number_of_segments]

    for video_path in video_files:
        if frame_count >= total_expected_frames and total_expected_frames > 0:
            print(f"Stopping processing as total frames reached expected limit of {total_expected_frames}.")
            break

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Warning: Could not open video file: {video_path}")
            continue

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        segment_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count += segment_frames

        rect = config['rectangle']
        start_x = int(rect['start'][0] * width)
        start_y = int(rect['start'][1] * height)
        end_x = int(rect['end'][0] * width)
        end_y = int(rect['end'][1] * height)

        for frame_idx in range(segment_frames):
            if frame_count > total_expected_frames and total_expected_frames > 0:
                print(f"Stopping frame extraction as total frames exceeded expected limit of {total_expected_frames}.")
                break
            ret, frame = cap.read()
            if not ret:
                break
            # Extract rectangle and calculate mean of R channel
            rect_frame = frame[start_y:end_y, start_x:end_x]
            r_mean = np.mean(rect_frame[:, :, 2])  # R channel in BGR format
            r_values.append(r_mean)

        cap.release()
        print(f"Processed video: {video_path.name} with {segment_frames} frames")

    if not r_values:
        raise ValueError(f"No frames processed from videos in {data_dir}")

    if total_expected_frames > 0 and frame_count != total_expected_frames:
        print(f"Warning: Total frames processed ({frame_count}) does not match expected total frames ({total_expected_frames}) from config.")
    return np.array(r_values)


def plot_data_window(data, start_idx, end_idx, sample_name, peaks=None, valleys=None, no_use_ranges=None):
    """Plot a window of R channel data with optional peaks, valleys, and no-use ranges, including index labels."""
    # Close any existing plot to avoid multiple windows
    plt.close('all')
    plt.figure(figsize=(12, 6))
    window_data = data[start_idx:end_idx]
    x_axis = range(start_idx, end_idx)
    plt.plot(x_axis, window_data, label='R Channel Intensity', color='red')
    
    if peaks is not None:
        window_peaks = peaks[(peaks >= start_idx) & (peaks < end_idx)]
        plt.plot(window_peaks, data[window_peaks], 'x', label='Peaks', color='green')
        # Add index labels for peaks
        for idx, peak in enumerate(window_peaks):
            plt.text(peak, data[peak], f'P{idx}', fontsize=8, ha='center', va='bottom', color='green')
    if valleys is not None:
        window_valleys = valleys[(valleys >= start_idx) & (valleys < end_idx)]
        plt.plot(window_valleys, data[window_valleys], 'o', label='Valleys', color='blue')
        # Add index labels for valleys
        for idx, valley in enumerate(window_valleys):
            plt.text(valley, data[valley], f'V{idx}', fontsize=8, ha='center', va='top', color='blue')
    if no_use_ranges:
        for start, end in no_use_ranges:
            if start < end_idx and end > start_idx:  # Check if range overlaps with window
                plt.axvspan(max(start, start_idx), min(end, end_idx), alpha=0.3, color='gray', 
                           label='No-Use' if start == no_use_ranges[0][0] else "")

    plt.title(f'Breath Data Annotation - {sample_name} (Frames {start_idx} to {end_idx-1})')
    plt.xlabel('Frame Index')
    plt.ylabel('R Channel Mean Intensity')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)  # Brief pause to allow plot to render without blocking


def find_peaks_valleys(data, distance=20, prominence=0.5):
    """Find peaks and valleys in the data using scipy with customizable parameters."""
    # Find peaks
    peaks, _ = find_peaks(data, distance=distance, prominence=prominence)
    # Find valleys by inverting the data
    valleys, _ = find_peaks(-data, distance=distance, prominence=prominence)
    return peaks, valleys


def save_annotations(config_path, no_use_ranges, peaks, valleys):
    """Save the annotated no-use ranges, peaks, and valleys to the config file."""
    config = load_config(config_path)
    if 'train-breathe' not in config:
        config['train-breathe'] = {}
    config['train-breathe']['no-use'] = no_use_ranges
    config['train-breathe']['peaks'] = peaks.tolist()  # Convert numpy array to list for JSON serialization
    config['train-breathe']['valleys'] = valleys.tolist()  # Convert numpy array to list for JSON serialization
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Updated config file with annotations: {config_path}")


def add_peak(peaks, frame_idx, total_frames):
    """Add a peak at the specified frame index if within bounds."""
    if 0 <= frame_idx < total_frames and frame_idx not in peaks:
        peaks = np.append(peaks, frame_idx)
        peaks = np.sort(peaks)
        print(f"Added peak at frame {frame_idx}")
    else:
        print(f"Invalid frame index {frame_idx} or peak already exists at this index. Frame index must be between 0 and {total_frames-1}.")
    return peaks


def add_valley(valleys, frame_idx, total_frames):
    """Add a valley at the specified frame index if within bounds."""
    if 0 <= frame_idx < total_frames and frame_idx not in valleys:
        valleys = np.append(valleys, frame_idx)
        valleys = np.sort(valleys)
        print(f"Added valley at frame {frame_idx}")
    else:
        print(f"Invalid frame index {frame_idx} or valley already exists at this index. Frame index must be between 0 and {total_frames-1}.")
    return valleys


def remove_peak(peaks, index, start_idx, end_idx):
    """Remove a peak by its index within the current window."""
    window_peaks = peaks[(peaks >= start_idx) & (peaks < end_idx)]
    if 0 <= index < len(window_peaks):
        frame_to_remove = window_peaks[index]
        peaks = peaks[peaks != frame_to_remove]
        print(f"Removed peak at frame {frame_to_remove}")
    else:
        print(f"Invalid peak index {index} for current window.")
    return peaks


def remove_valley(valleys, index, start_idx, end_idx):
    """Remove a valley by its index within the current window."""
    window_valleys = valleys[(valleys >= start_idx) & (valleys < end_idx)]
    if 0 <= index < len(window_valleys):
        frame_to_remove = window_valleys[index]
        valleys = valleys[valleys != frame_to_remove]
        print(f"Removed valley at frame {frame_to_remove}")
    else:
        print(f"Invalid valley index {index} for current window.")
    return valleys


def main():
    parser = argparse.ArgumentParser(description="Annotate no-use frames for breath data.")
    parser.add_argument('data_dir', type=str, help="Directory containing the split videos and config.json")
    parser.add_argument('--window_duration', type=float, default=5.0, help="Duration of each window in seconds (default: 5.0)")
    parser.add_argument('--peak_distance', type=int, default=20, help="Minimum distance between peaks/valleys (default: 20)")
    parser.add_argument('--peak_prominence', type=float, default=0.5, help="Prominence threshold for peaks/valleys (default: 0.5)")
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    config_path = data_path / 'config.json'

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found at: {data_path}")

    # Load configuration
    config = load_config(str(config_path))
    print(f"Loaded config from: {config_path}")

    # Extract R channel data from split videos
    r_data = extract_rectangle_data_from_split_videos(str(data_path), config)
    print(f"Extracted R channel data from {len(r_data)} frames.")

    # Find peaks and valleys with custom parameters
    peaks, valleys = find_peaks_valleys(r_data, distance=args.peak_distance, prominence=args.peak_prominence)
    print(f"Found {len(peaks)} peaks and {len(valleys)} valleys with distance={args.peak_distance} and prominence={args.peak_prominence}.")

    # Check if there are existing no-use ranges, peaks, and valleys
    train_data = config.get('train', {})
    existing_no_use = train_data.get('no-use', [])
    existing_peaks = np.array(train_data.get('peaks', [])) if 'peaks' in train_data else peaks.copy()
    existing_valleys = np.array(train_data.get('valleys', [])) if 'valleys' in train_data else valleys.copy()

    if existing_no_use:
        print("Existing no-use ranges found:")
        for start, end in existing_no_use:
            print(f"  - Frames {start} to {end}")
    if 'peaks' in train_data and len(existing_peaks) > 0:
        print(f"Loaded {len(existing_peaks)} existing peaks from config.")
        peaks = existing_peaks
    if 'valleys' in train_data and len(existing_valleys) > 0:
        print(f"Loaded {len(existing_valleys)} existing valleys from config.")
        valleys = existing_valleys

    # Calculate window size based on frame rate (assuming frame_rate_fps is in config)
    frame_rate = config.get('frame_rate_fps', 120)  # Default to 120 if not found
    window_size = int(args.window_duration * frame_rate)
    total_frames = len(r_data)
    print(f"Total frames in data: {total_frames}")
    current_start = 0
    no_use_ranges = existing_no_use.copy()

    # History for undo/redo functionality
    history = []
    history_index = -1

    def save_state(action, data_before):
        nonlocal history, history_index
        # Truncate history after current index (for redo after undo)
        history = history[:history_index + 1]
        history.append({'action': action, 'data_before': data_before})
        history_index += 1

    def undo():
        nonlocal history_index, peaks, valleys, no_use_ranges
        if history_index < 0:
            print("No actions to undo.")
            return
        state = history[history_index]
        action = state['action']
        data_before = state['data_before']
        if action == 'add_peak':
            peaks = data_before['peaks']
            print("Undid adding a peak.")
        elif action == 'add_valley':
            valleys = data_before['valleys']
            print("Undid adding a valley.")
        elif action == 'remove_peak':
            peaks = data_before['peaks']
            print("Undid removing a peak.")
        elif action == 'remove_valley':
            valleys = data_before['valleys']
            print("Undid removing a valley.")
        elif action == 'annotate':
            no_use_ranges = data_before['no_use_ranges']
            print("Undid annotating a no-use range.")
        history_index -= 1

    def redo():
        nonlocal history_index, peaks, valleys, no_use_ranges
        if history_index >= len(history) - 1:
            print("No actions to redo.")
            return
        history_index += 1
        state = history[history_index]
        action = state['action']
        # Since we don't store 'data_after', we can't directly redo, but we can simulate based on action
        # For simplicity, we'll just notify the user; full redo functionality would require storing 'data_after'
        print("Redo functionality is limited. Please perform the action again if needed.")
        history_index -= 1  # Revert to prevent confusion since we can't fully redo

    # Turn on interactive mode for non-blocking plots
    plt.ion()

    while True:
        current_end = min(current_start + window_size, total_frames)
        print(f"Displaying window: Frames {current_start} to {current_end-1}")
        plot_data_window(r_data, current_start, current_end, data_path.name, peaks, valleys, no_use_ranges)

        # CLI for navigation and annotation
        print("Commands:")
        print("  Navigation: 'n' or 'next' (next window), 'l' or 'last' (previous window)")
        print("  Annotation: 'ann' or 'annotate start,end' (mark no-use range)")
        print("  Add: 'a p frame' or 'add peak frame' (add peak), 'a v frame' or 'add valley frame' (add valley)")
        print("  Remove: 'r p index' or 'remove peak index' (remove peak), 'r v index' or 'remove valley index' (remove valley)")
        print("  Other: 'undo' (revert last action), 'redo' (reapply undone action), 'done' (finish and save), 'exit' (exit without saving)")
        user_input = input("Enter command: ").strip().lower()

        if user_input == 'done':
            plt.ioff()  # Turn off interactive mode before exiting
            if no_use_ranges != existing_no_use or not np.array_equal(peaks, existing_peaks) or not np.array_equal(valleys, existing_valleys):
                save_annotations(str(config_path), no_use_ranges, peaks, valleys)
            else:
                print("No changes were made to annotations.")
            break
        elif user_input == 'exit':
            plt.ioff()  # Turn off interactive mode before exiting
            print("Exiting without saving changes.")
            break
        elif user_input in ['n', 'next']:
            if current_end >= total_frames:
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
                # Extract the range part from either 'ann start,end' or 'annotate start,end'
                range_str = user_input.split(' ', 1)[1]
                start, end = map(int, range_str.split(','))
                # Adjust start and end to be within valid frame range
                start = max(0, min(start, total_frames - 1))
                end = max(0, min(end, total_frames - 1))
                if start > end:
                    print(f"Invalid range. Start must be <= end.")
                    continue
                save_state('annotate', {'no_use_ranges': no_use_ranges.copy()})
                no_use_ranges.append([start, end])
                print(f"Added no-use range: {start} to {end}")
            except (ValueError, IndexError):
                print("Invalid input. Please enter in the format 'ann start,end' or 'annotate start,end'.")
        elif user_input.startswith('a p ') or user_input.startswith('add peak '):
            try:
                # Extract the frame index from either 'a p frame' or 'add peak frame'
                frame_idx = int(user_input.split(' ', 2)[2])
                if frame_idx < 0 or frame_idx >= total_frames:
                    print(f"Invalid frame index. Must be between 0 and {total_frames-1}.")
                    continue
                save_state('add_peak', {'peaks': peaks.copy()})
                peaks = add_peak(peaks, frame_idx, total_frames)
            except (ValueError, IndexError):
                print("Invalid input. Please enter in the format 'a p frame' or 'add peak frame'.")
        elif user_input.startswith('a v ') or user_input.startswith('add valley '):
            try:
                # Extract the frame index from either 'a v frame' or 'add valley frame'
                frame_idx = int(user_input.split(' ', 2)[2])
                if frame_idx < 0 or frame_idx >= total_frames:
                    print(f"Invalid frame index. Must be between 0 and {total_frames-1}.")
                    continue
                save_state('add_valley', {'valleys': valleys.copy()})
                valleys = add_valley(valleys, frame_idx, total_frames)
            except (ValueError, IndexError):
                print("Invalid input. Please enter in the format 'a v frame' or 'add valley frame'.")
        elif user_input.startswith('r p ') or user_input.startswith('remove peak '):
            try:
                # Extract the index from either 'r p index' or 'remove peak index'
                index = int(user_input.split(' ', 2)[2])
                save_state('remove_peak', {'peaks': peaks.copy()})
                peaks = remove_peak(peaks, index, current_start, current_end)
            except (ValueError, IndexError):
                print("Invalid input. Please enter in the format 'r p index' or 'remove peak index'.")
        elif user_input.startswith('r v ') or user_input.startswith('remove valley '):
            try:
                # Extract the index from either 'r v index' or 'remove valley index'
                index = int(user_input.split(' ', 2)[2])
                save_state('remove_valley', {'valleys': valleys.copy()})
                valleys = remove_valley(valleys, index, current_start, current_end)
            except (ValueError, IndexError):
                print("Invalid input. Please enter in the format 'r v index' or 'remove valley index'.")
        else:
            print("Unknown command. Use 'n'/'next', 'l'/'last', 'ann'/'annotate start,end', 'a p'/'add peak frame', 'a v'/'add valley frame', 'r p'/'remove peak index', 'r v'/'remove valley index', 'undo', 'redo', 'done', or 'exit'.")

if __name__ == '__main__':
    main() 