import json
import math
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torch import nn
from torchvision.transforms import functional as VTF
from torch.nn import functional as F
import torchvision.io as vio  # For efficient video reading
import shutil
import matplotlib.pyplot as plt # Added for plotting
from scipy.interpolate import interp1d # Added for cubic interpolation

# --- Transform Classes (Copied from old_rat_dataset.py, may need adaptation) ---

class AffineTransform(nn.Module):
    def __init__(self, rotate_angle, scale_factor, shift_range):
        """
            Args:
                rotate_angle (tuple): (min_angle, max_angle) in degrees.
                scale_factor (tuple): (min_scale, max_scale).
                shift_range (tuple): (min_shift, max_shift) in normalized coordinates.
        """
        super().__init__()
        self.rotate_angle = rotate_angle
        self.scale_factor = scale_factor
        self.shift_range = shift_range

    def forward(self, *x):

        out = []
        rotate_angle = torch.FloatTensor(1).uniform_(*self.rotate_angle).item()
        scale_factor = torch.FloatTensor(1).uniform_(*self.scale_factor).item()
        shift_x = torch.FloatTensor(1).uniform_(*self.shift_range).item()
        shift_y = torch.FloatTensor(1).uniform_(*self.shift_range).item()

        angle_rad = torch.deg2rad(torch.tensor(rotate_angle))
        cos_val = torch.cos(angle_rad) * scale_factor
        sin_val = torch.sin(angle_rad) * scale_factor
        theta = torch.tensor([
            [cos_val, -sin_val, shift_x],
            [sin_val, cos_val, shift_y]
        ], dtype=torch.float).unsqueeze(0)

        for x_ in x:
            T, C, H, W = x_.shape

            theta_ = theta.repeat(T, 1, 1)

            grid = F.affine_grid(theta_, x_.size(), align_corners=False)
            x_ = F.grid_sample(x_, grid, align_corners=False)

            out.append(x_)

        if len(out) == 1:
            return out[0]

        return tuple(out)


class SpeedTransform(nn.Module):
    def __init__(self, speed_factor, out_len):
        super().__init__()
        self.speed_factor = speed_factor
        self.out_len = out_len # Renamed from out_frames for clarity

    def forward(self, video, *time_series_data):
        T_in, C, H, W = video.shape
        outputs = []

        # Determine speed factor
        speed_factor = torch.FloatTensor(1).uniform_(*self.speed_factor).item()
        T_new = int(round(T_in * speed_factor))
        T_new = max(T_new, 1)

        # --- Process Video ---
        video = video.permute(1, 0, 2, 3) # C, T, H, W
        video = F.interpolate(
            video.unsqueeze(0), # Add batch dim: 1, C, T_in, H, W
            size=(T_new, H, W),
            mode='trilinear',
            align_corners=True # Keep True as in original for now
        ).squeeze(0) # Remove batch dim: C, T_new, H, W
        video = video.permute(1, 0, 2, 3) # T_new, C, H, W

        # Pad or truncate video
        if T_new <= self.out_len:
            num_to_pad = self.out_len - T_new
            if num_to_pad > 0:
                last_frame = video[-1:, :, :, :] # Keep dim T=1
                padding = last_frame.repeat(num_to_pad, 1, 1, 1)
                video = torch.cat([video, padding], dim=0)
        else:
            start = random.randint(0, T_new - self.out_len)
            video = video[start:start + self.out_len, :, :, :]
        outputs.append(video)

        # --- Process Time Series Data (e.g., HR, Breath) ---
        for y in time_series_data:
            if y is None: # Handle cases where data might not be present
                outputs.append(None)
                continue

            y_in = y.shape[0]
            assert y_in == T_in, f"Time series length {y_in} must match input video length {T_in}"

            y = F.interpolate(
                y.unsqueeze(0).unsqueeze(0), # 1, 1, T_in
                size=(T_new),
                mode='linear',
                align_corners=False # Changed from True to False for consistency
            ).squeeze(0).squeeze(0) # T_new

            # Pad or truncate time series
            if T_new <= self.out_len:
                num_to_pad = self.out_len - T_new
                if num_to_pad > 0:
                    last_val = y[-1:] # Keep dim T=1
                    padding = last_val.repeat(num_to_pad)
                    y = torch.cat([y, padding], dim=0)
            else:
                # Use the same start index as the video for consistency
                start = random.randint(0, T_new - self.out_len) # Re-using the same start logic as video
                y = y[start:start + self.out_len]
            outputs.append(y)

        return tuple(outputs)


# --- Main Dataset Class ---

class RatDataset(Dataset):
    def __init__(self,
                 data_root="data/processed",
                 session_id="20250329_150340", # Specific session to load
                 target_fps=30, # Target frames per second for video and time series
                 input_frames=220, # Number of frames to load initially (before speed transform)
                 output_frames=150, # Desired number of frames *after* speed transform
                 speed_factor=(0.8, 1.25),
                 scale_factor=(0.5, 1.5),
                 rotate_angle=(-180, 180),
                 shift_range=(-0.2, 0.2),
                 ):
        super().__init__()
        
        if input_frames < output_frames:
            raise ValueError("input_frames must be greater than or equal to output_frames")

        self.session_path = os.path.join(data_root, session_id)
        if not os.path.isdir(self.session_path):
            raise FileNotFoundError(f"Session directory not found: {self.session_path}")

        self.target_fps = target_fps
        self.input_frames = input_frames # Number of frames to load
        self.output_frames = output_frames # Target length after transforms
        self.out_len = output_frames # Keep self.out_len for SpeedTransform

        self.video_files = sorted([os.path.join(self.session_path, f)
                                   for f in os.listdir(self.session_path) if f.endswith('.mp4')])
        if not self.video_files:
            raise FileNotFoundError(f"No MP4 files found in {self.session_path}")

        # --- Load Metadata ---
        config_path = os.path.join(self.session_path, "config.json")
        hr_annotations_path = os.path.join(self.session_path, "hr_annotations.json")
        heart_rate_path = os.path.join(self.session_path, "heart_rate.txt")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config.json not found in {self.session_path}")
        if not os.path.exists(hr_annotations_path):
            raise FileNotFoundError(f"hr_annotations.json not found in {self.session_path}")
        if not os.path.exists(heart_rate_path):
            raise FileNotFoundError(f"heart_rate.txt not found in {self.session_path}")

        with open(config_path, 'r') as f:
            self.config = json.load(f)
        with open(hr_annotations_path, 'r') as f:
            # Changed: Load raw hr_annotations, not processed ones with no_use_intervals key
            raw_hr_annotation_data = json.load(f)

        # --- Generate Heart Rate Signal from Annotations --- 
        print("Generating heart rate signal from annotations...")
        hr_times = []
        hr_values = [] # 1 for peak, 0 for valley
        
        # Handle potential structures in hr_annotations.json
        if isinstance(raw_hr_annotation_data, dict) and 'annotations' in raw_hr_annotation_data:
            annotation_list = raw_hr_annotation_data['annotations']
        elif isinstance(raw_hr_annotation_data, list):
            annotation_list = raw_hr_annotation_data
        else:
             raise ValueError(f"Unexpected JSON structure in {hr_annotations_path}. Expected a list or {{'annotations': [...]}}.")

        for item in annotation_list:
            try:
                # Annotations might be strings containing JSON, or dicts directly
                if isinstance(item, str):
                    ann = json.loads(item)
                elif isinstance(item, dict):
                    ann = item
                else:
                    # print(f"Warning: Skipping unrecognized annotation format: {item}")
                    continue
                    
                if 'time' in ann and 'type' in ann and ann['type'] in ['peak', 'valley']:
                    hr_times.append(float(ann['time']))
                    hr_values.append(1.0 if ann['type'] == 'peak' else 0.0)
                # else: # Optional: warn about missing keys or wrong types
                #     # print(f"Warning: Skipping annotation due to missing keys or invalid type: {ann}")
                    
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                pass
                # print(f"Warning: Could not process annotation item {item}: {e}")

        if not hr_times:
            raise ValueError("No valid peak/valley annotations found to generate heart rate signal.")

        # Sort by time and handle duplicates (keep first)
        sorted_indices = np.argsort(hr_times)
        hr_times_sorted = np.array(hr_times)[sorted_indices]
        hr_values_sorted = np.array(hr_values)[sorted_indices]
        unique_times, unique_indices = np.unique(hr_times_sorted, return_index=True)
        unique_values = hr_values_sorted[unique_indices]

        # Get interpolation parameters
        self.hr_sampling_rate = self.config.get('train-heart-rate', {}).get('sampling_rate_hz')
        if self.hr_sampling_rate is None:
            # print("Warning: 'train-heart-rate.sampling_rate_hz' not found in config. Using fallback 120 Hz.")
            self.hr_sampling_rate = 120 # Fallback HR rate
        else:
             print(f"Using HR sampling rate from config: {self.hr_sampling_rate} Hz")
             
        self.total_duration_seconds = self.config.get('total_duration_seconds')
        if self.total_duration_seconds is None:
             raise ValueError("Missing 'total_duration_seconds' in config.json")
             
        num_hr_points = int(self.total_duration_seconds * self.hr_sampling_rate)
        target_hr_times = np.linspace(0, self.total_duration_seconds, num_hr_points)

        # Perform cubic interpolation
        if len(unique_times) > 1:
            interp_func = interp1d(unique_times, unique_values, kind='cubic', bounds_error=False, fill_value=(unique_values[0], unique_values[-1]))
            interpolated_hr = interp_func(target_hr_times)
        elif len(unique_times) == 1:
             # print("Warning: Only one unique annotation time point. Filling HR signal with constant value.")
             interpolated_hr = np.full(num_hr_points, unique_values[0])
        else:
            # Should not happen if check passed earlier, but defensively:
             interpolated_hr = np.zeros(num_hr_points)

        self.heart_rate = torch.from_numpy(interpolated_hr).float()
        self.total_hr_points = len(self.heart_rate)
        print(f"Generated HR signal with {self.total_hr_points} points at {self.hr_sampling_rate} Hz.")
        
        # --- Load HR Validity Mask --- 
        self.hr_validity = torch.ones(self.total_hr_points, dtype=torch.bool)
        hr_config = self.config.get('train-heart-rate', {})
        hr_no_use_intervals = hr_config.get('no-use', [])
        saved_hr_rate = hr_config.get('sampling_rate_hz', None)

        if hr_no_use_intervals:
            if saved_hr_rate == self.hr_sampling_rate:
                print(f"Applying {len(hr_no_use_intervals)} HR no-use intervals (indices are for {self.hr_sampling_rate} Hz)...")
                for interval in hr_no_use_intervals:
                    start, end = interval
                    start = max(0, start)
                    end = min(self.total_hr_points, end) # Clip to HR signal length
                    if start < end:
                         self.hr_validity[start:end] = 0
            elif saved_hr_rate is not None:
                pass
                 # print(f"Warning: HR no-use intervals in config saved at {saved_hr_rate} Hz, but interpolating at {self.hr_sampling_rate} Hz. Cannot apply intervals.")
            else:
                pass
                 # print(f"Warning: HR no-use intervals found but sampling rate missing in config. Cannot reliably apply intervals.")

        # --- Generate Breath Signal from Annotations --- 
        print("Generating breath signal from annotations...")
        breath_times = [] # Using frame index as time marker
        breath_values = [] # 1 for peak, 0 for valley
        breath_config = self.config.get('train-breathe', {})
        breath_peaks = breath_config.get('peaks', [])
        breath_valleys = breath_config.get('valleys', [])
        
        for frame_idx in breath_peaks:
            breath_times.append(float(frame_idx))
            breath_values.append(1.0)
        for frame_idx in breath_valleys:
            breath_times.append(float(frame_idx))
            breath_values.append(0.0)
            
        if not breath_times:
             # print("Warning: No breath peak/valley annotations found in config. Breath signal will be zeros.")
             self.breath_signal = torch.zeros(self.total_hr_points)
        else:
            # Sort by time (frame index) and handle duplicates
            sorted_indices_breath = np.argsort(breath_times)
            breath_times_sorted = np.array(breath_times)[sorted_indices_breath]
            breath_values_sorted = np.array(breath_values)[sorted_indices_breath]
            unique_times_breath, unique_indices_breath = np.unique(breath_times_sorted, return_index=True)
            unique_values_breath = breath_values_sorted[unique_indices_breath]
            
            # Target time axis is the original video frame indices
            target_breath_times = np.arange(self.total_hr_points)
            
            # Perform cubic interpolation
            if len(unique_times_breath) > 1:
                 interp_func_breath = interp1d(unique_times_breath, unique_values_breath, kind='cubic', bounds_error=False, fill_value=(unique_values_breath[0], unique_values_breath[-1]))
                 interpolated_breath = interp_func_breath(target_breath_times)
            elif len(unique_times_breath) == 1:
                 # print("Warning: Only one unique breath annotation time point. Filling breath signal with constant value.")
                 interpolated_breath = np.full(self.total_hr_points, unique_values_breath[0])
            else:
                 interpolated_breath = np.zeros(self.total_hr_points)
                 
            self.breath_signal = torch.from_numpy(interpolated_breath).float()
            
        self.total_breath_points = len(self.breath_signal) # Should match total_hr_points
        print(f"Generated breath signal with {self.total_breath_points} points (at video FPS)." )

        # --- Load Breath Validity Mask --- 
        self.breath_validity = torch.ones(self.total_breath_points, dtype=torch.bool)
        breath_no_use_intervals = breath_config.get('no-use', [])
        if breath_no_use_intervals:
             print(f"Applying {len(breath_no_use_intervals)} breath no-use intervals (indices are original video frames)...")
             for interval in breath_no_use_intervals:
                 start, end = interval
                 start = max(0, start)
                 end = min(self.total_breath_points, end) # Clip to breath signal length
                 if start < end:
                      self.breath_validity[start:end] = 0

        # --- Determine Original Video FPS and Total Frames ---
        # Also, map global frame indices to video files
        # Try to get metadata from the first video file
        self.video_frame_counts = []
        self.video_cumulative_frames = [0]
        self.total_original_frames = 0
        self.original_fps = None
        num_video_files = len(self.video_files)

        print("Reading metadata from first video file...")
        if self.video_files:
            first_video_path = self.video_files[0]
            try:
                # Read the first video completely to get info (less efficient)
                vframes, aframes, info = vio.read_video(first_video_path, pts_unit='sec')
                self.original_fps = info.get('video_fps', self.config.get("video_fps", 30)) # Get FPS from info or config
                first_video_frames = vframes.shape[0]
                print(f"Read first video: {first_video_frames} frames at {self.original_fps} FPS.")
                
                # Estimate total frames and build mapping (assuming similar length files)
                # This is an approximation!
                self.total_original_frames = first_video_frames * num_video_files # Rough estimate
                print(f"Estimating total frames: {self.total_original_frames} (this might be inaccurate)")
                
                # Create approximate cumulative frames
                current_count = 0
                for i in range(num_video_files):
                     # Assume all files (except maybe last) have same frame count as first
                     # This is a major simplification!
                     frames_in_this_file = first_video_frames 
                     self.video_frame_counts.append(frames_in_this_file)
                     current_count += frames_in_this_file
                     self.video_cumulative_frames.append(current_count)
                # Adjust total frames based on calculation
                self.total_original_frames = self.video_cumulative_frames[-1]

            except Exception as e:
                # print(f"Warning: Could not read metadata from first video file {first_video_path}. Error: {e}")
                # Fallback to using HR data length and config FPS
                print("Using total frame count from heart_rate.txt and FPS from config/default as fallback.")
                self.total_original_frames = len(self.heart_rate)
                self.original_fps = self.config.get("video_fps", 30)
                self.video_frame_counts = [] # Invalidate frame counts
                self.video_cumulative_frames = [0, self.total_original_frames] # Cannot map accurately
                # print(f"Warning: Frame mapping will be incorrect due to metadata reading errors.")
        else:
            # No video files found earlier, use HR data
             self.total_original_frames = len(self.heart_rate)
             self.original_fps = self.config.get("video_fps", 30)
             # print("Warning: No video files found, using HR data length for total frames.")


        # Check consistency if total frames were derived from videos vs HR data
        if self.video_cumulative_frames and len(self.heart_rate) != self.total_original_frames:
            # print(f"Warning: Estimated total video frames ({self.total_original_frames}) does not match heart rate data length ({len(self.heart_rate)}).")
            # Adjust HR data and validity masks to match the estimated video length
            min_len = min(self.total_original_frames, len(self.heart_rate))
            if self.total_original_frames < len(self.heart_rate):
                print(f"Truncating heart rate data to estimated video length: {self.total_original_frames}")
                self.heart_rate = self.heart_rate[:self.total_original_frames]
            elif self.total_original_frames > len(self.heart_rate):
                 # print(f"Warning: Heart rate data is shorter than estimated video length. Sequences near the end might have invalid HR.")
                 # Consider padding HR data if needed, or let it fail later
                 pass # Proceeding with shorter HR data

            # Recreate validity mask based on the (potentially shorter) HR data length or estimated video length
            # Use the final self.total_original_frames as the reference length
            self.hr_validity = torch.ones(self.total_original_frames, dtype=torch.bool)
            # Apply annotations within the bounds of the new total_original_frames
            for interval in hr_no_use_intervals:
                 start, end = interval
                 start = max(0, start)
                 end = min(self.total_original_frames, end) # Clip to new total frames
                 if start < end:
                      self.hr_validity[start:end] = 0
            # Also adjust breath validity placeholder
            self.breath_validity = torch.ones(self.total_original_frames, dtype=torch.bool)


        print(f"Using Original FPS: {self.original_fps}, Total Original Frames: {self.total_original_frames}")

        self.frame_skip = int(round(self.original_fps / self.target_fps))
        if self.frame_skip < 1:
            # print(f"Warning: target_fps ({self.target_fps}) is higher than original_fps ({self.original_fps}). Using frame_skip=1.")
            self.frame_skip = 1
        self.effective_seq_len_original = self.output_frames * self.frame_skip # How many original frames needed for one sequence

        # --- Create Validity Masks ---
        self.hr_validity = torch.ones(self.total_original_frames, dtype=torch.bool)
        for interval in hr_no_use_intervals:
            start, end = interval
            # Ensure indices are within bounds
            start = max(0, start)
            end = min(self.total_original_frames, end)
            if start < end: # Make sure interval is valid
                 self.hr_validity[start:end] = 0

        # Placeholder for breath validity - assuming all valid for now
        self.breath_validity = torch.ones(self.total_original_frames, dtype=torch.bool)
        # TODO: Add logic if breath data has its own no-use intervals from config.json if needed

        # --- Initialize Transforms ---
        self.affine_transform = AffineTransform(rotate_angle, scale_factor, shift_range)
        # Pass target output length to SpeedTransform
        self.speed_transform = SpeedTransform(speed_factor, self.out_len)

        # --- Precompute Valid Start Indices (Based on Video Frames) ---
        self.valid_indices = []
        print("Precomputing valid start indices...")
        # Calculate number of HR points corresponding to the *input* video sequence duration
        hr_points_per_video_frame = self.hr_sampling_rate / self.original_fps
        # How many original video frames needed for the *input* sequence
        self.effective_input_seq_len_original = self.input_frames * self.frame_skip 
        # Calculate the number of HR points needed for the *input* sequence duration
        effective_seq_len_hr = int(round(self.effective_input_seq_len_original * hr_points_per_video_frame))

        for i in range(self.total_original_frames - self.effective_input_seq_len_original + 1):
            # Check if the *entire input sequence duration* is valid
            original_start_video_frame = i
            original_end_video_frame = i + self.effective_input_seq_len_original

            # Calculate corresponding HR indices
            start_hr_idx = int(round(original_start_video_frame * hr_points_per_video_frame))
            # Ensure end index doesn't exceed total HR points
            end_hr_idx = min(start_hr_idx + effective_seq_len_hr, self.total_hr_points)
            start_hr_idx = min(start_hr_idx, self.total_hr_points -1 ) # Ensure start is also within bounds
            
            # Check HR validity across the corresponding HR index range
            is_hr_valid = False # Default to invalid if range is bad
            if start_hr_idx < end_hr_idx: # Check if the calculated range is valid
                is_hr_valid = torch.all(self.hr_validity[start_hr_idx:end_hr_idx]).item()

            # Check breath validity across the original video frame range
            # Indices are original_start_video_frame to original_end_video_frame
            is_breath_valid = torch.all(self.breath_validity[original_start_video_frame:original_end_video_frame]).item()
            
            # TODO: Add check for video 'no-use' ranges from config.json if needed

            if is_hr_valid and is_breath_valid: # Now checking both
                self.valid_indices.append(original_start_video_frame)

        if not self.valid_indices:
            raise ValueError("No valid sequences found after filtering with no-use intervals.")

        print(f"Initialized RatDataset: Found {len(self.valid_indices)} valid sequences.")

    def _find_video_file_and_index(self, global_frame_index):
        """Maps a global frame index to its video file index and frame index within that file."""
        if not self.video_cumulative_frames or global_frame_index >= self.video_cumulative_frames[-1]:
            raise IndexError(f"Global frame index {global_frame_index} is out of bounds (total frames: {self.total_original_frames})")
        
        # Find the video file index using binary search on cumulative frames
        file_idx = np.searchsorted(self.video_cumulative_frames, global_frame_index, side='right') - 1
        
        # Calculate the frame index within that specific video file
        frame_idx_in_file = global_frame_index - self.video_cumulative_frames[file_idx]
        
        return file_idx, frame_idx_in_file

    def _get_frames(self, global_indices):
        """Loads specific frames based on global indices, reading from multiple video files efficiently."""
        frames_list = []
        # Group indices by the video file they belong to
        frames_by_file = {} 
        for global_idx in global_indices.tolist(): # Iterate through requested global indices
            try:
                file_idx, frame_idx_in_file = self._find_video_file_and_index(global_idx)
                if file_idx not in frames_by_file:
                    frames_by_file[file_idx] = []
                frames_by_file[file_idx].append((frame_idx_in_file, global_idx)) # Store local index and original global index
            except IndexError as e:
                # print(f"Warning: {e}")
                # Handle error: skip frame, return None, or raise?
                # For now, let's skip and continue, but this might lead to shape mismatches.
                continue 

        # Load frames from each relevant file
        loaded_frames_map = {} # Map global_idx back to its loaded frame tensor
        for file_idx, frame_info_list in sorted(frames_by_file.items()):
            video_path = self.video_files[file_idx]
            # Sort by frame index within the file to potentially read sequentially
            frame_info_list.sort(key=lambda x: x[0])
            local_indices_needed = [info[0] for info in frame_info_list]
            global_indices_in_file = [info[1] for info in frame_info_list]

            # --- Efficient loading is tricky with read_video by index --- 
            # Option A: Read the whole video (bad for memory)
            # Option B: Read chunks using PTS (if timestamps are reliable)
            # Option C: Iterate and read frame by frame (slow, but simple if PTS fails)
            # Let's try a simplified approach: read the whole video and select indices.
            # This is NOT memory-efficient for large videos but simplest to implement with torchvision.
            # Consider libraries like PyAV or decord for more efficient frame access if this is too slow/memory intensive.
            try:
                #print(f"Loading {len(local_indices_needed)} frames from {video_path}")
                # Note: pts_unit='sec' is often needed. Output is usually TCHW uint8 [0, 255]
                vframes, _, _ = vio.read_video(video_path, pts_unit='sec') 
                
                # Select the specific frames needed from this video
                selected_vframes = vframes[local_indices_needed]
                
                # Store loaded frames mapped by their original global index
                for i, global_idx in enumerate(global_indices_in_file):
                    loaded_frames_map[global_idx] = selected_vframes[i]

            except Exception as e:
                print(f"Error loading frames from {video_path}: {e}")
                # Handle error: potentially fill with dummy frames for missing ones?
                # Mark corresponding global indices as missing
                for global_idx in global_indices_in_file:
                     loaded_frames_map[global_idx] = None # Mark as None or provide a dummy

        # Assemble the final tensor in the order of the original global_indices
        frames_list = [] # Re-initialize frames_list here
        expected_shape_chw = None # To store C, H, W for dummy frames
        
        for global_idx in global_indices.tolist():
            frame = loaded_frames_map.get(global_idx)
            processed_frame = None # Use a temporary variable
            
            if frame is not None:
                # Process the loaded frame (likely HWC)
                if frame.ndim == 3: 
                    # Permute HWC -> CHW then unsqueeze -> TCHW
                    processed_frame = frame.permute(2, 0, 1).unsqueeze(0) 
                elif frame.ndim == 4 and frame.shape[0] == 1 and frame.shape[-1] == 3: # Check for THWC (T=1)
                    processed_frame = frame.permute(0, 3, 1, 2) # Permute THWC -> TCHW
                elif frame.ndim == 4 and frame.shape[1] == 3: # Assume already TCHW
                     processed_frame = frame
                # Else: Frame has unexpected dimensions
                
                # Validate processed frame shape (T=1, C=3, H, W)
                if processed_frame is not None and (processed_frame.ndim != 4 or processed_frame.shape[0] != 1 or processed_frame.shape[1] != 3):
                    # print(f"Warning: Frame for global index {global_idx} has unexpected shape {processed_frame.shape} after processing. Skipping.")
                    processed_frame = None # Mark as invalid
                elif processed_frame is not None and expected_shape_chw is None:
                     # Store shape C, H, W from the first valid frame
                     expected_shape_chw = processed_frame.shape[1:] 

            # If frame was None initially or processing failed, use dummy
            if processed_frame is None:
                # Create a dummy frame (TCHW)
                if expected_shape_chw is None: # If no valid frames found yet, use default
                     # Use a reasonable default, perhaps from config if possible? Or hardcode.
                     h_dummy, w_dummy = 224, 224 # Fallback default size
                     # print(f"Warning: Using default dummy frame size ({h_dummy}x{w_dummy}) for global index {global_idx}.")
                     dummy_shape_tchw = (1, 3, h_dummy, w_dummy) 
                else:
                     dummy_shape_tchw = (1,) + expected_shape_chw # Shape (1, C, H, W)
                     
                # print(f"Warning: Using dummy frame ({dummy_shape_tchw}) for global index {global_idx}.")
                processed_frame = torch.zeros(dummy_shape_tchw, dtype=torch.uint8) 
            
            # Append the processed (or dummy) frame
            frames_list.append(processed_frame)

        # Concatenate all loaded frames (should all be TCHW with T=1)
        try:
            final_frames_tchw = torch.cat(frames_list, dim=0)
        except RuntimeError as e:
             print("Error during final concatenation. Dumping shapes in frames_list:")
             for i, t in enumerate(frames_list):
                  print(f"  Frame {i}: {t.shape if isinstance(t, torch.Tensor) else type(t)}")
             raise e # Re-raise the error after printing debug info
        except Exception as e:
             print(f"Unexpected error during final concatenation: {e}")
             raise e
            
        return final_frames_tchw # Should be T, C, H, W uint8 [0, 255]

    def __len__(self):
        # Return the number of valid starting points
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # Map the index to a valid starting frame in the original video timeline
        original_start_frame = self.valid_indices[idx]
        # Calculate end frame based on the *input* sequence length needed
        original_end_frame = original_start_frame + self.effective_input_seq_len_original

        # --- Select Frame Indices for Loading (for input_frames) ---
        # Indices relative to the *original* video frame rate
        # Need self.input_frames number of indices after skipping
        frame_indices = torch.arange(original_start_frame, original_end_frame, self.frame_skip)
        # Ensure we don't exceed the desired number of input frames due to rounding/skipping
        if len(frame_indices) > self.input_frames:
             frame_indices = frame_indices[:self.input_frames]
        elif len(frame_indices) < self.input_frames:
             # This might happen near the end; could pad indices or rely on _get_frames padding
             # print(f"Warning: Got {len(frame_indices)} frame indices, expected {self.input_frames}. Check calculation or end of data.")
             # Pad indices (less ideal) or let loading handle it?
             # Let's proceed, _get_frames should handle loading fewer if needed.
             pass 

        # --- Load Video Frames ---
        video_frames = self._get_frames(frame_indices) # Expected output: T, C, H, W uint8 [0, 255]
        left_right = random.randint(0, 1)
        video_frames = video_frames[:, :, :, (video_frames.shape[3] // 2) * left_right : (video_frames.shape[3] // 2) * (left_right + 1)]

        # If _get_frames returns an empty or incorrect shape tensor, handle it
        if video_frames.shape[0] != len(frame_indices) or video_frames.ndim != 4:
             # print(f"Warning: _get_frames returned unexpected shape {video_frames.shape} for {len(frame_indices)} indices. Using placeholder.")
             # Fallback to placeholder if loading failed critically
             dummy_shape = (self.output_frames, 3, 224, 224) # T, C, H, W
             video_frames = torch.zeros(dummy_shape, dtype=torch.uint8)

        # --- Get Corresponding Time Series Data ---
        # Calculate time range corresponding to the selected *input* video frames
        start_time = original_start_frame / self.original_fps
        # End time corresponds to the *start* of the frame *after* the last input frame needed
        end_time = (original_start_frame + self.effective_input_seq_len_original) / self.original_fps
        
        # Find HR indices corresponding to this time range
        start_hr_idx = max(0, int(round(start_time * self.hr_sampling_rate)))
        end_hr_idx = min(self.total_hr_points, int(round(end_time * self.hr_sampling_rate)))
        
        # Select HR data and validity for this time range
        hr_segment = self.heart_rate[start_hr_idx:end_hr_idx]
        hr_validity_segment = self.hr_validity[start_hr_idx:end_hr_idx]
        
        # Resample the HR segment and its validity to match the number of *input* video frames (`self.input_frames`)
        # Use linear interpolation for resampling.
        if hr_segment.shape[0] > 1: # Need at least 2 points to interpolate
            # Resample HR data
            hr_sequence = F.interpolate(
                hr_segment.unsqueeze(0).unsqueeze(0), # Add Batch and Channel dims: [1, 1, L_in]
                size=self.input_frames, # Resample to match input frame count
                mode='linear',
                align_corners=False # Usually False for signal resampling
            ).squeeze(0).squeeze(0) # Remove Batch and Channel dims: [L_out]
            
            # Resample HR validity (treat as float 0.0 or 1.0, then threshold)
            hr_validity_sequence_float = F.interpolate(
                hr_validity_segment.float().unsqueeze(0).unsqueeze(0),
                size=self.input_frames, # Resample to match input frame count
                mode='linear',
                align_corners=False
            ).squeeze(0).squeeze(0)
            # Threshold back to boolean (consider > 0.5 as valid)
            hr_validity_sequence = hr_validity_sequence_float > 0.5 
        elif hr_segment.shape[0] == 1: # Only one point, just repeat
            hr_sequence = hr_segment.repeat(self.input_frames)
            hr_validity_sequence = hr_validity_segment.repeat(self.input_frames)
        else: # No HR points found for this segment, fill with zeros/False
            # print(f"Warning: No HR data points found for video frame range {original_start_frame}-{original_start_frame + self.effective_input_seq_len_original}. Returning zeros.")
            hr_sequence = torch.zeros(self.input_frames)
            hr_validity_sequence = torch.zeros(self.input_frames, dtype=torch.bool)
            
        # --- Get Corresponding Breath Data ---
        # Breath signal has the same sampling rate as video originally
        start_breath_idx = original_start_frame
        end_breath_idx = original_start_frame + self.effective_input_seq_len_original
        
        # Ensure indices are within bounds
        start_breath_idx = max(0, start_breath_idx)
        end_breath_idx = min(self.total_breath_points, end_breath_idx)

        breath_segment = self.breath_signal[start_breath_idx:end_breath_idx]
        breath_validity_segment = self.breath_validity[start_breath_idx:end_breath_idx]
        
        # Resample breath segment and validity to match input_frames
        if breath_segment.shape[0] > 1:
             breath_sequence = F.interpolate(
                 breath_segment.unsqueeze(0).unsqueeze(0), 
                 size=self.input_frames, 
                 mode='linear',
                 align_corners=False
             ).squeeze(0).squeeze(0)
             
             breath_validity_sequence_float = F.interpolate(
                 breath_validity_segment.float().unsqueeze(0).unsqueeze(0),
                 size=self.input_frames,
                 mode='linear',
                 align_corners=False
             ).squeeze(0).squeeze(0)
             breath_validity_sequence = breath_validity_sequence_float > 0.5
        elif breath_segment.shape[0] == 1:
             breath_sequence = breath_segment.repeat(self.input_frames)
             breath_validity_sequence = breath_validity_segment.repeat(self.input_frames)
        else:
             # print(f"Warning: No breath data points found for video frame range {start_breath_idx}-{end_breath_idx}. Returning zeros.")
             breath_sequence = torch.zeros(self.input_frames)
             breath_validity_sequence = torch.zeros(self.input_frames, dtype=torch.bool)
             
        # Double-check length after resampling
        if breath_sequence.shape[0] != self.input_frames:
             breath_sequence = torch.zeros(self.input_frames)
             # print(f"Warning: Breath sequence shape mismatch after resampling ({breath_sequence.shape[0]} vs {self.input_frames})")
        if breath_validity_sequence.shape[0] != self.input_frames:
             breath_validity_sequence = torch.zeros(self.input_frames, dtype=torch.bool)
             # print(f"Warning: Breath validity shape mismatch after resampling ({breath_validity_sequence.shape[0]} vs {self.input_frames})")

        # --- Apply Transforms ---
        # Normalize video frames to [0, 1] float for transforms
        # Video frames tensor has shape [input_frames, C, H, W] here
        video_frames = video_frames.float() / 255.0
        
        # Apply Affine Transform (to video only)
        video_frames = self.affine_transform(video_frames)

        # Apply Speed Transform (to video and time series)
        # Input sequences have length self.input_frames
        hr_sequence_in = hr_sequence 
        breath_sequence_in = breath_sequence 

        # Apply speed transform. Note: time series validity needs careful handling if transformed.
        # SpeedTransform will output sequences of length self.output_frames (self.out_len)
        video_frames, hr_sequence_out, breath_sequence_out = self.speed_transform(
            video_frames, hr_sequence_in, breath_sequence_in
        )

        # Output validity flags correspond to the sequence *before* speed transform,
        # but should match the final output length (self.output_frames).
        # We need to select the appropriate part of hr_validity_sequence 
        # OR resample it again after speed transform (complex). 
        # Let's keep the simpler approach: return validity corresponding to the *input* sequence
        # length, truncated/padded to the *output* length. This is an approximation.
        hr_validity_out = hr_validity_sequence[:self.output_frames]
        breath_validity_out = breath_validity_sequence[:self.output_frames]

        # --- Prepare Output --- 
        # video_frames is currently float [0, 1], TCHW after transforms
        # hr_sequence_out, breath_sequence_out are float, length self.output_frames
        # *_validity_out are bool, length self.output_frames

        return {
            "frames": video_frames,             # (output_frames, C, H, W), float [0, ~1]
            "heart_rate": hr_sequence_out[1:] - hr_sequence_out[:-1],      # (output_frames,)
            "breathe": breath_sequence_out[1:] - breath_sequence_out[:-1],      # (output_frames,) - Now real data
            "no_use_heart_rate": hr_validity_out[1:], # (output_frames,) - True if should NOT be used
            "no_use_breathe": breath_validity_out[1:]   # (output_frames,) - True if should NOT be used
        }

# Example Usage (requires implementing video loading logic)
if __name__ == "__main__":
    print("Attempting to initialize dataset...")
    try:
        dataset = RatDataset(
            data_root="../data/processed",
            session_id="20250329_150340",
            target_fps=30,
            input_frames=220, # Request 220 frames at 30 FPS
            output_frames=150 # Desired 150 frames at 30 FPS
        )
        print(f"Dataset length: {len(dataset)}")

        if len(dataset) > 10:
            if os.path.isdir("dataset/test"):
                shutil.rmtree("dataset/test")
            os.makedirs("dataset/test")

            for i in range(10):
                item = dataset[i]
                print("Sample item keys:", item.keys())
                print("Frames shape:", item["frames"].shape)
                print("Heart rate shape:", item["heart_rate"].shape)
                print("Breath shape:", item["breath"].shape)
                print("No use HR shape:", item["no_use_heart_rate"].shape)
                print("No use Breath shape:", item["no_use_breathe"].shape)
                print("Sample no_use_heart_rate:", item["no_use_heart_rate"]) # Show full tensor

                # save frames to mp4 video
                video_path = os.path.join("dataset/test", f"{i}.mp4")
                # Convert float [0, 1] back to uint8 [0, 255] for saving
                frames_to_save = (item["frames"] * 255.0).byte()
                # Ensure THWC format for write_video
                vio.write_video(video_path, frames_to_save.permute(0, 2, 3, 1), dataset.target_fps)
                
                # Save heart rate and breath plot
                hr_data = item["heart_rate"].cpu().detach().numpy()
                breath_data = item["breath"].cpu().detach().numpy()
                time_axis = np.arange(len(hr_data))
                
                plt.figure(figsize=(10, 4))
                plt.plot(time_axis, hr_data, label='Heart Rate (post-transform)')
                plt.plot(time_axis, breath_data, label='Breath (post-transform)', linestyle='--')
                plt.title(f"Sample {i} - Heart Rate and Breath Data")
                plt.xlabel("Time Steps (post-transform)")
                plt.ylabel("Value")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plot_path = os.path.join("dataset/test", f"plot_{i}.png")
                plt.savefig(plot_path)
                plt.close() # Close the figure to free memory

    except FileNotFoundError as e:
        print(f"Error initializing dataset: {e}")
    except ValueError as e:
        print(f"Error initializing dataset: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}") 