import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import cv2
import torchvision.transforms as transforms
from train.train_rat import LightningRatModel as RatEstimator3DCNN
import json
from scipy.interpolate import interp1d
import torchvision.io as vio
import torch.nn.functional as F
from datetime import datetime

RECORD_PATH = "data/processed"
RECORD_ID = "20250329_150340"
CKPT_PATH = "train/saves/3d-cnn-gan/3d-cnn-gan-epoch=35-val_loss=0.13443.ckpt"
TARGET_FPS = 30
PLOT_OUTPUT_DIR = "plot_outputs"  # Directory for storing output plots and videos

def load_config_and_annotations(session_path):
    config_path = os.path.join(session_path, "config.json")
    hr_annotations_path = os.path.join(session_path, "hr_annotations.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in {session_path}")
    if not os.path.exists(hr_annotations_path):
        raise FileNotFoundError(f"hr_annotations.json not found in {session_path}")
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    with open(hr_annotations_path, 'r') as f:
        hr_annotations_data = json.load(f)
    return config_data, hr_annotations_data

def load_and_resample_video(session_path, target_fps, config_data):
    video_files = sorted([os.path.join(session_path, f) for f in os.listdir(session_path) if f.endswith('.mp4')])
    if not video_files:
        print(f"Warning: No MP4 files found in {session_path}. Video generator will be empty.")
        return

    # print(f"Note: 'vio.read_video' loads each entire video file. Memory issues might still occur if a single video file is excessively large.")

    for video_path in video_files:
        current_frames_tensor_orig = None
        info = {}
        try:
            vframes, _, info = vio.read_video(video_path, pts_unit='sec', output_format="TCHW")
            current_frames_tensor_orig = vframes
        except RuntimeError:
            try:
                vframes, _, info = vio.read_video(video_path, pts_unit='sec') 
                current_frames_tensor_orig = torch.from_numpy(vframes).permute(0, 3, 1, 2)
            except Exception as e_default:
                print(f"Warning: Error reading video {video_path}: {e_default}. Skipping.")
                yield torch.empty((0,3,0,0), dtype=torch.uint8), None, 0
                continue
        except Exception as e: 
            print(f"Warning: General error reading video {video_path}: {e}. Skipping.")
            yield torch.empty((0,3,0,0), dtype=torch.uint8), None, 0
            continue
        
        if current_frames_tensor_orig is None or current_frames_tensor_orig.shape[0] == 0:
            # print(f"Warning: Video file {video_path} is empty or could not be loaded. Skipping.")
            yield torch.empty((0, current_frames_tensor_orig.shape[1] if current_frames_tensor_orig is not None and current_frames_tensor_orig.ndim==4 else 3 ,0,0), dtype=torch.uint8), None, 0
            continue

        num_original_frames_in_file = current_frames_tensor_orig.shape[0]
        current_file_original_fps_val = info.get('video_fps')
        current_file_original_fps = None
        if current_file_original_fps_val is not None:
            try:
                current_file_original_fps = float(current_file_original_fps_val)
                if current_file_original_fps <= 0: current_file_original_fps = None 
            except ValueError: pass
        
        if current_file_original_fps is None:
            config_fps_str = config_data.get("video_fps")
            if config_fps_str is not None:
                try:
                    current_file_original_fps = float(config_fps_str)
                    if current_file_original_fps <= 0: current_file_original_fps = None
                except ValueError: pass
            if current_file_original_fps is None: 
                current_file_original_fps = float(target_fps)

        processed_frames_for_yield = current_frames_tensor_orig
        if abs(current_file_original_fps - target_fps) > 1e-2: 
            if current_file_original_fps > target_fps:
                frame_skip = round(current_file_original_fps / target_fps)
                if frame_skip < 1: frame_skip = 1
                processed_frames_for_yield = current_frames_tensor_orig[::int(frame_skip)]
            # else: Upsampling not handled by frame skipping, using all frames
        
        # if processed_frames_for_yield.shape[0] == 0 and num_original_frames_in_file > 0:
            # print(f"Warning: After processing, no frames from {video_path} to yield...")
        
        yield processed_frames_for_yield, current_file_original_fps, num_original_frames_in_file

def get_video_metadata_for_signals(session_path, target_fps, config_data):
    # print("Gathering video metadata for signal generation (Pass 1)...")
    video_generator_metadata = load_and_resample_video(session_path, target_fps, config_data)
    total_resampled_frames_count = 0
    accumulated_total_original_frames = 0
    session_determined_original_fps = None
    first_valid_fps_found = False

    for i, (frame_chunk_uint8, chunk_original_fps, chunk_num_original_frames) in enumerate(video_generator_metadata):
        if frame_chunk_uint8 is not None: 
            total_resampled_frames_count += frame_chunk_uint8.shape[0]
        accumulated_total_original_frames += chunk_num_original_frames
        if not first_valid_fps_found and chunk_original_fps is not None and chunk_original_fps > 0:
            session_determined_original_fps = chunk_original_fps
            first_valid_fps_found = True
    
    if session_determined_original_fps is None or session_determined_original_fps <= 0:
        config_fps_val = config_data.get("video_fps")
        if config_fps_val is not None: 
            try: session_determined_original_fps = float(config_fps_val)
            except ValueError: pass
        if session_determined_original_fps is None or session_determined_original_fps <= 0:             
            session_determined_original_fps = float(target_fps)
            print(f"Warning: Session original FPS defaulted to target FPS {session_determined_original_fps}.")

    video_total_duration_seconds = 0
    if target_fps > 0 and total_resampled_frames_count > 0:
        video_total_duration_seconds = total_resampled_frames_count / target_fps
    
    # print(f"Metadata collection complete: ")
    # print(f"  Total original frames: {accumulated_total_original_frames}, Session original FPS: {session_determined_original_fps}")
    # print(f"  Total resampled frames at ~{target_fps} FPS: {total_resampled_frames_count}, Estimated duration: {video_total_duration_seconds:.2f}s")
    return total_resampled_frames_count, session_determined_original_fps, accumulated_total_original_frames, video_total_duration_seconds

def generate_hr_signal(hr_annotations_data, config_data, video_total_duration_seconds, num_frames_in_final_video):
    hr_times_raw, hr_values_raw = [], []
    annotation_list = hr_annotations_data.get('annotations', []) if isinstance(hr_annotations_data, dict) else hr_annotations_data if isinstance(hr_annotations_data, list) else []
    for item in annotation_list:
        try:
            ann = json.loads(item) if isinstance(item, str) else item
            if 'time' in ann and 'type' in ann and ann['type'] in ['peak', 'valley']:
                hr_times_raw.append(float(ann['time']))
                hr_values_raw.append(1.0 if ann['type'] == 'peak' else 0.0)
        except (json.JSONDecodeError, TypeError, ValueError): pass
    if not hr_times_raw: return torch.zeros(num_frames_in_final_video)
    unique_times, unique_indices = np.unique(np.array(hr_times_raw)[np.argsort(hr_times_raw)], return_index=True)
    unique_values = np.array(hr_values_raw)[np.argsort(hr_times_raw)][unique_indices]
    if len(unique_times) < 2: return torch.full((num_frames_in_final_video,), unique_values[0] if len(unique_times) == 1 else 0.0, dtype=torch.float32)
    hr_sampling_rate = float(config_data.get('train-heart-rate', {}).get('sampling_rate_hz', 120.0))
    num_hr_points_high_res = int(video_total_duration_seconds * hr_sampling_rate)
    if num_hr_points_high_res <= 0: return torch.zeros(num_frames_in_final_video)
    times_high_res = np.linspace(0, video_total_duration_seconds, num_hr_points_high_res, endpoint=False)
    interp_func = interp1d(unique_times, unique_values, kind='cubic', bounds_error=False, fill_value=(unique_values[0], unique_values[-1]))
    hr_signal_high_res = torch.from_numpy(interp_func(times_high_res)).float()
    if num_frames_in_final_video == 0: return torch.empty(0)
    return F.interpolate(hr_signal_high_res.unsqueeze(0).unsqueeze(0), size=num_frames_in_final_video, mode='linear', align_corners=False).squeeze()

def generate_breath_signal(config_data, total_original_video_frames, original_video_fps, num_frames_in_final_video):
    if total_original_video_frames == 0 or original_video_fps == 0: return torch.zeros(num_frames_in_final_video)
    breath_config, breath_times_raw, breath_values_raw = config_data.get('train-breathe', {}), [], []
    for frame_idx in breath_config.get('peaks', []): breath_times_raw.append(float(frame_idx)); breath_values_raw.append(1.0)
    for frame_idx in breath_config.get('valleys', []): breath_times_raw.append(float(frame_idx)); breath_values_raw.append(0.0)
    if not breath_times_raw: return torch.zeros(num_frames_in_final_video)
    unique_times, unique_indices = np.unique(np.array(breath_times_raw)[np.argsort(breath_times_raw)], return_index=True)
    unique_values = np.array(breath_values_raw)[np.argsort(breath_times_raw)][unique_indices]
    if len(unique_times) < 2: return torch.full((num_frames_in_final_video,), unique_values[0] if len(unique_times) == 1 else 0.0, dtype=torch.float32)
    unique_times = np.clip(unique_times, 0, total_original_video_frames - 1)
    sorted_indices_clipped = np.argsort(unique_times); unique_times = unique_times[sorted_indices_clipped]; unique_values = unique_values[sorted_indices_clipped]
    unique_times, unique_indices_clipped_final = np.unique(unique_times, return_index=True); unique_values = unique_values[unique_indices_clipped_final]
    if len(unique_times) < 2: return torch.full((num_frames_in_final_video,), unique_values[0] if len(unique_times) == 1 else 0.0, dtype=torch.float32)
    interp_func = interp1d(unique_times, unique_values, kind='cubic', bounds_error=False, fill_value=(unique_values[0], unique_values[-1]))
    breath_signal_orig_fps = torch.from_numpy(interp_func(np.arange(total_original_video_frames))).float()
    if num_frames_in_final_video == 0: return torch.empty(0)
    return F.interpolate(breath_signal_orig_fps.unsqueeze(0).unsqueeze(0), size=num_frames_in_final_video, mode='linear', align_corners=False).squeeze()

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    session_path = os.path.join(RECORD_PATH, RECORD_ID)
    print(f"Processing session: {session_path}")
    config_data, hr_annotations_data = load_config_and_annotations(session_path)

    # Create output directory if it doesn't exist
    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video_path = os.path.join(PLOT_OUTPUT_DIR, f"signal_animation_{timestamp}.mp4")
    
    total_final_frames, session_original_fps, total_original_frames_sum, video_duration_for_signals = \
        get_video_metadata_for_signals(session_path, TARGET_FPS, config_data)
    print(f"Video metadata: Total resampled frames = {total_final_frames}, Original FPS = {session_original_fps:.2f}, Duration for signals = {video_duration_for_signals:.2f}s")

    print("Generating full HR and breath signals...")
    full_hr_signal_30fps = torch.empty(0)
    if total_final_frames > 0 and video_duration_for_signals > 0:
        full_hr_signal_30fps = generate_hr_signal(hr_annotations_data, config_data, video_duration_for_signals, total_final_frames)
    print(f"Full HR signal: {full_hr_signal_30fps.shape}")
    full_breath_signal_30fps = torch.empty(0)
    if total_final_frames > 0 and total_original_frames_sum > 0 and session_original_fps is not None and session_original_fps > 0:
        full_breath_signal_30fps = generate_breath_signal(config_data, total_original_frames_sum, session_original_fps, total_final_frames)
    print(f"Full breath signal: {full_breath_signal_30fps.shape}")

    print("Loading 3D CNN model...")
    lightning_module = None
    try:
        lightning_module = RatEstimator3DCNN.load_from_checkpoint(CKPT_PATH)
        lightning_module.eval()
        if torch.cuda.is_available():
            lightning_module.cuda()
            print("Model loaded to CUDA.")
        else:
            print("Model loaded to CPU.")
    except Exception as e:
        print(f"Error loading model: {e}. Proceeding without model.")

    print("\nProcessing video chunks and signals (Pass 2)..." )
    video_generator_main = load_and_resample_video(session_path, TARGET_FPS, config_data)
    
    last_frame = None  # Store the last frame from previous chunk for continuity
    
    # Initialize containers for continuous data
    all_frames = []  # To store actual video frames for reference
    all_b_hat = []   # Predicted breath signals
    all_hr_hat = []  # Predicted heart rate signals
    all_b_true = []  # Ground truth breath signals
    all_hr_true = [] # Ground truth heart rate signals
    time_axis = []   # For keeping track of time/frame indices
    frame_count = 0  # Count of frames processed so far
    
    # Setup figure for plotting and video creation
    fig = plt.figure(figsize=(12, 6), dpi=100)
    
    # Start frame index for cumulative viewing
    vis_start_idx = 0  # Window start index for visualization
    window_size = 300  # Show 10 seconds of data at 30fps in the visualization
    
    for i, (frame_chunk_uint8, chunk_original_fps, _) in enumerate(video_generator_main):
        if frame_chunk_uint8 is None or frame_chunk_uint8.nelement() == 0:
            continue

        # Process frames with continuity from previous chunk
        frames = frame_chunk_uint8[..., :800]
        frames = frames.float() / 255.0
        
        # If we have a last frame from previous chunk, prepend it to the current frames
        if last_frame is not None:
            frames = torch.cat([last_frame.unsqueeze(0), frames], dim=0)
        
        # Keep last frame for next iteration
        last_frame = frames[-1].clone()
        
        # Store original frames for reference/visualization
        for f in range(frames.shape[0]):
            # Store downsampled version to save memory
            frame_rgb = frames[f].permute(1, 2, 0).cpu().numpy()  # Convert to HWC format
            # Resize for memory efficiency
            frame_small = cv2.resize(frame_rgb, (320, 240))
            all_frames.append(frame_small)
        
        num_frames_in_chunk = frames.shape[0]
        
        # Get corresponding signal segments
        segment_start = i * 150
        segment_end = segment_start + 150
        hr = full_hr_signal_30fps[segment_start:segment_end]
        b = full_breath_signal_30fps[segment_start:segment_end]  # Original breath signal
        
        if hr.shape[0] > 1:
            hr = hr[1:] - hr[:-1]  # Compute the difference for heart rate
        if b.shape[0] > 1:
            b = b[1:] - b[:-1]  # Compute the difference for breath signal

        # Run inference
        with torch.no_grad():
            try:
                y_hat = lightning_module(frames.unsqueeze(0).to(lightning_module.device))
                b_hat = y_hat[0, 0].cpu().detach().numpy()  # Raw breath signal from model
                hr_hat = y_hat[0, 1].cpu().detach().numpy()  # Heart rate signal from model
                
                # Store results for continuous plotting
                all_b_hat.extend(b_hat)
                all_hr_hat.extend(hr_hat)
                
                # Store ground truth signals, handling potential length differences
                pad_length = len(b_hat)
                if len(b) < pad_length:
                    # Pad with zeros if ground truth is shorter than model output
                    b_padded = np.zeros(pad_length)
                    b_padded[:len(b)] = b.cpu().numpy()
                    all_b_true.extend(b_padded)
                else:
                    # Truncate if ground truth is longer
                    all_b_true.extend(b.cpu().numpy()[:pad_length])
                
                if len(hr) < pad_length:
                    hr_padded = np.zeros(pad_length)
                    hr_padded[:len(hr)] = hr.cpu().numpy()
                    all_hr_true.extend(hr_padded)
                else:
                    all_hr_true.extend(hr.cpu().numpy()[:pad_length])
                
                # Update time axis
                time_axis.extend(range(frame_count, frame_count + len(b_hat)))
                frame_count += len(b_hat)
                
                # Create individual plots for this chunk (optional)
                if i % 5 == 0:  # Save fewer images to avoid clutter
                    plt.clf()  # Clear the figure
                    
                    # Breath signal subplot
                    plt.subplot(2, 1, 1)
                    plt.plot(b_hat, label="Model Breath", color='blue')
                    plt.plot(b.cpu().numpy()[:min(len(b), len(b_hat))], label="Ground Truth", color='orange')
                    plt.title(f"Breath Signal (Chunk {i})")
                    plt.xlabel("Frame")
                    plt.ylabel("Amplitude")
                    plt.legend()
                    
                    # Heart rate signal subplot
                    plt.subplot(2, 1, 2)
                    plt.plot(hr_hat, label="Model Heart Rate", color='green')
                    plt.plot(hr.cpu().numpy()[:min(len(hr), len(hr_hat))], label="Ground Truth", color='red')
                    plt.title("Heart Rate Signal")
                    plt.xlabel("Frame")
                    plt.ylabel("Amplitude")
                    plt.legend()
                    
                    plt.tight_layout()
                    
                    # Save the figure as an image
                    plt.savefig(os.path.join(PLOT_OUTPUT_DIR, f"chunk_{i:04d}.png"))
                    
                # Update visualization window for continuous plotting
                if len(all_b_hat) > window_size:
                    vis_start_idx = len(all_b_hat) - window_size
                
            except Exception as e:
                print(f"Error during inference on chunk {i}: {e}")
                continue
        
        # Print progress
        if i % 10 == 0:
            print(f"Processed chunk {i}: Total frames collected: {len(all_b_hat)}")

    print("\nFinished processing all video chunks. Creating continuous animation...")
    
    # Generate comprehensive continuous animation
    if len(all_b_hat) > 0:
        print("Creating animation video...")
        
        # Video parameters
        fps = 30  # Output FPS
        width, height = 1200, 600  # Output size
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        # Function to create a plot frame at specific index
        def create_plot_frame(idx, window_size=300):
            # Define window: We want to show a fixed window size that moves with the data
            start_idx = max(0, idx - window_size//2)
            end_idx = min(len(all_b_hat), start_idx + window_size)
            
            # Adjust start_idx to maintain window size if we're near the end
            if end_idx - start_idx < window_size and start_idx > 0:
                start_idx = max(0, end_idx - window_size)
            
            # Create plot
            fig = plt.figure(figsize=(12, 6), dpi=100)
            
            # Breath signal subplot
            ax1 = plt.subplot(2, 1, 1)
            ax1.plot(time_axis[start_idx:end_idx], all_b_hat[start_idx:end_idx], label="Model", color='blue')
            ax1.plot(time_axis[start_idx:end_idx], all_b_true[start_idx:end_idx], label="Ground Truth", color='orange')
            
            # Highlight current point
            if start_idx <= idx < end_idx:
                ax1.axvline(x=time_axis[idx], color='red', linestyle='--', alpha=0.7)
                # Mark the current point with a point marker
                ax1.plot(time_axis[idx], all_b_hat[idx], 'ro', markersize=5)
            
            ax1.set_title("Breath Signal")
            ax1.set_xlabel("Frame")
            ax1.set_ylabel("Rate of Change")
            ax1.legend(loc='upper left')
            
            # Heart rate signal subplot
            ax2 = plt.subplot(2, 1, 2)
            ax2.plot(time_axis[start_idx:end_idx], all_hr_hat[start_idx:end_idx], label="Model", color='green')
            ax2.plot(time_axis[start_idx:end_idx], all_hr_true[start_idx:end_idx], label="Ground Truth", color='red')
            
            # Highlight current point
            if start_idx <= idx < end_idx:
                ax2.axvline(x=time_axis[idx], color='red', linestyle='--', alpha=0.7)
                # Mark the current point with a point marker
                ax2.plot(time_axis[idx], all_hr_hat[idx], 'ro', markersize=5)
            
            ax2.set_title("Heart Rate Signal")
            ax2.set_xlabel("Frame")
            ax2.set_ylabel("Rate of Change")
            ax2.legend(loc='upper left')
            
            plt.tight_layout()
            
            # Convert plot to image - using the modern approach that works across matplotlib versions
            canvas = FigureCanvas(fig)
            canvas.draw()
            
            # Get the RGBA buffer from the figure
            buf = fig.canvas.buffer_rgba()
            # Convert to numpy array - modern, compatible approach
            plot_img = np.asarray(buf)
            
            plt.close(fig)
            
            # Resize to desired dimensions - convert to RGB if CV2 expects it
            plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGBA2BGR)
            plot_img = cv2.resize(plot_img, (width, height))
            
            return plot_img
        
        # Create animation frames - show actual progress through the data
        stride = max(1, len(all_b_hat) // (60 * fps))  # Aim for around 1 minute of video
        
        for j in range(0, len(all_b_hat), stride):
            plot_img = create_plot_frame(j)
            video_writer.write(plot_img)
            
            # Show progress
            if j % (stride * 10) == 0:
                print(f"Creating video: {j}/{len(all_b_hat)} frames ({j/len(all_b_hat)*100:.1f}%)")
        
        video_writer.release()
        print(f"Animation video saved to: {output_video_path}")
    
    print("Processing complete!")

