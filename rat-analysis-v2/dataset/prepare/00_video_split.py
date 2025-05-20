import os
import subprocess
import math
import json
import re

# Define paths
RAW_DIR = 'data/raw'
PROCESSED_DIR = 'data/processed'

# Create processed directory if it doesn't exist
os.makedirs(PROCESSED_DIR, exist_ok=True)

def get_video_info(video_path):
    """Get the duration, frame rate, and total frames of a video using ffmpeg."""
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-hide_banner'
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output = result.stderr
        duration_line = [line for line in output.splitlines() if 'Duration' in line]
        fps_line = [line for line in output.splitlines() if ' fps,' in line]
        duration = 0
        fps = 0
        total_frames = 0
        if duration_line:
            duration_str = duration_line[0].split('Duration: ')[1].split(',')[0]
            h, m, s = map(float, duration_str.split(':'))
            duration = h * 3600 + m * 60 + s
        else:
            print(f"Error: Could not retrieve duration for {video_path}.")
        if fps_line:
            fps_str = fps_line[0].split(' fps,')[0].split(' ')[-1]
            fps = float(fps_str)
        else:
            print(f"Error: Could not retrieve frame rate for {video_path}.")
        # Estimate total frames if possible
        if duration > 0 and fps > 0:
            total_frames = int(round(duration * fps))
        # Try to get exact frame count with a more precise method
        cmd_count = [
            'ffmpeg',
            '-i', video_path,
            '-map', 'v:0',
            '-c', 'copy',
            '-f', 'null',
            '-'
        ]
        result_count = subprocess.run(cmd_count, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output_count = result_count.stderr
        match = re.search(r'frame=\s*(\d+)', output_count)
        if match:
            total_frames = int(match.group(1))
        return duration, fps, total_frames
    except FileNotFoundError:
        print("Error: ffmpeg not found. Please ensure FFmpeg is installed and added to PATH.")
        return 0, 0, 0
    except Exception as e:
        print(f"Error retrieving video info for {video_path}: {e}")
        return duration, fps, total_frames

def get_frame_count(video_path):
    """Get the exact frame count of a video segment using ffmpeg."""
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-map', 'v:0',
        '-c', 'copy',
        '-f', 'null',
        '-'
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output = result.stderr
        match = re.search(r'frame=\s*(\d+)', output)
        if match:
            return int(match.group(1))
        return 0
    except Exception as e:
        print(f"Error getting frame count for {video_path}: {e}")
        return 0

def load_original_config(raw_dir, video_name):
    """Load the original config.json from the raw directory if it exists."""
    config_path = os.path.join(raw_dir, video_name, 'config.json')
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading original config from {config_path}: {e}")
            return {}
    return {}

def split_video(video_path, output_dir, target_duration=5):
    """Split a video into segments of target_duration seconds (aiming for 600 frames at 120 FPS), without sound."""
    total_duration, fps, total_frames = get_video_info(video_path)
    if total_duration == 0 or fps == 0:
        print(f"Skipping {video_path} due to duration or frame rate retrieval failure.")
        return
    if total_frames == 0:
        print(f"Warning: Could not determine total frames for {video_path}, estimating based on duration and FPS.")
        total_frames = int(round(total_duration * fps))
    num_segments = math.ceil(total_duration / target_duration)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    target_frames = int(target_duration * fps) if fps > 0 else 600  # Target frames based on FPS or default to 600
    
    print(f"Processing {video_name}: Total duration = {total_duration:.2f} seconds, FPS = {fps:.2f}, Total frames = {total_frames}, Segments = {num_segments}")
    
    # Load original config if available
    raw_dir = os.path.dirname(video_path)
    original_config = load_original_config(raw_dir, video_name)
    
    # Write config.json, including original config details
    config = {
        'original_video': video_path,
        'total_duration_seconds': total_duration,
        'frame_rate_fps': fps,
        'total_frames': total_frames,
        'target_duration_per_segment_seconds': target_duration,
        'target_frames_per_segment': target_frames,
        'number_of_segments': num_segments
    }
    # Merge original config into the new config
    if original_config:
        config.update(original_config)
        print(f"Merged original configuration from {raw_dir}/{video_name}/config.json")
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Wrote configuration to {config_path}")
    
    validation_log = []
    for i in range(num_segments):
        output_file = os.path.join(output_dir, f"video_{i+1:03d}.mp4")
        start_time = i * target_duration
        duration = min(target_duration, total_duration - start_time)
        
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-ss', str(start_time),
            '-t', str(duration),
            '-c:v', 'libx264',
            '-an',  # Disable audio in output
            '-y',  # Overwrite output if exists
            output_file
        ]
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode == 0:
                print(f"Created segment: {output_file} (Start time {start_time:.2f}s, Duration {duration:.2f}s)")
                # Validate frame count
                actual_frames = get_frame_count(output_file)
                if actual_frames > 0:
                    expected_frames = target_frames
                    if abs(actual_frames - expected_frames) > 10:  # Allow small deviation
                        validation_log.append({
                            'segment': output_file,
                            'expected_frames': expected_frames,
                            'actual_frames': actual_frames,
                            'status': 'Invalid - Frame count mismatch'
                        })
                        print(f"  ✗ Invalid: Segment has {actual_frames} frames, expected around {expected_frames}")
                    else:
                        validation_log.append({
                            'segment': output_file,
                            'expected_frames': expected_frames,
                            'actual_frames': actual_frames,
                            'status': 'Valid'
                        })
                else:
                    validation_log.append({
                        'segment': output_file,
                        'expected_frames': target_frames,
                        'actual_frames': 0,
                        'status': 'Error counting frames'
                    })
                    print(f"  ✗ Error: Could not determine frame count for {output_file}")
            else:
                print(f"Error creating segment {output_file}: {result.stderr}")
                validation_log.append({
                    'segment': output_file,
                    'expected_frames': target_frames,
                    'actual_frames': 0,
                    'status': 'Error creating segment'
                })
        except FileNotFoundError:
            print("Error: ffmpeg not found. Please ensure FFmpeg is installed and added to PATH.")
            return
    
    # Write validation log
    validation_path = os.path.join(output_dir, 'validation_log.json')
    with open(validation_path, 'w') as f:
        json.dump(validation_log, f, indent=4)
    print(f"Wrote validation log to {validation_path}")

def main():
    """Process all videos in the raw directory."""
    for root, dirs, files in os.walk(RAW_DIR):
        for file in files:
            if file.endswith('.mp4'):
                video_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, RAW_DIR)
                output_dir = os.path.join(PROCESSED_DIR, relative_path)
                os.makedirs(output_dir, exist_ok=True)
                split_video(video_path, output_dir)

if __name__ == '__main__':
    main()

