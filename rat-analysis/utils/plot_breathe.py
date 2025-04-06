import json
import os
from os import remove

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.signal import find_peaks
from torchvision import transforms

RECORD_PATH = "../data/records"
RECORD_ID = "20241123_121043"

PLOT = True


def smooth(data, alpha):
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]  # Initialize the first value
    for i in range(1, len(data)):
        smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i-1]
    return smoothed


def fix_peaks_and_valleys(peaks, valleys):
    # get each distance between peaks
    distances = np.concatenate([np.diff(peaks), np.diff(valleys)])
    distance = (distances - distances.mean()) / distances.std()

    add_peaks = []
    add_valleys = []

    # remove_peaks = []
    # remove_valleys = []

    # last = 0
    for i in range(0, len(peaks) -1):
        if distance[i] > 3:
            add_peaks.append((peaks[i] + peaks[i+1]) // 2)

        # if distance[i] - last < -1.5:
        #     if distances[i] > distances[i-1]:
        #         remove_peaks.append(peaks[i])
        #     else:
        #         remove_peaks.append(peaks[i + 1])
        #     last = distance[i]
        # else:
        #     last = 0

    # last = 0
    for i in range(0, len(valleys) -1):
        if distance[len(peaks) - 1 + i] > 3:
            add_valleys.append((valleys[i] + valleys[i+1]) // 2)

        # if distance[len(peaks) - 1 + i] - last < -1.5:
        #     if distances[len(peaks) - 1 + i] > distances[len(peaks) - 1]:
        #         remove_valleys.append(valleys[i])
        #     else:
        #         remove_valleys.append(valleys[i + 1])
        #     last = distance[len(peaks) - 1 + i]
        # else:
        #     last = 0

    return np.array(add_peaks, dtype=int), np.array(add_valleys, dtype=int)
    # return np.array(remove_peaks, dtype=int), np.array(remove_valleys, dtype=int)


if __name__ == "__main__":
    print("Starting the process...")

    video_path = os.path.join(RECORD_PATH, RECORD_ID, "video.mp4")
    config = os.path.join(RECORD_PATH, RECORD_ID, "config.json")
    out_path = os.path.join(RECORD_PATH, RECORD_ID, "rect_rgb_mean.npy")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 448)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if not os.path.isfile(out_path):
        print("Loading configuration...")
        with open(config, "r") as f:
            config = json.load(f)
            rect_start = config["rectangle"]["start"]
            rect_end = config["rectangle"]["end"]

        print("Opening video file...")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Cannot open video file")
            exit(1)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        rect_rgb_mean = []
        frames = []

        print("Processing video frames...")
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            print(f"\rProcessing frame {frame_count} / {total_frames}...", end="")
            frames.append(transform(frame))
            frame = frame[
                int(rect_start[1] * frame.shape[0]):int(rect_end[1] * frame.shape[0]),
                int(rect_start[0] * frame.shape[1]):int(rect_end[0] * frame.shape[1])
            ]
            mean_pixel = np.mean(frame, axis=(0, 1))
            rect_rgb_mean.append(mean_pixel)

        cap.release()
        print("\nFinished processing video frames, saving...")

        frames = np.stack(frames)
        np.save(os.path.join(RECORD_PATH, RECORD_ID, "frames.npy"), frames)

        print("Standardizing the data...")
        rect_rgb_mean = np.array(rect_rgb_mean)

        np.save(out_path, rect_rgb_mean)

    else:
        print("Loading the data...")
        rect_rgb_mean = np.load(out_path)

    mean = rect_rgb_mean.mean(axis=0)
    std = rect_rgb_mean.std(axis=0)
    rect_rgb_mean = (rect_rgb_mean - rect_rgb_mean.mean(axis=0)) / rect_rgb_mean.std(axis=0)


    r = rect_rgb_mean[:, 0]
    g = rect_rgb_mean[:, 1]
    b = rect_rgb_mean[:, 2]

    r_smoothed = smooth(r, alpha=0.2)

    # Find peaks and valleys
    peaks, _ = find_peaks(r_smoothed, distance=30, prominence=0.2)
    valleys, _ = find_peaks(-r_smoothed, distance=30, prominence=0.2)

    # Refine peaks and valleys
    # refined_peaks, refined_valleys = fix_peaks_and_valleys(peaks, valleys)

    r_binary = np.zeros_like(r_smoothed)
    r_binary[peaks] = 1
    r_binary[valleys] = 0
    # r_binary[refined_peaks] = 1
    # r_binary[refined_valleys] = 0

    # points = np.sort(np.concatenate([peaks, valleys, refined_peaks, refined_valleys]))
    points = np.sort(np.concatenate([peaks, valleys]))
    values = r_binary[points]

    # interpolator = PchipInterpolator(points, values, extrapolate=False)
    interpolator = interp1d(points, values, kind="cubic", fill_value=0, bounds_error=False)
    # interpolator = interp1d(points, values, kind="linear", fill_value=0, bounds_error=False)

    # Generate smooth signal
    r_smoothed = np.array(interpolator(np.arange(r_smoothed.shape[0])))
    r_smoothed[np.isnan(r_smoothed)] = 0

    np.save(os.path.join(RECORD_PATH, RECORD_ID, "r_smoothed.npy"), r_smoothed)

    if PLOT:
        for i in range(0, r_smoothed.shape[0] // 1000 + 1):
            if i == r_smoothed.shape[0] // 1000:
                plot_position = (i * 1000, r_smoothed.shape[0])
            else:
                plot_position = (i * 1000, (i + 1) * 1000)
            # plot_position = (6500, 7500)

            _peaks = peaks[(peaks >= plot_position[0]) & (peaks <= plot_position[1])]
            _valleys = valleys[(valleys >= plot_position[0]) & (valleys <= plot_position[1])]

            # Plot the results
            plt.figure(figsize=(21, 5))
            plt.plot(r_smoothed[plot_position[0]: plot_position[1]], label="Smoothed R", color="red")
            plt.scatter(_peaks - plot_position[0], r_smoothed[_peaks], color="blue", label="Original Peaks", zorder=5)
            plt.scatter(_valleys - plot_position[0], r_smoothed[_valleys], color="green", label="Original Valleys", zorder=5)

            for p in _peaks:
                plt.text(p - plot_position[0], r_smoothed[p] + 0.05, str(p), fontsize=8, color="blue", ha="center", va="bottom")
            for v in _valleys:
                plt.text(v - plot_position[0], r_smoothed[v] - 0.05, str(v), fontsize=8, color="green", ha="center", va="top")

            plt.axhline(0, color="black", linestyle="--")
            plt.legend()
            plt.title("Red Channel with Refined Peaks and Valleys")
            plt.xlabel("Frame")
            plt.ylabel("Standardized Value")
            plt.show()
            print()
    else:
        pass


#
