import collections
import json
import os
import shutil
import tempfile

import cv2
import numpy as np
import asyncio

from dns.e164 import query
from fastapi import FastAPI, APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse
from starlette.requests import Request

app = FastAPI()
router = APIRouter(prefix="/breathe")

RECORD_DIR = "data/records"


def create_temp_copy(video_path):
    temp_dir = tempfile.gettempdir()
    temp_video_path = os.path.join(temp_dir, os.path.basename(video_path))
    shutil.copy(video_path, temp_video_path)
    return temp_video_path


# Video streaming endpoint
@router.get("/video-stream/{video_id}")
async def video_stream(video_id: str, request: Request):
    video_path = f"data/records/{video_id}/video.mp4"  # Path to the video file

    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video not found")

    range_header = request.headers.get("range")
    file_size = os.path.getsize(video_path)

    def generate_video():
        video_temp = create_temp_copy(video_path)
        with open(video_temp, "rb") as video:
            if range_header:
                # Parse range header (e.g., "bytes=1234-")
                range_start = int(range_header.replace("bytes=", "").split("-")[0])
                video.seek(range_start)
            while True:
                data = video.read(1024 * 1024)  # Stream 1 MB chunks
                if not data:
                    break
                yield data

    # Handle range requests
    if range_header:
        range_start = int(range_header.replace("bytes=", "").split("-")[0])
        headers = {
            "Content-Range": f"bytes {range_start}-{file_size - 1}/{file_size}",
            "Accept-Ranges": "bytes",
        }
        return StreamingResponse(generate_video(), status_code=206, headers=headers)
    else:
        # Serve the entire file if no range header is present
        headers = {"Accept-Ranges": "bytes"}
        return StreamingResponse(generate_video(), headers=headers)


@router.websocket("/square-rgb-steam/{record_id}")
async def f(websocket: WebSocket, record_id: str):
    await websocket.accept()
    video_path = os.path.join(RECORD_DIR, record_id, "video.mp4")
    config_path = os.path.join(RECORD_DIR, record_id, "config.json")
    if not os.path.exists(video_path):
        await websocket.send_json({"error": "File not found"})
        await websocket.close()
        return
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            await websocket.send_json({"error": "Cannot open video file"})
            await websocket.close()
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = None
        start = 0
        last_size = 0

        while True:
            data = await websocket.receive_json()
            current_time = data.get("current_time", 0)
            frame_size = data.get("frame_size", 5)
            rect_start = data.get("rect_start", [0, 0])
            rect_end = data.get("rect_end", [1, 1])

            if frames is None:
                frames = collections.deque(maxlen=int(fps * frame_size))

            current_start = current_time - frame_size

            if len(frames) == 0 or last_size != frame_size or frames[0]["time"] < current_start or frames[-1]["time"] - current_time > frame_size:
                frames.clear()
                frames = collections.deque(maxlen=int(fps * frame_size))
                last_size = frame_size
                start = current_start

                cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
                with open(config_path, "r") as f:
                    config = json.load(f)
                    config["rectangle"] = {
                        "start": rect_start,
                        "end": rect_end
                    }
                    with open(config_path, "w") as f:
                        json.dump(config, f, indent=2)

            while len(frames) == 0 or frames[-1]["time"] < current_time:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame[
                    int(rect_start[1] * frame.shape[0]):int(rect_end[1] * frame.shape[0]),
                    int(rect_start[0] * frame.shape[1]):int(rect_end[0] * frame.shape[1])
                ]
                mean_pixel = np.mean(frame, axis=(0, 1))
                frames.append({"time": cap.get(cv2.CAP_PROP_POS_MSEC) / 1000, "pixel": mean_pixel})
            if frames:
                all_r = [frame["pixel"][0] for frame in frames]
                all_g = [frame["pixel"][1] for frame in frames]
                all_b = [frame["pixel"][2] for frame in frames]
                all_mean = [(r + g + b) / 3 for r, g, b in zip(all_r, all_g, all_b)]

                min_r, max_r = min(all_r), max(all_r)
                min_g, max_g = min(all_g), max(all_g)
                min_b, max_b = min(all_b), max(all_b)
                min_all, max_all = min(all_mean), max(all_mean)

                d_r = 1 if max_r - min_r == 0 else max_r - min_r
                d_g = 1 if max_g - min_g == 0 else max_g - min_g
                d_b = 1 if max_b - min_b == 0 else max_b - min_b
                d_all = 1 if max_all - min_all == 0 else max_all - min_all

                all_r = [(r - min_r) / (d_r) for r in all_r]
                all_g = [(g - min_g) / (d_g) for g in all_g]
                all_b = [(b - min_b) / (d_b) for b in all_b]
                all_mean = [(m - min_all) / (d_all) for m in all_mean]

                time = [int(frame["time"] * fps) for frame in frames]

                output = {
                    "time": time[::10],
                    "mean_r": all_r[::10],
                    "mean_g": all_g[::10],
                    "mean_b": all_b[::10],
                    "mean_all": all_mean[::10]
                }
                await websocket.send_json(output)

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    finally:
        cap.release()


def setup(app: FastAPI, config: dict):
    app.include_router(router)

