import asyncio
import json
import os.path
import platform
import subprocess
import time
from datetime import datetime

import cv2
from fastapi import APIRouter, FastAPI
from threading import Thread
from pydantic import BaseModel
from starlette.responses import StreamingResponse

RECORD_PATH = "data/records"
FRAME = [{
    "width": 3200,
    "height": 1200,
    "rate": [60]
}, {
    "width": 2560,
    "height": 720,
    "rate": [60]
}, {
    "width": 1600,
    "height": 600,
    "rate": [120, 60]
}, {
    "width": 1280,
    "height": 480,
    "rate": [120, 60]
}, {
    "width": 640,
    "height": 240,
    "rate": [120, 60]
}]
LIGHT = [
    "2500k",
    "3000k",
    "3500k",
    "4000k",
    "4500k",
    "5000k",
    "5500k",
    "RED"
]

camera_devices = None
camera_index = 0

camera_frame_index = 2
camera_frame_rate_index = 0

light_index = 0

capture = None
latest_frame = None
is_running = False
is_recording = False
video_writer = None

config_update_clients = []


def get_camera_config():
    global light_index, camera_frame_rate_index, camera_frame_index
    return {
        "light": LIGHT[light_index],
        "width": FRAME[camera_frame_index]["width"],
        "height": FRAME[camera_frame_index]["height"],
        "rate": FRAME[camera_frame_index]["rate"][camera_frame_rate_index],
    }


def start_camera():
    global capture, latest_frame, is_running, camera_index
    print("Starting camera...", camera_index)
    capture = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME[camera_frame_index]["width"])
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME[camera_frame_index]["height"])
    capture.set(cv2.CAP_PROP_FPS, FRAME[camera_frame_index]["rate"][camera_frame_rate_index])

    if not capture.isOpened():
        raise RuntimeError("Camera could not be opened.")

    print("Camera OK.")
    is_running = True
    while is_running:
        ret, frame = capture.read()
        if ret:
            rotated_frame = cv2.rotate(frame, cv2.ROTATE_180)
            latest_frame = cv2.resize(rotated_frame, (800, 300))
            if is_recording and video_writer:
                video_writer.write(rotated_frame)  # Write the original rotated frame
        else:
            latest_frame = None


def stop_camera():
    global is_running, capture, is_recording, video_writer
    print("Stopping camera...")
    is_running = False
    is_recording = False
    time.sleep(.5)
    if video_writer:
        video_writer.release()
        video_writer = None
    if capture:
        capture.release()


async def start_recording(output_path):
    global is_recording, video_writer
    if not is_running:
        raise RuntimeError("Camera is not running. Cannot start recording.")

    if is_recording:
        raise RuntimeError("Already recording.")

    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # You can change the codec if needed
    width = FRAME[camera_frame_index]["width"]
    height = FRAME[camera_frame_index]["height"]
    fps = FRAME[camera_frame_index]["rate"][camera_frame_rate_index]

    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    is_recording = True
    print(f"Camera started recording: {output_path}")


async def stop_recording():
    global is_recording, video_writer
    if not is_recording:
        raise RuntimeError("Camera not currently recording.")

    is_recording = False
    await asyncio.sleep(.5)
    if video_writer:
        video_writer.release()
        video_writer = None

    print("Camera recording stopped.")


async def push_config_update():
    update = json.dumps({
        "CAMERAS": camera_devices,
        "FRAMES": FRAME,
        "LIGHTS": LIGHT,
        "camera_index": camera_index,
        "camera_frame_index": camera_frame_index,
        "camera_frame_rate_index": camera_frame_rate_index,
        "light_index": light_index
    })
    for queue in config_update_clients:
        await queue.put(update)


def refresh_devices():
    global camera_devices, camera_index
    try:
        if platform.system() == "Darwin":
            result = subprocess.run(
                ["system_profiler", "SPCameraDataType"],
                capture_output=True,
                text=True,
            )
            output = result.stdout.strip()
            devices = []
            lines = output.split("\n")
            for line in lines:
                line = line.strip()
                if len(line) > 0 and line[-1] == ":":
                    devices.append(line[:-1])

            camera_devices = devices[1:]
            camera_devices.sort()
            camera_index = 0

            return {
                "code": "OK",
                "message": "Camera list retrieved successfully.",
                "data": camera_devices,
            }
        else:
            return {
                "code": "ERROR",
                "message": f"Not implemented platform: {platform.system()}",
            }
    except Exception as e:
        return {
            "code": "ERROR",
            "message": f"Failed to retrieve camera list: {str(e)}",
        }


router = APIRouter(prefix="/camera")


@router.get("/config")
def get_config():
    global camera_index, camera_frame_index, camera_frame_rate_index, camera_devices, light_index
    return {
        "code": "OK",
        "message": "Camera configuration retrieved successfully.",
        "data": {
            "CAMERAS": camera_devices,
            "FRAMES": FRAME,
            "LIGHTS": LIGHT,
            "camera_index": camera_index,
            "camera_frame_index": camera_frame_index,
            "camera_frame_rate_index": camera_frame_rate_index,
            "light_index": light_index
        },
    }

@router.get("/config-realtime", response_class=StreamingResponse)
async def config_realtime():
    client_queue = asyncio.Queue()
    config_update_clients.append(client_queue)

    async def event_stream():
        try:
            while True:
                update = await client_queue.get()
                yield f"data: {update}\n\n"
        finally:
            config_update_clients.remove(client_queue)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


class UpdateRequest(BaseModel):
    camera_index: int
    camera_frame_index: int
    camera_frame_rate_index: int
    light_index: int


@router.post("/update")
async def update(update_request: UpdateRequest):
    global camera_index, camera_frame_index, camera_frame_rate_index, light_index
    camera_index = update_request.camera_index
    camera_frame_index = update_request.camera_frame_index
    camera_frame_rate_index = update_request.camera_frame_rate_index
    light_index = update_request.light_index

    stop_camera()
    Thread(target=start_camera, daemon=True).start()

    await push_config_update()

    return {
        "code": "OK",
        "message": "Camera configuration updated successfully.",
    }


@router.get("/refresh_devices")
async def api_refresh_devices():
    output = refresh_devices()
    await push_config_update()
    return output


async def generate_frames():
    global latest_frame
    frame_interval = 1 / 30  # Interval for 30 FPS
    while True:
        start_time = asyncio.get_event_loop().time()
        if latest_frame is not None:
            _, buffer = cv2.imencode('.jpg', latest_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        # Enforce 30 FPS
        elapsed_time = asyncio.get_event_loop().time() - start_time
        await asyncio.sleep(max(0, frame_interval - elapsed_time))


@router.get("/stream", response_class=StreamingResponse)
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@router.post("/record/start")
async def api_start_recording():
    try:
        filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".avi"
        filepath = os.path.join(RECORD_PATH, filename)
        await start_recording(filepath)
        return {"code": "OK", "message": "Camera recording started successfully."}
    except Exception as e:
        return {"code": "ERROR", "message": str(e)}


@router.post("/record/stop")
async def api_stop_recording():
    try:
        await stop_recording()
        return {"code": "OK", "message": "Camera recording stopped successfully."}
    except Exception as e:
        return {"code": "ERROR", "message": str(e)}


def startup(config):
    if "record" not in config or not config["record"]:
        return
    if not os.path.isdir(RECORD_PATH):
        os.makedirs(RECORD_PATH)
    refresh_devices()
    Thread(target=start_camera, daemon=True).start()


def shutdown():
    stop_camera()


def setup(app: FastAPI, config: dict):
    if "record" not in config or not config["record"]:
        return
    app.include_router(router)