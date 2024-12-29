import asyncio
from datetime import datetime
import json
import os
import time
from typing import Optional, Union

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, FastAPI
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from routes import camera, heart_rate
from routes.heart_rate import sample_rate_index

router = APIRouter(prefix="/record")

RECORD_DIR = "data/records"


is_recording = False
duration = None
recording_start = None
recording_title = None

status_update_clients = []
records_update_clients = []


async def push_status_update():
    update = json.dumps({
        "is_recording": is_recording,
        "recording_title": recording_title,
        "start_at": recording_start
    })
    for queue in status_update_clients:
        await queue.put(update)


async def push_records_update():
    update = json.dumps([item for item in os.listdir(RECORD_DIR) if item != ".DS_Store"])
    for queue in records_update_clients:
        await queue.put(update)


@router.get("/records-realtime", response_class=StreamingResponse)
async def records_realtime():
    client_queue = asyncio.Queue()
    records_update_clients.append(client_queue)

    async def event_stream():
        try:
            while True:
                update = await client_queue.get()
                yield f"data: {update}\n\n"
        finally:
            records_update_clients.remove(client_queue)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/status-realtime", response_class=StreamingResponse)
async def status_realtime():
    client_queue = asyncio.Queue()
    status_update_clients.append(client_queue)

    async def event_stream():
        try:
            while True:
                update = await client_queue.get()
                yield f"data: {update}\n\n"
        finally:
            status_update_clients.remove(client_queue)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/status")
def api_get_recording_status():
    global is_recording, recording_start, duration, recording_title
    return {
        "code": "OK",
        "message": "Recording status retrieved successfully.",
        "data": {
            "is_recording": is_recording,
            "recording_title": recording_title,
            "start_at": recording_start
        }
    }

@router.get("/records")
def api_get_records():
    return {
        "code": "OK",
        "message": "Records retrieved successfully.",
        "data": [item for item in os.listdir(RECORD_DIR) if item != ".DS_Store"]
    }


async def stop_recording_after_duration():
    global is_recording, duration
    try:
        await asyncio.sleep(duration)
        if is_recording:
            await api_stop_recording()
            await push_records_update()
    except Exception as e:
        print(f"Error in stop_recording_after_duration: {e}")


class StartRecordRequest(BaseModel):
    duration: int

@router.post("/start")
async def api_start_recording(request: StartRecordRequest):
    global is_recording, recording_start, duration, recording_title
    if is_recording:
        return {"code": "ERROR", "message": "Already recording."}
    try:
        out_dir = f"{RECORD_DIR}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(out_dir, exist_ok=True)

        camera_config = camera.get_camera_config()
        sensor_config = heart_rate.get_sensor_config()

        with open(f"{out_dir}/config.json", "w") as f:
            json.dump({
                "duration": request.duration,
                "light": camera_config["light"],
                "camera": {
                    "width": camera_config["width"],
                    "height": camera_config["height"],
                    "rate": camera_config["rate"],
                },
                "heart-rate-sensor": {
                    "sample_rate": sensor_config["sample_rate"],
                }
            }, f, indent=2)

        await asyncio.gather(
            camera.start_recording(out_dir + "/video.mp4"),
            heart_rate.start_recording(out_dir + "/heart_rate.txt")
        )
        recording_start = datetime.now().strftime("%M:%S")
        is_recording = True
        duration = request.duration + .5
        if request.duration != -1:
            recording_title = f"({request.duration} seconds)"
            asyncio.create_task(stop_recording_after_duration())

        await push_status_update()

        return {"code": "OK", "message": "Recording started successfully."}
    except Exception as e:
        return {"code": "ERROR", "message": str(e)}


@router.post("/stop")
async def api_stop_recording():
    global is_recording, recording_start, recording_title, duration
    if not is_recording:
        return {"code": "ERROR", "message": "Not recording."}
    try:
        await asyncio.gather(
            camera.stop_recording(),
            heart_rate.stop_recording()
        )
        is_recording = False
        duration = None
        recording_start = None
        recording_title = None

        await push_status_update()
        await push_records_update()
        return {"code": "OK", "message": "Recording stopped successfully."}
    except Exception as e:
        return {"code": "ERROR", "message": str(e)}


def setup(app: FastAPI, config: dict):
    app.include_router(router)
