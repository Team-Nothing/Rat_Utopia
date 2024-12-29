import json
import os
from collections import deque
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, FastAPI
from pydantic import BaseModel
import asyncio
import time

from starlette.responses import StreamingResponse


router = APIRouter(prefix="/heart-rate")

RECORD_PATH = "data/records"
SAMPLE_RATE = [150, 100, 50, 25, 10, 5, 1]
FRAME_SIZE = [0.5, 1, 2, 5, 8, 10]
FRAME_DELAY = [0.02, 0.05, 0.1, 0.2, 0.5, 1]
FRAME_REDUCE = [0, 1, 2, 3, 4]

sample_rate_index = 1
frame_size_index = 2
frame_delay_index = 2
frame_reduce_index = 4

current_client = None
device_latency = None
connected_device_id = "ESP32_HEARTRATE_PRESSURE_SENSOR"

sensor_data_cache = deque(maxlen=SAMPLE_RATE[sample_rate_index] * FRAME_SIZE[frame_size_index] // (FRAME_REDUCE[frame_reduce_index] + 1))

ping_start = 0
config_update_clients = []

is_recording = False
file_out = None


def get_sensor_config():
    global sample_rate_index
    return {
        "sample_rate": SAMPLE_RATE[sample_rate_index],
    }


async def start_recording(output_path):
    global is_recording, file_out, current_client
    if current_client is None:
        return

    if is_recording:
        return

    file_out = open(output_path, "w")

    is_recording = True
    print(f"Heart rate started recording: {output_path}")


async def stop_recording():
    global is_recording, file_out
    if not is_recording:
        return
    await asyncio.sleep(.5)
    is_recording = False
    await asyncio.sleep(.5)
    if file_out:
        file_out.close()
        file_out = None

    print("Heart rate recording stopped.")


async def push_config_update():
    update = json.dumps({
        "SAMPLE_RATES": SAMPLE_RATE,
        "FRAME_SIZES": FRAME_SIZE,
        "FRAME_DELAYS": FRAME_DELAY,
        "FRAME_REDUCES": FRAME_REDUCE,
        "sample_rate_index": sample_rate_index,
        "frame_size_index": frame_size_index,
        "frame_delay_index": frame_delay_index,
        "frame_reduce_index": frame_reduce_index,
    })
    for queue in config_update_clients:
        await queue.put(update)


class HeartRateConfig(BaseModel):
    sample_rate_index: int
    frame_size_index: int
    frame_delay_index: int
    frame_reduce_index: int


@router.post("/update")
async def update(rate: HeartRateConfig):
    global sample_rate_index, current_client, sensor_data_cache, frame_delay_index, frame_reduce_index, frame_size_index
    sample_rate_index = rate.sample_rate_index
    frame_size_index = rate.frame_size_index
    frame_delay_index = rate.frame_delay_index
    frame_reduce_index = rate.frame_reduce_index

    sensor_data_cache = deque(maxlen=SAMPLE_RATE[sample_rate_index] * FRAME_SIZE[frame_size_index]//(FRAME_REDUCE[frame_reduce_index] + 1))

    await push_config_update()

    if current_client is not None:
        await current_client.send_json({"sample_rate": SAMPLE_RATE[sample_rate_index]})

    return {
        "code": "OK",
        "message": "Sample rate updated.",
        "sample_rate": SAMPLE_RATE[sample_rate_index],
    }


@router.get("/config")
def get_config():
    return {
        "code": "OK",
        "message": "Heart Rate Sensor configuration retrieved successfully.",
        "data": {
            "SAMPLE_RATES": SAMPLE_RATE,
            "FRAME_SIZES": FRAME_SIZE,
            "FRAME_DELAYS": FRAME_DELAY,
            "FRAME_REDUCES": FRAME_REDUCE,
            "sample_rate_index": sample_rate_index,
            "frame_size_index": frame_size_index,
            "frame_delay_index": frame_delay_index,
            "frame_reduce_index": frame_reduce_index,
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


@router.get("/latency")
def get_latency():
    global device_latency
    return {
        "code": "OK",
        "message": "Device latency retrieved successfully.",
        "data": device_latency,
    }


@router.get("/stream", response_class=StreamingResponse)
async def stream_cache():
    async def event_stream():
        while True:
            if len(sensor_data_cache) > 0:
                sanitized_data = json.dumps(list(sensor_data_cache))
                yield f"data: {sanitized_data}\n\n"
            await asyncio.sleep(FRAME_DELAY[frame_delay_index])
    return StreamingResponse(event_stream(), media_type="text/event-stream")


class SensorData(BaseModel):
    t: float
    p: float
    T: int


class Identity(BaseModel):
    device_id: str


@router.websocket("/device-connect")
async def websocket_endpoint(websocket: WebSocket):
    global sample_rate_index, current_client, connected_device_id, device_latency, ping_start, is_recording, file_out

    await websocket.accept()

    try:
        init_message = await websocket.receive_json()
        identity = Identity(**init_message)

        if identity.device_id != connected_device_id:
            await websocket.send_json({"code": "ERROR", "message": "Unauthorized device_id"})
            await websocket.close()
            return

        if current_client is not None:
            await current_client.close()
            print(f"Replaced previous connection with new connection for device_id: {identity.device_id}")

        current_client = websocket

        await websocket.send_json(
            {"sample_rate": SAMPLE_RATE[sample_rate_index], "server_time": asyncio.get_event_loop().time() * 1000000}
        )

        async def ping_device():
            global device_latency, ping_start
            while current_client == websocket:
                ping_start = time.time()
                try:
                    await websocket.send_json({"type": "ping"})
                except asyncio.TimeoutError:
                    device_latency = -1
                await asyncio.sleep(3)

        asyncio.create_task(ping_device())

        s = 0
        while True:
            message = await websocket.receive_json()
            message_type = message.get("type")

            if message_type == "pong":
                device_latency = int((time.time() - ping_start) * 1000)
            if message_type == "sensor_data":
                sensor_data = SensorData(**message).model_dump()
                if s == 0:
                    sensor_data_cache.append(sensor_data)
                    s = FRAME_REDUCE[frame_reduce_index] + 1

                if is_recording and file_out is not None:
                    file_out.write(json.dumps(sensor_data) + "\n")

                s -= 1


    except WebSocketDisconnect:
        print(f"WebSocket disconnected for device_id: {identity.device_id}")
    finally:
        if current_client == websocket:
            current_client = None
            device_latency = None


@router.post("/record/start")
async def api_start_recording():
    try:
        filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt"
        filepath = os.path.join(RECORD_PATH, filename)
        await start_recording(filepath)
        return {"code": "OK", "message": "Heart rate recording started successfully."}
    except Exception as e:
        return {"code": "ERROR", "message": str(e)}


@router.post("/record/stop")
async def api_stop_recording():
    try:
        await stop_recording()
        return {"code": "OK", "message": "Heart rate recording stopped successfully."}
    except Exception as e:
        return {"code": "ERROR", "message": str(e)}


def setup(app: FastAPI, config: dict):
    if "record" not in config or not config["record"]:
        return
    app.include_router(router)