import json
import os
from collections import deque
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, FastAPI, HTTPException, Query, Body
from pydantic import BaseModel, Field
import asyncio
import time
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from typing import List, Dict, Any, Literal

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


# --- Pydantic Models ---
class Annotation(BaseModel):
    time: float
    value: float
    type: Literal['peak', 'valley']

class AnnotationsPayload(BaseModel):
    annotations: List[Annotation]

class AutoAnnotatePayload(BaseModel):
    frame_start_time: float = Field(..., description="Start time of the frame for auto-annotation")
    frame_size: float = Field(..., description="Duration of the frame for auto-annotation")

# --- Helper Functions ---
def get_record_path(record_id: str):
    """Constructs path to record directory and checks existence."""
    path = os.path.join(RECORD_PATH, record_id)
    if not os.path.isdir(path):
        raise HTTPException(status_code=404, detail=f"Record '{record_id}' not found")
    return path

def read_config(record_id: str) -> Dict[str, Any]:
    """Reads and parses the config.json file for a record."""
    record_path = get_record_path(record_id)
    config_path = os.path.join(record_path, "config.json")
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Config file not found for record '{record_id}'")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail=f"Error reading config file for record '{record_id}'")

def read_heart_rate_raw(record_id: str) -> np.ndarray:
    """Reads the raw heart_rate.txt data."""
    record_path = get_record_path(record_id)
    hr_file = os.path.join(record_path, "heart_rate.txt")
    try:
        # Assuming one float value per line
        data = np.loadtxt(hr_file)
        if data.ndim == 0: # Handle case where file might have only one number
            data = np.array([data.item()])
        elif data.ndim > 1:
             raise HTTPException(status_code=500, detail="heart_rate.txt should contain one value per line")
        return data
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"heart_rate.txt not found for record '{record_id}'")
    except ValueError:
         raise HTTPException(status_code=500, detail="Error reading heart_rate.txt: Contains non-numeric data")
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Error reading heart_rate.txt: {e}")

# --- API Endpoints ---

@router.get("/{record_id}/config", include_in_schema=True)
async def get_record_config_alternative(record_id: str):
    """Alternative endpoint to get the config for a specific record.
    This matches the client's expected URL pattern.
    """
    try:
        config = read_config(record_id)
        return {"code": "OK", "message": f"Config retrieved for record {record_id}", "data": config}
    except HTTPException as he:
        # Re-raise HTTP exceptions from read_config
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving config: {str(e)}")

@router.get("/{record_id}/config")
async def get_record_config_endpoint(record_id: str):
    """Endpoint to get the config for a specific record."""
    try:
        config = read_config(record_id)
        return {"code": "OK", "message": f"Config retrieved for record {record_id}", "data": config}
    except HTTPException as he:
        # Re-raise HTTP exceptions from read_config
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving config: {str(e)}")

@router.get("/{record_id}/interpolated", include_in_schema=True)
async def get_heart_rate_interpolated_alternative(
    record_id: str,
    frame_start_time: float = Query(..., description="Start time of the frame in seconds"),
    frame_size: float = Query(..., gt=0, description="Duration of the frame in seconds (must be positive)")
):
    """Alternative endpoint to get interpolated heart rate data.
    This matches the client's expected URL pattern.
    """
    return await get_heart_rate_interpolated(record_id, frame_start_time, frame_size)

@router.get("/{record_id}/annotations", include_in_schema=True)
async def get_heart_rate_annotations_alternative(record_id: str):
    """Alternative endpoint to get heart rate annotations.
    This matches the client's expected URL pattern.
    """
    return await get_heart_rate_annotations(record_id)

@router.post("/{record_id}/annotations", include_in_schema=True)
async def save_heart_rate_annotations_alternative(record_id: str, payload: AnnotationsPayload):
    """Alternative endpoint to save heart rate annotations.
    This matches the client's expected URL pattern.
    """
    return await save_heart_rate_annotations(record_id, payload)

@router.post("/{record_id}/auto-annotate", include_in_schema=True)
async def auto_annotate_heart_rate_alternative(record_id: str, payload: AutoAnnotatePayload = Body(...)):
    """Alternative endpoint for auto-annotating heart rate data.
    This matches the client's expected URL pattern.
    """
    return await auto_annotate_heart_rate(record_id, payload)

@router.get("/{record_id}/annotations")
async def get_heart_rate_annotations(record_id: str):
    """Gets saved heart rate annotations for a record."""
    record_path = get_record_path(record_id)
    annot_path = os.path.join(record_path, "hr_annotations.json")

    if not os.path.exists(annot_path):
        return {"annotations": []} # Return empty list if file doesn't exist

    try:
        with open(annot_path, 'r') as f:
            data = json.load(f)
            # Basic validation
            if isinstance(data, dict) and isinstance(data.get("annotations"), list):
                # Further validation could check structure of each annotation dict
                 validated_annotations = []
                 for item in data["annotations"]:
                      try:
                           # Use Pydantic model for validation
                           validated_annotations.append(Annotation(**item))
                      except Exception:
                           # Skip invalid items, maybe log a warning
                           pass
                 return {"annotations": [a.dict() for a in validated_annotations]}
            else:
                # Log warning: Invalid format
                return {"annotations": []}
    except (json.JSONDecodeError, Exception) as e:
         # Log error e
         raise HTTPException(status_code=500, detail=f"Error reading annotations file: {e}")

@router.post("/{record_id}/annotations")
async def save_heart_rate_annotations(record_id: str, payload: AnnotationsPayload):
    """Saves heart rate annotations for a record, overwriting existing file."""
    record_path = get_record_path(record_id)
    annot_path = os.path.join(record_path, "hr_annotations.json")
    try:
        # Ensure parent directory exists (though get_record_path should handle record dir)
        os.makedirs(os.path.dirname(annot_path), exist_ok=True)
        with open(annot_path, 'w') as f:
            # Use Pydantic's .dict() for serialization
            json.dump(payload.dict(), f, indent=2)
        return {"status": "success", "message": f"Annotations saved for {record_id}"}
    except Exception as e:
        # Log error e
        raise HTTPException(status_code=500, detail=f"Failed to save annotations: {e}")

@router.post("/{record_id}/auto-annotate")
async def auto_annotate_heart_rate(record_id: str, payload: AutoAnnotatePayload = Body(...)):
    """Automatically detects peaks and valleys in a heart rate data frame."""
    config = read_config(record_id)
    hr_config = config.get("heart-rate-sensor")
    if not hr_config or "sample_rate" not in hr_config:
        raise HTTPException(status_code=500, detail="Heart rate sample rate not found in config")

    try:
        sample_rate = float(hr_config["sample_rate"])
        if sample_rate <= 0:
             raise ValueError("Sample rate must be positive")
    except (ValueError, TypeError):
        raise HTTPException(status_code=500, detail="Invalid sample rate in config")

    raw_data = read_heart_rate_raw(record_id)
    total_samples = len(raw_data)
    if total_samples < 2:
        return {"annotations": []}

    start_sample = max(0, int(payload.frame_start_time * sample_rate))
    end_sample = min(total_samples, int((payload.frame_start_time + payload.frame_size) * sample_rate))

    if start_sample >= end_sample -1: # Need at least 2 points for find_peaks
         return {"annotations": []}

    frame_data = raw_data[start_sample:end_sample]
    frame_indices = np.arange(start_sample, end_sample)

    # --- Peak/Valley Detection Parameters (ADJUST AS NEEDED) ---
    # distance: Minimum horizontal distance (in samples) between peaks.
    # height: Minimum height of peaks.
    # prominence: Required prominence of peaks.
    min_peak_distance_sec = 0.3 # Minimum seconds between heartbeats (adjust based on expected HR)
    min_peak_distance_samples = int(min_peak_distance_sec * sample_rate)

    # Consider normalizing data or setting height relative to frame mean/std
    # height_threshold = np.mean(frame_data) + np.std(frame_data) # Example height threshold

    # --- Find Peaks and Valleys ---
    try:
        peaks, _ = find_peaks(frame_data, distance=min_peak_distance_samples) # Add other params like height, prominence
        # Find valleys by inverting data
        valleys, _ = find_peaks(-frame_data, distance=min_peak_distance_samples) # Add other params
    except Exception as e:
        # Log error during peak finding
        raise HTTPException(status_code=500, detail=f"Error during peak detection: {e}")


    annotations: List[Annotation] = []
    for p_idx in peaks:
        abs_idx = frame_indices[p_idx]
        time = abs_idx / sample_rate
        value = float(frame_data[p_idx])
        annotations.append(Annotation(time=time, value=value, type="peak"))

    for v_idx in valleys:
         abs_idx = frame_indices[v_idx]
         time = abs_idx / sample_rate
         value = float(frame_data[v_idx])
         annotations.append(Annotation(time=time, value=value, type="valley"))

    # Sort by time
    annotations.sort(key=lambda a: a.time)

    return {"annotations": [a.dict() for a in annotations]}

# Original endpoint for interpolated data
async def get_heart_rate_interpolated(
    record_id: str,
    frame_start_time: float,
    frame_size: float
):
    """Gets interpolated heart rate data for a specific time frame."""
    config = read_config(record_id)
    hr_config = config.get("heart-rate-sensor")
    if not hr_config or "sample_rate" not in hr_config:
        raise HTTPException(status_code=500, detail="Heart rate sample rate not found in config")

    try:
        sample_rate = float(hr_config["sample_rate"])
        if sample_rate <= 0:
            raise ValueError("Sample rate must be positive")
    except (ValueError, TypeError):
        raise HTTPException(status_code=500, detail="Invalid sample rate in config")

    raw_data = read_heart_rate_raw(record_id)
    total_samples = len(raw_data)
    if total_samples == 0:
        return {"time": [], "value": []}

    # Calculate sample indices for the frame
    start_sample = int(frame_start_time * sample_rate)
    end_sample = int((frame_start_time + frame_size) * sample_rate)

    # Clamp indices to valid range
    start_sample = max(0, start_sample)
    end_sample = min(total_samples, end_sample)

    if start_sample >= end_sample:
         return {"time": [], "value": []} # No data in range or invalid range

    frame_data = raw_data[start_sample:end_sample]
    frame_time_raw = np.arange(start_sample, end_sample) / sample_rate

    # Interpolate to a fixed number of points (e.g., 500) for consistent plotting
    num_interp_points = 500
    interp_time = np.linspace(max(frame_start_time, frame_time_raw[0] if len(frame_time_raw)>0 else frame_start_time),
                              min(frame_start_time + frame_size, frame_time_raw[-1] if len(frame_time_raw)>0 else frame_start_time + frame_size),
                              num_interp_points)

    if len(frame_data) > 1:
        interp_func = interp1d(frame_time_raw, frame_data, kind='linear', bounds_error=False, fill_value=np.nan)
        interp_value = interp_func(interp_time)
    elif len(frame_data) == 1:
         # Interpolate single point across the requested time range
         interp_value = np.full(num_interp_points, frame_data[0])
    else:
        interp_value = np.full(num_interp_points, np.nan)

    # Replace NaNs from interpolation or fill_value with None for JSON compatibility
    interp_value_json = [v if not np.isnan(v) else None for v in interp_value]

    return {"time": interp_time.tolist(), "value": interp_value_json}