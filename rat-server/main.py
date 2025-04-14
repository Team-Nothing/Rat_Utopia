import importlib
import json
import os
from fastapi import FastAPI, Query, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, TypeVar, Generic, Dict, Any
from pydantic import BaseModel
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

# --- Define ApiResponse Pydantic Model --- 
T = TypeVar('T')

class ApiResponse(BaseModel, Generic[T]):
    code: str = "OK"
    message: str = "Success"
    data: T | None = None
# ----------------------------------------

# Import route modules
from routes import breath, heart_rate, record, camera # Added record, camera assuming they exist

description = """
Rat Recorder API helps you do awesome stuff. 🚀
"""

DATA_DIR = "data/records"

with open("service_config.json", "r") as f:
    config = json.load(f)

# Lifecycle function (modified to handle routes modules)
def app_lifespan(app: FastAPI):
    # Include all relevant modules with potential lifecycle hooks
    modules_to_manage = [breath, heart_rate, record, camera]
    try:
        for module in modules_to_manage:
            if hasattr(module, "startup"):
                getattr(module, "startup")(config)

        yield # Application runs here

    finally:
        for module in modules_to_manage:
            if hasattr(module, "shutdown"):
                getattr(module, "shutdown")()


app = FastAPI(
    title="Rat Recorder API",
    description=description,
    summary="API",
    version="0.0.1",
    contact={
        "name": "Nothing Chang",
        "url": "https://github.com/I-am-nothing",
        "email": "jdps99119@gmail.com",
    },
    lifespan=app_lifespan
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Be more specific in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {
        "code": "OK",
        "message": "Medical Record API is running!"
    }

# --- NEW Endpoint for Listing Records ---
@app.get("/records", tags=["Records"], response_model=ApiResponse[List[str]])
async def list_records():
    """Lists available record IDs based on directory names."""
    try:
        if not os.path.isdir(DATA_DIR):
            # Return empty list if data directory doesn't exist
            return ApiResponse[List[str]](code="OK", message="Data directory not found", data=[])
        
        record_ids = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d)) and not d.startswith('.')]
        record_ids.sort(reverse=True) # Sort by name (usually date based), newest first
        return ApiResponse[List[str]](code="OK", message="Records listed successfully", data=record_ids)
    except Exception as e:
        # Log the error e
        # Return an empty list with an error message, or raise HTTPException
         return JSONResponse(
             status_code=500,
             content=ApiResponse[List[str]](code="ERROR", message=f"Failed to list records: {e}", data=[]).dict()
         )

@app.get("/records/{record_id}", tags=["Records"], response_model=ApiResponse[Dict[str, Any]])
async def get_record_by_id(record_id: str):
    """Gets configuration and metadata for a specific record by ID."""
    try:
        record_path = os.path.join(DATA_DIR, record_id)
        if not os.path.isdir(record_path):
            return JSONResponse(
                status_code=404,
                content=ApiResponse[Dict[str, Any]](code="ERROR", message=f"Record {record_id} not found", data=None).dict()
            )
        
        config_path = os.path.join(record_path, "config.json")
        if not os.path.exists(config_path):
            return JSONResponse(
                status_code=404,
                content=ApiResponse[Dict[str, Any]](code="ERROR", message=f"Config not found for record {record_id}", data=None).dict()
            )
            
        # Read the config file
        with open(config_path, 'r') as f:
            config_data = json.load(f)
            
        return ApiResponse[Dict[str, Any]](code="OK", message=f"Record {record_id} data retrieved", data=config_data)
    except Exception as e:
        # Log the error e
        return JSONResponse(
            status_code=500,
            content=ApiResponse[Dict[str, Any]](code="ERROR", message=f"Failed to retrieve record: {e}", data=None).dict()
        )

# Include routers from modules
app.include_router(breath.router, tags=["Breathe"], prefix="/breathe")
app.include_router(heart_rate.router, tags=["Heart Rate"], prefix="/heart-rate")
app.include_router(record.router, tags=["Recording"], prefix="/record")
app.include_router(camera.router, tags=["Camera"], prefix="/camera")

# Setup routes from modules (if they have a setup function)
modules_to_setup = [breath, heart_rate, record, camera]
for module in modules_to_setup:
     if hasattr(module, "setup"):
         getattr(module, "setup")(app, config)

# Note: The dynamic importlib loop below is commented out as we now
# explicitly import and include routers/lifecycles/setups.
# If you prefer dynamic loading, uncomment and adapt the lifespan/setup logic.

# for file_name in os.listdir("routes"):
#     if file_name.endswith(".py") and file_name != "__init__.py":
#         module = importlib.import_module(f"routes.{file_name[:-3]}")
#         # Including router dynamically based on presence
#         if hasattr(module, "router") and hasattr(module, "TAGS") and hasattr(module, "PREFIX"):
#              app.include_router(module.router, tags=module.TAGS, prefix=module.PREFIX)
#         # Setup function call remains the same conceptually
#         if hasattr(module, "setup"):
#             getattr(module, "setup")(app, config)

class Annotation(BaseModel):
    time: float
    value: float
    type: str  # 'peak' or 'valley'

class AnnotationsPayload(BaseModel):
    annotations: List[Annotation]

class AutoAnnotatePayload(BaseModel):
    frame_start_time: float
    frame_size: float

def get_record_path(record_id: str):
    """Constructs path to record directory and checks existence."""
    path = os.path.join(DATA_DIR, record_id)
    if not os.path.isdir(path):
        raise HTTPException(status_code=404, detail=f"Record '{record_id}' not found")
    return path

def read_config(record_id: str) -> Dict:
    """Read the config.json file for a record."""
    record_path = get_record_path(record_id)
    config_path = os.path.join(record_path, "config.json")
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Config file not found for record {record_id}")
        return {"sampling_rate": 100}  # Default config
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Error parsing config.json: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading config.json: {str(e)}")

def read_heart_rate_raw(record_id: str) -> np.ndarray:
    """Reads the raw heart_rate.txt data."""
    record_path = get_record_path(record_id)
    hr_file = os.path.join(record_path, "heart_rate.txt")
    try:
        # Read each line as JSON and extract the 'p' value (pressure/heart rate)
        pressure_values = []
        with open(hr_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    # Extract the pressure value
                    pressure_values.append(float(data.get('p', 0)))
                except json.JSONDecodeError:
                    print(f"Warning: Invalid JSON line in {hr_file}")
                    continue
                except (ValueError, KeyError) as e:
                    print(f"Warning: Error extracting pressure value: {e}")
                    continue

        if not pressure_values:
            print(f"No valid pressure values found in {hr_file}")
            return np.array([])

        return np.array(pressure_values)
    except FileNotFoundError:
        # Return empty array instead of raising a 404 error
        print(f"heart_rate.txt not found for record '{record_id}', returning empty array")
        return np.array([])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading heart_rate.txt: {e}")

@app.get("/records-interpolated/{record_id}", tags=["Records"], response_model=Dict[str, List | float])
async def get_interpolated_heart_rate(
    record_id: str, frame_start_time: float = 0, frame_size: float = 100
) -> Dict[str, List | float]:
    """Fetch heart rate data from a record, using a sliding frame for interpolation."""
    print(f"Processing record: {record_id}, frame: {frame_start_time}-{frame_start_time + frame_size}")
    heart_rate_data = read_heart_rate_raw(record_id)
    
    # Return empty data if heart_rate.txt doesn't exist
    if len(heart_rate_data) == 0:
        print(f"No heart rate data found for record: {record_id}")
        return {"time": [], "value": [], "normalized": [], "mean": 0, "std": 0, "dominant_freq_hz": 0}
    
    # Get config to determine the sample rate
    config = read_config(record_id)
    try:
        # Check for the sample rate in the heart-rate-sensor section first
        if "heart-rate-sensor" in config and "sample_rate" in config["heart-rate-sensor"]:
            sample_rate = config["heart-rate-sensor"]["sample_rate"]
            print(f"Using heart-rate-sensor sample rate: {sample_rate}")
        # Fall back to top-level sampling_rate if needed
        elif "sampling_rate" in config:
            sample_rate = config["sampling_rate"]
            print(f"Using top-level sampling_rate: {sample_rate}")
        else:
            # Default sample rate if not found
            sample_rate = 100
            print(f"No sample rate found in config, using default: {sample_rate}")
    except KeyError as e:
        print(f"Error accessing sample rate in config: {e}")
        sample_rate = 100  # Use a default instead of failing
    
    print(f"Using sample rate: {sample_rate}, data length: {len(heart_rate_data)}")
    
    # Convert frame times to indices
    start_sample = int(frame_start_time * sample_rate)
    end_sample = int((frame_start_time + frame_size) * sample_rate)

    # Clamp indices to valid range
    start_sample = max(0, start_sample)
    end_sample = min(len(heart_rate_data), end_sample)

    if start_sample >= end_sample:
        print(f"Invalid range: start_sample={start_sample} >= end_sample={end_sample}")
        return {"time": [], "value": [], "normalized": [], "mean": 0, "std": 0, "dominant_freq_hz": 0}  # No data in range or invalid range

    frame_data = heart_rate_data[start_sample:end_sample]
    frame_time_raw = np.arange(start_sample, end_sample) / sample_rate

    print(f"Frame data range: {frame_data.min():.2f} - {frame_data.max():.2f}, points: {len(frame_data)}")

    # Calculate statistics for the frame
    data_mean = float(np.mean(frame_data))
    data_std = float(np.std(frame_data)) if len(frame_data) > 1 else 1.0
    
    # Normalize the data (z-score normalization)
    normalized_data = (frame_data - data_mean) / (data_std if data_std > 0 else 1.0)
    
    # Calculate dominant frequency using FFT
    dominant_freq_hz = 0.0
    try:
        if len(frame_data) > 10:  # Need sufficient data for FFT
            # Compute FFT
            yf = np.fft.rfft(normalized_data)
            xf = np.fft.rfftfreq(len(normalized_data), 1.0 / sample_rate)
            
            # Consider only frequencies in the rat heart rate range (typically 200-450 BPM = 3.33-7.5 Hz)
            min_freq = 3.33  # 200 BPM
            max_freq = 7.5   # 450 BPM
            
            # Find frequencies in the expected range
            mask = (xf >= min_freq) & (xf <= max_freq)
            if np.any(mask):
                # Find the frequency with maximum power in range
                power = np.abs(yf) ** 2
                max_power_idx = np.argmax(power[mask])
                dominant_freq_hz = float(xf[mask][max_power_idx])
                print(f"Detected dominant frequency: {dominant_freq_hz:.2f} Hz ({dominant_freq_hz * 60:.1f} BPM)")
    except Exception as e:
        print(f"Error calculating dominant frequency: {e}")

    # Interpolate to a fixed number of points (e.g., 500) for consistent plotting
    num_interp_points = 500
    interp_time = np.linspace(
        max(frame_start_time, frame_time_raw[0] if len(frame_time_raw) > 0 else frame_start_time),
        min(frame_start_time + frame_size, frame_time_raw[-1] if len(frame_time_raw) > 0 else frame_start_time + frame_size),
        num_interp_points
    )

    # Interpolate both raw and normalized data
    if len(frame_data) > 1:
        try:
            # Interpolate raw data
            interp_func = interp1d(frame_time_raw, frame_data, kind='linear', bounds_error=False, fill_value=np.nan)
            interp_value = interp_func(interp_time)
            
            # Interpolate normalized data
            norm_interp_func = interp1d(frame_time_raw, normalized_data, kind='linear', bounds_error=False, fill_value=np.nan)
            norm_interp_value = norm_interp_func(interp_time)
            
            print(f"Interpolated {len(frame_data)} points to {len(interp_value)} points")
        except Exception as e:
            print(f"Error during interpolation: {e}")
            # Fallback to simple resampling if interpolation fails
            interp_value = np.full(num_interp_points, np.nan)
            norm_interp_value = np.full(num_interp_points, np.nan)
    elif len(frame_data) == 1:
        # Interpolate single point across the requested time range
        interp_value = np.full(num_interp_points, frame_data[0])
        norm_interp_value = np.full(num_interp_points, 0.0)  # Single point normalized is 0
        print("Created constant value from single data point")
    else:
        interp_value = np.full(num_interp_points, np.nan)
        norm_interp_value = np.full(num_interp_points, np.nan)
        print("No data points in frame, filling with NaN")

    # Replace NaNs from interpolation or fill_value with None for JSON compatibility
    interp_value_json = [float(v) if not np.isnan(v) else None for v in interp_value]
    norm_interp_value_json = [float(v) if not np.isnan(v) else None for v in norm_interp_value]

    print(f"Returning {len(interp_time)} time points and {sum(1 for v in interp_value_json if v is not None)} valid values")
    print(f"Mean: {data_mean:.2f}, Std: {data_std:.2f}, Dominant Freq: {dominant_freq_hz:.2f} Hz")
    
    return {
        "time": interp_time.tolist(), 
        "value": interp_value_json, 
        "normalized": norm_interp_value_json,
        "mean": data_mean,
        "std": data_std,
        "dominant_freq_hz": dominant_freq_hz
    }

@app.get("/records-annotations/{record_id}", tags=["Records"], response_model=Dict[str, List])
async def get_heart_rate_annotations(record_id: str):
    """Gets saved heart rate annotations for a record."""
    try:
        record_path = get_record_path(record_id)
        annot_path = os.path.join(record_path, "hr_annotations.json")

        if not os.path.exists(annot_path):
            return {"annotations": []}  # Return empty list if file doesn't exist

        try:
            with open(annot_path, 'r') as f:
                data = json.load(f)
                # Basic validation
                if isinstance(data, dict) and isinstance(data.get("annotations"), list):
                    # Simply return the data for now
                    return {"annotations": data["annotations"]}
                else:
                    # Log warning: Invalid format
                    return {"annotations": []}
        except (json.JSONDecodeError, Exception) as e:
            # Log error e
            raise HTTPException(status_code=500, detail=f"Error reading annotations file: {e}")
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving annotations: {str(e)}")

@app.post("/records-annotations/{record_id}", tags=["Records"])
async def save_heart_rate_annotations(record_id: str, payload: AnnotationsPayload):
    """Saves heart rate annotations for a record, overwriting existing file."""
    try:
        record_path = get_record_path(record_id)
        annot_path = os.path.join(record_path, "hr_annotations.json")
        
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(annot_path), exist_ok=True)
        
        with open(annot_path, 'w') as f:
            json.dump(payload.dict(), f, indent=2)
            
        return {"status": "success", "message": f"Annotations saved for {record_id}"}
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save annotations: {e}")

@app.post("/records-auto-annotate/{record_id}", tags=["Records"], response_model=Dict[str, List])
async def auto_annotate_heart_rate(record_id: str, payload: AutoAnnotatePayload):
    """Automatically detects peaks and valleys in heart rate data."""
    try:
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

        if start_sample >= end_sample - 1:  # Need at least 2 points for find_peaks
            return {"annotations": []}

        frame_data = raw_data[start_sample:end_sample]
        frame_indices = np.arange(start_sample, end_sample)

        # --- Peak/Valley Detection Parameters (ADJUST AS NEEDED) ---
        min_peak_distance_sec = 0.3  # Minimum seconds between heartbeats
        min_peak_distance_samples = int(min_peak_distance_sec * sample_rate)

        # --- Find Peaks and Valleys ---
        try:
            peaks, _ = find_peaks(frame_data, distance=min_peak_distance_samples)
            # Find valleys by inverting data
            valleys, _ = find_peaks(-frame_data, distance=min_peak_distance_samples)
        except Exception as e:
            # Log error during peak finding
            raise HTTPException(status_code=500, detail=f"Error during peak detection: {e}")

        annotations = []
        for p_idx in peaks:
            abs_idx = frame_indices[p_idx]
            time = float(abs_idx / sample_rate)
            value = float(frame_data[p_idx])
            annotations.append({"time": time, "value": value, "type": "peak"})

        for v_idx in valleys:
            abs_idx = frame_indices[v_idx]
            time = float(abs_idx / sample_rate)
            value = float(frame_data[v_idx])
            annotations.append({"time": time, "value": value, "type": "valley"})

        # Sort by time
        annotations.sort(key=lambda x: x["time"])
        
        return {"annotations": annotations}
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during auto-annotation: {str(e)}")