import os
import shutil
import json
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# Import the processing function from the previous step
from volumen.estimator import estimate_volume

app = FastAPI(title="Plant Volume Estimator")

# Ensure the base data directory exists
VOLUME_DIR = Path.home() / "Info" / "volume"
VOLUME_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)

# Mount the static directory to serve the frontend website
app.mount("/static", StaticFiles(directory="./website/static"), name="static")

@app.post("/api/calculate-volume")
async def calculate_volume(
    photo_xy: UploadFile = File(...),
    photo_yz: UploadFile = File(...),
    photo_xz: UploadFile = File(...)
):
    # 1. Create a timestamped directory for this specific calculation run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(VOLUME_DIR, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Define local file paths
    xy_path = os.path.join(run_dir, "photo_xy.jpg")
    yz_path = os.path.join(run_dir, "photo_yz.jpg")
    xz_path = os.path.join(run_dir, "photo_xz.jpg")
    result_path = os.path.join(run_dir, "result.json")

    # 2. Save the uploaded files efficiently using shutil.copyfileobj
    # This prevents loading massive image files directly into RAM all at once
    try:
        with open(xy_path, "wb") as buffer:
            shutil.copyfileobj(photo_xy.file, buffer)
        with open(yz_path, "wb") as buffer:
            shutil.copyfileobj(photo_yz.file, buffer)
        with open(xz_path, "wb") as buffer:
            shutil.copyfileobj(photo_xz.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save images: {str(e)}")
    finally:
        # Always close the uploaded files to free up resources
        photo_xy.file.close()
        photo_yz.file.close()
        photo_xz.file.close()

    # 3. Process the images and calculate volume
    try:
        volume_m3 = estimate_volume(xy_path, yz_path, xz_path, resolution=256)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")

    # 4. Save the calculated results and metadata to a JSON file
    result_data = {
        "timestamp": timestamp,
        "estimated_volume_m3": volume_m3,
        "files_used": {
            "xy": xy_path,
            "yz": yz_path,
            "xz": xz_path
        }
    }

    with open(result_path, "w") as f:
        json.dump(result_data, f, indent=4)

    # 5. Return the data to the frontend
    return JSONResponse(content=result_data)
