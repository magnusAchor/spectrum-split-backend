import os
import uuid
import shutil
import subprocess
import threading
import time
import sys

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import librosa
import soundfile as sf

from instrument_split import splitOtherStem


app = FastAPI()

# ================= CORS =================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= FOLDERS =================
UPLOAD_DIR = "temp"
OUTPUT_DIR = "output"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")


# ================= MEMORY JOB STORE =================
jobs = {}


# ================= CLEANUP =================
def delete_later(paths: list):
    time.sleep(120)
    for path in paths:
        try:
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            elif os.path.isfile(path):
                os.remove(path)
        except Exception as e:
            print("Cleanup error:", e)


# ================= DEMUCS BACKGROUND =================
def run_demucs(job_id, input_path):
    try:
        command = [
            sys.executable,
            "-m",
            "demucs",
            "-n",
            "mdx_extra_q",  # 🔥 lighter model
            "-o",
            OUTPUT_DIR,
            input_path
        ]

        print("Running:", command)

        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode != 0:
            jobs[job_id] = {"status": "failed"}
            print(result.stderr)
            return

        base_name = os.path.splitext(os.path.basename(input_path))[0]
        stem_folder = os.path.join(OUTPUT_DIR, "mdx_extra_q", base_name)

        stems = {}

        for f in os.listdir(stem_folder):
            if f.endswith(".wav"):
                name = f.replace(".wav", "")
                stems[name] = f"https://spectrum-split-backend.onrender.com/output/mdx_extra_q/{base_name}/{f}"

        jobs[job_id] = {
            "status": "done",
            "stems": stems
        }

        threading.Thread(
            target=delete_later,
            args=([input_path, stem_folder],),
            daemon=True
        ).start()

    except Exception as e:
        print("DEMUX ERROR:", e)
        jobs[job_id] = {"status": "failed"}


# ================= START SPLIT =================
@app.post("/separate")
async def separate(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())

    # 🚫 limit file size (~10MB)
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(400, "File too large (max 10MB)")

    input_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")

    with open(input_path, "wb") as f:
        f.write(content)

    jobs[job_id] = {"status": "processing"}

    # 🔥 run in background
    threading.Thread(
        target=run_demucs,
        args=(job_id, input_path),
        daemon=True
    ).start()

    return {
        "job_id": job_id,
        "status": "processing"
    }


# ================= CHECK STATUS =================
@app.get("/status/{job_id}")
def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    return jobs[job_id]


# ================= INSTRUMENT SPLIT =================
class SplitRequest(BaseModel):
    audio: str


@app.post("/split-instruments")
async def split_instruments(payload: SplitRequest):
    try:
        audio_url = payload.audio

        path = audio_url.replace("https://spectrum-split-backend.onrender.com/", "")
        file_path = os.path.join(os.getcwd(), path)

        if not os.path.exists(file_path):
            raise HTTPException(404, "Audio file not found")

        # 🔥 reduce memory (mono)
        y, sr = librosa.load(file_path, sr=None, mono=False)

        if y.ndim == 1:
            raise HTTPException(400, "Audio must be stereo")

        left, right = y[0], y[1]

        result = splitOtherStem(left, right, sr)

        job_id = str(uuid.uuid4())
        out_dir = os.path.join(OUTPUT_DIR, "instruments", job_id)
        os.makedirs(out_dir, exist_ok=True)

        instruments = {}

        for name, data in result.items():
            path = os.path.join(out_dir, f"{name}.wav")

            stereo = list(zip(data["left"], data["right"]))
            sf.write(path, stereo, sr)

            instruments[name] = f"https://spectrum-split-backend.onrender.com/output/instruments/{job_id}/{name}.wav"

        return {"instruments": instruments}

    except Exception as e:
        print("INSTRUMENT ERROR:", e)
        raise HTTPException(500, str(e))