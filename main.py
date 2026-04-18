import os
import uuid
import asyncio
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
import json

app = FastAPI(title="Shortzify API")

# CORS — frontend se connect ho sake
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Folders banao
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Job status track karne ke liye (production me Redis use karo)
job_store: dict = {}

# ── Static files (frontend) ──
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")


# ─────────────────────────────────────────────
# 1. VIDEO UPLOAD ENDPOINT
# ─────────────────────────────────────────────
@app.post("/api/upload")
async def upload_video(
    file: UploadFile = File(...),
    n_clips: int = Form(5),
    max_duration: int = Form(60),
    detect_peaks: bool = Form(True),
    vertical_crop: bool = Form(True),
    captions: bool = Form(True),
    hook_headline: bool = Form(False),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    job_id = str(uuid.uuid4())

    # File save karo
    ext = Path(file.filename).suffix
    video_path = UPLOAD_DIR / f"{job_id}{ext}"
    with open(video_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Job register karo
    job_store[job_id] = {
        "status": "queued",
        "step": "Waiting to start",
        "progress": 0,
        "clips": []
    }

    # Background me processing shuru karo
    background_tasks.add_task(
        process_video,
        job_id=job_id,
        video_path=str(video_path),
        n_clips=n_clips,
        max_duration=max_duration,
        options={
            "detect_peaks": detect_peaks,
            "vertical_crop": vertical_crop,
            "captions": captions,
            "hook_headline": hook_headline,
        }
    )

    return {"job_id": job_id, "message": "Processing started"}


# ─────────────────────────────────────────────
# 2. STATUS POLLING ENDPOINT
# ─────────────────────────────────────────────
@app.get("/api/status/{job_id}")
def get_status(job_id: str):
    if job_id not in job_store:
        return JSONResponse(status_code=404, content={"error": "Job not found"})
    return job_store[job_id]


# ─────────────────────────────────────────────
# 3. DOWNLOAD CLIP ENDPOINT
# ─────────────────────────────────────────────
@app.get("/api/download/{filename}")
def download_clip(filename: str):
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        return JSONResponse(status_code=404, content={"error": "File not found"})
    return FileResponse(file_path, media_type="video/mp4", filename=filename)


# ─────────────────────────────────────────────
# 4. MAIN PROCESSING PIPELINE
# ─────────────────────────────────────────────
async def process_video(job_id: str, video_path: str, n_clips: int, max_duration: int, options: dict):
    try:
        # ── STEP 1: Transcribe with Whisper ──
        update_job(job_id, step="Transcribing audio with Whisper", progress=5)
        transcript = await run_whisper(video_path)
        update_job(job_id, step="Transcription done", progress=20)

        # ── STEP 2: Detect emotional peaks via Gemini ──
        update_job(job_id, step="Detecting emotional peaks via Gemini 1.5", progress=25)
        peaks = await detect_peaks_gemini(transcript, n_clips, max_duration)
        update_job(job_id, step="Peaks detected", progress=45)

        # ── STEP 3: Face tracking with MediaPipe ──
        if options["vertical_crop"]:
            update_job(job_id, step="Face tracking with MediaPipe", progress=50)
            face_data = await run_face_tracking(video_path)
        else:
            face_data = None
        update_job(job_id, step="Face tracking done", progress=60)

        # ── STEP 4: Crop clips to 9:16 using MoviePy ──
        update_job(job_id, step="Smart-cropping to 9:16 via MoviePy", progress=65)
        clip_files = await cut_and_crop_clips(video_path, peaks, face_data, job_id, max_duration)
        update_job(job_id, step="Clips cropped", progress=80)

        # ── STEP 5: Add captions and hook headlines ──
        if options["captions"]:
            update_job(job_id, step="Rendering captions & hook headlines", progress=85)
            clip_files = await add_captions(clip_files, transcript, peaks, options["hook_headline"])
        update_job(job_id, step="Captions added", progress=95)

        # ── DONE ──
        job_store[job_id]["status"] = "done"
        job_store[job_id]["progress"] = 100
        job_store[job_id]["step"] = "Completed"
        job_store[job_id]["clips"] = clip_files

    except Exception as e:
        job_store[job_id]["status"] = "error"
        job_store[job_id]["step"] = f"Error: {str(e)}"


def update_job(job_id, step, progress):
    job_store[job_id]["status"] = "processing"
    job_store[job_id]["step"] = step
    job_store[job_id]["progress"] = progress


# ─────────────────────────────────────────────
# 5. WHISPER — Audio Transcription
# ─────────────────────────────────────────────
async def run_whisper(video_path: str) -> dict:
    import whisper
    loop = asyncio.get_event_loop()

    def _transcribe():
        model = whisper.load_model("base")  # "small" ya "medium" for better accuracy
        result = model.transcribe(video_path, word_timestamps=True)
        return result

    result = await loop.run_in_executor(None, _transcribe)
    return result


# ─────────────────────────────────────────────
# 6. GEMINI — Emotional Peak Detection
# ─────────────────────────────────────────────
async def detect_peaks_gemini(transcript: dict, n_clips: int, max_duration: int) -> list:
    import google.generativeai as genai

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Transcript text + timestamps nikalo
    segments = transcript.get("segments", [])
    transcript_text = "\n".join(
        f"[{seg['start']:.1f}s - {seg['end']:.1f}s]: {seg['text']}"
        for seg in segments
    )

    prompt = f"""
You are a viral content editor. Analyze this video transcript and find the {n_clips} most emotionally powerful, insightful, or "viral-worthy" moments.

TRANSCRIPT:
{transcript_text}

Rules:
- Each clip must be between 30 and {max_duration} seconds long
- Pick moments that are self-contained (have a clear start and end)
- Prefer moments with strong opinions, surprising facts, or emotional highs
- For each clip, write a short "hook headline" (max 8 words) that would stop someone from scrolling

Respond ONLY with valid JSON like this:
{{
  "clips": [
    {{
      "start": 45.2,
      "end": 98.7,
      "viral_score": 94,
      "hook": "Nobody tells you this about success",
      "tags": ["mindset", "career"],
      "reason": "Speaker delivers a powerful counterintuitive insight"
    }}
  ]
}}
"""

    response = model.generate_content(prompt)
    text = response.text.strip()

    # JSON extract karo
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    data = json.loads(text)
    return data["clips"]


# ─────────────────────────────────────────────
# 7. MEDIAPIPE — Face Tracking
# ─────────────────────────────────────────────
async def run_face_tracking(video_path: str) -> dict:
    import cv2
    import mediapipe as mp
    loop = asyncio.get_event_loop()

    def _track():
        mp_face = mp.solutions.face_detection
        face_positions = {}  # frame_number -> (cx, cy)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_num = 0

        with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Har 5th frame process karo (speed ke liye)
                if frame_num % 5 == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = detector.process(rgb)
                    h, w = frame.shape[:2]

                    if results.detections:
                        bbox = results.detections[0].location_data.relative_bounding_box
                        cx = (bbox.xmin + bbox.width / 2) * w
                        cy = (bbox.ymin + bbox.height / 2) * h
                        face_positions[frame_num] = (cx, cy)

                frame_num += 1

        cap.release()
        return {"fps": fps, "face_positions": face_positions}

    return await loop.run_in_executor(None, _track)


# ─────────────────────────────────────────────
# 8. MOVIEPY — Cut + Smart Crop to 9:16
# ─────────────────────────────────────────────
async def cut_and_crop_clips(video_path: str, peaks: list, face_data: dict, job_id: str, max_duration: int) -> list:
    from moviepy.editor import VideoFileClip
    loop = asyncio.get_event_loop()

    def _cut():
        clips_info = []
        video = VideoFileClip(video_path)
        w, h = video.size

        # Target: 9:16 vertical
        target_h = h
        target_w = int(h * 9 / 16)

        for i, peak in enumerate(peaks):
            start = float(peak["start"])
            end = min(float(peak["end"]), start + max_duration)

            clip = video.subclip(start, end)

            # Smart crop: face position ke around crop karo
            if face_data and face_data.get("face_positions"):
                fps = face_data["fps"]
                mid_frame = int((start + end) / 2 * fps)
                # Nearest frame dhundo
                positions = face_data["face_positions"]
                if positions:
                    nearest = min(positions.keys(), key=lambda k: abs(k - mid_frame))
                    face_cx, _ = positions[nearest]
                    # Crop x position
                    x1 = max(0, int(face_cx - target_w / 2))
                    x1 = min(x1, w - target_w)
                else:
                    x1 = (w - target_w) // 2  # center crop fallback
            else:
                x1 = (w - target_w) // 2  # center crop

            cropped = clip.crop(x1=x1, y1=0, x2=x1 + target_w, y2=target_h)

            # Save
            filename = f"{job_id}_clip_{i+1}.mp4"
            out_path = str(OUTPUT_DIR / filename)
            cropped.write_videofile(out_path, codec="libx264", audio_codec="aac", logger=None)

            clips_info.append({
                "filename": filename,
                "download_url": f"/api/download/{filename}",
                "start": start,
                "end": end,
                "viral_score": peak.get("viral_score", 75),
                "hook": peak.get("hook", ""),
                "tags": peak.get("tags", []),
            })

        video.close()
        return clips_info

    return await loop.run_in_executor(None, _cut)


# ─────────────────────────────────────────────
# 9. CAPTIONS — Word-by-word overlays
# ─────────────────────────────────────────────
async def add_captions(clip_files: list, transcript: dict, peaks: list, add_hook: bool) -> list:
    from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
    loop = asyncio.get_event_loop()

    # Word-level timestamps from Whisper
    words = []
    for seg in transcript.get("segments", []):
        for w in seg.get("words", []):
            words.append(w)

    def _caption(clip_info, peak):
        clip_start = float(peak["start"])
        clip_end = float(peak["end"])

        # Words sirf is clip ke
        clip_words = [w for w in words if clip_start <= w["start"] <= clip_end]

        video = VideoFileClip(str(OUTPUT_DIR / clip_info["filename"]))
        clips_list = [video]

        # Hook headline (top pe)
        if add_hook and peak.get("hook"):
            hook_clip = (
                TextClip(peak["hook"], fontsize=28, color="white", font="Arial-Bold",
                         stroke_color="black", stroke_width=2, method="caption", size=(video.w - 40, None))
                .set_position(("center", 60))
                .set_duration(min(3.0, video.duration))
            )
            clips_list.append(hook_clip)

        # Word-by-word karaoke captions
        for i, word in enumerate(clip_words):
            w_start = word["start"] - clip_start
            w_end   = word["end"]   - clip_start
            text    = word["word"].strip().upper()
            if not text:
                continue

            txt_clip = (
                TextClip(text, fontsize=52, color="yellow", font="Arial-Bold",
                         stroke_color="black", stroke_width=3)
                .set_position(("center", "bottom"))
                .set_start(w_start)
                .set_end(w_end)
            )
            clips_list.append(txt_clip)

        final = CompositeVideoClip(clips_list)
        captioned_filename = clip_info["filename"].replace(".mp4", "_captioned.mp4")
        out_path = str(OUTPUT_DIR / captioned_filename)
        final.write_videofile(out_path, codec="libx264", audio_codec="aac", logger=None)
        video.close()

        clip_info["filename"] = captioned_filename
        clip_info["download_url"] = f"/api/download/{captioned_filename}"
        return clip_info

    def _run_all():
        updated = []
        for clip_info, peak in zip(clip_files, peaks):
            try:
                updated.append(_caption(clip_info, peak))
            except Exception:
                updated.append(clip_info)  # fallback: bina caption ke
        return updated

    return await loop.run_in_executor(None, _run_all)
