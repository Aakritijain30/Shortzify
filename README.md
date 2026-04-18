# Shortzify 🎬
**Automated Content Repurposing Engine** — AttentionX AI Hackathon

Upload a long-form video → get viral 9:16 clips with captions, automatically.

---

## Tech Stack
- **Backend**: Python + FastAPI
- **Transcription**: OpenAI Whisper
- **AI Analysis**: Google Gemini 1.5 Flash
- **Face Tracking**: MediaPipe
- **Video Editing**: MoviePy
- **Frontend**: Vanilla HTML/CSS/JS

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/upload` | Video upload karo, job start karo |
| GET | `/api/status/{job_id}` | Processing status check karo |
| GET | `/api/download/{filename}` | Clip download karo |

---

## Pipeline
1. **Whisper** — Audio transcription with word-level timestamps
2. **Gemini 1.5 Flash** — Emotional peak detection + viral scoring + hook generation
3. **MediaPipe** — Face detection for smart crop positioning
4. **MoviePy** — Cut video + 9:16 vertical crop
5. **MoviePy** — Karaoke captions + hook headline overlay
