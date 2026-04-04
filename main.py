# ================== IMPORTS ==================
import os
import io
import uuid
import logging
import re
import traceback
from contextlib import asynccontextmanager
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

# ================== LOGGING SETUP ==================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ================== ENV ==================
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "vidmind-ai")

# ================== IMPORTS FROM YOUTUBE_CHATBOT ==================
from youtube_chatbot import (
    get_video_id,
    get_transcript_with_timestamps,
    generate_summary,
    generate_quiz,
    generate_notes,
    generate_flashcards,
    generate_exam_plan,
    generate_important_questions,
    generate_revision_notes,
    merge_multi_video_summary,
    detect_confusion,
    build_rag_chain,
    generate_pdf_from_notes,
    answer_with_hybrid_rag,
    add_video_to_session,
    answer_cross_video,
    export_chat_history,
    remove_video_session,
    store,
    cross_video_store,
    _item_to_dict,
    _fetch_via_youtube_transcript_api,
    _fetch_via_ytdlp,
    _fetch_via_innertube,
)

# ================== STATIC DIR ==================
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)


# ================== LIBRARY CHECKER ==================
def _check_libraries() -> dict:
    """Return version strings (or None) for all transcript/PDF libraries."""
    import importlib.metadata
    result = {}
    for lib in ["youtube-transcript-api", "yt-dlp", "fpdf2", "reportlab"]:
        try:
            result[lib] = importlib.metadata.version(lib)
        except Exception:
            result[lib] = None
    return result


# ================== LIFESPAN ==================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 55)
    logger.info("VidMind AI v3.5.0 Starting")
    logger.info(f"Static dir : {static_dir}")
    logger.info(f"Docs       : http://127.0.0.1:8000/docs")
    logger.info(f"Health     : http://127.0.0.1:8000/health")
    logger.info(f"Diagnose   : http://127.0.0.1:8000/diagnose/{{video_id}}")
    logger.info("=" * 55)
    _log_available_libraries()
    yield
    logger.info("VidMind AI shutting down")


def _log_available_libraries():
    libs = _check_libraries()
    for lib, ver in libs.items():
        if ver:
            logger.info(f"  ✅ {lib}: {ver}")
        else:
            logger.warning(f"  ⚠️  {lib}: NOT installed")

    if not libs.get("youtube-transcript-api") and not libs.get("yt-dlp"):
        logger.error(
            "❌ CRITICAL: Neither youtube-transcript-api nor yt-dlp is installed. "
            "Transcript fetching will rely only on InnerTube scraping."
        )


# ================== FASTAPI APP ==================
app = FastAPI(
    title="VidMind AI API",
    version="3.5.0",
    description="AI-powered YouTube video learning assistant",
    lifespan=lifespan,
)

# ================== CORS ==================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================== SESSION STORE ==================
sessions: dict = {}


# ================== LOGGING MIDDLEWARE ==================
@app.middleware("http")
async def log_requests(request: Request, call_next):
    if request.method == "POST":
        body = await request.body()
        logger.info(f"📨 {request.method} {request.url.path}")
        logger.info(f"   Body: {body.decode('utf-8', errors='ignore')[:300]}")

        async def receive():
            return {"type": "http.request", "body": body}

        request._receive = receive

    response = await call_next(request)
    return response


# ================== EXCEPTION HANDLERS ==================
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors       = exc.errors()
    logger.error(f"Validation Error: {errors}")
    error_details = [
        f"{' -> '.join(str(x) for x in e['loc'])}: {e['msg']}"
        for e in errors
    ]
    return JSONResponse(
        status_code=422,
        content={
            "error":   "Request Validation Failed",
            "details": error_details,
            "hint":    "Send JSON with Content-Type: application/json",
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(
        f"Unexpected error on {request.url.path}: {str(exc)}",
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={
            "error":   "Internal Server Error",
            "message": str(exc),
            "type":    type(exc).__name__,
        },
    )


# ================== REQUEST MODELS ==================

class LoadVideoRequest(BaseModel):
    url: str = Field(..., min_length=1)

    @field_validator('url')
    @classmethod
    def validate_url(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError('URL cannot be empty')
        return v.strip()


class LoadMultipleVideosRequest(BaseModel):
    urls: List[str] = Field(..., min_length=1, max_length=5)

    @field_validator('urls')
    @classmethod
    def validate_urls(cls, v: List[str]) -> List[str]:
        cleaned = [url.strip() for url in v if url.strip()]
        if not cleaned:
            raise ValueError('At least one valid URL required')
        return cleaned


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    question:   str = Field(..., min_length=1)


class SessionRequest(BaseModel):
    session_id: str           = Field(..., min_length=1)
    mode:       Optional[str] = None
    section:    Optional[str] = None
    topic:      Optional[str] = None


class SetModeRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    mode:       str = Field(..., pattern="^(exam|deep_learn)$")


class ExamRequest(BaseModel):
    session_id: str           = Field(..., min_length=1)
    topic:      Optional[str] = None
    hours:      float         = Field(..., gt=0, le=24)


class ImportantQuestionsRequest(BaseModel):
    session_id: str           = Field(..., min_length=1)
    topic:      Optional[str] = None


class RevisionNotesRequest(BaseModel):
    session_id: str           = Field(..., min_length=1)
    topic:      Optional[str] = None


class ExportRequest(BaseModel):
    session_id: str = Field(..., min_length=1)


class CustomNotesPDFRequest(BaseModel):
    notes_text: str           = Field(..., min_length=1)
    title:      Optional[str] = Field(default="My Study Notes", max_length=100)


class AddVideoRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    url:        str = Field(..., min_length=1)


class CrossVideoRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    question:   str = Field(..., min_length=1)


# ================== HELPERS ==================

def _get_session(session_id: str) -> dict:
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Session '{session_id}' not found. "
                "Please load a video first."
            )
        )
    return session


def _combined_transcript(session: dict) -> str:
    all_t = session.get("all_transcripts", [session.get("transcript", "")])
    return "\n\n".join(t for t in all_t if t)


def _derived_topic(session: dict) -> str:
    return session.get("summary", {}).get("title", "Main Topic")


def _lang_display_name(lang_code: str) -> str:
    """Convert BCP-47 language code to human-readable name."""
    _LANG_MAP = {
        "en":    "English", "en-US": "English", "en-GB": "English",
        "en-IN": "English", "en-AU": "English", "en-CA": "English",
        "hi":    "Hindi",   "es":    "Spanish",  "fr":   "French",
        "de":    "German",  "pt":    "Portuguese","ar":   "Arabic",
        "zh":    "Chinese", "ja":    "Japanese", "ko":   "Korean",
        "ru":    "Russian", "it":    "Italian",  "nl":   "Dutch",
        "ta":    "Tamil",   "te":    "Telugu",   "bn":   "Bengali",
        "mr":    "Marathi", "gu":    "Gujarati", "pa":   "Punjabi",
        "ml":    "Malayalam","kn":   "Kannada",  "ur":   "Urdu",
        "und":   "Unknown",
    }
    if not lang_code:
        return "Unknown"
    return (
        _LANG_MAP.get(lang_code)
        or _LANG_MAP.get(lang_code[:2])
        or lang_code.upper()
    )


def _is_non_english(lang_code: str) -> bool:
    if not lang_code:
        return False
    english_codes = {"en", "en-US", "en-GB", "en-IN", "en-AU", "en-CA"}
    return lang_code not in english_codes and not lang_code.startswith("en-")


def _safe_fetch_transcript(video_id: str) -> tuple:
    """
    Resilient wrapper around get_transcript_with_timestamps.
    Always returns a 4-tuple, never raises.
    """
    try:
        logger.info(f"[fetch] Starting for video_id={video_id}")
        result = get_transcript_with_timestamps(video_id)

        if not isinstance(result, (list, tuple)) or len(result) != 4:
            logger.error(
                f"[fetch] Unexpected return type: "
                f"type={type(result)}, repr={repr(result)[:200]}"
            )
            return None, [], [], None

        transcript, chunks, raw, lang = result

        if not transcript or not isinstance(transcript, str) or not transcript.strip():
            logger.warning(f"[fetch] Empty/invalid transcript for {video_id}")
            return None, [], [], None

        logger.info(
            f"[fetch] ✅ {video_id} | "
            f"{len(transcript):,} chars | "
            f"{len(chunks)} chunks | "
            f"{len(raw) if raw else 0} raw items | "
            f"lang={lang}"
        )
        return transcript, chunks or [], raw or [], lang or "en"

    except Exception as e:
        logger.error(
            f"[fetch] CRASHED for {video_id}: {type(e).__name__}: {e}\n"
            f"{traceback.format_exc()}"
        )
        return None, [], [], None


def _build_chapters(chunks: list) -> list:
    """Convert transcript chunks into frontend-ready chapter list."""
    chapters = []
    for c in chunks:
        try:
            text    = c.get("text", "")
            words   = text.split()
            preview = " ".join(words[:8]) + ("..." if len(words) > 8 else "")
            chapters.append({
                "time":  c["start_timestamp"],
                "label": c["timestamp"],
                "name":  preview,
            })
        except Exception:
            continue
    return chapters


def _run_diagnose_method(
    method_fn, video_id: str, method_name: str
) -> dict:
    """
    Run a single transcript fetch method and return a standardised result dict.
    """
    try:
        items, lang = method_fn(video_id)
        success     = bool(items)

        first_item_preview = None
        if items:
            try:
                d = _item_to_dict(items[0])
                first_item_preview = d
            except Exception as pe:
                first_item_preview = {"error": str(pe)}

        return {
            "success":            success,
            "item_count":         len(items) if items else 0,
            "lang":               lang,
            "first_item_preview": first_item_preview,
            "error":              None,
        }
    except Exception as e:
        return {
            "success":            False,
            "item_count":         0,
            "lang":               None,
            "first_item_preview": None,
            "error":              f"{type(e).__name__}: {e}",
            "traceback":          traceback.format_exc()[-800:],
        }


# ================== ROUTES ==================

# ── HEALTH ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    libs = _check_libraries()

    transcript_methods = []
    if libs.get("youtube-transcript-api"):
        transcript_methods.append("youtube-transcript-api")
    if libs.get("yt-dlp"):
        transcript_methods.append("yt-dlp")
    transcript_methods.append("innertube-scraper")

    return {
        "status":             "ok",
        "service":            "VidMind AI",
        "version":            "3.5.0",
        "active_sessions":    len(sessions),
        "libraries":          libs,
        "transcript_methods": transcript_methods,
    }


# ── DIAGNOSE ──────────────────────────────────────────────────────────────────
@app.get("/diagnose/{video_id}")
async def diagnose(video_id: str):
    """
    Live per-method transcript fetch test.
    Returns exactly which method succeeded/failed and why.
    """
    if not re.match(r'^[A-Za-z0-9_-]{11}$', video_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid video_id — must be exactly 11 alphanumeric characters."
        )

    logger.info(f"[diagnose] Running all methods for video_id={video_id}")

    report = {
        "video_id":    video_id,
        "libraries":   _check_libraries(),
        "methods":     {},
        "winner":      None,
        "any_success": False,
    }

    method_map = [
        ("youtube_transcript_api", _fetch_via_youtube_transcript_api),
        ("ytdlp",                  _fetch_via_ytdlp),
        ("innertube",              _fetch_via_innertube),
    ]

    for method_key, method_fn in method_map:
        logger.info(f"[diagnose] Testing method: {method_key}")
        result = _run_diagnose_method(method_fn, video_id, method_key)
        report["methods"][method_key] = result

        if result["success"] and not report["winner"]:
            report["winner"]      = method_key
            report["any_success"] = True

    if not report["any_success"]:
        report["advice"] = [
            "All 3 transcript methods failed for this video.",
            "Check: Does the video have CC/subtitles enabled on YouTube?",
            "Check: Is the video age-restricted or region-locked?",
            "Check: Was the video uploaded very recently (< 1 hour)?",
            "Try: pip install -U youtube-transcript-api yt-dlp",
            "Try a different video to confirm the server is working.",
        ]
    else:
        report["advice"] = [
            f"Method '{report['winner']}' succeeded.",
            "If load-video still fails, check the normalization of raw items.",
            (
                f"First item preview: "
                f"{report['methods'][report['winner']].get('first_item_preview')}"
            ),
        ]

    logger.info(
        f"[diagnose] {video_id} → winner={report['winner']} | "
        f"methods={[(k, v['success']) for k, v in report['methods'].items()]}"
    )
    return report


# ── SESSION INFO ──────────────────────────────────────────────────────────────
@app.get("/session-info/{session_id}")
async def session_info(session_id: str):
    session = _get_session(session_id)
    lang    = session.get("detected_language", "en")
    return {
        "session_id":        session_id,
        "video_ids":         session.get("video_ids", []),
        "video_count":       session.get("video_count", 1),
        "detected_language": lang,
        "language_name":     _lang_display_name(lang),
        "is_non_english":    _is_non_english(lang),
        "transcript_length": len(session.get("transcript", "")),
        "transcript_chunks": len(session.get("chunks", [])),
        "raw_item_count":    len(session.get("raw_transcript", [])),
        "mode":              session.get("mode"),
        "confused_topics":   session.get("confused_topics", {}),
        "has_notes":         session.get("cached_notes") is not None,
        "summary_title":     session.get("summary", {}).get("title", ""),
    }


# ── LOAD SINGLE VIDEO ──────────────────────────────────────────────────────────
@app.post("/load-video")
async def load_video(req: LoadVideoRequest):
    logger.info(f"[load-video] URL: {req.url}")

    # 1. Validate video ID
    video_id = get_video_id(req.url)
    if not video_id or not re.match(r'^[A-Za-z0-9_-]{11}$', video_id):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Could not extract a valid YouTube video ID from: {req.url}"
            )
        )
    logger.info(f"[load-video] video_id={video_id}")

    # 2. Fetch transcript
    transcript, chunks, raw, lang = _safe_fetch_transcript(video_id)

    if not transcript:
        logger.warning(f"[load-video] NO_TRANSCRIPT for {video_id}")
        return JSONResponse(status_code=200, content={
            "success":  False,
            "error":    "NO_TRANSCRIPT",
            "video_id": video_id,
            "message": (
                "No captions could be fetched. "
                "All 3 methods were tried: "
                "youtube-transcript-api → yt-dlp → InnerTube scraping."
            ),
            "debug_tips": [
                f"Run GET /diagnose/{video_id} for a per-method breakdown",
                "Check if the video has CC/subtitles enabled on YouTube",
                "Some region-locked or age-restricted videos block transcript access",
                "Very new uploads (< 1 hour) may not have auto-captions yet",
                "Try: pip install -U youtube-transcript-api yt-dlp",
            ],
        })

    lang = lang or "en"
    logger.info(
        f"[load-video] Transcript OK: "
        f"{len(transcript):,} chars | {len(chunks)} chunks | lang={lang}"
    )

    # 3. Build RAG chain
    try:
        chain, get_ts, retriever = build_rag_chain(transcript, chunks, "Video 1")
    except Exception as e:
        logger.error(f"[load-video] build_rag_chain failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to build knowledge base: {str(e)}"
        )

    # 4. Generate summary
    try:
        summary = generate_summary(transcript)
    except Exception as e:
        logger.warning(f"[load-video] generate_summary failed: {e}")
        summary = {
            "title":              "Video Summary",
            "summary":            "Summary could not be generated.",
            "why_it_matters":     "",
            "key_concepts":       [],
            "difficulty":         "Intermediate",
            "study_time_minutes": 15,
        }

    # 5. Create session
    # raw is already a list of plain dicts (normalized in get_transcript_with_timestamps)
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "chain":             chain,
        "get_timestamps":    get_ts,
        "retriever":         retriever,
        "transcript":        transcript,
        "raw_transcript":    raw,       # already plain dicts — safe to serialize
        "chunks":            chunks,
        "all_transcripts":   [transcript],
        "all_raw":           [raw],
        "video_id":          video_id,
        "video_ids":         [video_id],
        "video_count":       1,
        "detected_language": lang,
        "chat_history":      [],
        "summary":           summary,
        "video_summaries":   [summary],
        "cached_notes":      None,
        "confused_topics":   {},
        "mode":              None,
    }

    # 6. Register for cross-video queries
    try:
        add_video_to_session(session_id, video_id, transcript, chunks, "Video 1")
    except Exception as e:
        logger.warning(f"[load-video] add_video_to_session non-fatal: {e}")

    chapters = _build_chapters(chunks)

    logger.info(
        f"[load-video] ✅ session={session_id} | "
        f"lang={lang} | chapters={len(chapters)}"
    )
    return {
        "success":           True,
        "session_id":        session_id,
        "video_id":          video_id,
        "detected_language": lang,
        "language_name":     _lang_display_name(lang),
        "translated":        _is_non_english(lang),
        "chapters":          chapters,
        **summary,
    }


# ── LOAD MULTIPLE VIDEOS ───────────────────────────────────────────────────────
@app.post("/load-multiple-videos")
async def load_multiple_videos(req: LoadMultipleVideosRequest):
    logger.info(f"[load-multiple] {len(req.urls)} URLs")

    session_id      = str(uuid.uuid4())
    all_transcripts = []
    all_raw         = []
    all_summaries   = []
    all_video_ids   = []
    all_chunks      = []
    all_chapters    = []
    all_langs       = []
    skipped         = []

    for idx, url in enumerate(req.urls):
        label = f"Video {idx + 1}"
        vid   = get_video_id(url)

        if not vid:
            logger.warning(f"[load-multiple] Invalid URL: {url}")
            skipped.append({"url": url, "reason": "Invalid YouTube URL"})
            continue

        transcript, chunks, raw, lang = _safe_fetch_transcript(vid)

        if not transcript:
            logger.warning(f"[load-multiple] No transcript for {vid}")
            skipped.append({
                "url":      url,
                "video_id": vid,
                "reason": (
                    f"No captions available (all 3 methods failed). "
                    f"Run /diagnose/{vid}"
                ),
            })
            continue

        lang = lang or "en"
        all_transcripts.append(transcript)
        all_raw.append(raw)
        all_video_ids.append(vid)
        all_langs.append(lang)
        all_chunks.append(chunks)

        try:
            summary = generate_summary(transcript)
        except Exception as e:
            logger.warning(f"[load-multiple] Summary failed for {vid}: {e}")
            summary = {
                "title":              label,
                "summary":            "",
                "why_it_matters":     "",
                "key_concepts":       [],
                "difficulty":         "Intermediate",
                "study_time_minutes": 15,
            }
        all_summaries.append(summary)

        try:
            add_video_to_session(session_id, vid, transcript, chunks, label)
        except Exception as e:
            logger.warning(f"[load-multiple] add_video_to_session {vid}: {e}")

        for c in _build_chapters(chunks):
            c["video_index"] = idx
            c["video_id"]    = vid
            all_chapters.append(c)

        logger.info(f"[load-multiple] Loaded {label}: {vid} | lang={lang}")

    if not all_transcripts:
        raise HTTPException(
            status_code=400,
            detail={
                "error":   "No valid videos loaded",
                "message": "Could not load transcripts from any of the provided URLs.",
                "skipped": skipped,
                "tip":     (
                    "Use GET /diagnose/{video_id} to investigate individual failures."
                ),
            }
        )

    merged   = merge_multi_video_summary(all_summaries)
    combined = "\n\n".join(all_transcripts)

    try:
        chain, get_ts, retriever = build_rag_chain(
            combined, [], "Combined Videos"
        )
    except Exception as e:
        logger.error(f"[load-multiple] build_rag_chain failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to build combined knowledge base: {str(e)}"
        )

    primary_lang = all_langs[0] if all_langs else "en"

    sessions[session_id] = {
        "chain":             chain,
        "get_timestamps":    get_ts,
        "retriever":         retriever,
        "transcript":        combined,
        "raw_transcript":    all_raw[0] if all_raw else [],
        "chunks":            all_chunks[0] if all_chunks else [],
        "all_transcripts":   all_transcripts,
        "all_raw":           all_raw,
        "video_id":          all_video_ids[0],
        "video_ids":         all_video_ids,
        "video_count":       len(all_video_ids),
        "detected_language": primary_lang,
        "chat_history":      [],
        "summary":           merged,
        "video_summaries":   all_summaries,
        "cached_notes":      None,
        "confused_topics":   {},
        "mode":              None,
    }

    return {
        "success":        True,
        "session_id":     session_id,
        "video_count":    len(all_video_ids),
        "video_ids":      all_video_ids,
        "languages":      all_langs,
        "skipped":        skipped,
        "chapters":       all_chapters,
        "summaries":      all_summaries,
        "merged_summary": merged,
        **merged,
    }


# ── CHAT ───────────────────────────────────────────────────────────────────────
@app.post("/chat")
async def chat(req: ChatRequest):
    logger.info(
        f"[chat] session={req.session_id[:8]} | q={req.question[:80]}"
    )
    session = _get_session(req.session_id)

    # Confusion detection
    try:
        confusion       = detect_confusion(req.question)
        is_confused     = confusion.get("score", 1) >= 4
        confusion_score = confusion.get("score", 1)
        confused_topic  = confusion.get("confused_topic", "")
    except Exception as e:
        logger.warning(f"[chat] detect_confusion failed: {e}")
        confusion, is_confused, confusion_score, confused_topic = {}, False, 1, ""

    if is_confused and confused_topic:
        ct = session["confused_topics"]
        ct[confused_topic] = ct.get(confused_topic, 0) + 1

    # Answer generation
    try:
        result = answer_with_hybrid_rag(
            question=req.question,
            retriever=session["retriever"],
            session_id=req.session_id,
            confused=is_confused,
        )
    except Exception as e:
        logger.error(
            f"[chat] answer_with_hybrid_rag failed: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Answer generation failed: {str(e)}"
        )

    answer       = result["answer"]
    mode         = result["mode"]
    source_label = result["source_label"]
    is_grounded  = mode == "video_grounded"

    # Timestamps — safe callable check
    timestamps = []
    if is_grounded:
        try:
            get_ts_fn = session.get("get_timestamps")
            if callable(get_ts_fn):
                timestamps = get_ts_fn(req.question)
        except Exception as e:
            logger.warning(f"[chat] get_timestamps failed: {e}")

    # Update chat history
    session["chat_history"].append({"role": "user", "content": req.question})
    session["chat_history"].append({
        "role":            "assistant",
        "content":         answer,
        "timestamps":      timestamps,
        "outside_video":   not is_grounded,
        "simplified":      is_confused,
        "source_label":    source_label,
        "confusion_score": confusion_score,
    })

    # Normalize timestamps for frontend
    normalized_ts = []
    for ts in timestamps:
        try:
            normalized_ts.append({
                "time":     ts.get("timestamp") or ts.get("label") or "",
                "seconds":  int(ts.get("time", 0)),
                "video_id": ts.get("video_id", session.get("video_id", "")),
            })
        except Exception:
            continue

    return {
        "answer":     answer,
        "source":     "video" if is_grounded else "general",
        "timestamps": normalized_ts,
        "confusion": {
            "score":  confusion_score,
            "reason": confusion.get("reason", ""),
            "topic":  confused_topic,
        },
        "confusion_tracking": session["confused_topics"],
        "simplified":         is_confused,
        "source_label":       source_label,
        "mode":               mode,
    }


# ── SET MODE ────────────────────────────────────────────────────────────────────
@app.post("/set-mode")
async def set_mode(req: SetModeRequest):
    session = _get_session(req.session_id)
    session["mode"] = req.mode
    logger.info(f"[set-mode] session={req.session_id[:8]} | mode={req.mode}")
    return {"success": True, "mode": req.mode}


# ── GENERATE QUIZ ──────────────────────────────────────────────────────────────
@app.post("/generate-quiz")
async def gen_quiz(req: SessionRequest):
    session    = _get_session(req.session_id)
    transcript = _combined_transcript(session)
    topic      = req.topic   or _derived_topic(session)
    section    = req.section

    try:
        quiz = generate_quiz(transcript, section=section, topic=topic)
    except Exception as e:
        logger.error(f"[generate-quiz] failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Quiz generation failed: {str(e)}"
        )

    logger.info(
        f"[generate-quiz] {len(quiz)} Qs | session={req.session_id[:8]}"
    )
    return {"quiz": quiz, "count": len(quiz)}


# ── GENERATE NOTES ─────────────────────────────────────────────────────────────
@app.post("/generate-notes")
async def gen_notes(req: SessionRequest):
    session    = _get_session(req.session_id)
    transcript = _combined_transcript(session)

    try:
        notes = generate_notes(transcript)
    except Exception as e:
        logger.error(f"[generate-notes] failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Notes generation failed: {str(e)}"
        )

    session["cached_notes"] = notes
    logger.info(f"[generate-notes] done | session={req.session_id[:8]}")
    return {"notes": notes}


# ── GENERATE FLASHCARDS ────────────────────────────────────────────────────────
@app.post("/generate-flashcards")
async def gen_flashcards(req: SessionRequest):
    session    = _get_session(req.session_id)
    transcript = _combined_transcript(session)

    try:
        cards = generate_flashcards(transcript)
    except Exception as e:
        logger.error(f"[generate-flashcards] failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Flashcard generation failed: {str(e)}"
        )

    logger.info(
        f"[generate-flashcards] {len(cards)} cards | session={req.session_id[:8]}"
    )
    return {"flashcards": cards, "count": len(cards)}


# ── GENERATE STUDY PLAN ───────────────────────────────────────────────────────
@app.post("/generate-study-plan")
async def gen_study_plan(req: ExamRequest):
    session    = _get_session(req.session_id)
    transcript = _combined_transcript(session)
    topic      = req.topic or _derived_topic(session)

    try:
        plan = generate_exam_plan(transcript, topic, req.hours)
    except Exception as e:
        logger.error(f"[generate-study-plan] failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Study plan generation failed: {str(e)}"
        )

    logger.info(
        f"[generate-study-plan] done | session={req.session_id[:8]}"
    )
    return {
        "plan":          plan.get("sections", []),
        "total_minutes": plan.get("total_minutes", int(req.hours * 60)),
        "quick_tips":    plan.get("quick_tips", []),
    }


# ── GENERATE IMPORTANT QUESTIONS ──────────────────────────────────────────────
@app.post("/generate-important-questions")
async def gen_important_questions(req: ImportantQuestionsRequest):
    session    = _get_session(req.session_id)
    transcript = _combined_transcript(session)
    topic      = req.topic or _derived_topic(session)

    try:
        questions = generate_important_questions(transcript, topic)
    except Exception as e:
        logger.error(
            f"[generate-important-questions] failed: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Question generation failed: {str(e)}"
        )

    logger.info(
        f"[generate-important-questions] {len(questions)} Qs | "
        f"session={req.session_id[:8]}"
    )
    return {"questions": questions, "count": len(questions)}


# ── GENERATE REVISION NOTES ───────────────────────────────────────────────────
@app.post("/generate-revision-notes")
async def gen_revision_notes(req: RevisionNotesRequest):
    session    = _get_session(req.session_id)
    transcript = _combined_transcript(session)
    topic      = req.topic or _derived_topic(session)

    try:
        revision = generate_revision_notes(transcript, topic)
    except Exception as e:
        logger.error(f"[generate-revision-notes] failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Revision notes generation failed: {str(e)}"
        )

    logger.info(
        f"[generate-revision-notes] done | session={req.session_id[:8]}"
    )
    return {"revision": revision}


# ── GET TRANSCRIPT ─────────────────────────────────────────────────────────────
@app.get("/transcript/{session_id}")
async def get_transcript(session_id: str):
    session    = _get_session(session_id)
    lang       = session.get("detected_language", "en")
    # raw_transcript is already plain dicts (normalized at load time)
    serialized = session.get("raw_transcript", [])

    logger.info(
        f"[transcript] session={session_id[:8]} | "
        f"{len(serialized)} items | lang={lang}"
    )
    return {
        "transcript":        serialized,
        "item_count":        len(serialized),
        "detected_language": lang,
        "language_name":     _lang_display_name(lang),
        "translated":        _is_non_english(lang),
    }


# ── GET CONFUSED TOPICS ───────────────────────────────────────────────────────
@app.get("/confused-topics/{session_id}")
async def get_confused_topics(session_id: str):
    session  = _get_session(session_id)
    tracking = session.get("confused_topics", {})
    topics   = [{"topic": t, "count": c} for t, c in tracking.items() if t]
    topics.sort(key=lambda x: x["count"], reverse=True)
    return {"confused_topics": topics}


# ── EXPORT NOTES PDF ──────────────────────────────────────────────────────────
@app.post("/export-notes-pdf")
async def export_notes_pdf(req: SessionRequest):
    session = _get_session(req.session_id)
    notes   = session.get("cached_notes")

    if not notes:
        transcript = _combined_transcript(session)
        try:
            notes = generate_notes(transcript)
        except Exception as e:
            logger.error(
                f"[export-notes-pdf] generate_notes failed: {e}", exc_info=True
            )
            raise HTTPException(
                status_code=500,
                detail=f"Notes generation failed: {str(e)}"
            )
        session["cached_notes"] = notes

    title = session.get("summary", {}).get("title", "VidMind AI Notes")

    try:
        pdf_bytes = generate_pdf_from_notes(notes, title=title)
    except Exception as e:
        logger.error(
            f"[export-notes-pdf] PDF generation failed: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"PDF generation failed: {str(e)}"
        )

    logger.info(f"[export-notes-pdf] done | session={req.session_id[:8]}")
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=VidMind_Notes.pdf"},
    )


# ── EXPORT CUSTOM NOTES PDF ──────────────────────────────────────────────────
@app.post("/export-custom-notes-pdf")
async def export_custom_notes_pdf(req: CustomNotesPDFRequest):
    try:
        pdf_bytes = generate_pdf_from_notes(req.notes_text, title=req.title)
    except Exception as e:
        logger.error(f"[export-custom-notes-pdf] failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"PDF generation failed: {str(e)}"
        )

    filename = (
        re.sub(r'[^\w\s-]', '', req.title or "notes")
        .strip()
        .replace(' ', '_') + ".pdf"
    )
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# ── ADD VIDEO TO SESSION ──────────────────────────────────────────────────────
@app.post("/add-video")
async def add_video(req: AddVideoRequest):
    session  = _get_session(req.session_id)
    video_id = get_video_id(req.url)

    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")

    if video_id in session.get("video_ids", []):
        return {
            "success":  True,
            "message":  "Video already in session",
            "video_id": video_id,
        }

    transcript, chunks, raw, lang = _safe_fetch_transcript(video_id)

    if not transcript:
        raise HTTPException(
            status_code=400,
            detail={
                "error":    "NO_TRANSCRIPT",
                "video_id": video_id,
                "message":  (
                    "No captions available. All 3 transcript methods failed."
                ),
                "diagnose": f"GET /diagnose/{video_id}",
            }
        )

    lang  = lang or "en"
    idx   = session.get("video_count", 1) + 1
    label = f"Video {idx}"

    session["all_transcripts"].append(transcript)
    session["all_raw"].append(raw)
    session["video_ids"].append(video_id)
    session["video_count"] = idx

    combined = "\n\n".join(session["all_transcripts"])

    try:
        chain, get_ts, retriever = build_rag_chain(
            combined, [], f"Combined ({idx} videos)"
        )
        session["chain"]          = chain
        session["get_timestamps"] = get_ts
        session["retriever"]      = retriever
        session["transcript"]     = combined
    except Exception as e:
        logger.error(f"[add-video] build_rag_chain failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to rebuild knowledge base: {str(e)}"
        )

    try:
        add_video_to_session(req.session_id, video_id, transcript, chunks, label)
    except Exception as e:
        logger.warning(f"[add-video] add_video_to_session non-fatal: {e}")

    try:
        summary = generate_summary(transcript)
    except Exception as e:
        logger.warning(f"[add-video] generate_summary failed: {e}")
        summary = {
            "title":              label,
            "summary":            "",
            "why_it_matters":     "",
            "key_concepts":       [],
            "difficulty":         "Intermediate",
            "study_time_minutes": 15,
        }

    session["video_summaries"].append(summary)
    session["summary"] = merge_multi_video_summary(session["video_summaries"])

    logger.info(
        f"[add-video] ✅ {video_id} → "
        f"session={req.session_id[:8]} (total={idx})"
    )
    return {
        "success":     True,
        "video_id":    video_id,
        "video_count": idx,
        "language":    _lang_display_name(lang),
        "summary":     summary,
    }


# ── CROSS-VIDEO CHAT ──────────────────────────────────────────────────────────
@app.post("/cross-video-chat")
async def cross_video_chat(req: CrossVideoRequest):
    logger.info(
        f"[cross-video-chat] session={req.session_id[:8]} | "
        f"q={req.question[:60]}"
    )
    _get_session(req.session_id)

    try:
        result = answer_cross_video(req.session_id, req.question)
    except Exception as e:
        logger.error(f"[cross-video-chat] failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Cross-video answer failed: {str(e)}"
        )

    return result


# ── EXPORT CHAT HISTORY ───────────────────────────────────────────────────────
@app.post("/export-chat-history")
async def export_chat(req: ExportRequest):
    _get_session(req.session_id)

    try:
        text = export_chat_history(req.session_id)
    except Exception as e:
        logger.error(f"[export-chat-history] failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Chat export failed: {str(e)}"
        )

    return {"history": text}


# ── DELETE SESSION ─────────────────────────────────────────────────────────────
@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """
    Explicitly clean up a session and all associated resources.
    Prevents memory leaks in long-running deployments.
    """
    existed = session_id in sessions
    sessions.pop(session_id, None)
    remove_video_session(session_id)   # cleans cross_video_store
    store.pop(session_id, None)        # cleans LangChain chat history

    logger.info(
        f"[delete-session] {'Cleaned up' if existed else 'Not found'}: "
        f"{session_id[:8]}"
    )
    return {
        "success":  True,
        "deleted":  session_id,
        "existed":  existed,
    }


# ── LIST SESSIONS ──────────────────────────────────────────────────────────────
@app.get("/sessions")
async def list_sessions():
    """List all active sessions with basic metadata."""
    result = []
    for sid, sess in sessions.items():
        result.append({
            "session_id":    sid,
            "video_count":   sess.get("video_count", 1),
            "video_ids":     sess.get("video_ids", []),
            "summary_title": sess.get("summary", {}).get("title", ""),
            "language":      sess.get("detected_language", "en"),
            "has_notes":     sess.get("cached_notes") is not None,
        })
    return {"sessions": result, "count": len(result)}


# ================== STATIC FILES & FRONTEND ==================
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=HTMLResponse)
async def read_root():
    index_path = os.path.join(static_dir, "vidmind.html")
    if not os.path.exists(index_path):
        return HTMLResponse(f"""
            <html>
            <body style="font-family:Arial;padding:40px;text-align:center;
                         background:#0f0f0f;color:#f1f1f1;">
                <h1>VidMind AI Running ✅</h1>
                <p>Place <code>vidmind.html</code> in: <code>{static_dir}</code></p>
                <p>
                  <a href="/docs"     style="color:#ff0000;">📖 API Docs</a> &nbsp;|&nbsp;
                  <a href="/health"   style="color:#ff0000;">❤️ Health</a>   &nbsp;|&nbsp;
                  <a href="/diagnose/dQw4w9WgXcQ"
                     style="color:#ff0000;">🔬 Diagnose Test</a>
                </p>
            </body>
            </html>
        """)
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()


# ================== ENTRY POINT ==================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080)