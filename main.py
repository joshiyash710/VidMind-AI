import os
import io
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Optional

# ── LangSmith must be configured before any LangChain imports ─────
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"]    = os.getenv("LANGCHAIN_PROJECT", "vidmind-ai")

from youtube_chatbot import (
    get_video_id,
    get_transcript_with_timestamps,
    generate_summary,
    generate_quiz,
    generate_notes,
    generate_flashcards,
    generate_exam_plan,
    detect_confusion,
    build_rag_chain,
    generate_pdf_from_notes,
    answer_with_hybrid_rag,     # NEW — Feature 2: Hybrid RAG routing
    add_video_to_session,       # NEW — Feature 5: Cross-Video index builder
    answer_cross_video,         # NEW — Feature 5: Cross-Video QA
    export_chat_history,        # NEW — Supporting feature: chat export helper
    store,
)

app = FastAPI(title="VidMind AI API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory session store ───────────────────────────────────────
sessions: dict = {}

# ── Request models ────────────────────────────────────────────────
class LoadVideoRequest(BaseModel):
    url: str

class ChatRequest(BaseModel):
    session_id: str
    question: str

class SessionRequest(BaseModel):
    session_id: str
    section: Optional[str] = None
    topic: Optional[str] = None

class ExamRequest(BaseModel):
    session_id: str
    topic: str
    hours: float

class ExportRequest(BaseModel):
    session_id: str

class CustomNotesPDFRequest(BaseModel):
    notes_text: str
    title: Optional[str] = "My Study Notes"

# NEW — Feature 5: Add a second/third video to an existing session
class AddVideoRequest(BaseModel):
    session_id: str
    url: str

# NEW — Feature 5: Cross-video question
class CrossVideoRequest(BaseModel):
    session_id: str
    question: str


# ── ROUTES ────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "VidMind AI", "version": "2.0.0"}


@app.post("/load-video")
async def load_video(req: LoadVideoRequest):
    """
    Load a YouTube video:
    - Extracts transcript (multilingual, auto-translates to English)
    - Builds FAISS vector store + timestamp index
    - Registers video in cross-video store (Feature 5)
    - Auto-generates summary
    - Returns session_id + summary data + detected language + chapters
    """
    video_id = get_video_id(req.url)
    if not video_id:
        raise HTTPException(
            status_code=400,
            detail="Invalid YouTube URL. Please check the URL and try again."
        )

    try:
        transcript, timestamped_chunks, raw_transcript, detected_language = \
            get_transcript_with_timestamps(video_id)
    except RuntimeError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # FIX: build_rag_chain now returns 3 values — (chain, get_timestamps_fn, retriever)
    # The retriever is needed by answer_with_hybrid_rag for relevance scoring (Feature 2)
    chain, get_timestamps_fn, retriever = build_rag_chain(
        transcript, timestamped_chunks, video_label="Video 1"
    )

    # Auto-generate summary on load
    summary_data = generate_summary(transcript)

    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "chain":             chain,
        "get_timestamps":    get_timestamps_fn,
        "retriever":         retriever,          # stored for hybrid RAG routing
        "transcript":        transcript,
        "raw_transcript":    raw_transcript,
        "all_transcripts":   [transcript],
        "all_raw_transcripts": [raw_transcript],
        "video_id":          video_id,
        "video_count":       1,                  # tracks how many videos are loaded
        "detected_language": detected_language,
        "chat_history":      [],
        "summary":           summary_data,
        "video_summaries":  [summary_data],
        "cached_notes":      None,
    }

    # Register this video in the cross-video store so /add-video works from the start
    add_video_to_session(
        session_id=session_id,
        video_id=video_id,
        transcript=transcript,
        chunks=timestamped_chunks,
        label="Video 1",
    )

    # Build chapters for sidebar from timestamped chunks
    chapters = []
    for chunk in timestamped_chunks:
        ts    = chunk["start_timestamp"]
        lbl   = chunk["timestamp"]
        words = chunk["text"].split()[:8]
        name  = " ".join(words) + ("..." if len(chunk["text"].split()) > 8 else "")
        chapters.append({"time": ts, "label": lbl, "name": name})

    # Triage study time if not provided by summary
    if "study_time_minutes" not in summary_data:
        # rough estimate: 150 words per minute
        predicted = max(15, min(120, int(len(transcript.split()) / 150)))
        summary_data["study_time_minutes"] = predicted

    return {
        "session_id":        session_id,
        "video_id":          video_id,
        "detected_language": detected_language,
        "translated":        detected_language != "en",
        "chapters":          chapters,
        "study_time_minutes": summary_data.get("study_time_minutes", 0),
        "key_topics":        summary_data.get("key_concepts", []),
        "difficulty":        summary_data.get("difficulty", "Intermediate"),
        **summary_data,
    }


@app.post("/chat")
async def chat(req: ChatRequest):
    """
    RAG chat with:
    - Hybrid RAG (Feature 2): routes between video-grounded and world-knowledge mode
    - Confusion detection (Feature 4): auto-simplifies if confusion score >= 4
    - Timestamp extraction for every video-grounded answer
    - Full chat history stored for export
    """
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Please load a video first."
        )

    # ── Feature 4: Confusion Detection ──────────────────────────
    confusion    = detect_confusion(req.question)
    is_confused  = confusion.get("score", 1) >= 4
    confusion_score = confusion.get("score", 1)

    # ── Feature 2: Hybrid RAG routing ───────────────────────────
    # answer_with_hybrid_rag handles both routing logic AND the confusion
    # flag injection into the prompt (simplified analogy-first explanation)
    retriever = session["retriever"]
    hybrid_result = answer_with_hybrid_rag(
        question=req.question,
        retriever=retriever,
        session_id=req.session_id,
        confused=is_confused,
    )

    answer       = hybrid_result["answer"]
    mode         = hybrid_result["mode"]           # "video_grounded" | "world_knowledge"
    source_label = hybrid_result["source_label"]   # shown as a UI badge

    # Timestamps only make sense for video-grounded answers
    timestamps = []
    if mode == "video_grounded":
        get_timestamps = session["get_timestamps"]
        timestamps = get_timestamps(req.question)

    outside_video = (mode == "world_knowledge")

    # ── Store turn for /export-chat ──────────────────────────────
    session["chat_history"].append({"role": "user", "content": req.question})
    session["chat_history"].append({
        "role":           "assistant",
        "content":        answer,
        "timestamps":     timestamps,
        "outside_video":  outside_video,
        "simplified":     is_confused,
        "source_label":   source_label,
        "confusion_score": confusion_score,
    })

    # normalize timestamps payload for frontend API spec
    normalized_timestamps = [
        {
            "time": ts.get("timestamp") or ts.get("label") or "",
            "seconds": int(ts.get("time", ts.get("start_timestamp", 0))),
            "video_id": ts.get("video_id", session.get("video_id"))
        }
        for ts in timestamps
    ]

    return {
        "answer":        answer,
        "source":        "video" if mode == "video_grounded" else "general",
        "timestamps":    normalized_timestamps,
        "confusion":     {"score": confusion_score, "reason": confusion.get("reason", "")},
        "simplified":    is_confused,
        "source_label":  source_label,
        "mode":          mode,
    }


@app.post("/generate-quiz")
async def generate_quiz_endpoint(req: SessionRequest):
    """Generate MCQs from the transcript(s)."""
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    all_transcripts = session.get("all_transcripts", [session.get("transcript", "")])
    combined_transcript = "\n\n".join([t for t in all_transcripts if t])

    quiz = generate_quiz(combined_transcript, req.section, req.topic)
    if not isinstance(quiz, list):
        quiz = []

    # Normalize to minimum structure
    normalized = []
    for item in quiz:
        if not isinstance(item, dict):
            continue

        q = item.get("question", "")
        options = item.get("options") or {}
        if isinstance(options, list):
            # old prompt: list of options
            options_map = {}
            for idx, opt in enumerate(options[:4]):
                options_map["ABCD"[idx]] = opt
            options = options_map

        normalized.append({
            "question": q,
            "options": options,
            "correct": item.get("correct", ""),
            "explanation": item.get("explanation", ""),
        })

    return {"quiz": normalized, "count": len(normalized)}


@app.post("/generate-notes")
async def generate_notes_endpoint(req: SessionRequest):
    """
    Generate detailed exam-ready markdown study notes.
    Notes are cached in session for instant PDF export.
    """
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    all_transcripts = session.get("all_transcripts", [session.get("transcript", "")])
    combined_transcript = "\n\n".join([t for t in all_transcripts if t])

    notes = generate_notes(combined_transcript)
    session["cached_notes"] = notes
    return {"notes": notes}


@app.post("/generate-flashcards")
async def generate_flashcards_endpoint(req: SessionRequest):
    """Generate 8 concept/answer flashcards from the video transcript."""
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    flashcards = generate_flashcards(session["transcript"])
    return {"flashcards": flashcards}


@app.post("/generate-study-plan")
async def generate_study_plan(req: ExamRequest):
    """
    Feature 3 — Study Plan Generator endpoint.
    Input: topic + hours available before exam.
    Output: structured plan array.
    """
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if req.hours <= 0:
        raise HTTPException(status_code=400, detail="Hours must be greater than 0")

    all_transcripts = session.get("all_transcripts", [session.get("transcript", "")])
    combined_transcript = "\n\n".join([t for t in all_transcripts if t])

    plan_payload = generate_exam_plan(combined_transcript, req.topic, req.hours)
    sections = plan_payload.get("sections") if isinstance(plan_payload, dict) else []

    normalized_plan = []
    for sec in sections:
        if not isinstance(sec, dict):
            continue
        normalized_plan.append({
            "section": sec.get("title", "Untitled"),
            "duration_mins": int(sec.get("duration_mins", 0)),
            "concepts": sec.get("concepts", []),
            "checkpoint_question": sec.get("checkpoint_question", ""),
        })

    session["last_study_plan"] = normalized_plan

    return {
        "plan": normalized_plan,
        "quick_tips": plan_payload.get("quick_tips", []),
        "total_minutes": plan_payload.get("total_minutes", int(req.hours * 60)),
    }


@app.post("/exam-mode")
async def exam_mode(req: ExamRequest):
    """
    Backward-compatible alias to /generate-study-plan.
    """
    return await generate_study_plan(req)


@app.get("/transcript/{session_id}")
async def get_transcript_endpoint(session_id: str):
    """Return the raw timestamped transcript for the Transcript tab."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    raw = session["raw_transcript"]
    serialized = []
    for item in raw:
        if isinstance(item, dict):
            serialized.append({
                "text":  item.get("text", ""),
                "start": item.get("start", 0),
            })
        else:
            serialized.append({
                "text":  getattr(item, "text", str(item)),
                "start": getattr(item, "start", 0),
            })

    return {
        "transcript":        serialized,
        "detected_language": session.get("detected_language", "en"),
        "translated":        session.get("detected_language", "en") != "en",
    }


@app.post("/export-chat")
async def export_chat(req: ExportRequest):
    """
    Export full chat history as formatted plain text.
    Uses the helper from youtube_chatbot for LangChain message history,
    plus the richer session chat_history for tags (simplified, outside_video, timestamps).
    """
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    history = session["chat_history"]
    if not history:
        raise HTTPException(status_code=400, detail="No chat history to export yet")

    lines = [
        "VidMind AI — Chat Export",
        "=" * 40,
        f"Video ID : {session['video_id']}",
        f"Language : {session.get('detected_language', 'en').upper()}",
        "",
    ]

    for msg in history:
        if msg["role"] == "user":
            lines.append(f"You:\n{msg['content']}\n")
        else:
            tags = []
            if msg.get("simplified"):    tags.append("Simplified")
            if msg.get("outside_video"): tags.append("General Knowledge")
            tag_str = f" [{', '.join(tags)}]" if tags else ""
            lines.append(f"VidMind AI{tag_str}:\n{msg['content']}")
            if msg.get("timestamps"):
                ts_labels = ", ".join(
                    t.get("timestamp", t.get("label", "")) for t in msg["timestamps"]
                )
                lines.append(f"  Referenced timestamps: {ts_labels}")
            lines.append("")

    return {"content": "\n".join(lines)}


# ── Feature 5 — Cross-Video Playlist Intelligence ─────────────────

@app.post("/add-video")
async def add_video(req: AddVideoRequest):
    """
    Feature 5 — Add a second/third video to an existing session.
    Builds a new FAISS index for the new video and merges it into
    the session's cross-video store. Frontend shows loaded videos as chips.
    """
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    video_id = get_video_id(req.url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")

    # Prevent duplicate video loads within the same session
    from youtube_chatbot import cross_video_store
    existing = cross_video_store.get(req.session_id, {})
    if video_id in existing:
        raise HTTPException(
            status_code=409,
            detail="This video is already loaded in the current session."
        )

    try:
        transcript, timestamped_chunks, raw_transcript, detected_language = \
            get_transcript_with_timestamps(video_id)
    except RuntimeError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Auto-label as "Video N" where N = next video number
    video_count = session["video_count"] + 1
    session["video_count"] = video_count
    label = f"Video {video_count}"

    add_video_to_session(
        session_id=req.session_id,
        video_id=video_id,
        transcript=transcript,
        chunks=timestamped_chunks,
        label=label,
    )

    # Keep cross-video transcript context updated for quiz/notes/exam
    if "all_transcripts" not in session:
        session["all_transcripts"] = [session.get("transcript", "")]
    if "all_raw_transcripts" not in session:
        session["all_raw_transcripts"] = [session.get("raw_transcript", [])]

    session["all_transcripts"].append(transcript)
    session["all_raw_transcripts"].append(raw_transcript)

    mini_summary = generate_summary(transcript)
    session.setdefault("video_summaries", []).append(mini_summary)

    return {
        "video_id":   video_id,
        "label":      label,
        "title":      mini_summary.get("title", label),
        "difficulty": mini_summary.get("difficulty", "Intermediate"),
        "message":    f"{label} loaded and indexed for cross-video queries.",
    }


@app.post("/cross-video-chat")
async def cross_video_chat(req: CrossVideoRequest):
    """
    Feature 5 — Cross-Video Playlist Intelligence.
    Searches across ALL video indexes loaded in the session simultaneously.
    Answer includes per-source attribution ('According to Video 2 at 7:15...').
    """
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    from youtube_chatbot import cross_video_store
    loaded_videos = cross_video_store.get(req.session_id, {})
    if len(loaded_videos) < 2:
        raise HTTPException(
            status_code=400,
            detail="Cross-video chat requires at least 2 videos loaded. "
                   "Use /add-video to load more videos."
        )

    result = answer_cross_video(session_id=req.session_id, question=req.question)

    # Store in chat history for export
    session["chat_history"].append({"role": "user", "content": req.question})
    session["chat_history"].append({
        "role":          "assistant",
        "content":       result["answer"],
        "timestamps":    result.get("sources", []),
        "outside_video": False,
        "simplified":    False,
        "source_label":  "🎬 Cross-video answer",
    })

    # Convert sources into citation array for strict frontend contract
    citations = []
    for s in result.get("sources", []):
        citations.append({
            "video": s.get("label", ""),
            "time": s.get("timestamp", ""),
            "seconds": int(s.get("time", 0)),
            "video_id": s.get("video_id", "")
        })

    return {
        "answer":    result["answer"],
        "citations": citations,
        "mode":      "cross_video",
    }


# ── PDF EXPORT — AI NOTES ─────────────────────────────────────────

@app.post("/export-notes-pdf")
async def export_notes_pdf(req: SessionRequest):
    """
    Export AI-generated study notes as a downloadable PDF.
    Uses cached notes if available, otherwise generates fresh.

    NOTE: generate_pdf_from_notes in youtube_chatbot.py only accepts
    notes_text (no title param). Title is prepended manually here.
    """
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        notes = session.get("cached_notes")
        if not notes:
            notes = generate_notes(session["transcript"])
            session["cached_notes"] = notes

        if not notes or notes.strip() == "":
            raise HTTPException(status_code=400, detail="No notes available to export")

        video_title = session.get("summary", {}).get("title", "Study Notes")

        # Prepend title as a header line so it appears in the PDF
        notes_with_title = f"{video_title}\n{'=' * len(video_title)}\n\n{notes}"

        pdf_bytes = generate_pdf_from_notes(notes_with_title)

        if not pdf_bytes or len(pdf_bytes) == 0:
            raise HTTPException(status_code=500, detail="Failed to generate PDF")

        safe_title = video_title.replace(" ", "_").replace("/", "_")
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{safe_title}_notes.pdf"',
                "Content-Length": str(len(pdf_bytes)),
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")


# ── PDF EXPORT — USER NOTES ───────────────────────────────────────

@app.post("/export-custom-notes-pdf")
async def export_custom_notes_pdf(req: CustomNotesPDFRequest):
    """
    Export user-written notes as a downloadable PDF.
    Accepts plain text or markdown. No session required.
    """
    notes_text = req.notes_text.strip()
    if not notes_text:
        raise HTTPException(status_code=400, detail="Notes content is empty")

    try:
        title = req.title or "My Study Notes"
        notes_with_title = f"{title}\n{'=' * len(title)}\n\n{notes_text}"

        pdf_bytes = generate_pdf_from_notes(notes_with_title)

        if not pdf_bytes or len(pdf_bytes) == 0:
            raise HTTPException(status_code=500, detail="Failed to generate PDF")

        safe_title = title.replace(" ", "_").replace("/", "_")
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{safe_title}.pdf"',
                "Content-Length": str(len(pdf_bytes)),
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")


# ── STATIC FILES & FRONTEND ───────────────────────────────────────

static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    index_path = os.path.join(static_dir, "index.html")
    if not os.path.exists(index_path):
        return HTMLResponse(
            "<h2>VidMind AI backend is running. "
            "Place your index.html in the static/ folder.</h2>"
        )
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)