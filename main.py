import os
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv

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

class ExamRequest(BaseModel):
    session_id: str
    topic: str
    hours: float

class ExportRequest(BaseModel):
    session_id: str

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
    - Auto-generates summary
    - Returns session_id + summary data + detected language
    """
    video_id = get_video_id(req.url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL. Please check the URL and try again.")

    try:
        transcript, timestamped_chunks, raw_transcript, detected_language = \
            get_transcript_with_timestamps(video_id)
    except RuntimeError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Build RAG chain (this also builds FAISS indexes — takes 10-30s)
    chain, get_timestamps_fn = build_rag_chain(transcript, timestamped_chunks)

    # Auto-generate summary on load
    summary_data = generate_summary(transcript)

    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "chain":             chain,
        "get_timestamps":    get_timestamps_fn,
        "transcript":        transcript,
        "raw_transcript":    raw_transcript,
        "video_id":          video_id,
        "detected_language": detected_language,
        "chat_history":      [],
        "summary":           summary_data,
    }

    # Build chapters for sidebar from timestamped chunks
    chapters = []
    for chunk in timestamped_chunks:
        ts = chunk["start_timestamp"]
        m, s = divmod(int(ts), 60)
        h, m_val = divmod(m, 60)
        label = f"{h}:{m_val:02d}:{s:02d}" if h else f"{m}:{s:02d}"
        # First ~8 words as chapter name
        words = chunk["text"].split()[:8]
        name = " ".join(words)
        if len(chunk["text"].split()) > 8:
            name += "…"
        chapters.append({"time": ts, "label": label, "name": name})

    return {
        "session_id":        session_id,
        "video_id":          video_id,
        "detected_language": detected_language,
        "translated":        detected_language != "en",
        "chapters":          chapters,
        **summary_data,
    }


@app.post("/chat")
async def chat(req: ChatRequest):
    """
    RAG chat with:
    - Hybrid RAG (video-grounded + world knowledge fallback)
    - Confusion detection (auto-simplifies if score >= 4)
    - Timestamp extraction for every answer
    - Full chat history for export
    """
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Please load a video first."
        )

    # ── Confusion detection (lightweight GPT call) ────────────────
    confusion    = detect_confusion(req.question)
    is_confused  = confusion.get("score", 1) >= 4

    # Prepend confusion flag so the RAG prompt handles it
    question = f"[STUDENT IS CONFUSED] {req.question}" if is_confused else req.question

    chain          = session["chain"]
    get_timestamps = session["get_timestamps"]

    answer = chain.invoke(
        {"question": question},
        config={"configurable": {"session_id": req.session_id}},
    )

    # ── Hybrid RAG — detect outside-video answers ─────────────────
    outside_video = answer.startswith("[OUTSIDE VIDEO]")
    if outside_video:
        answer = answer.replace("[OUTSIDE VIDEO]", "").strip()

    # ── Timestamp retrieval ───────────────────────────────────────
    timestamps = get_timestamps(req.question)

    # ── Store turn for export ─────────────────────────────────────
    session["chat_history"].append({
        "role":    "user",
        "content": req.question,
    })
    session["chat_history"].append({
        "role":          "assistant",
        "content":       answer,
        "timestamps":    timestamps,
        "outside_video": outside_video,
        "simplified":    is_confused,
    })

    return {
        "answer":        answer,
        "timestamps":    timestamps,
        "outside_video": outside_video,
        "simplified":    is_confused,
    }


@app.post("/generate-quiz")
async def generate_quiz_endpoint(req: SessionRequest):
    """Generate 5 MCQs with answers and explanations from the video transcript."""
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    quiz = generate_quiz(session["transcript"])
    return {"quiz": quiz}


@app.post("/generate-notes")
async def generate_notes_endpoint(req: SessionRequest):
    """Generate structured markdown study notes from the video transcript."""
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    notes = generate_notes(session["transcript"])
    return {"notes": notes}


@app.post("/generate-flashcards")
async def generate_flashcards_endpoint(req: SessionRequest):
    """Generate 8 concept/answer flashcards from the video transcript."""
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    flashcards = generate_flashcards(session["transcript"])
    return {"flashcards": flashcards}


@app.post("/exam-mode")
async def exam_mode(req: ExamRequest):
    """
    Generate a personalized study plan.
    Input: topic + hours available before exam.
    Output: time-boxed sections with concepts and checkpoint questions.
    """
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if req.hours <= 0:
        raise HTTPException(status_code=400, detail="Hours must be greater than 0")
    plan = generate_exam_plan(
        session["transcript"],
        req.topic,
        req.hours,
    )
    return {"plan": plan}


@app.get("/transcript/{session_id}")
async def get_transcript_endpoint(session_id: str):
    """Return the raw timestamped transcript for the Transcript tab."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Convert raw transcript items to plain dicts for JSON serialization
    raw = session["raw_transcript"]
    serialized = []
    for item in raw:
        if isinstance(item, dict):
            serialized.append({"text": item.get("text", ""), "start": item.get("start", 0)})
        else:
            serialized.append({"text": getattr(item, "text", str(item)), "start": getattr(item, "start", 0)})

    return {
        "transcript":        serialized,
        "detected_language": session.get("detected_language", "en"),
        "translated":        session.get("detected_language", "en") != "en",
    }


@app.post("/export-chat")
async def export_chat(req: ExportRequest):
    """Export the full chat history as formatted plain text."""
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
                ts = ", ".join(t["timestamp"] for t in msg["timestamps"])
                lines.append(f"  Referenced timestamps: {ts}")
            lines.append("")

    return {"content": "\n".join(lines)}


# ── STATIC FILES & FRONTEND ───────────────────────────────────────
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    index_path = os.path.join(static_dir, "index.html")
    if not os.path.exists(index_path):
        return HTMLResponse("<h2>VidMind AI backend is running. Place your index.html in the static/ folder.</h2>")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)