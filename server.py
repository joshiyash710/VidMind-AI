from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os

# Import the existing backend logic without modifying it
from youtube_chatbot import get_video_id, get_transcript, build_chain_with_history

app = FastAPI(title="YouTube GenAI Chat")

# We will mount static files for the frontend, but let's make sure the directory exists
# Mount the static directory
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

app.mount("/static", StaticFiles(directory=static_dir), name="static")

# In-memory store for chains (simulating session state)
chains = {}

class LoadRequest(BaseModel):
    url: str
    session_id: str

class ChatRequest(BaseModel):
    question: str
    session_id: str

@app.get("/", response_class=HTMLResponse)
async def read_root():
    # Serve the index.html from static folder
    index_path = os.path.join(static_dir, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/load")
def load_transcript(req: LoadRequest):
    print(f"\\n--- New Request Received ---", flush=True)
    print(f"URL: {req.url}", flush=True)
    try:
        video_id = get_video_id(req.url)
        print(f"1. Extracted Video ID: {video_id}", flush=True)
        if not video_id:
            raise HTTPException(status_code=400, detail="Invalid YouTube URL")

        print("2. Fetching Transcript (this may take a moment)...", flush=True)
        transcript = get_transcript(video_id)
        print(f"   -> Fetched {len(transcript)} characters.", flush=True)
        
        print("3. Building AI Brain & OpenAI Embeddings (This can take 30s-120s for long videos!)...", flush=True)
        chain = build_chain_with_history(transcript)
        print("   -> Success! Chain built.", flush=True)
        
        # Store the chain for this session
        chains[req.session_id] = chain
        print("--- Request Completed ---", flush=True)
        return {"status": "success", "message": "Transcript loaded and AI prepared!"}
    except Exception as e:
        print(f"*** ERROR Exception Blocked Request: {e}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
def chat(req: ChatRequest):
    if req.session_id not in chains:
        raise HTTPException(status_code=400, detail="Transcript not loaded for this session. Please load a URL first.")
    
    chain = chains[req.session_id]
    
    try:
        # Expected input format by `youtube_chatbot.py`
        answer = chain.invoke(
            {"question": req.question},
            config={"configurable": {"session_id": req.session_id}}
        )
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
