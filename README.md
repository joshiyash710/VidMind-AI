<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=0,2,30&height=220&section=header&text=VidMind%20AI&fontSize=80&fontColor=ffffff&animation=fadeIn&fontAlignY=40&desc=Learn%20What%20Matters.%20Faster.&descAlignY=62&descSize=22" width="100%"/>

<br/>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white"/>
  <img src="https://img.shields.io/badge/FAISS-00BFFF?style=for-the-badge&logo=meta&logoColor=white"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Hackathon-OceanLab_×_CHARUSAT_Hacks_2026-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/Team-ReACTors-crimson?style=flat-square"/>
  <img src="https://img.shields.io/badge/Built_in-48_Hours-orange?style=flat-square"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square"/>
</p>

<br/>

> **Paste a YouTube URL. Get a context-aware AI tutor, instant summaries, smart quizzes, and a full learning workspace — all grounded strictly in the video.**

<br/>

[🧠 What is VidMind?](#-what-is-vidmind-ai) · [✨ Features](#-features) · [🏗️ Architecture](#-architecture) · [⚙️ Setup](#-setup--installation) · [🖥️ Two Interfaces](#-two-ways-to-run) · [📁 Project Structure](#-project-structure) · [🧑‍💻 Team](#-team)

</div>

---

## 🧠 What is VidMind AI?

YouTube has billions of hours of educational content — but passive watching doesn't equal learning. **VidMind AI** transforms any YouTube video into an interactive, AI-powered learning session.

Paste a URL, and VidMind automatically fetches the transcript, chunks and embeds it into a local **FAISS vector store**, then builds a **retrieval-augmented generation (RAG) chain with persistent conversation memory**. The result: an AI that answers questions about the video with pinpoint accuracy, never hallucinating beyond the source content.

Built in **48 hours** at **OceanLab × CHARUSAT Hacks 2026** by Team **ReACTors**.

---

## ✨ Features

| Feature | Description |
|---|---|
| 💬 **Context-Aware Q&A** | Ask anything about the video. The LLM answers *only* using transcript content — zero hallucination. |
| 🧠 **Conversation Memory** | Full multi-turn chat with `RunnableWithMessageHistory` — the AI remembers everything discussed in the session. |
| 🔍 **FAISS Semantic Search** | Transcript is chunked (1000 chars, 200 overlap), embedded with `text-embedding-ada-002`, and stored in FAISS for top-k=4 retrieval. |
| 📊 **Summary & Key Points UI** | Rich learning workspace with summary, chapters, and key concept cards in the custom HTML frontend. |
| 📝 **Smart Notes Panel** | In-app notes editor with toolbar — take notes alongside your video-grounded AI conversation. |
| 🎯 **Interactive Quiz Tab** | Built-in MCQ quiz UI in the frontend for active recall and self-testing. |
| 🖥️ **Dual Interface** | Run as a **Streamlit** app for rapid use, or a full **FastAPI + custom HTML** web app for the complete experience. |
| 🎨 **YouTube-native Design** | Custom dark-mode UI with `DM Sans`, `Bebas Neue`, and a full YouTube-inspired red-on-black design system. |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         VidMind AI — System Flow                     │
│                                                                      │
│   User Input: YouTube URL                                            │
│        │                                                             │
│        ▼                                                             │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │              youtube_chatbot.py  (Core Engine)              │   │
│   │                                                             │   │
│   │  get_video_id()  →  get_transcript()                        │   │
│   │                          │                                  │   │
│   │                   YouTubeTranscriptApi                      │   │
│   │                   (fetches English transcript)              │   │
│   │                          │                                  │   │
│   │            RecursiveCharacterTextSplitter                   │   │
│   │            (chunk_size=1000, overlap=200)                   │   │
│   │                          │                                  │   │
│   │            OpenAIEmbeddings (text-embedding-ada-002)        │   │
│   │                          │                                  │   │
│   │            FAISS VectorStore → Retriever (top-k=4)          │   │
│   │                          │                                  │   │
│   │            RunnableParallel                                 │   │
│   │            ├─ context  → retrieved chunks                   │   │
│   │            ├─ question → passthrough                        │   │
│   │            └─ history  → session memory                     │   │
│   │                          │                                  │   │
│   │            ChatPromptTemplate → GPT-4o-mini                 │   │
│   │                          │                                  │   │
│   │            RunnableWithMessageHistory (InMemoryHistory)     │   │
│   └─────────────────────────────────────────────────────────────┘   │
│              │                              │                        │
│              ▼                              ▼                        │
│   ┌──────────────────┐          ┌──────────────────────────┐        │
│   │    app.py        │          │       server.py           │        │
│   │   (Streamlit)    │          │  (FastAPI + static HTML)  │        │
│   │   Port: 8501     │          │  POST /api/load           │        │
│   └──────────────────┘          │  POST /api/chat           │        │
│                                 │  Port: 8000               │        │
│                                 └──────────────────────────┘        │
│                                              │                       │
│                                   static/index.html                  │
│                                   (Custom YouTube-style UI)          │
└──────────────────────────────────────────────────────────────────────┘
```

### Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | OpenAI `gpt-4o-mini` via `langchain-openai` |
| **Embeddings** | OpenAI `text-embedding-ada-002` |
| **Vector Store** | FAISS (`faiss-cpu`) |
| **RAG Chain** | LangChain `RunnableParallel` + `ChatPromptTemplate` + `StrOutputParser` |
| **Memory** | `RunnableWithMessageHistory` with custom `InMemoryHistory` (per `session_id`) |
| **Transcript API** | `youtube-transcript-api` (English, auto-fetched) |
| **Interface A** | Streamlit (`app.py`) |
| **Interface B** | FastAPI + Uvicorn (`server.py`) serving static HTML/CSS/JS |
| **Frontend UI** | Custom single-file `vidmind.html` / `static/index.html` |
| **Env Management** | `python-dotenv` |

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.10+
- An OpenAI API key

### 1. Clone the Repository

```bash
git clone https://github.com/joshiyash710/VidMind-AI.git
cd VidMind-AI
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

---

## 🖥️ Two Ways to Run

VidMind AI ships with **two fully functional interfaces** — pick whichever suits your workflow.

---

### 🅰️ Option 1 — Streamlit App

The fastest way to get started. Clean, minimal UI for rapid testing.

```bash
streamlit run app.py
```

Open **`http://localhost:8501`** in your browser.

**Workflow:**
1. Paste a YouTube URL in the text input
2. Click **📥 Load Transcript** — VidMind fetches the transcript and builds the FAISS index *(30–120s for longer videos)*
3. Ask questions in the chat input — the AI answers with full session memory

---

### 🅱️ Option 2 — FastAPI + Custom Web UI

Full-stack web application with the polished YouTube-native dark-mode interface.

```bash
python server.py
```

Or with uvicorn directly:

```bash
uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```

Open **`http://localhost:8000`** in your browser.

This serves the complete `static/index.html` frontend featuring:
- 🎬 Landing page with animated hero, URL input, and stats
- 📊 Full 3-column learning dashboard (Summary · Quiz · Notes tabs)
- 💬 Real-time chat panel with quick-prompt chips and typing indicator
- 🗂️ Sidebar with video metadata, thumbnail, and chapter navigation

#### REST API

| Method | Endpoint | Body | Description |
|---|---|---|---|
| `GET` | `/` | — | Serves the HTML frontend |
| `POST` | `/api/load` | `{ "url": "...", "session_id": "..." }` | Fetches transcript, builds FAISS index, stores chain |
| `POST` | `/api/chat` | `{ "question": "...", "session_id": "..." }` | Returns AI answer for the session |

**Example — Load a video:**
```json
POST /api/load
{
  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "session_id": "user-session-001"
}
```

**Example — Chat:**
```json
POST /api/chat
{
  "question": "What is the main topic of this video?",
  "session_id": "user-session-001"
}
```

---

## 📁 Project Structure

```
VidMind-AI/
│
├── youtube_chatbot.py      # ⭐ Core AI engine
│                           #    - get_video_id(), get_transcript()
│                           #    - InMemoryHistory, get_session_history()
│                           #    - build_chain_with_history()
│                           #    - FAISS + OpenAI RAG pipeline
│
├── app.py                  # Interface A: Streamlit UI (59 lines)
│
├── server.py               # Interface B: FastAPI server (83 lines)
│                           #    - GET  /
│                           #    - POST /api/load
│                           #    - POST /api/chat
│
├── vidmind.html            # Standalone UI prototype (1411 lines)
│                           # Full YouTube-style learning dashboard
│
├── static/
│   └── index.html          # Production frontend served by FastAPI
│
├── requirements.txt        # All Python dependencies
├── .gitignore
└── README.md
```

---

## 🔬 How the RAG Pipeline Works

```
1. URL Parsing
   └── get_video_id(url)
       Parses ?v= param from any YouTube URL format

2. Transcript Fetching
   └── get_transcript(video_id)
       YouTubeTranscriptApi().fetch(video_id, languages=["en"])
       Joins all segments into a single text string

3. Chunking
   └── RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
       Creates overlapping Document chunks to preserve context at boundaries

4. Embedding & Indexing
   └── OpenAIEmbeddings(model="text-embedding-ada-002")
       └── FAISS.from_documents(docs, embeddings)
           └── retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

5. Chain Assembly
   └── RunnableParallel({
         "context":  lambda x → format_docs(retriever.invoke(x["question"])),
         "question": lambda x → x["question"],
         "history":  lambda x → x["history"]
       })
       | ChatPromptTemplate([
           ("system", "Answer ONLY using the transcript below.\n\n{context}"),
           MessagesPlaceholder("history"),
           ("human", "{question}")
         ])
       | ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
       | StrOutputParser()

6. Memory Wrapping
   └── RunnableWithMessageHistory(
         chain,
         get_session_history,          # Dict[session_id → InMemoryHistory]
         input_messages_key="question",
         history_messages_key="history"
       )
```

---

## 📦 Full Dependency List

```txt
# LangChain
langchain==1.2.4
langchain-core==1.2.4
langchain-community==0.2.19
langchain-text-splitters==0.3.11
langchain-openai
langchain-huggingface==1.2.0

# Embeddings & Vector Store
huggingface-hub==0.36.0
sentence-transformers
faiss-cpu

# YouTube Transcript
youtube-transcript-api

# Web Frameworks
streamlit
fastapi
uvicorn

# Utilities
python-dotenv
```

---

## 🧑‍💻 Team

<div align="center">

Built with ❤️ by **Team ReACTors** at OceanLab × CHARUSAT Hacks 2026

| | Name | Role | Links |
|---|---|---|---|
| 👨‍💻 | **Yash Hiren Joshi** | Backend · LangChain RAG Pipeline · FastAPI · AI Architecture | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/in/yashjoshi710) [![Portfolio](https://img.shields.io/badge/Portfolio-FF5722?style=flat-square&logo=google-chrome&logoColor=white)](https://joshiyash710.github.io) |
| 👩‍💻 | **Bansi Deepak Kanani** | Frontend · UI/UX Design · Integration | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/in/bansi-kanani) |

*Final-year B.Tech Information Technology students · DEPSTAR, CHARUSAT University, Anand, Gujarat*

</div>

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

```bash
# Fork the repo, then:
git checkout -b feature/your-feature-name
git commit -m "feat: add your feature"
git push origin feature/your-feature-name
# Open a Pull Request
```

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](./LICENSE) file for details.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=0,2,30&height=120&section=footer" width="100%"/>

**VidMind AI** — *Because great learning deserves great tools.*

⭐ If this project helped you, drop a star!

</div>
