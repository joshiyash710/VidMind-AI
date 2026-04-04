<div align="center">

# 🎥 VidMind AI

### *Turn any YouTube video into an intelligent conversation*

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-🦜-1C3C3C?style=for-the-badge)](https://langchain.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-blue?style=for-the-badge)](https://faiss.ai)

<br/>

> **Paste a YouTube URL. Ask anything. Get instant, context-aware answers — grounded 100% in the video.**

<br/>

</div>

---

## ✨ What is VidMind AI?

**VidMind AI** is an intelligent YouTube video Q&A chatbot powered by **Retrieval-Augmented Generation (RAG)**. It automatically fetches a video's transcript, embeds it into a vector store, and lets you have a natural, memory-aware conversation about the video content — all powered by OpenAI's GPT-4o-mini.

Whether you want to extract key insights from a lecture, get a quick summary of a long tutorial, or deep-dive into a documentary without watching the whole thing — VidMind AI has you covered.

---

## 🚀 Features

| Feature | Description |
|---|---|
| 🔗 **YouTube Transcript Extraction** | Auto-fetches transcripts directly from any YouTube URL |
| 🧠 **RAG-Powered Q&A** | Retrieves the most relevant transcript chunks before answering |
| 💬 **Conversation Memory** | Maintains full chat history within a session — ask follow-up questions naturally |
| ⚡ **Dual Frontend Support** | Run as a sleek **Streamlit** app or a full **FastAPI + HTML** web app |
| 🔍 **FAISS Vector Search** | Semantically searches the transcript for the most relevant context |
| 🎯 **Grounded Answers** | Responses are strictly based on the video transcript — no hallucinations |

---

## 🏗️ Architecture

```
YouTube URL
     │
     ▼
┌─────────────────────┐
│  YouTube Transcript  │  ← youtube-transcript-api
│       API            │
└────────┬────────────┘
         │  Raw transcript text
         ▼
┌─────────────────────┐
│  Text Splitter       │  ← RecursiveCharacterTextSplitter
│  (chunks: 1000 tok)  │     chunk_overlap: 200
└────────┬────────────┘
         │  Document chunks
         ▼
┌─────────────────────┐
│  OpenAI Embeddings   │  ← text-embedding-ada-002
│  + FAISS Vectorstore │
└────────┬────────────┘
         │  Semantic retrieval (top-4 chunks)
         ▼
┌──────────────────────────────┐
│  LangChain RAG Chain          │
│  ┌────────────────────────┐  │
│  │  ChatPromptTemplate    │  │  ← System: "Answer ONLY using transcript"
│  │  + MessagesPlaceholder │  │  ← Full conversation history injected
│  └────────────┬───────────┘  │
│               │               │
│  ┌────────────▼───────────┐  │
│  │  GPT-4o-mini (LLM)     │  │
│  └────────────────────────┘  │
└──────────────┬───────────────┘
               │  Answer
               ▼
         User Interface
      (Streamlit / FastAPI)
```

---

## 📁 Project Structure

```
VidMind-AI/
│
├── youtube_chatbot.py   # 🧠 Core RAG engine — transcript fetch, embeddings, chain builder
├── app.py               # 🎨 Streamlit frontend (simple, standalone UI)
├── server.py            # ⚡ FastAPI backend — REST API with session management
├── vidmind.html         # 🌐 Custom HTML frontend (served via FastAPI)
├── static/              # 📂 Static assets for the web frontend
├── requirements.txt     # 📦 Python dependencies
└── .gitignore
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | OpenAI GPT-4o-mini |
| **Embeddings** | OpenAI `text-embedding-ada-002` |
| **Orchestration** | LangChain + LangChain Core |
| **Vector Store** | FAISS (Facebook AI Similarity Search) |
| **Transcript** | `youtube-transcript-api` |
| **UI (Option A)** | Streamlit |
| **UI (Option B)** | FastAPI + Vanilla HTML/CSS/JS |
| **Memory** | `RunnableWithMessageHistory` (in-memory) |

---

## ⚙️ Getting Started

### Prerequisites

- Python 3.9+
- An [OpenAI API Key](https://platform.openai.com/api-keys)

### 1. Clone the Repository

```bash
git clone https://github.com/joshiyash710/VidMind-AI.git
cd VidMind-AI
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The `requirements.txt` includes LangChain, FAISS, HuggingFace tools, and more. Also install these packages separately:
> ```bash
> pip install openai langchain-openai youtube-transcript-api fastapi uvicorn streamlit python-dotenv
> ```

### 4. Set Up Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

---

## ▶️ Running the App

### Option A — Streamlit UI (Quickest Start)

```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501`

### Option B — FastAPI + HTML Web App

```bash
python server.py
```

Then open `http://127.0.0.1:8000` in your browser.

The FastAPI backend exposes two endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/api/load` | `POST` | Load transcript from a YouTube URL |
| `/api/chat` | `POST` | Send a question and receive an AI answer |

---

## 💡 How to Use

1. **Paste** any YouTube video URL into the input field
2. Click **"Load Transcript"** — the AI will fetch and index the video content
3. **Ask anything** about the video in the chat box
4. Get instant, grounded answers with full **conversation memory**

> 🎯 *Example questions you can ask:*
> - *"What is the main argument of this video?"*
> - *"Summarize the key points mentioned after the 10-minute mark."*
> - *"What did the speaker say about machine learning?"*

---

## 🧩 Core Module — `youtube_chatbot.py`

This is the brain of VidMind AI. Here's what it does:

- **`get_video_id(url)`** — Parses a YouTube URL and extracts the video ID
- **`get_transcript(video_id)`** — Fetches the English transcript using `YouTubeTranscriptApi`
- **`InMemoryHistory`** — A custom `BaseChatMessageHistory` implementation that stores conversation turns per session
- **`build_chain_with_history(transcript)`** — The main builder that:
  1. Splits the transcript into overlapping chunks
  2. Embeds them with OpenAI embeddings into a FAISS index
  3. Builds a `RunnableParallel` chain that retrieves context + injects history
  4. Wraps everything in `RunnableWithMessageHistory` for session-aware conversations

---

## 🔒 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | ✅ Yes | Your OpenAI API key for GPT-4o-mini and embeddings |

---

## 🗺️ Roadmap

- [ ] Support for multilingual transcripts
- [ ] Timestamp-aware answers (cite the exact moment in the video)
- [ ] Multi-video comparison mode
- [ ] Persistent vector store (save and reload sessions)
- [ ] Export Q&A as PDF/Markdown
- [ ] Deploy to Streamlit Cloud / Railway / Render

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

<div align="center">

**Built by [Yash Joshi & Bansi Kanani(ReACTors)](https://github.com/joshiyash710)**

*If you found this useful, please consider giving it a ⭐ on GitHub!*

</div>
