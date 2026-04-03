# ================== IMPORTS ==================
import os
import json
import re
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlparse, parse_qs

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from youtube_transcript_api import YouTubeTranscriptApi

# ================== ENV ==================
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
assert OPENAI_API_KEY, "OPENAI_API_KEY not set"

# ================== LLM ==================
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, api_key=OPENAI_API_KEY)
llm_precise = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2, api_key=OPENAI_API_KEY)

# ================== UTILS ==================
def get_video_id(url: str) -> Optional[str]:
    parsed = urlparse(url)
    if parsed.hostname == "youtu.be":
        return parsed.path.lstrip("/")
    return parse_qs(parsed.query).get("v", [None])[0]

def _format_time(seconds: int) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

# ================== TRANSCRIPT ==================
def get_transcript_with_timestamps(video_id: str):
    try:
        ytt = YouTubeTranscriptApi()
        transcript_list = ytt.list(video_id)

        fetched = None

        # Try English manual
        try:
            fetched = transcript_list.find_manually_created_transcript(["en"])
        except:
            pass

        # Try English auto
        if not fetched:
            try:
                fetched = transcript_list.find_generated_transcript(["en"])
            except:
                pass

        # Try any language
        if not fetched:
            try:
                available = [t.language_code for t in transcript_list]
                fetched = transcript_list.find_generated_transcript(available)
            except:
                pass

        if not fetched:
            raise Exception("No transcript available")

        raw = fetched.fetch()

        text = " ".join([item.text for item in raw])

        chunks = []
        current = []
        start = 0

        for item in raw:
            if not current:
                start = int(item.start)

            current.append(item.text)

            if item.start - start >= 30:
                chunks.append({
                    "timestamp": _format_time(start),
                    "start_timestamp": start,
                    "text": " ".join(current),
                    "video_id": video_id
                })
                current = []

        if current:
            chunks.append({
                "timestamp": _format_time(start),
                "start_timestamp": start,
                "text": " ".join(current),
                "video_id": video_id
            })

        return text, chunks, raw, "en"

    except Exception as e:
        raise RuntimeError(f"Transcript error: {str(e)}")


# ================== QUIZ ==================
def generate_quiz(transcript: str, section: Optional[str] = None, topic: Optional[str] = None):
    # NOTE: All literal { } in the JSON schema inside the prompt are escaped as {{ }}
    # to prevent LangChain from treating them as template variables.

    topic_prompt = ""
    if topic:
        topic_prompt = (
            f"Exam Topic: {topic.strip()}\n"
            "Create quiz questions strictly on this topic, using transcript context only.\n"
        )

    section_prompt = ""
    if section:
        section_prompt = (
            f"Focus Area: {section.strip()}\n"
            "Generate questions primarily about this area while still using the transcript context.\n"
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert exam paper setter for technical content.

Generate EXACTLY 5 high-quality MCQs or short-answer questions based ONLY on transcript content.

STRICT RULES:
- NO generic questions
- NO "what is the video about" type questions
- Prefer concept-checking, scenario-based, and application questions
- If a question cannot have 4 valid options, return it as descriptive with an empty options object
- Always include a brief explanation for the correct answer

Return ONLY valid JSON with no markdown fences:
[
  {{
    "question": "...",
    "options": {{ "A": "...", "B": "...", "C": "...", "D": "..." }},
    "correct": "A",
    "explanation": "..."
  }}
]
"""),
        ("human", f"{topic_prompt}{section_prompt}Transcript:\n{{transcript}}")
    ])

    chain = prompt | llm_precise | StrOutputParser()
    raw = chain.invoke({"transcript": transcript[:12000]})

    try:
        clean = raw.strip().replace("```json", "").replace("```", "")
        data = json.loads(clean)
        return data
    except:
        return []


# ================== NOTES ==================
def generate_notes(transcript: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert academic tutor. Generate structured, exam-ready study notes from the transcript."),
        ("human", "Given the transcript, produce notes with these sections:\n- Topic Overview\n- Key Concepts (bullet list)\n- Step-by-step explanations\n- Example applications\n- Quick review summary\n\nUse concise sentences and clear headings."
        ),
        ("human", "Transcript:\n{transcript}")
    ])

    raw = (prompt | llm | StrOutputParser()).invoke({"transcript": transcript[:12000]})

    if not raw or raw.strip() == "":
        return "Notes could not be generated. Please try again."

    return raw


# ================== PDF ==================
def generate_pdf_from_notes(notes_text: str):
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph
    from io import BytesIO

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    for line in notes_text.split('\n'):
        if line.strip():
            p = Paragraph(line, styles['Normal'])
            story.append(p)

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ================== MEMORY ==================
class InMemoryHistory(BaseChatMessageHistory):
    def __init__(self):
        self._messages = []

    @property
    def messages(self):
        return self._messages

    def add_message(self, message):
        self._messages.append(message)

store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]


# ================== SUMMARY (FIXED) ==================
# BUG FIX: The original prompt used raw { } around the JSON schema keys.
# LangChain's ChatPromptTemplate parses anything inside single { } as a
# template input variable. This caused:
#   KeyError: 'Input to ChatPromptTemplate is missing variables {"title"}'
# Solution: Escape ALL literal curly braces in the prompt string as {{ and }}
# so LangChain treats them as plain text, not variable placeholders.

def generate_summary(transcript: str) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert content analyst and learning strategist.

Analyze the transcript and return ONLY valid JSON with no markdown fences:
{{
  "title": "A concise, descriptive title for the video",
  "summary": "A 4-sentence educational summary with main ideas and examples",
  "why_it_matters": "A 1-2 sentence explanation of real-world relevance",
  "key_concepts": ["concept1", "concept2", "concept3", "concept4", "concept5"],
  "difficulty": "Beginner or Intermediate or Advanced"
}}

Return ONLY the JSON object. No extra text. No markdown fences."""),
        ("human", "Transcript:\n{transcript}")
    ])

    chain = prompt | llm_precise | StrOutputParser()
    raw = chain.invoke({"transcript": transcript[:6000]})

    try:
        clean = raw.strip().replace("```json", "").replace("```", "")
        parsed = json.loads(clean)
        if "why_it_matters" not in parsed:
            parsed["why_it_matters"] = "This content is important for mastering related concepts and exam readiness."
        return parsed
    except:
        return {
            "title": "Video Summary",
            "summary": raw[:280],
            "why_it_matters": "This content improves understanding of key concepts and exam preparation.",
            "key_concepts": [],
            "difficulty": "Intermediate"
        }


# ================== FLASHCARDS (FIXED) ==================
def generate_flashcards(transcript: str) -> list:
    # Escaped {{ }} to prevent LangChain template variable parsing
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert teacher.

Create EXACTLY 8 high-quality flashcards from the transcript.

Rules:
- Focus on core concepts only
- Each flashcard should help in revision
- Keep front concise, back explanatory

Return ONLY valid JSON with no markdown fences:
[
  {{"front": "Question or concept", "back": "Clear explanation"}},
  {{"front": "Question or concept", "back": "Clear explanation"}}
]
"""),
        ("human", "Transcript:\n{transcript}")
    ])

    chain = prompt | llm_precise | StrOutputParser()
    raw = chain.invoke({"transcript": transcript[:6000]})

    try:
        clean = raw.strip().replace("```json", "").replace("```", "")
        data = json.loads(clean)
        return data if isinstance(data, list) else []
    except:
        return []


# ================== EXAM PLAN (FIXED) ==================
def generate_exam_plan(transcript: str, topic: str, hours: float) -> dict:
    # Escaped {{ }} to prevent LangChain template variable parsing
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert exam strategist.

A student has limited time before an exam. Create a HIGHLY OPTIMIZED study plan based on the transcript.

Rules:
- Focus only on high-weight, exam-relevant concepts
- Break time into structured sections
- Include revision checkpoints
- Keep it practical and actionable

Return ONLY valid JSON with no markdown fences:
{{
  "total_minutes": 120,
  "sections": [
    {{
      "title": "Topic Name",
      "duration_mins": 20,
      "concepts": ["concept1", "concept2"],
      "checkpoint_question": "Quick test question to verify understanding"
    }}
  ],
  "quick_tips": ["tip1", "tip2", "tip3"]
}}
"""),
        ("human", "Topic: {topic}\nHours Available: {hours}\n\nTranscript:\n{transcript}")
    ])

    chain = prompt | llm_precise | StrOutputParser()

    raw = chain.invoke({
        "transcript": transcript[:6000],
        "topic": topic,
        "hours": str(hours)
    })

    try:
        clean = raw.strip().replace("```json", "").replace("```", "")
        data = json.loads(clean)

        if "total_minutes" not in data:
            data["total_minutes"] = int(hours * 60)

        return data

    except:
        return {
            "total_minutes": int(hours * 60),
            "sections": [],
            "quick_tips": []
        }


# ================== CONFUSION DETECTOR (FIXED) ==================
def detect_confusion(question: str) -> dict:
    # Escaped {{ }} to prevent LangChain template variable parsing
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an AI assistant that detects how confused a student is based on their question.

Rate confusion level from 1 to 5:
1 = clear and confident
3 = somewhat confused
5 = very confused

Consider:
- vague or unclear phrasing
- very short queries like "what?", "huh?"
- repeated questions
- explicit confusion ("I don't understand")
- filler-heavy phrasing ("I don't really get... like... how does it work?")
- contradictory follow-ups ("But you said X, so why is it Y?")

Return ONLY valid JSON with no markdown fences:
{{
  "score": 3,
  "reason": "short explanation of detected confusion signal"
}}
"""),
        ("human", "Student question: {question}")
    ])

    chain = prompt | llm_precise | StrOutputParser()
    raw = chain.invoke({"question": question})

    try:
        clean = raw.strip().replace("```json", "").replace("```", "")
        return json.loads(clean)
    except:
        return {"score": 1, "reason": "could not detect"}


# ================== HYBRID RAG ANSWER (NEW - Feature 2) ==================
# Routes between Video-Grounded mode and World Knowledge mode based on
# retrieval relevance score. If top chunks score below threshold (0.75),
# falls back to GPT general knowledge and tags the response accordingly.

def answer_with_hybrid_rag(
    question: str,
    retriever,
    session_id: str,
    confused: bool = False
) -> dict:
    """
    Returns a dict with keys:
      answer      - str: the generated answer
      mode        - str: 'video_grounded' or 'world_knowledge'
      source_label - str: shown in UI to indicate answer source
    """
    # Retrieve top chunks WITH scores to decide routing
    try:
        docs_and_scores = retriever.vectorstore.similarity_search_with_score(question, k=4)
    except Exception:
        # Fallback if vectorstore attribute not accessible
        docs_and_scores = []

    RELEVANCE_THRESHOLD = 0.75
    use_world_knowledge = False

    if docs_and_scores:
        # FAISS returns L2 distance — lower is more similar.
        # Convert to a 0-1 similarity: sim = 1 / (1 + distance)
        best_distance = min(score for _, score in docs_and_scores)
        similarity = 1.0 / (1.0 + best_distance)
        if similarity < RELEVANCE_THRESHOLD:
            use_world_knowledge = True
    else:
        use_world_knowledge = True

    if use_world_knowledge:
        # World Knowledge mode — answer from GPT general knowledge
        wk_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful tutor. Answer the student's question using your general knowledge. "
             "Be clear, concise, and educational."),
            ("human", "{question}")
        ])
        chain = wk_prompt | llm | StrOutputParser()
        answer = chain.invoke({"question": question})
        return {
            "answer": answer,
            "mode": "world_knowledge",
            "source_label": "⚠️ Not covered in this video — general answer"
        }
    else:
        # Video-Grounded mode — answer from retrieved transcript chunks
        context = "\n".join([doc.page_content for doc, _ in docs_and_scores])

        confusion_instruction = ""
        if confused:
            confusion_instruction = (
                "\n\nIMPORTANT: The student seems confused. "
                "Start with a simple real-world analogy, then give the technical explanation."
            )

        vg_prompt = ChatPromptTemplate.from_messages([
            ("system",
             f"Answer ONLY from the provided transcript context. "
             f"If the answer is not in the context, say so.{confusion_instruction}"),
            ("human", "Context:\n{context}\n\nQuestion: {question}")
        ])
        chain = vg_prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})
        return {
            "answer": answer,
            "mode": "video_grounded",
            "source_label": "📹 Answered from video"
        }


# ================== RAG ==================
def build_rag_chain(transcript, chunks, video_label: str = "Video 1"):
    """
    Builds the RAG chain for a single video.
    video_label is used for cross-video source attribution (Feature 5).
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    docs = splitter.create_documents([transcript])

    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Attach vectorstore reference for hybrid scoring
    retriever.vectorstore = vectorstore

    # Build timestamp index from chunks
    if chunks:
        ts_docs = splitter.create_documents(
            [c["text"] for c in chunks],
            metadatas=[{**c, "source_label": video_label} for c in chunks]
        )
        ts_vectorstore = FAISS.from_documents(ts_docs, embeddings)
        ts_retriever = ts_vectorstore.as_retriever(search_kwargs={"k": 3})
    else:
        ts_retriever = None

    def get_timestamps(question: str) -> list:
        """Extract timestamps for relevant sections of the video."""
        if not ts_retriever:
            return []
        try:
            results = ts_retriever.invoke(question)
            timestamps = []
            seen = set()
            for doc in results:
                ts = doc.metadata.get("timestamp", "")
                time = doc.metadata.get("start_timestamp", 0)
                vid = doc.metadata.get("video_id", "")
                label = doc.metadata.get("source_label", video_label)
                if ts and ts not in seen:
                    seen.add(ts)
                    timestamps.append({
                        "timestamp": ts,
                        "time": time,
                        "label": f"{label} @ {ts}",
                        "video_id": vid,
                    })
            return timestamps
        except:
            return []

    # Standard RAG chain with chat history (used for simple /chat endpoint)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer from transcript only. Be concise and educational."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])

    chain = (
        RunnableParallel({
            "context": lambda x: "\n".join(
                [d.page_content for d in retriever.invoke(x["question"])]
            ),
            "question": lambda x: x["question"],
            "history": lambda x: x["history"],
        })
        | prompt
        | llm
        | StrOutputParser()
    )

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )

    return chain_with_history, get_timestamps, retriever


# ================== CROSS-VIDEO SESSION STORE (Feature 5) ==================
# Tracks multiple FAISS indexes per session for Cross-Video Playlist Intelligence.
# Each entry: { video_id: { "retriever": ..., "label": ..., "get_timestamps": ... } }

cross_video_store: Dict[str, Dict] = {}

def add_video_to_session(session_id: str, video_id: str, transcript: str, chunks: list, label: str):
    """Adds a new video's index to the cross-video store for a session."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    docs = splitter.create_documents([transcript])
    vectorstore = FAISS.from_documents(docs, embeddings)

    if chunks:
        ts_docs = splitter.create_documents(
            [c["text"] for c in chunks],
            metadatas=[{**c, "source_label": label} for c in chunks]
        )
        ts_vectorstore = FAISS.from_documents(ts_docs, embeddings)
    else:
        ts_vectorstore = None

    if session_id not in cross_video_store:
        cross_video_store[session_id] = {}

    cross_video_store[session_id][video_id] = {
        "vectorstore": vectorstore,
        "ts_vectorstore": ts_vectorstore,
        "label": label,
        "video_id": video_id,
    }


def answer_cross_video(session_id: str, question: str) -> dict:
    """
    Feature 5 — Cross-Video Playlist Intelligence.
    Searches across ALL video indexes in the session simultaneously.
    Returns answer with per-source attribution (which video + timestamp).
    """
    if session_id not in cross_video_store or not cross_video_store[session_id]:
        return {
            "answer": "No videos loaded for this session.",
            "sources": []
        }

    all_docs = []
    for vid_id, entry in cross_video_store[session_id].items():
        try:
            results = entry["vectorstore"].similarity_search(question, k=2)
            for doc in results:
                doc.metadata["source_label"] = entry["label"]
                doc.metadata["video_id"] = vid_id
            all_docs.extend(results)
        except:
            continue

    if not all_docs:
        return {"answer": "Could not retrieve relevant content.", "sources": []}

    # Build context with source attribution
    context_parts = []
    for doc in all_docs:
        label = doc.metadata.get("source_label", "Unknown Video")
        context_parts.append(f"[{label}]: {doc.page_content}")
    context = "\n\n".join(context_parts)

    # Gather timestamps from ts_vectorstores
    sources = []
    seen_ts = set()
    for vid_id, entry in cross_video_store[session_id].items():
        if entry.get("ts_vectorstore"):
            try:
                ts_results = entry["ts_vectorstore"].similarity_search(question, k=2)
                for doc in ts_results:
                    ts = doc.metadata.get("timestamp", "")
                    if ts and (vid_id, ts) not in seen_ts:
                        seen_ts.add((vid_id, ts))
                        sources.append({
                            "label": entry["label"],
                            "timestamp": ts,
                            "time": doc.metadata.get("start_timestamp", 0),
                            "video_id": vid_id
                        })
            except:
                continue

    cv_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert tutor synthesizing information from multiple video transcripts. "
         "Each piece of context is labeled with its source video. "
         "Provide a clear, comprehensive answer that integrates information from all relevant videos. "
         "Structure your response logically: start with a direct answer, then provide supporting details, "
         "and cite sources using 'According to [Video Label]...' format. "
         "If videos have conflicting information, note the differences. "
         "Be educational and easy to understand."),
        ("human", "Context from multiple videos:\n{context}\n\nQuestion: {question}")
    ])
    chain = cv_prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    return {"answer": answer, "sources": sources}


# ================== CHAT HISTORY EXPORT (Supporting Feature) ==================
def export_chat_history(session_id: str) -> str:
    """
    Exports the full conversation history as a formatted .txt string
    with per-message timestamps referenced.
    """
    history = get_session_history(session_id)
    if not history.messages:
        return "No conversation history found for this session."

    lines = [f"=== VidMind AI — Chat Export ===", f"Session: {session_id}", ""]
    for i, msg in enumerate(history.messages):
        role = "You" if msg.type == "human" else "VidMind AI"
        lines.append(f"[{role}]")
        lines.append(str(msg.content))
        lines.append("")

    return "\n".join(lines)