import os
import json
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

# ── LangSmith picks up LANGCHAIN_API_KEY from .env automatically ──
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "vidmind-ai")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
assert OPENAI_API_KEY, "OPENAI_API_KEY not set in .env"

# ── LLMs ──────────────────────────────────────────────────────────
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    api_key=OPENAI_API_KEY,
)
llm_precise = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.2,
    api_key=OPENAI_API_KEY,
)

# ── UTILS ─────────────────────────────────────────────────────────
def get_video_id(url: str) -> Optional[str]:
    parsed = urlparse(url)
    if parsed.hostname in ("youtu.be",):
        return parsed.path.lstrip("/")
    return parse_qs(parsed.query).get("v", [None])[0]

def _format_time(seconds: int) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

def get_transcript_with_timestamps(video_id: str) -> Tuple[str, list, list, str]:
    """
    Returns (plain_text, timestamped_chunks, raw_transcript, detected_language)

    Priority order:
    1. Manual English transcript
    2. Auto-generated English transcript
    3. Manual transcript in any language → translated to English
    4. Auto-generated transcript in any language → translated to English
    5. Raises RuntimeError if nothing is available
    """
    try:
        ytt = YouTubeTranscriptApi()
        transcript_list = ytt.list(video_id)

        fetched = None
        detected_language = "en"

        # ── Priority 1: Manual English ────────────────────────────
        try:
            fetched = transcript_list.find_manually_created_transcript(["en"])
            detected_language = "en"
        except Exception:
            pass

        # ── Priority 2: Auto-generated English ───────────────────
        if not fetched:
            try:
                fetched = transcript_list.find_generated_transcript(["en"])
                detected_language = "en"
            except Exception:
                pass

        # ── Priority 3: Manual in any language → translate ────────
        if not fetched:
            try:
                available_codes = [t.language_code for t in transcript_list]
                manual = transcript_list.find_manually_created_transcript(available_codes)
                detected_language = manual.language_code
                try:
                    fetched = manual.translate("en")
                except Exception:
                    fetched = manual
            except Exception:
                pass

        # ── Priority 4: Auto-generated in any language → translate ─
        if not fetched:
            try:
                available_codes = [t.language_code for t in transcript_list]
                generated = transcript_list.find_generated_transcript(available_codes)
                detected_language = generated.language_code
                try:
                    fetched = generated.translate("en")
                except Exception:
                    fetched = generated
            except Exception:
                pass

        # ── Nothing available ─────────────────────────────────────
        if not fetched:
            raise RuntimeError(
                "No transcript available for this video. "
                "The video may have transcripts disabled by the creator."
            )

        # ── Fetch raw data ────────────────────────────────────────
        raw = fetched.fetch()
        
        def _get_val(obj, key):
            if isinstance(obj, dict): return obj.get(key)
            return getattr(obj, key, None)
            
        plain_text = " ".join(str(_get_val(item, "text")) for item in raw)

        # ── Build 30-second timestamped chunks ────────────────────
        timestamped, chunk_text, chunk_start = [], [], 0
        for item in raw:
            i_text = _get_val(item, "text")
            i_start = _get_val(item, "start")
            if i_text is None or i_start is None: continue
            
            if not chunk_text:
                chunk_start = int(i_start)
            chunk_text.append(str(i_text))
            if i_start - chunk_start >= 30:
                timestamped.append({
                    "start_timestamp": chunk_start,
                    "text": " ".join(chunk_text),
                    "video_id": video_id
                })
                chunk_text = []
        if chunk_text:
            timestamped.append({
                "start_timestamp": chunk_start,
                "text": " ".join(chunk_text),
                "video_id": video_id
            })

        return plain_text, timestamped, raw, detected_language

    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Transcript error: {e}")

# ── MEMORY ────────────────────────────────────────────────────────
class InMemoryHistory(BaseChatMessageHistory):
    def __init__(self):
        self._messages: List = []

    @property
    def messages(self): return self._messages

    def add_message(self, message): self._messages.append(message)

    def clear(self): self._messages.clear()

store: Dict[str, BaseChatMessageHistory] = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]

# ── SUMMARY CHAIN ─────────────────────────────────────────────────
def generate_summary(transcript: str) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert content analyst. Analyze this YouTube video transcript and return ONLY valid JSON with no extra text:
{{
  "title": "A concise, descriptive title for this video (max 10 words)",
  "summary": "3 sentence overview of the video",
  "key_concepts": ["concept1", "concept2", "concept3", "concept4"],
  "why_it_matters": "2 sentence explanation of real-world importance",
  "key_topics": ["topic1", "topic2", "topic3", "topic4", "topic5"],
  "difficulty": "Beginner or Intermediate or Advanced",
  "study_time_minutes": 15
}}"""),
        ("human", "Transcript:\n{transcript}")
    ])
    chain = prompt | llm_precise | StrOutputParser()
    raw = chain.invoke({"transcript": transcript[:6000]})
    try:
        clean = raw.strip().strip("```json").strip("```").strip()
        return json.loads(clean)
    except Exception:
        return {
            "summary": raw[:300],
            "key_concepts": [],
            "why_it_matters": "",
            "key_topics": [],
            "difficulty": "Intermediate",
            "study_time_minutes": 10,
        }

# ── QUIZ CHAIN ────────────────────────────────────────────────────
def generate_quiz(transcript: str) -> list:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert educator creating a comprehensive knowledge assessment.
Analyze this transcript and generate as many multiple choice questions as needed to
thoroughly test the student's understanding of ALL concepts covered.

Guidelines:
- Generate between 5 and 20 questions depending on the depth and breadth of the content.
- Cover every key concept, definition, comparison, and example mentioned.
- Mix difficulty levels: some basic recall, some understanding, some application.
- Each question should test a distinct concept — avoid redundancy.
- Make wrong options plausible but clearly incorrect to someone who understood the material.

Return ONLY a valid JSON array with no extra text:
[
  {{
    "question": "...",
    "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
    "correct": "A",
    "explanation": "Brief explanation of why this is correct"
  }}
]"""),
        ("human", "Transcript:\n{transcript}")
    ])
    chain = prompt | llm_precise | StrOutputParser()
    raw = chain.invoke({"transcript": transcript[:8000]})
    try:
        clean = raw.strip().strip("```json").strip("```").strip()
        return json.loads(clean)
    except Exception:
        return []

# ── NOTES CHAIN ───────────────────────────────────────────────────
def generate_notes(transcript: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert academic tutor. Create DETAILED, exam-ready study notes
from this transcript. These notes should be thorough enough that if a student
revises ONLY these notes before an exam, they can score well.

Structure your notes in markdown using this format:

# [Video Topic Title]

## 1. Introduction & Context
- Why this topic matters
- Prerequisites and background

## 2-N. [Main Topic Sections]
For EACH concept covered in the video:
- **Definition**: Clear, precise definition
- **Explanation**: How it works in detail (2-3 sentences minimum)
- **Key Formula/Rule**: If any mathematical or logical rules apply
- **Example**: A concrete example or analogy
- **Common Mistakes**: What students typically get wrong

## Important Comparisons
- Create comparison tables where relevant (use markdown tables)

## Formulas & Key Rules
- List all formulas, rules, or important relationships

## Key Takeaways
- 5-8 essential points to remember

## Quick Revision Checklist
- A bulleted checklist of everything to review before the exam

**Important**: Be thorough and detailed. Use **bold** for terms, `code` for technical terms.
Do NOT summarize — EXPLAIN. Every concept should be exam-ready."""),
        ("human", "Transcript:\n{transcript}")
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"transcript": transcript[:8000]})

# ── FLASHCARDS CHAIN ──────────────────────────────────────────────
def generate_flashcards(transcript: str) -> list:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Create exactly 8 flashcards from this transcript.
Return ONLY valid JSON array with no extra text:
[{{"front": "concept or question", "back": "definition or answer"}}]"""),
        ("human", "Transcript:\n{transcript}")
    ])
    chain = prompt | llm_precise | StrOutputParser()
    raw = chain.invoke({"transcript": transcript[:6000]})
    try:
        clean = raw.strip().strip("```json").strip("```").strip()
        return json.loads(clean)
    except Exception:
        return []

# ── EXAM MODE CHAIN ───────────────────────────────────────────────
def generate_exam_plan(transcript: str, topic: str, hours: float) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert tutor. The student has {hours} hours before their exam on "{topic}".
Based on this video transcript, create a focused study plan.
Prioritize the most exam-relevant concepts ruthlessly.
Return ONLY valid JSON with no extra text:
{{
  "total_minutes": 120,
  "sections": [
    {{
      "title": "section name",
      "duration_mins": 20,
      "concepts": ["concept1", "concept2"],
      "checkpoint_question": "What is...?"
    }}
  ],
  "quick_tips": ["tip1", "tip2", "tip3"]
}}"""),
        ("human", "Topic: {topic}\nHours available: {hours}\n\nTranscript:\n{transcript}")
    ])
    chain = prompt | llm_precise | StrOutputParser()
    raw = chain.invoke({
        "transcript": transcript[:6000],
        "topic":      topic,
        "hours":      str(hours),
    })
    try:
        clean = raw.strip().strip("```json").strip("```").strip()
        return json.loads(clean)
    except Exception:
        return {
            "total_minutes": int(hours * 60),
            "sections":      [],
            "quick_tips":    [],
        }

# ── CONFUSION DETECTOR ────────────────────────────────────────────
def detect_confusion(question: str) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Rate this student question for confusion level 1-5.
1 = clear and confident
3 = somewhat unclear
5 = very confused or lost

Signals of confusion: vague phrasing, very short follow-ups ("ok", "what", "huh"),
repeated asking of the same thing, contradictory statements, "I don't understand".

Return ONLY valid JSON with no extra text: {{"score": 3, "reason": "brief reason"}}"""),
        ("human", "Student question: {question}")
    ])
    chain = prompt | llm_precise | StrOutputParser()
    raw = chain.invoke({"question": question})
    try:
        clean = raw.strip().strip("```json").strip("```").strip()
        return json.loads(clean)
    except Exception:
        return {"score": 1, "reason": "could not assess"}

# ── RAG CHAIN ─────────────────────────────────────────────────────
def build_rag_chain(transcript: str, timestamped_chunks: list):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        api_key=OPENAI_API_KEY,
    )

    # Main retriever
    docs = splitter.create_documents([transcript])
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Timestamp retriever — separate index with metadata
    ts_docs = splitter.create_documents(
        [c["text"] for c in timestamped_chunks],
        metadatas=timestamped_chunks,
    )
    ts_store = FAISS.from_documents(ts_docs, embeddings)
    ts_retriever = ts_store.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    def get_timestamps(question: str) -> list:
        results = ts_retriever.invoke(question)
        seen, timestamps = set(), []
        for doc in results:
            start = doc.metadata.get("start_timestamp", 0)
            vid = doc.metadata.get("video_id", "")
            ts = _format_time(start)
            if ts and ts not in seen:
                seen.add(ts)
                timestamps.append({"time": start, "label": ts, "video_id": vid})
        return timestamps

    # ── Hybrid RAG prompt ─────────────────────────────────────────
    # Answers from video → normal response
    # Answers outside video → prefixed with [OUTSIDE VIDEO]
    # Confused student → simplified explanation with analogy
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are VidMind AI — an intelligent video learning assistant.

Instructions:
1. Check if the answer exists in the Video Transcript Context below.
   - If YES: answer using ONLY the transcript context. Respond normally.
   - If NO: answer from your general knowledge and start your response with exactly: [OUTSIDE VIDEO]

2. If the question starts with [STUDENT IS CONFUSED]:
   - First give a simple real-world analogy
   - Then give the technical explanation
   - Use simple language throughout

Video Transcript Context:
{context}"""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

    chain = (
        RunnableParallel({
            "context":  lambda x: format_docs(retriever.invoke(x["question"])),
            "question": lambda x: x["question"],
            "history":  lambda x: x["history"],
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

    return chain_with_history, get_timestamps