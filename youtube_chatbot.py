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

def _get_val(obj, key):
    """Safely get value from dict or object attribute."""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)

# ── TRANSCRIPT ────────────────────────────────────────────────────
def get_transcript_with_timestamps(video_id: str) -> Tuple[str, list, list, str]:
    """
    Returns (plain_text, timestamped_chunks, raw_transcript, detected_language)

    Priority order:
    1. Manual English transcript
    2. Auto-generated English transcript
    3. Manual transcript in any language -> translated to English
    4. Auto-generated transcript in any language -> translated to English
    5. Raises RuntimeError if nothing is available
    """
    try:
        ytt = YouTubeTranscriptApi()
        transcript_list = ytt.list(video_id)

        fetched = None
        detected_language = "en"

        # Priority 1: Manual English
        try:
            fetched = transcript_list.find_manually_created_transcript(["en"])
            detected_language = "en"
        except Exception:
            pass

        # Priority 2: Auto-generated English
        if not fetched:
            try:
                fetched = transcript_list.find_generated_transcript(["en"])
                detected_language = "en"
            except Exception:
                pass

        # Priority 3: Manual in any language -> translate to English
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

        # Priority 4: Auto-generated in any language -> translate to English
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

        if not fetched:
            raise RuntimeError(
                "No transcript available for this video. "
                "The video may have transcripts disabled by the creator."
            )

        # Fetch raw data
        raw = fetched.fetch()
        plain_text = " ".join(str(_get_val(item, "text") or "") for item in raw)

        # Build 30-second timestamped chunks with both keys for compatibility
        timestamped, chunk_text, chunk_start = [], [], 0
        for item in raw:
            i_text  = _get_val(item, "text")
            i_start = _get_val(item, "start")
            if i_text is None or i_start is None:
                continue
            if not chunk_text:
                chunk_start = int(i_start)
            chunk_text.append(str(i_text))
            if i_start - chunk_start >= 30:
                timestamped.append({
                    "start_timestamp": chunk_start,
                    "timestamp":       _format_time(chunk_start),
                    "text":            " ".join(chunk_text),
                    "video_id":        video_id,
                })
                chunk_text = []
        if chunk_text:
            timestamped.append({
                "start_timestamp": chunk_start,
                "timestamp":       _format_time(chunk_start),
                "text":            " ".join(chunk_text),
                "video_id":        video_id,
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
  "title": "A concise descriptive title for this video (max 10 words)",
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
            "title": "Video Study Notes",
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
- Generate between 5 and 20 questions depending on depth and breadth of content.
- Cover every key concept, definition, comparison, and example mentioned.
- Mix difficulty: some basic recall, some understanding, some application.
- Each question tests a distinct concept — no redundancy.
- Make wrong options plausible but clearly incorrect to someone who understood the material.
- Write detailed explanations that teach, not just confirm the answer.

Return ONLY a valid JSON array with no extra text:
[
  {{
    "question": "...",
    "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
    "correct": "A",
    "explanation": "Explain why this is correct AND why the other options are wrong."
  }}
]"""),
        ("human", "Transcript:\n{transcript}")
    ])
    chain = prompt | llm_precise | StrOutputParser()
    raw = chain.invoke({"transcript": transcript[:8000]})
    try:
        clean = raw.strip().strip("```json").strip("```").strip()
        result = json.loads(clean)
        if isinstance(result, dict):
            return result.get("questions", [])
        return result
    except Exception:
        return []

# ── NOTES CHAIN ───────────────────────────────────────────────────
def generate_notes(transcript: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert academic tutor. Create DETAILED, exam-ready study notes
from this transcript. These notes must be thorough enough that a student who reads
ONLY these notes before an exam can score well without re-watching the video.

Structure your notes in markdown using this EXACT format:

# [Video Topic Title]

## 1. Introduction & Context
- What this topic is about
- Why it matters in the real world
- What prior knowledge is assumed

## 2. [First Major Concept Name]
- **Definition**: Clear, precise definition in simple language
- **How it works**: Step-by-step explanation (minimum 3-4 sentences)
- **Key rule/formula**: Any mathematical or logical rules (if applicable)
- **Real-world example**: A concrete, relatable example
- **Common mistake**: What students typically get wrong about this

(Repeat ## section for EVERY major concept in the video)

## Comparisons & Relationships
Create a markdown table comparing concepts that are often confused:
| Concept A | Concept B | Key Difference |
|-----------|-----------|----------------|
| ...       | ...       | ...            |

## All Formulas & Key Rules
List every formula, algorithm, or important rule with explanation.

## Important Terms & Definitions
| Term | Definition |
|------|------------|
| ...  | ...        |

## Likely Exam Questions & Hints
1. [Question a professor would ask] — Hint: focus on [key point]
2. [Another likely question] — Hint: remember [key point]
(Generate 6-8 likely exam questions with hints)

## Key Takeaways
- The 6-8 most critical things to remember
- What you absolutely cannot forget for the exam

## Quick Revision Checklist
- [ ] I can define [concept 1]
- [ ] I understand how [concept 2] works
- [ ] I know the difference between [X] and [Y]
(one checkbox per major concept)

Rules:
- Be THOROUGH and DETAILED — do not summarize, EXPLAIN
- Use **bold** for key terms
- Use tables wherever comparison helps
- Every section must have enough detail to answer exam questions"""),
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

# ── PDF EXPORT ────────────────────────────────────────────────────
def generate_pdf_from_notes(notes_text: str, title: str = "VidMind AI Notes") -> bytes:
    """
    Converts markdown-style notes to a clean, well-formatted PDF.
    Handles: H1, H2, H3, bullet points, numbered lists, tables,
             checklist items, bold sub-labels, normal paragraphs.
    Returns PDF as bytes.
    """
    from fpdf import FPDF
    import re

    class NotesPDF(FPDF):
        def header(self):
            self.set_font("Helvetica", "B", 9)
            self.set_text_color(120, 120, 120)
            self.cell(0, 8, "VidMind AI \u2014 Study Notes", align="R")
            self.ln(2)
            self.set_draw_color(220, 220, 220)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(4)

        def footer(self):
            self.set_y(-14)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(160, 160, 160)
            self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def safe(t: str) -> str:
        t = re.sub(r"\*\*(.+?)\*\*", r"\1", t)
        t = re.sub(r"\*(.+?)\*",     r"\1", t)
        t = re.sub(r"`(.+?)`",       r"\1", t)
        for old, new in [("\u2019","'"),("\u2018","'"),("\u201c",'"'),
                         ("\u201d",'"'),("\u2014","-"),("\u2013","-")]:
            t = t.replace(old, new)
        return t

    pdf = NotesPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    pdf.set_margins(15, 15, 15)

    in_table   = False
    table_cols = 0

    for line in notes_text.split("\n"):
        raw_line = line.rstrip()
        s = raw_line.strip()

        if not s:
            if in_table:
                in_table = False
            pdf.ln(2)
            continue

        # H1
        if s.startswith("# ") and not s.startswith("## "):
            in_table = False
            pdf.set_font("Helvetica", "B", 18)
            pdf.set_text_color(30, 58, 95)
            pdf.ln(2)
            pdf.multi_cell(0, 10, safe(s[2:]))
            pdf.set_draw_color(30, 58, 95)
            pdf.line(15, pdf.get_y(), 195, pdf.get_y())
            pdf.ln(4)

        # H2
        elif s.startswith("## "):
            in_table = False
            pdf.set_font("Helvetica", "B", 13)
            pdf.set_text_color(46, 117, 182)
            pdf.ln(5)
            pdf.multi_cell(0, 8, safe(s[3:]))
            pdf.set_draw_color(180, 210, 240)
            pdf.line(15, pdf.get_y(), 195, pdf.get_y())
            pdf.ln(3)

        # H3
        elif s.startswith("### "):
            in_table = False
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(50, 50, 80)
            pdf.ln(3)
            pdf.multi_cell(0, 7, safe(s[4:]))
            pdf.ln(1)

        # Table separator row |---|---| — skip silently
        elif s.startswith("|") and not re.search(r"[a-zA-Z0-9]", s):
            continue

        # Table data row
        elif s.startswith("|") and s.endswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            table_cols = len(cells)
            col_w = max(int(180 / max(table_cols, 1)), 20)
            if not in_table:
                in_table = True
                pdf.set_font("Helvetica", "B", 9)
                pdf.set_fill_color(214, 228, 245)
                pdf.set_text_color(30, 58, 95)
            else:
                pdf.set_font("Helvetica", "", 9)
                pdf.set_fill_color(245, 248, 252)
                pdf.set_text_color(40, 40, 40)
            for c in cells:
                pdf.cell(col_w, 7, safe(c)[:35], border=1, fill=True)
            pdf.ln()

        # Checklist
        elif s.startswith("- [ ]") or s.startswith("- [x]"):
            in_table = False
            checked  = s.startswith("- [x]")
            content  = s[5:].strip()
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(60, 60, 60)
            mark = "\u2611" if checked else "\u2610"
            pdf.multi_cell(0, 6, f"  {mark}  {safe(content)}")

        # Bullet with bold label e.g. "- **Definition**: ..."
        elif s.startswith("- **") and "**:" in s:
            in_table    = False
            inner       = s[2:]
            label_end   = inner.index("**:", 2)
            label       = inner[2:label_end]
            rest        = inner[label_end + 3:].strip()
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(30, 58, 95)
            lbl_w = pdf.get_string_width(f"  \u2022  {label}: ") + 1
            pdf.cell(lbl_w, 6, f"  \u2022  {safe(label)}: ")
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(50, 50, 50)
            pdf.multi_cell(0, 6, safe(rest))

        # Regular bullet
        elif s.startswith("- "):
            in_table = False
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(50, 50, 50)
            pdf.multi_cell(0, 6, f"  \u2022  {safe(s[2:])}")

        # Numbered list
        elif len(s) > 2 and s[0].isdigit() and s[1] in ".)":
            in_table = False
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(50, 50, 50)
            pdf.multi_cell(0, 6, f"  {safe(s)}")

        # Normal paragraph
        else:
            in_table = False
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(40, 40, 40)
            pdf.multi_cell(0, 6, safe(s))

    return bytes(pdf.output())

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

    # Timestamp retriever — separate FAISS index with metadata
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
            ts    = doc.metadata.get("timestamp", _format_time(start))
            vid   = doc.metadata.get("video_id", "")
            if ts and ts not in seen:
                seen.add(ts)
                timestamps.append({
                    "timestamp": ts,    # used by export-chat
                    "time":      start, # used by frontend chips
                    "label":     ts,    # alias
                    "video_id":  vid,
                })
        return timestamps

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are VidMind AI — an intelligent video learning assistant.

Instructions:
1. Check if the answer exists in the Video Transcript Context below.
   - If YES: answer using ONLY the transcript context. Respond normally.
   - If NO: answer from your general knowledge and start your response with exactly: [OUTSIDE VIDEO]

2. If the question starts with [STUDENT IS CONFUSED]:
   - First give a simple real-world analogy
   - Then give the technical explanation
   - Use simple, clear language throughout

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