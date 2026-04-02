import os
from typing import List, Dict
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


# ------------------------------------------------------------------
# ENV SETUP
# ------------------------------------------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
assert OPENAI_API_KEY is not None, "OPENAI_API_KEY not set"


# ------------------------------------------------------------------
# LLM
# ------------------------------------------------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    api_key=OPENAI_API_KEY,
)


# ------------------------------------------------------------------
# UTILS
# ------------------------------------------------------------------
def get_video_id(url: str) -> str | None:
    parsed = urlparse(url)
    return parse_qs(parsed.query).get("v", [None])[0]


def get_transcript(video_id):
    try:
        ytt_api = YouTubeTranscriptApi()
        fetched_transcript = ytt_api.fetch(video_id, languages=["en"])
        raw_transcript = fetched_transcript.to_raw_data()
        return " ".join(item["text"] for item in raw_transcript)
    except Exception as e:
        raise RuntimeError(f"❌ Transcript error: {e}")


# ------------------------------------------------------------------
# MEMORY
# ------------------------------------------------------------------
class InMemoryHistory(BaseChatMessageHistory):
    def __init__(self):
        self._messages: List = []

    @property
    def messages(self):
        return self._messages

    def add_message(self, message):
        self._messages.append(message)

    def clear(self):
        self._messages.clear()


store: Dict[str, BaseChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]


# ------------------------------------------------------------------
# CORE BUILDER
# ------------------------------------------------------------------
def build_chain_with_history(transcript: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    docs = splitter.create_documents([transcript])

    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        api_key=OPENAI_API_KEY,
    )

    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Answer ONLY using the transcript below.\n\n{context}"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    chain = (
        RunnableParallel(
            {
                "context": lambda x: format_docs(
                    retriever.invoke(x["question"])
                ),
                "question": lambda x: x["question"],
                "history": lambda x: x["history"],
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )
