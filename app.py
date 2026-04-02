import streamlit as st
from youtube_chatbot import get_video_id, get_transcript, build_chain_with_history


st.set_page_config(
    page_title="YouTube GenAI Chat",
    page_icon="🎥",
)

st.title("🎥 YouTube Video Q&A (GenAI)")

SESSION_ID = "youtube_chat_1"

if "chain" not in st.session_state:
    st.session_state.chain = None

if "ready" not in st.session_state:
    st.session_state.ready = False


url = st.text_input("🔗 Enter YouTube Video URL")

if st.button("📥 Load Transcript"):
    if not url:
        st.error("Please enter a YouTube URL")
    else:
        with st.spinner("Fetching transcript and preparing AI..."):
            video_id = get_video_id(url)

            if not video_id:
                st.error("Invalid YouTube URL")
            else:
                transcript = get_transcript(video_id)
                st.session_state.chain = build_chain_with_history(transcript)
                st.session_state.ready = True
                st.success("Transcript loaded. Start chatting 🎉")


if st.session_state.ready:
    st.divider()
    st.subheader("🤖 Ask questions about the video")

    user_question = st.chat_input("Type your question...")

    if user_question:
        with st.chat_message("user"):
            st.write(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state.chain.invoke(
                    {"question": user_question},
                    config={
                        "configurable": {
                            "session_id": SESSION_ID
                        }
                    },
                )
                st.write(answer)
