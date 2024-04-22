import streamlit as st

from src.qna import QnA
from dataclasses import dataclass


@dataclass
class Message:
    actor: str
    payload: str


def main():
    st.set_page_config(
        page_title="KnowledgeGPT",
        page_icon="📖",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    st.header("📖KnowledgeGPT")

    USER = "user"
    ASSISTANT = "ai"
    MESSAGES = "messages"

    with st.spinner(text="Initializing..."):
        st.session_state["qna"] = QnA()

    qna = st.session_state["qna"]
    if MESSAGES not in st.session_state:
        st.session_state[MESSAGES] = [
            Message(
                actor=ASSISTANT,
                payload="Hi! How can I help you?",
            )
        ]
    msg: Message
    for msg in st.session_state[MESSAGES]:
        st.chat_message(msg.actor).write(msg.payload)

    prompt: str = st.chat_input("Enter a prompt here")

    if prompt:
        st.session_state[MESSAGES].append(Message(actor=USER, payload=prompt))
        st.chat_message(USER).write(prompt)
        with st.spinner(text="Thinking..."):
            response = qna.ask_question(
                query=prompt, session_id="AWDAA-adawd-ADAFAEF"
            )

        st.session_state[MESSAGES].append(Message(actor=ASSISTANT, payload=response))
        st.chat_message(ASSISTANT).write(response)


if __name__ == "__main__":
    main()
