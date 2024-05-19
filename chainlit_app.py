import os
import chainlit as cl

from src.qna import QnA

@cl.on_chat_start
async def on_chat_start():
    msg = cl.Message(content=f"Initializing chatbot...", disable_feedback=True)
    qna = QnA()
    cl.user_session.set("chain", qna)
    # Let the user know that the system is ready
    msg.content = f"Initialization done. You can now ask questions!"
    await msg.update()


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler()

    res = await chain.ask_question(message.content, callbacks=[cb], session_id="1321431", verbose=False,)
    answer = res["answer"]
    source_documents = res["source_documents"]

    text_elements = []

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()
