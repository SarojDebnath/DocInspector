import os
from typing import List, Optional

from langchain_core.messages import AIMessage, HumanMessage, BaseMessage

from src.document_ingestion.data_ingestion import ChatIngestor
from src.document_chat.retrieval import ConversationalRAG

import chainlit as cl


# Simple adapter so our ingestion pipeline can read Chainlit uploads
class ChainlitFileAdapter:
    def __init__(self, name: str, path: str):
        self.name = name
        self._path = path

    def read(self) -> bytes:
        with open(self._path, "rb") as f:
            return f.read()


FAISS_BASE = os.getenv("FAISS_BASE", "faiss_index")
UPLOAD_BASE = os.getenv("UPLOAD_BASE", "data")


@cl.on_chat_start
async def on_chat_start() -> None:
    files = None

    while files is None:
        files = await cl.AskFileMessage(
            content="Upload PDF/DOCX/TXT/XLSX/PPTX to start a document chat.",
            accept=[
                "application/pdf",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "text/plain",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ],
            max_size_mb=50,
            max_files=20,
            timeout=180,
        ).send()

    msg = cl.Message(content="Indexing documents...")
    await msg.send()

    # Build retriever (and persist FAISS) using existing ingestion pipeline
    adapters: List[ChainlitFileAdapter] = [
        ChainlitFileAdapter(f.name, f.path) for f in files
    ]

    chat_ingestor = ChatIngestor(
        temp_base=UPLOAD_BASE,
        faiss_base=FAISS_BASE,
        use_session_dirs=True,
        session_id=None,
    )

    # Larger chunks improve semantic coherence for sections like flowcharts or procedures
    # Run heavy ingestion asynchronously to avoid UI freeze/timeouts
    retriever = await cl.make_async(chat_ingestor.built_retriver)(
        adapters, chunk_size=2000, chunk_overlap=800, k=60
    )

    # Create RAG with retriever attached (uses Azure LLM + prompts via ModelLoader)
    rag = ConversationalRAG(session_id=chat_ingestor.session_id, retriever=retriever)

    # Initialize chat memory
    chat_history: List[BaseMessage] = []

    cl.user_session.set("rag", rag)
    cl.user_session.set("chat_history", chat_history)
    cl.user_session.set("session_id", chat_ingestor.session_id)

    msg.content = f"Index ready for session `{chat_ingestor.session_id}`. Ask your question!"
    await msg.update()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    rag: Optional[ConversationalRAG] = cl.user_session.get("rag")
    chat_history: List[BaseMessage] = cl.user_session.get("chat_history") or []

    # Allow runtime upload: user can type /upload and attach new files to create a NEW session
    if message.content.strip().lower() == "/upload":
        files = await cl.AskFileMessage(
            content="Upload files (PDF/DOCX/TXT/XLSX/PPTX) to create a NEW session and analyze these documents.",
            accept=[
                "application/pdf",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "text/plain",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ],
            max_size_mb=50,
            max_files=20,
            timeout=180,
        ).send()

        if not files:
            await cl.Message(content="No files received.").send()
            return

        # Create NEW session instead of reusing existing one
        ci = ChatIngestor(
            temp_base=UPLOAD_BASE,
            faiss_base=FAISS_BASE,
            use_session_dirs=True,
            session_id=None,  # This creates a new session every time
        )
        adapters = [ChainlitFileAdapter(f.name, f.path) for f in files]
        # Run heavy ingestion asynchronously to avoid UI freeze/timeouts
        retriever = await cl.make_async(ci.built_retriver)(adapters, chunk_size=2000, chunk_overlap=800, k=60)
        rag = ConversationalRAG(session_id=ci.session_id, retriever=retriever)
        
        # Reset chat history for new session
        chat_history: List[BaseMessage] = []
        
        cl.user_session.set("rag", rag)
        cl.user_session.set("chat_history", chat_history)
        cl.user_session.set("session_id", ci.session_id)
        await cl.Message(content=f"🆕 NEW session created: `{ci.session_id}`\n\nYour documents have been indexed. This is a fresh session - ask your questions!").send()
        return

    if rag is None:
        await cl.Message(content="Session not initialized. Please restart the chat.").send()
        return

    # Update memory with user turn
    chat_history.append(HumanMessage(content=message.content))

    # Invoke RAG with memory - simple and general
    answer: str = rag.invoke(message.content, chat_history=chat_history)

    # Update memory with assistant turn (using original query for memory)
    chat_history.append(AIMessage(content=answer))
    cl.user_session.set("chat_history", chat_history)

    await cl.Message(content=answer).send()


