import os
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_community.vectorstores import FAISS

from utils.config_loader import load_config
from utils.file_io import generate_session_id
from utils.document_ops import FastAPIFileAdapter
from src.document_ingestion.data_ingestion import ChatIngestor
from src.document_chat.retrieval import ConversationalRAG
from exception.custom_exception import DocumentPortalException
from logger import GLOBAL_LOGGER as log


class ChatSessionService:
    """
    Manages multi-document chat sessions with conversational memory and retriever caching.
    """

    def __init__(
        self,
        *,
        faiss_base: str = "faiss_index",
        upload_base: str = "data",
        index_name: str = "index",
    ) -> None:
        self.faiss_base = faiss_base
        self.upload_base = upload_base
        self.index_name = index_name

        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._config: Dict[str, Any] = load_config()

        retr_conf = self._config.get("retriever", {})
        self.default_k: int = int(retr_conf.get("top_k", 5))
        self.search_type: str = str(retr_conf.get("search_type", "mmr"))
        self.lambda_mult: float = float(retr_conf.get("mmr_lambda", 0.7))
        self.fetch_k_factor: int = int(retr_conf.get("fetch_k_factor", 4))

        memory_conf = self._config.get("memory", {})
        self.max_turns: int = int(memory_conf.get("max_turns", 20))

    # -------- Session lifecycle --------
    def get_or_create_session(self, session_id: Optional[str]) -> str:
        sid = session_id or generate_session_id()
        if sid not in self._sessions:
            self._sessions[sid] = {
                "history": [],
                "rag": None,
            }
            log.info("Chat session created", session_id=sid)
        return sid

    def reset(self, session_id: str) -> None:
        sess = self._sessions.get(session_id)
        if not sess:
            return
        sess["history"] = []
        log.info("Chat session memory reset", session_id=session_id)

    # -------- Indexing / documents --------
    def add_documents(
        self,
        *,
        session_id: str,
        files: List[FastAPIFileAdapter],
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        k: Optional[int] = None,
    ) -> Dict[str, Any]:
        try:
            chat_ingestor = ChatIngestor(
                temp_base=self.upload_base,
                faiss_base=self.faiss_base,
                use_session_dirs=True,
                session_id=session_id,
            )
            retriever = chat_ingestor.built_retriver(
                files, chunk_size=chunk_size, chunk_overlap=chunk_overlap, k=k or self.default_k
            )

            # Attach/refresh RAG on session after ingestion
            rag = ConversationalRAG(session_id=session_id, retriever=retriever)
            self._sessions[session_id]["rag"] = rag

            return {
                "session_id": session_id,
                "k": k or self.default_k,
            }
        except Exception as e:
            log.exception("Failed to add documents to session", session_id=session_id)
            raise DocumentPortalException("Failed to add documents", e)

    def _ensure_rag(self, session_id: str, k: Optional[int] = None) -> ConversationalRAG:
        sess = self._sessions.get(session_id)
        if not sess:
            raise DocumentPortalException(f"Session not found: {session_id}", None)

        rag: Optional[ConversationalRAG] = sess.get("rag")
        if rag is not None:
            return rag

        # Build from existing FAISS index on disk
        index_dir = os.path.join(self.faiss_base, session_id)
        if not os.path.isdir(index_dir):
            raise DocumentPortalException(f"FAISS index not found for session: {index_dir}", None)

        rag = ConversationalRAG(session_id=session_id)
        rag.load_retriever_from_faiss(
            index_path=index_dir,
            k=k or self.default_k,
            index_name=self.index_name,
            search_type=self.search_type,
            search_kwargs=self._search_kwargs(k),
        )
        sess["rag"] = rag
        return rag

    def _search_kwargs(self, k: Optional[int]) -> Dict[str, Any]:
        if self.search_type == "mmr":
            kk = k or self.default_k
            return {
                "k": kk,
                "fetch_k": max(20, kk * self.fetch_k_factor),
                "lambda_mult": self.lambda_mult,
            }
        return {"k": k or self.default_k}

    # -------- Chat --------
    def send_message(self, *, session_id: str, content: str, k: Optional[int] = None) -> str:
        try:
            rag = self._ensure_rag(session_id, k=k)
            history: List[BaseMessage] = self._sessions[session_id]["history"]

            # Trim memory to max_turns (each turn is Human+AI)
            if self.max_turns > 0 and len(history) > self.max_turns * 2:
                history[:] = history[-self.max_turns * 2 :]

            history.append(HumanMessage(content=content))
            answer = rag.invoke(content, chat_history=history)
            history.append(AIMessage(content=answer))

            return answer
        except Exception as e:
            log.exception("Failed to handle chat message", session_id=session_id)
            raise DocumentPortalException("Failed to handle chat message", e)

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        sess = self._sessions.get(session_id)
        if not sess:
            return []
        items: List[Dict[str, str]] = []
        for m in sess["history"]:
            role = "assistant" if isinstance(m, AIMessage) else ("user" if isinstance(m, HumanMessage) else "other")
            items.append({"role": role, "content": m.content})
        return items


