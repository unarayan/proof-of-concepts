from __future__ import annotations

import json
import logging
import threading
import uuid
from dataclasses import dataclass
from typing import Generator

import torch
from langchain_chroma import Chroma
from langchain_core.documents import Document
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from components.chunker_component import SemanticChunker
from components.embedding_component import EmbeddingComponent
from utils.config_loader import config
from utils.ensure_model import ensure_llm_model


logger = logging.getLogger(__name__)

_SHARED_PIPELINE: "RagPipeline | None" = None


class ChromaEmbeddingAdapter:
    def __init__(self, component: EmbeddingComponent) -> None:
        self.component = component

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.component.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.component.embed_query(text)


@dataclass(slots=True)
class RetrievalRecord:
    source: str
    content: str
    score: float | None
    metadata: dict


class RagPipeline:
    def __init__(self) -> None:
        ensure_llm_model()

        self.embedding_component = EmbeddingComponent()

        storage_cfg = config.storage
        self.persist_directory = storage_cfg.persist_directory
        self.collection_name = storage_cfg.collection_name
        self.top_k = int(getattr(config.retrieval, "top_k", 5))
        self.fetch_k = int(getattr(config.retrieval, "fetch_k", 10))
        self.max_context_chars = int(getattr(config.retrieval, "max_context_chars", 12000))
        self.score_threshold = getattr(config.retrieval, "score_threshold", None)
        self.include_source_markers = bool(getattr(config.answering, "include_source_markers", False))

        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            embedding_function=ChromaEmbeddingAdapter(self.embedding_component),
        )

        llm_cfg = config.models.llm
        logger.info("Loading PyTorch LLM: %s", llm_cfg.hf_id)

        configured_device = str(getattr(llm_cfg, "device", "auto")).strip().lower()
        if configured_device in {"gpu", "cuda"}:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                logger.warning("GPU requested for LLM but CUDA is unavailable; falling back to CPU")
                device = "cpu"
        elif configured_device == "cpu":
            device = "cpu"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Using device: %s", device)

        # Load tokenizer and model
        self._tokenizer = AutoTokenizer.from_pretrained(llm_cfg.hf_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            llm_cfg.hf_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        self._model.to(device)
        self._device = device

        self.chunker = SemanticChunker(
            self.embedding_component,
            self._generate_text,
            llm_tokenizer=self._tokenizer,
        )

    def ingest_text(self, text: str, source: str = "api", metadata: dict | None = None) -> int:
        chunks = self.chunker.chunk_text(text)
        if not chunks:
            return 0

        docs = [
            Document(
                page_content=chunk.text,
                metadata={
                    "source": source,
                    "chunk_index": chunk.index,
                    **(metadata or {}),
                },
                id=str(uuid.uuid4()),
            )
            for chunk in chunks
        ]
        self.vectorstore.add_documents(docs)
        return len(docs)

    def clear_context(self) -> None:
        client = getattr(self.vectorstore, "_client", None)
        if client is None:
            raise RuntimeError("Vector store client is not available")
        try:
            client.delete_collection(self.collection_name)
        except Exception:  # noqa: BLE001
            logger.info("Collection %s did not exist yet during clear_context", self.collection_name)
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            embedding_function=ChromaEmbeddingAdapter(self.embedding_component),
        )

    def get_stats(self) -> dict:
        collection = getattr(self.vectorstore, "_collection", None)
        count = collection.count() if collection is not None else None
        return {
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory,
            "document_count": count,
            "chunking_strategy": config.chunking.strategy,
            "llm_model": config.models.llm.hf_id,
            "embedding_model": config.models.embedding.hf_id,
        }

    def answer_question(
        self,
        question: str,
        context_text: str | None = None,
        top_k: int | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
    ) -> dict:
        prompt, sources = self.plan_answer(
            question,
            context_text=context_text,
            top_k=top_k,
            system_prompt=system_prompt,
        )
        answer = self.generate_from_prompt(prompt, max_tokens=max_tokens, temperature=temperature)
        return {
            "answer": answer.strip(),
            "sources": [self._source_payload(record) for record in sources],
        }

    def stream_answer(
        self,
        question: str,
        context_text: str | None = None,
        top_k: int | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
    ) -> Generator[str, None, None]:
        prompt, _ = self.plan_answer(
            question,
            context_text=context_text,
            top_k=top_k,
            system_prompt=system_prompt,
        )
        yield from self.stream_from_prompt(prompt, max_tokens=max_tokens, temperature=temperature)

    def plan_answer(
        self,
        question: str,
        context_text: str | None = None,
        top_k: int | None = None,
        system_prompt: str | None = None,
    ) -> tuple[str, list[RetrievalRecord]]:
        sources = self.retrieve(question, top_k=top_k)
        prompt = self._build_prompt(question, sources, context_text=context_text, system_prompt=system_prompt)
        return prompt, sources

    def generate_from_prompt(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        return self._generate_text(prompt, max_tokens=max_tokens, temperature=temperature)

    def stream_from_prompt(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Generator[str, None, None]:
        yield from self._stream_generate(prompt, max_tokens=max_tokens, temperature=temperature)

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievalRecord]:
        desired_k = top_k or self.top_k
        docs_with_scores = self.vectorstore.similarity_search_with_score(question, k=max(desired_k, self.fetch_k))
        records: list[RetrievalRecord] = []
        for document, score in docs_with_scores:
            if self.score_threshold is not None and score is not None and score > self.score_threshold:
                continue
            records.append(
                RetrievalRecord(
                    source=str(document.metadata.get("source", "context")),
                    content=document.page_content,
                    score=float(score) if score is not None else None,
                    metadata=document.metadata,
                )
            )
            if len(records) >= desired_k:
                break
        return records

    def _build_prompt(
        self,
        question: str,
        sources: list[RetrievalRecord],
        context_text: str | None = None,
        system_prompt: str | None = None,
    ) -> str:
        prompt_system = system_prompt or config.answering.system_prompt
        retrieved_context = self._build_context_block(sources)
        extra_context = (context_text or "").strip()
        fallback_hint = (
            "If the retrieved store context is insufficient, you may use general retail knowledge but state uncertainty clearly."
            if bool(getattr(config.answering, "fallback_to_general_knowledge", True))
            else "If the context is insufficient, say you do not have enough store context."
        )

        prompt = [prompt_system.strip(), "", f"Customer question:\n{question.strip()}"]
        if retrieved_context:
            prompt.extend(["", f"Retrieved store context:\n{retrieved_context}"])
        if extra_context:
            prompt.extend(["", f"Runtime context passed by caller:\n{extra_context}"])
        prompt.extend(["", fallback_hint, "Answer:"])
        return "\n".join(prompt).strip()

    def _build_context_block(self, sources: list[RetrievalRecord]) -> str:
        parts: list[str] = []
        total_chars = 0
        for index, record in enumerate(sources, start=1):
            label = f"[{index}] {record.source}" if self.include_source_markers else record.source
            block = f"### SOURCE {label}\n{record.content.strip()}"
            if total_chars + len(block) > self.max_context_chars:
                break
            parts.append(block)
            total_chars += len(block)
        return "\n\n".join(parts)

    def _generate_text(self, prompt: str, max_tokens: int | None = None, temperature: float | None = None) -> str:
        llm_cfg = config.models.llm
        temp = temperature if temperature is not None else float(getattr(llm_cfg, "temperature", 0.0))
        
        # Tokenize input
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        
        # Generate kwargs — no max_new_tokens cap; model stops at its own EOS
        gen_kwargs: dict = dict(
            temperature=max(temp, 1e-7),
            do_sample=temp > 0.0,
            top_p=0.9 if temp > 0.0 else 1.0,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        if max_tokens is not None:
            gen_kwargs["max_new_tokens"] = max_tokens
        
        # Generate
        with torch.no_grad():
            outputs = self._model.generate(**inputs, **gen_kwargs)
        
        # Decode
        generated_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the new part (remove the input prompt)
        if prompt in generated_text:
            answer = generated_text.split(prompt, 1)[1]
        else:
            answer = generated_text[len(self._tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]
        
        return answer

    def _stream_generate(self, prompt: str, max_tokens: int | None = None, temperature: float | None = None) -> Generator[str, None, None]:
        llm_cfg = config.models.llm
        temp = temperature if temperature is not None else float(getattr(llm_cfg, "temperature", 0.0))
        
        # Use TextIteratorStreamer for streaming
        streamer = TextIteratorStreamer(self._tokenizer, skip_special_tokens=True, skip_prompt=True)
        
        # Tokenize input
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        
        # Generate kwargs — no max_new_tokens cap; model stops at its own EOS
        gen_kwargs: dict = dict(
            temperature=max(temp, 1e-7),
            do_sample=temp > 0.0,
            top_p=0.9 if temp > 0.0 else 1.0,
            streamer=streamer,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        if max_tokens is not None:
            gen_kwargs["max_new_tokens"] = max_tokens
        
        def _generate_thread():
            with torch.no_grad():
                self._model.generate(**inputs, **gen_kwargs)
        
        # Run generation in thread
        thread = threading.Thread(target=_generate_thread, daemon=True)
        thread.start()
        
        # Stream tokens
        for text in streamer:
            yield text

    @staticmethod
    def _source_payload(record: RetrievalRecord) -> dict:
        return {
            "source": record.source,
            "score": record.score,
            "metadata": record.metadata,
            "content": record.content,
        }


def set_shared_pipeline(pipeline: RagPipeline) -> None:
    global _SHARED_PIPELINE
    _SHARED_PIPELINE = pipeline


def get_shared_pipeline() -> RagPipeline:
    global _SHARED_PIPELINE
    if _SHARED_PIPELINE is None:
        _SHARED_PIPELINE = RagPipeline()
    return _SHARED_PIPELINE
