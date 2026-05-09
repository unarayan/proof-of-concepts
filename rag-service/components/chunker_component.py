from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from utils.config_loader import config


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ChunkRecord:
    text: str
    index: int


class SemanticChunker:
    def __init__(
        self,
        embedding_component,
        llm_text_generator: Callable[[str, int | None, float | None], str],
        llm_tokenizer=None,
    ) -> None:
        chunk_cfg = config.chunking
        self.strategy = getattr(chunk_cfg, "strategy", "semantic_embedding")
        self.max_chunk_chars = int(getattr(chunk_cfg, "max_chunk_chars", 1200))
        self.min_chunk_chars = int(getattr(chunk_cfg, "min_chunk_chars", 180))
        self.overlap_chars = int(getattr(chunk_cfg, "overlap_chars", 120))
        self.semantic_similarity_threshold = float(getattr(chunk_cfg, "semantic_similarity_threshold", 0.72))
        self.llm_passage_chars = int(getattr(chunk_cfg, "llm_passage_chars", 6000))
        self.llm_passage_tokens = int(getattr(chunk_cfg, "llm_passage_tokens", 0))
        self.llm_passage_overlap_tokens = int(getattr(chunk_cfg, "llm_passage_overlap_tokens", 0))
        self.llm_text_generator = llm_text_generator
        self.embedding_component = embedding_component
        self.llm_tokenizer = llm_tokenizer

    def chunk_text(self, text: str) -> list[ChunkRecord]:
        normalized = self._normalize_text(text)
        if not normalized:
            return []

        t0 = time.monotonic()
        logger.info(
            "[CHUNKER] Starting chunking | strategy=%s | input_chars=%d",
            self.strategy,
            len(normalized),
        )

        if self.strategy == "semantic_llm":
            chunks = self._semantic_llm_chunks(normalized)
        elif self.strategy == "semantic_embedding":
            chunks = self._semantic_embedding_chunks(normalized)
        else:
            chunks = self._recursive_chunks(normalized)

        chunks = self._apply_overlap(chunks)
        records = [ChunkRecord(text=chunk, index=index) for index, chunk in enumerate(chunks)]
        elapsed = time.monotonic() - t0
        logger.info(
            "[CHUNKER] Done | total_chunks=%d | elapsed=%.1fs",
            len(records),
            elapsed,
        )
        return records

    # Hard token cap for the chunking LLM call — prevents it from running forever.
    # Chunking only needs a JSON array back, not a long essay.
    _CHUNKING_MAX_TOKENS = 1024

    def _semantic_llm_chunks(self, text: str) -> list[str]:
        if self.llm_tokenizer is not None and self.llm_passage_tokens > 0:
            coarse_passages = self._split_by_tokens(
                text,
                self.llm_passage_tokens,
                self.llm_passage_overlap_tokens,
            )
        else:
            coarse_passages = self._split_by_size(text, self.llm_passage_chars)

        total_passages = len(coarse_passages)
        logger.info("[CHUNKER] semantic_llm | total_passages=%d to process", total_passages)

        results: list[str] = []
        for p_idx, passage in enumerate(coarse_passages, start=1):
            passage_chars = len(passage)
            if len(passage) <= self.max_chunk_chars:
                logger.info(
                    "[CHUNKER] Passage %d/%d | chars=%d | fits in one chunk, skipping LLM",
                    p_idx, total_passages, passage_chars,
                )
                results.append(passage)
                continue

            logger.info(
                "[CHUNKER] Passage %d/%d | chars=%d | sending to LLM for semantic split",
                p_idx, total_passages, passage_chars,
            )
            t_passage = time.monotonic()

            prompt = (
                "You are preparing chunks for a retail knowledge-base RAG system.\n"
                "Split the passage into semantically coherent chunks.\n"
                "Rules:\n"
                "- Keep the original wording exactly.\n"
                "- Split only on natural sentence boundaries.\n"
                f"- Each chunk should be roughly {self.min_chunk_chars}-{self.max_chunk_chars} characters.\n"
                "- Return ONLY a JSON array of strings, nothing else.\n\n"
                f"PASSAGE:\n{passage}"
            )
            try:
                # Always cap chunking generation — we only need a JSON array back
                raw = self.llm_text_generator(
                    prompt,
                    self._CHUNKING_MAX_TOKENS,
                    0.0,
                )
                parsed = self._parse_llm_chunk_output(raw)
                elapsed_p = time.monotonic() - t_passage
                if parsed:
                    logger.info(
                        "[CHUNKER] Passage %d/%d | LLM produced %d sub-chunks | %.1fs",
                        p_idx, total_passages, len(parsed), elapsed_p,
                    )
                    results.extend(parsed)
                    continue
                logger.warning(
                    "[CHUNKER] Passage %d/%d | LLM output unparseable after %.1fs, using embedding fallback",
                    p_idx, total_passages, elapsed_p,
                )
            except Exception as exc:  # noqa: BLE001
                elapsed_p = time.monotonic() - t_passage
                logger.warning(
                    "[CHUNKER] Passage %d/%d | LLM error after %.1fs (%s), using embedding fallback",
                    p_idx, total_passages, elapsed_p, exc,
                )

            fb_chunks = self._semantic_embedding_chunks(passage)
            logger.info(
                "[CHUNKER] Passage %d/%d | embedding fallback produced %d chunks",
                p_idx, total_passages, len(fb_chunks),
            )
            results.extend(fb_chunks)

        return self._cleanup_chunks(results)

    def _semantic_embedding_chunks(self, text: str) -> list[str]:
        sentences = self._split_sentences(text)
        if not sentences:
            return []
        if len(sentences) == 1:
            return [sentences[0]]

        sentence_vectors = np.array(self.embedding_component.embed_documents(sentences), dtype=np.float32)
        chunks: list[str] = []
        current_sentences = [sentences[0]]

        for index in range(1, len(sentences)):
            prev_vector = sentence_vectors[index - 1]
            current_vector = sentence_vectors[index]
            similarity = self._cosine_similarity(prev_vector, current_vector)
            projected = " ".join(current_sentences + [sentences[index]])

            if len(projected) > self.max_chunk_chars or similarity < self.semantic_similarity_threshold:
                candidate = " ".join(current_sentences).strip()
                if candidate:
                    chunks.append(candidate)
                current_sentences = [sentences[index]]
                continue

            current_sentences.append(sentences[index])

        final_chunk = " ".join(current_sentences).strip()
        if final_chunk:
            chunks.append(final_chunk)
        return self._cleanup_chunks(chunks)

    def _recursive_chunks(self, text: str) -> list[str]:
        chunks = self._split_by_size(text, self.max_chunk_chars)
        return self._cleanup_chunks(chunks)

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        if self.overlap_chars <= 0 or len(chunks) < 2:
            return chunks

        with_overlap: list[str] = [chunks[0]]
        for index in range(1, len(chunks)):
            prefix = chunks[index - 1][-self.overlap_chars :].strip()
            current = chunks[index]
            combined = f"{prefix}\n{current}" if prefix else current
            with_overlap.append(combined.strip())
        return with_overlap

    def _split_by_size(self, text: str, max_chars: int) -> list[str]:
        paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]
        if not paragraphs:
            return []

        chunks: list[str] = []
        current = ""
        for paragraph in paragraphs:
            candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
            if len(candidate) <= max_chars:
                current = candidate
                continue
            if current:
                chunks.append(current)
            if len(paragraph) <= max_chars:
                current = paragraph
                continue

            sentence_buffer = ""
            for sentence in self._split_sentences(paragraph):
                sentence_candidate = f"{sentence_buffer} {sentence}".strip() if sentence_buffer else sentence
                if len(sentence_candidate) <= max_chars:
                    sentence_buffer = sentence_candidate
                    continue
                if sentence_buffer:
                    chunks.append(sentence_buffer)
                sentence_buffer = sentence
            current = sentence_buffer

        if current:
            chunks.append(current)
        return chunks

    def _split_by_tokens(self, text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
        tokenizer = self.llm_tokenizer
        if tokenizer is None:
            return self._split_by_size(text, self.llm_passage_chars)

        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if not token_ids:
            return []

        chunks: list[str] = []
        start = 0
        step = max(max_tokens - max(overlap_tokens, 0), 1)

        while start < len(token_ids):
            end = min(start + max_tokens, len(token_ids))
            window_ids = token_ids[start:end]
            chunk_text = tokenizer.decode(window_ids, skip_special_tokens=True).strip()
            if chunk_text:
                chunks.append(chunk_text)
            if end >= len(token_ids):
                break
            start += step

        return chunks

    def _parse_llm_chunk_output(self, raw_output: str) -> list[str]:
        match = re.search(r"\[[\s\S]*\]", raw_output)
        if not match:
            return []
        parsed = json.loads(match.group(0))
        if not isinstance(parsed, list):
            return []
        return [str(item).strip() for item in parsed if str(item).strip()]

    def _cleanup_chunks(self, chunks: list[str]) -> list[str]:
        cleaned = [re.sub(r"\s+", " ", chunk).strip() for chunk in chunks if chunk and chunk.strip()]
        merged: list[str] = []
        for chunk in cleaned:
            if merged and len(chunk) < self.min_chunk_chars:
                merged[-1] = f"{merged[-1]} {chunk}".strip()
            else:
                merged.append(chunk)
        return merged

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        normalized = text.replace("\n", " ")
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", normalized)
        return [part.strip() for part in parts if part.strip()]

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
        left_norm = np.linalg.norm(left)
        right_norm = np.linalg.norm(right)
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return float(np.dot(left, right) / (left_norm * right_norm))
