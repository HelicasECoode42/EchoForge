"""Auditable document chunking strategies for retrieval experiments."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple


TOKEN_RE = re.compile(r"[\u3400-\u9fff]|[A-Za-z0-9_]+|[^\s]", re.UNICODE)
SENTENCE_RE = re.compile(r"[^。！？!?；;\n]+[。！？!?；;]?|\n", re.UNICODE)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def estimate_tokens(text: str) -> int:
    """Deterministic language-agnostic proxy used for budgets and offline cost."""

    return len(TOKEN_RE.findall(text or ""))


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    title: str
    content: str
    strategy: str
    index: int
    total_chunks: int
    char_start: int
    char_end: int
    token_count: int
    content_sha256: str
    document_version: str = "1"
    section_path: str = ""
    previous_chunk_id: str = ""
    next_chunk_id: str = ""

    def metadata(self) -> Dict[str, Any]:
        data = asdict(self)
        data.pop("content")
        return data


class Chunker(Protocol):
    name: str

    def chunk(
        self,
        *,
        document_id: str,
        title: str,
        text: str,
        document_version: str = "1",
    ) -> List[Chunk]: ...


@dataclass(frozen=True)
class _Draft:
    content: str
    char_start: int
    char_end: int
    section_path: str = ""


class _BaseChunker:
    name = "base"

    def _finalize(
        self,
        drafts: Sequence[_Draft],
        *,
        document_id: str,
        title: str,
        document_version: str,
    ) -> List[Chunk]:
        clean = [draft for draft in drafts if draft.content.strip()]
        ids = [
            hashlib.sha256(
                f"{document_id}:{document_version}:{self.name}:{i}:{draft.char_start}:{draft.content}".encode(
                    "utf-8", errors="ignore"
                )
            ).hexdigest()[:24]
            for i, draft in enumerate(clean)
        ]
        chunks: List[Chunk] = []
        for i, draft in enumerate(clean):
            chunks.append(Chunk(
                chunk_id=ids[i],
                document_id=document_id,
                title=title,
                content=draft.content,
                strategy=self.name,
                index=i,
                total_chunks=len(clean),
                char_start=draft.char_start,
                char_end=draft.char_end,
                token_count=estimate_tokens(draft.content),
                content_sha256=hashlib.sha256(draft.content.encode("utf-8", errors="ignore")).hexdigest(),
                document_version=document_version,
                section_path=draft.section_path,
                previous_chunk_id=ids[i - 1] if i > 0 else "",
                next_chunk_id=ids[i + 1] if i + 1 < len(ids) else "",
            ))
        return chunks


class FixedCharacterChunker(_BaseChunker):
    """Baseline: strict fixed character windows without overlap."""

    name = "fixed_char"

    def __init__(self, chunk_size: int = 500):
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        self.chunk_size = chunk_size

    def chunk(self, *, document_id: str, title: str, text: str, document_version: str = "1") -> List[Chunk]:
        drafts = [
            _Draft(text[start:start + self.chunk_size], start, min(len(text), start + self.chunk_size))
            for start in range(0, len(text), self.chunk_size)
        ]
        return self._finalize(
            drafts,
            document_id=document_id,
            title=title,
            document_version=document_version,
        )


class SlidingWindowChunker(_BaseChunker):
    """Character windows with overlap to protect evidence at hard boundaries."""

    name = "sliding_window"

    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, *, document_id: str, title: str, text: str, document_version: str = "1") -> List[Chunk]:
        step = self.chunk_size - self.overlap
        drafts: List[_Draft] = []
        for start in range(0, len(text), step):
            end = min(len(text), start + self.chunk_size)
            drafts.append(_Draft(text[start:end], start, end))
            if end == len(text):
                break
        return self._finalize(
            drafts,
            document_id=document_id,
            title=title,
            document_version=document_version,
        )


class StructureAwareTokenChunker(_BaseChunker):
    """Markdown/paragraph-aware chunks packed under a deterministic token budget."""

    name = "structure_token"

    def __init__(self, max_tokens: int = 320):
        if max_tokens < 8:
            raise ValueError("max_tokens must be >= 8")
        self.max_tokens = max_tokens

    def chunk(self, *, document_id: str, title: str, text: str, document_version: str = "1") -> List[Chunk]:
        units = self._semantic_units(text)
        drafts: List[_Draft] = []
        current: List[Tuple[str, int, int, str]] = []
        current_tokens = 0

        def flush() -> None:
            nonlocal current, current_tokens
            if not current:
                return
            content = "\n".join(unit[0].strip() for unit in current if unit[0].strip())
            drafts.append(_Draft(
                content=content,
                char_start=current[0][1],
                char_end=current[-1][2],
                section_path=current[-1][3],
            ))
            current = []
            current_tokens = 0

        for unit_text, start, end, section_path in units:
            unit_tokens = estimate_tokens(unit_text)
            if unit_tokens > self.max_tokens:
                flush()
                for piece_text, piece_start, piece_end in self._split_oversized(unit_text, start):
                    drafts.append(_Draft(piece_text, piece_start, piece_end, section_path))
                continue
            if current and current_tokens + unit_tokens > self.max_tokens:
                flush()
            current.append((unit_text, start, end, section_path))
            current_tokens += unit_tokens
        flush()

        return self._finalize(
            drafts,
            document_id=document_id,
            title=title,
            document_version=document_version,
        )

    def _semantic_units(self, text: str) -> List[Tuple[str, int, int, str]]:
        units: List[Tuple[str, int, int, str]] = []
        heading_stack: List[str] = []
        offset = 0
        for line in text.splitlines(keepends=True):
            line_without_break = line.rstrip("\r\n")
            heading = HEADING_RE.match(line_without_break.strip())
            if heading:
                level = len(heading.group(1))
                heading_stack = heading_stack[:level - 1]
                heading_stack.append(heading.group(2).strip())
                offset += len(line)
                continue

            section_path = " > ".join(heading_stack)
            for match in SENTENCE_RE.finditer(line_without_break):
                content = match.group(0).strip()
                if content:
                    start = offset + match.start()
                    units.append((content, start, offset + match.end(), section_path))
            offset += len(line)
        return units

    def _split_oversized(self, text: str, base_offset: int) -> Iterable[Tuple[str, int, int]]:
        matches = list(TOKEN_RE.finditer(text))
        for index in range(0, len(matches), self.max_tokens):
            group = matches[index:index + self.max_tokens]
            if not group:
                continue
            local_start, local_end = group[0].start(), group[-1].end()
            yield text[local_start:local_end], base_offset + local_start, base_offset + local_end


def create_chunker(name: str, **kwargs: Any) -> Chunker:
    normalized = (name or "").strip().lower().replace("-", "_")
    if normalized in {"fixed", "fixed_char", "character"}:
        return FixedCharacterChunker(**kwargs)
    if normalized in {"sliding", "sliding_window", "overlap"}:
        return SlidingWindowChunker(**kwargs)
    if normalized in {"structure", "structure_token", "token"}:
        return StructureAwareTokenChunker(**kwargs)
    raise ValueError(f"unknown chunking strategy: {name}")
