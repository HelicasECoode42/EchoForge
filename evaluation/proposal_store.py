"""Small local proposal ledger for offline review state."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path
from threading import RLock
from typing import Optional

from evaluation.improvement import ImprovementProposal, ProposalStatus


class ProposalStore:
    """Persist proposal artifacts without exposing a production apply path.

    The file is an offline review ledger, not a runtime configuration store.
    Store instances for the same canonical path share one process-level lock.
    Writes use a temporary file and replace so an interrupted review cannot
    leave a partially written JSON document.
    """

    SCHEMA_VERSION = "1"
    _LOCKS_GUARD = RLock()
    _LOCKS: dict[Path, RLock] = {}

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        with self._LOCKS_GUARD:
            self._lock = self._LOCKS.setdefault(self.path, RLock())

    def list(self) -> list[ImprovementProposal]:
        with self._lock:
            payload = self._read()
        items = [ImprovementProposal.from_dict(item) for item in payload.get("items", [])]
        return sorted(items, key=lambda item: (item.created_at, item.proposal_id), reverse=True)

    def get(self, proposal_id: str) -> Optional[ImprovementProposal]:
        return next((item for item in self.list() if item.proposal_id == proposal_id), None)

    def save(self, proposal: ImprovementProposal) -> ImprovementProposal:
        with self._lock:
            payload = self._read()
            items = [ImprovementProposal.from_dict(item) for item in payload.get("items", [])]
            existing = next((item for item in items if item.proposal_id == proposal.proposal_id), None)
            if existing is None:
                items.append(proposal)
            else:
                self._validate_identity(existing, proposal)
                if proposal.status is ProposalStatus.PROPOSED:
                    # Proposal generation is idempotent. Never replace an
                    # evaluated artifact with a fresh proposed shell.
                    return existing
                self._validate_replacement(existing, proposal)
                proposal = self._preserve_evaluation_history(existing, proposal)
                items = [proposal if item.proposal_id == proposal.proposal_id else item for item in items]
            self._write(items)
        return proposal

    def approve(self, proposal_id: str, *, approved_at: Optional[str] = None) -> ImprovementProposal:
        proposal = self._require(proposal_id)
        return self.save(proposal.approve(approved_at=approved_at))

    def reject(self, proposal_id: str) -> ImprovementProposal:
        proposal = self._require(proposal_id)
        return self.save(proposal.reject())

    def _require(self, proposal_id: str) -> ImprovementProposal:
        proposal = self.get(proposal_id)
        if proposal is None:
            raise KeyError(f"proposal not found: {proposal_id}")
        return proposal

    @staticmethod
    def _validate_identity(
        existing: ImprovementProposal,
        replacement: ImprovementProposal,
    ) -> None:
        immutable = (
            "proposal_version",
            "source_trace_ids",
            "target",
            "parameters",
            "failure_types",
            "description",
        )
        if any(getattr(existing, name) != getattr(replacement, name) for name in immutable):
            raise ValueError("proposal identity fields cannot change in the ledger")

    @staticmethod
    def _validate_replacement(
        existing: ImprovementProposal,
        replacement: ImprovementProposal,
    ) -> None:
        if existing.status in {
            ProposalStatus.APPROVED,
            ProposalStatus.REJECTED,
            ProposalStatus.NO_CHANGE,
        }:
            if replacement.to_dict() == existing.to_dict():
                return
            raise ValueError(f"{existing.status.value} proposals are terminal and immutable")
        allowed = {
            ProposalStatus.PROPOSED: {ProposalStatus.PROPOSED, ProposalStatus.CANDIDATE, ProposalStatus.NO_CHANGE, ProposalStatus.REJECTED},
            ProposalStatus.CANDIDATE: {ProposalStatus.CANDIDATE, ProposalStatus.APPROVED, ProposalStatus.REJECTED},
        }
        if replacement.status not in allowed[existing.status]:
            raise ValueError(
                f"invalid proposal transition: {existing.status.value} -> {replacement.status.value}"
            )

    @staticmethod
    def _preserve_evaluation_history(
        existing: ImprovementProposal,
        replacement: ImprovementProposal,
    ) -> ImprovementProposal:
        """Merge new evidence without discarding prior evaluation artifacts."""
        history = list(existing.evaluation_history)
        incoming = replacement.evaluation_history
        if replacement.evaluation is not None and not incoming:
            incoming = (replacement.evaluation,)
        for evaluation in incoming:
            if evaluation not in history:
                history.append(evaluation)
        return replace(replacement, evaluation_history=tuple(history))

    def _read(self) -> dict:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return {"schema_version": self.SCHEMA_VERSION, "items": []}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != self.SCHEMA_VERSION:
            raise ValueError(f"unsupported proposal ledger schema: {payload.get('schema_version')}")
        if not isinstance(payload.get("items"), list):
            raise ValueError("proposal ledger items must be a list")
        return payload

    def _write(self, items: list[ImprovementProposal]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "items": [item.to_dict() for item in items],
        }
        fd, temp_name = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            os.replace(temp_name, self.path)
        except Exception:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
            raise
