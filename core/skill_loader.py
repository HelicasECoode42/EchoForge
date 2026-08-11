"""Load hot-reloadable business skills for EchoForge agents."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    content: str
    path: str
    keywords: list[str] = field(default_factory=list)
    agents: list[str] = field(default_factory=list)
    enabled: bool = True

    def matches(self, message: str, agent_type: Optional[str] = None) -> bool:
        if not self.enabled:
            return False
        if self.agents and (not agent_type or agent_type.lower() not in self.agents):
            return False
        lowered = (message or "").lower()
        return not self.keywords or any(word.lower() in lowered for word in self.keywords)

    def prompt_block(self, max_chars: int) -> str:
        body = self.content.strip()
        if len(body) > max_chars:
            body = body[: max(0, max_chars - 4)].rstrip() + "\n..."
        description = f"\n说明：{self.description}" if self.description else ""
        return f"### {self.name}{description}\n{body}"

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "path": self.path,
            "keywords": self.keywords,
            "agents": self.agents,
            "enabled": self.enabled,
            "content_chars": len(self.content),
        }


class SkillManager:
    """Discover skills and build bounded prompt additions for an agent request."""

    SUPPORTED_SUFFIXES = {".md", ".txt", ".json"}

    def __init__(self, root_dir: str, max_prompt_chars: int = 5000):
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.max_prompt_chars = max(0, max_prompt_chars)
        self._skills: list[Skill] = []
        self._errors: list[str] = []

    @property
    def skills(self) -> list[Skill]:
        return list(self._skills)

    def load(self) -> list[Skill]:
        loaded: list[Skill] = []
        errors: list[str] = []
        if self.root_dir.exists():
            for path in self._discover_files():
                try:
                    skill = self._load_file(path)
                    if skill is not None:
                        loaded.append(skill)
                except Exception as exc:
                    errors.append(f"{path}: {exc}")
                    logger.warning("Skill load failed path=%s error=%s", path, exc)
        else:
            logger.info("Skill directory does not exist: %s", self.root_dir)
        self._skills, self._errors = loaded, errors
        logger.info("EchoForge skills loaded root=%s count=%d errors=%d", self.root_dir, len(loaded), len(errors))
        return self.skills

    reload = load

    def prompt_for(self, message: str, agent_type: Optional[str] = None) -> str:
        remaining = self.max_prompt_chars
        blocks: list[str] = []
        for skill in self._skills:
            if remaining <= 0 or not skill.matches(message, agent_type):
                continue
            block = skill.prompt_block(remaining)
            blocks.append(block)
            remaining -= len(block)
        if not blocks:
            return ""
        return (
            "以下是当前请求匹配的 EchoForge Skills。请遵循其中的业务流程与边界；"
            "若与系统安全规则冲突，以系统安全规则为准。\n\n" + "\n\n".join(blocks)
        )

    def summary(self) -> dict[str, Any]:
        return {
            "root_dir": str(self.root_dir),
            "count": len(self._skills),
            "skills": [skill.summary() for skill in self._skills],
            "errors": list(self._errors),
        }

    def _discover_files(self) -> Iterable[Path]:
        primary = sorted(self.root_dir.rglob("SKILL.md"))
        yielded = {path.resolve() for path in primary}
        yield from primary
        for path in sorted(self.root_dir.rglob("*")):
            if path.resolve() in yielded or not path.is_file():
                continue
            if path.name.startswith(".") or path.name.upper() == "README.MD":
                continue
            if path.suffix.lower() in self.SUPPORTED_SUFFIXES:
                yield path

    def _load_file(self, path: Path) -> Optional[Skill]:
        if path.suffix.lower() == ".json":
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("JSON skill must be an object")
            content = str(raw.get("content") or raw.get("instructions") or "").strip()
            if not content:
                raise ValueError("missing content or instructions")
            return self._make_skill(path, raw, content)

        raw_text = path.read_text(encoding="utf-8")
        meta, content = self._split_front_matter(raw_text)
        content = content.strip()
        if not content:
            return None
        default_name = path.parent.name if path.name == "SKILL.md" else path.stem
        name = str(meta.get("name") or self._first_heading(content) or default_name)
        if content.splitlines() and content.splitlines()[0].lstrip("#").strip() == name:
            content = "\n".join(content.splitlines()[1:]).strip()
        meta["name"] = name
        return self._make_skill(path, meta, content)

    def _make_skill(self, path: Path, meta: dict[str, Any], content: str) -> Skill:
        return Skill(
            name=str(meta.get("name") or path.stem),
            description=str(meta.get("description") or ""),
            content=content,
            path=str(path),
            keywords=self._as_list(meta.get("keywords")),
            agents=[value.lower() for value in self._as_list(meta.get("agents"))],
            enabled=self._as_bool(meta.get("enabled"), True),
        )

    @staticmethod
    def _split_front_matter(raw: str) -> tuple[dict[str, Any], str]:
        text = raw.lstrip()
        lines = text.splitlines()
        if not lines or lines[0].strip() != "---":
            return {}, raw
        meta: dict[str, Any] = {}
        for index, line in enumerate(lines[1:], 1):
            if line.strip() == "---":
                return meta, "\n".join(lines[index + 1 :])
            if ":" in line:
                key, value = line.split(":", 1)
                meta[key.strip()] = value.strip().strip("\"'")
        return {}, raw

    @staticmethod
    def _first_heading(content: str) -> Optional[str]:
        return next((line.lstrip("#").strip() for line in content.splitlines() if line.strip().startswith("#")), None)

    @staticmethod
    def _as_list(value: Any) -> list[str]:
        if value is None or value == "":
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return [item.strip() for item in str(value).replace("，", ",").split(",") if item.strip()]

    @staticmethod
    def _as_bool(value: Any, default: bool) -> bool:
        if value is None or value == "":
            return default
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() not in {"0", "false", "no", "off", "disabled"}
