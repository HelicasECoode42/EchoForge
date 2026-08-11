from pathlib import Path

from core.skill_loader import SkillManager


def test_skill_manager_filters_by_agent_and_keyword(tmp_path: Path):
    skill_dir = tmp_path / "refund"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        """---
name: Refund Guard
description: billing policy
keywords: 退款，refund
agents: billing
enabled: true
---
Never promise an unverified refund.
""",
        encoding="utf-8",
    )

    manager = SkillManager(str(tmp_path), max_prompt_chars=500)
    manager.load()

    assert "Refund Guard" in manager.prompt_for("我要退款", "billing")
    assert manager.prompt_for("我要退款", "technical") == ""
    assert manager.prompt_for("普通咨询", "billing") == ""


def test_skill_summary_excludes_instruction_content(tmp_path: Path):
    (tmp_path / "global.txt").write_text("private instructions", encoding="utf-8")
    manager = SkillManager(str(tmp_path))
    manager.load()

    summary = manager.summary()
    assert summary["count"] == 1
    assert "content" not in summary["skills"][0]
    assert summary["skills"][0]["content_chars"] == len("private instructions")
