# EchoForge Skills

Skills are hot-reloadable business instructions injected into the matching Agent prompt. Put each skill in `skills/<name>/SKILL.md`, then call `POST /skills/reload`. `GET /skills` returns metadata and parse errors without exposing full instruction text.

Supported front matter fields are `name`, `description`, comma-separated `keywords`, comma-separated `agents` (`general`, `technical`, `billing`), and `enabled`.

Keep authorization, privacy, escalation, and forbidden actions explicit. Skills guide model output; they do not grant tools or bypass system safety rules.
