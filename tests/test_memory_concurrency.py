import asyncio
import json
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from memory.conversation_memory import MemoryManager, MsgRole


def _raw(index: int) -> str:
    return json.dumps({
        "role": MsgRole.USER.value,
        "content": f"message-{index}",
        "ts": datetime.now().isoformat(),
        "metadata": {},
    })


class _Redis:
    def __init__(self, key: str, values: list[str]):
        self.lists = {key: values}
        self.values = {}

    def lrange(self, key, start, end):
        values = self.lists[key]
        return list(values[start:] if end == -1 else values[start:end + 1])

    def eval(self, script, key_count, key, summary_key, summary, *oldest_first):
        values = self.lists[key]
        for index, expected in enumerate(oldest_first, start=1):
            if values[-index] != expected:
                return 0
        del values[-len(oldest_first):]
        old_summary = self.values.get(summary_key, "")
        self.values[summary_key] = f"{old_summary}\n{summary}".strip()
        return 1


def test_compression_preserves_messages_appended_while_summary_is_generated():
    key = MemoryManager._wm_key("user", "conversation")
    initial = list(reversed([_raw(index) for index in range(15)]))
    redis_client = _Redis(key, initial)
    manager = MemoryManager.__new__(MemoryManager)
    manager._redis = redis_client
    manager._client = object()
    manager._model = "test-model"
    manager._compression_locks = {}
    manager._store_episodic = AsyncMock()
    concurrent_message = _raw(99)

    async def create_summary(*args, **kwargs):
        redis_client.lists[key].insert(0, concurrent_message)
        await asyncio.sleep(0)
        return SimpleNamespace(content=[SimpleNamespace(text="summary")])

    with patch("memory.conversation_memory.create_message", side_effect=create_summary):
        asyncio.run(manager._compress("user", "conversation"))

    assert redis_client.lists[key][0] == concurrent_message
    assert len(redis_client.lists[key]) == 6
    assert redis_client.values[MemoryManager._summary_key("user", "conversation")] == "summary"
    manager._store_episodic.assert_awaited_once()
