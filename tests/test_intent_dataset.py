from evaluation.evaluator import load_intent_cases
from core.intent_recognizer import IntentRecognizer


class _Embedding:
    def embed_queries(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]


class _Messages:
    def __init__(self):
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        from types import SimpleNamespace
        return SimpleNamespace(content=[SimpleNamespace(text='{"intent":"query","confidence":0.9,"reasoning":"查询订单","entities":{"order_id":["123"],"product":[],"date":[],"amount":[],"error_code":[]}}')])


class _Client:
    def __init__(self):
        self.messages = _Messages()


def test_default_intent_dataset_is_auditable_and_covers_core_labels():
    cases, provenance = load_intent_cases()
    assert len(cases) >= 40
    assert provenance["dataset_id"] == "echoforge-intent-seed-v1"
    assert provenance["label_status"] == "needs_human_validation"
    labels = {case.expected_intent for case in cases}
    assert {"query", "request", "technical", "billing", "greeting"} <= labels


def test_intent_and_entities_share_one_model_call():
    import asyncio

    client = _Client()
    recognizer = IntentRecognizer(api_key="test", embedding_provider=_Embedding())
    recognizer.client = client
    result = asyncio.run(recognizer.recognize("订单 123 到哪里了？"))
    assert result.intent.value == "query"
    assert result.entities["order_id"] == ["123"]
    assert client.messages.calls == 1
