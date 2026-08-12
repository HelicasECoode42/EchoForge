from __future__ import annotations

import httpx
import pytest

import api.main as main


@pytest.fixture(autouse=True)
def controlled_offline_environment(monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")


@pytest.mark.anyio
async def test_improvement_api_evaluates_lists_and_approves_candidate(tmp_path, monkeypatch):
    monkeypatch.setenv("IMPROVEMENT_PROPOSAL_PATH", str(tmp_path / "proposals.json"))
    monkeypatch.setenv(
        "IMPROVEMENT_DATASET_PATH",
        str(main.pathlib.Path(main._ROOT) / "data" / "chunking" / "cases.json"),
    )
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/improvement/evaluate",
            json={
                "proposal_version": "api.chunking.v1",
                "source_trace_ids": ["trace-api-1"],
                "target": "chunk_strategy",
                "parameters": {"strategy": "structure_token", "max_tokens": 140},
                "baseline_parameters": {"strategy": "fixed_char", "chunk_size": 220},
                "failure_types": ["no_recall"],
                "description": "Use structure-aware chunks to protect retrievable evidence boundaries.",
            },
        )
        assert response.status_code == 200
        payload = response.json()
        proposal_id = payload["proposal"]["proposal_id"]
        assert payload["proposal"]["status"] == "candidate"
        assert payload["evaluation"]["passed_regression"] is True

        generated = await client.post(
            "/improvement/proposals/generate",
            json={
                "proposal_version": "api.chunking.v1",
                "traces": [
                    {
                        "trace_id": "trace-api-1",
                        "status": "blocked",
                        "verification_reason": "insufficient_evidence",
                        "should_retrieve": True,
                        "evidence_ids": [],
                    }
                ],
            },
        )
        assert generated.status_code == 200
        assert generated.json()["items"][0]["proposal_id"] == proposal_id
        assert generated.json()["items"][0]["status"] == "candidate"

        listed = await client.get("/improvement/proposals?status=candidate")
        assert listed.status_code == 200
        assert [item["proposal_id"] for item in listed.json()["items"]] == [proposal_id]

        approved = await client.post(f"/improvement/proposals/{proposal_id}/approve")
        assert approved.status_code == 200
        assert approved.json()["proposal"]["status"] == "approved"


@pytest.mark.anyio
async def test_improvement_api_rejects_unsupported_replay_target(tmp_path, monkeypatch):
    monkeypatch.setenv("IMPROVEMENT_PROPOSAL_PATH", str(tmp_path / "proposals.json"))
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/improvement/evaluate",
            json={
                "proposal_version": "api.prompt.v1",
                "source_trace_ids": ["trace-api-2"],
                "target": "prompt_version",
                "parameters": {"version": "prompt-v2"},
                "baseline_parameters": {"strategy": "fixed_char", "chunk_size": 220},
            },
        )
    assert response.status_code == 400
    assert "no isolated replay adapter" in response.json()["detail"]


@pytest.mark.anyio
async def test_improvement_api_generates_proposals_from_trace_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("IMPROVEMENT_PROPOSAL_PATH", str(tmp_path / "proposals.json"))
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/improvement/proposals/generate",
            json={
                "traces": [
                    {
                        "trace_id": "trace-api-no-recall",
                        "status": "blocked",
                        "verification_reason": "insufficient_evidence",
                        "should_retrieve": True,
                        "evidence_ids": [],
                    }
                ]
            },
        )
    assert response.status_code == 200
    assert response.json()["items"][0]["failure_types"] == ["no_recall"]


@pytest.mark.anyio
async def test_improvement_api_is_disabled_in_production(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/improvement/proposals")
    assert response.status_code == 403
