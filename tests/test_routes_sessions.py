"""Integration tests for session and factor-library routes."""

import uuid

import pytest

pytestmark = pytest.mark.asyncio


class TestSessionCRUD:
    async def test_create_session(self, client):
        resp = await client.post("/api/v1/sessions", json={"name": "My Session"})
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "My Session"
        assert "id" in data

    async def test_list_sessions(self, client):
        await client.post("/api/v1/sessions", json={"name": "S1"})
        await client.post("/api/v1/sessions", json={"name": "S2"})
        resp = await client.get("/api/v1/sessions")
        assert resp.status_code == 200
        sessions = resp.json()["sessions"]
        assert len(sessions) >= 2

    async def test_rename_session(self, client):
        create_resp = await client.post("/api/v1/sessions", json={"name": "Old"})
        sid = create_resp.json()["id"]
        resp = await client.patch(f"/api/v1/sessions/{sid}", json={"name": "New"})
        assert resp.status_code == 200
        assert resp.json()["name"] == "New"

    async def test_delete_session(self, client):
        create_resp = await client.post("/api/v1/sessions", json={"name": "ToDelete"})
        sid = create_resp.json()["id"]
        resp = await client.delete(f"/api/v1/sessions/{sid}")
        assert resp.status_code == 204

    async def test_delete_nonexistent_session_404(self, client):
        fake_id = str(uuid.uuid4())
        resp = await client.delete(f"/api/v1/sessions/{fake_id}")
        assert resp.status_code == 404


class TestFactorLibrary:
    async def test_save_factor(self, client):
        resp = await client.post("/api/v1/factor-library", json={
            "expression": "rank(close)",
            "name": "Test Factor",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["expression"] == "rank(close)"
        assert data["name"] == "Test Factor"

    async def test_duplicate_factor_rejected(self, client):
        await client.post("/api/v1/factor-library", json={
            "expression": "rank(volume)",
        })
        resp = await client.post("/api/v1/factor-library", json={
            "expression": "rank(volume)",
        })
        assert resp.status_code == 409

    async def test_list_factors(self, client):
        await client.post("/api/v1/factor-library", json={
            "expression": "ts_mean(close, 5)",
        })
        resp = await client.get("/api/v1/factor-library")
        assert resp.status_code == 200
        factors = resp.json()["factors"]
        assert len(factors) >= 1

    async def test_update_factor(self, client):
        create_resp = await client.post("/api/v1/factor-library", json={
            "expression": "zscore(close)",
        })
        fid = create_resp.json()["id"]
        resp = await client.patch(f"/api/v1/factor-library/{fid}", json={
            "name": "Updated Name",
            "note": "Added a note",
        })
        assert resp.status_code == 200
        assert resp.json()["name"] == "Updated Name"
        assert resp.json()["note"] == "Added a note"

    async def test_delete_factor(self, client):
        create_resp = await client.post("/api/v1/factor-library", json={
            "expression": "rank(amount)",
        })
        fid = create_resp.json()["id"]
        resp = await client.delete(f"/api/v1/factor-library/{fid}")
        assert resp.status_code == 204
