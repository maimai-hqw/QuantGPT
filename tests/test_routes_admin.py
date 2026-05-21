"""Integration tests for admin routes."""

import pytest

pytestmark = pytest.mark.asyncio


class TestAdminLogin:
    async def test_login_always_succeeds(self, client):
        resp = await client.post("/api/v1/admin/login", json={"password": "anything"})
        assert resp.status_code == 200
        assert "token" in resp.json()


class TestAdminOverview:
    @pytest.mark.skip(reason="admin_overview uses date_trunc which is PostgreSQL-only")
    async def test_accessible_without_token(self, client):
        resp = await client.get("/api/v1/admin/overview")
        assert resp.status_code == 200


class TestAdminUsers:
    async def test_list_users(self, client):
        resp = await client.get("/api/v1/admin/users")
        assert resp.status_code == 200
        data = resp.json()
        assert "users" in data
