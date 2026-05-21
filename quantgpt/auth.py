"""Minimal auth stub — QuantGPT runs without authentication.

All endpoints receive a fixed local user. No JWT, no passwords, no API keys.
"""

import uuid as _uuid_mod

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from .db import get_db
from .models import User

_DEV_USER_ID = _uuid_mod.UUID("00000000-0000-0000-0000-000000000099")
GUEST_USER_ID = "00000000-0000-0000-0000-000000000001"


def _get_dev_user() -> User:
    return User(id=_DEV_USER_ID, email="dev@localhost", nickname="Local User")


async def get_current_user(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> User:
    return _get_dev_user()


async def get_optional_user(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> User | None:
    return _get_dev_user()


async def require_admin(request: Request) -> bool:
    return True


def decode_token(token: str) -> dict:
    return {"sub": str(_DEV_USER_ID), "type": "access"}
