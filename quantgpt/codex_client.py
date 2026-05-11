"""Codex CLI subprocess wrapper — drives the OpenAI Codex agent through
a ChatGPT subscription instead of paying per-token via API.

Provides a `chat_complete()` callable shaped like a minimal subset of
openai.OpenAI().chat.completions.create() so callers can swap providers
by switching LLM_PROVIDER=codex in .env.

Requires `codex` CLI on PATH (install via npm: `npm i -g @openai/codex`).
The user must be logged in (`codex login`) with an active ChatGPT
subscription.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass
from typing import Iterable

logger = logging.getLogger(__name__)

CODEX_BIN = os.environ.get("CODEX_BIN", "codex")
DEFAULT_TIMEOUT = int(os.environ.get("CODEX_TIMEOUT_SEC", "180"))


@dataclass
class CodexResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0


def _flatten_messages(messages: Iterable[dict]) -> str:
    """Concatenate chat-format messages into a single string prompt.

    Codex exec takes a single prompt; we render system/user roles inline
    so the agent sees the same instructions it would via the API."""
    parts = []
    for m in messages:
        role = m.get("role", "user").upper()
        content = m.get("content", "")
        if isinstance(content, list):
            content = "\n".join(c.get("text", "") for c in content if isinstance(c, dict))
        parts.append(f"[{role}]\n{content}")
    return "\n\n".join(parts)


def chat_complete(
    messages: list[dict],
    model: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> CodexResponse:
    """One-shot chat completion via `codex exec --json`.

    `messages` is a chat-completions style list. `model` is passed via -m.
    Returns CodexResponse with extracted text and usage.

    Raises RuntimeError on non-zero exit or missing agent_message.
    """
    prompt = _flatten_messages(messages)

    cmd = [
        CODEX_BIN,
        "exec",
        "--skip-git-repo-check",
        "--json",
        "-c", 'sandbox_mode="read-only"',
        "-c", 'approval_policy="never"',
    ]
    if model:
        cmd += ["-m", model]
    effort = os.environ.get("CODEX_REASONING_EFFORT")
    if effort:
        cmd += ["-c", f'model_reasoning_effort="{effort}"']

    try:
        proc = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"codex exec timed out after {timeout}s")

    if proc.returncode != 0:
        stderr_tail = (proc.stderr or "")[-500:]
        raise RuntimeError(f"codex exec failed (exit={proc.returncode}): {stderr_tail}")

    text = ""
    input_tokens = output_tokens = reasoning_tokens = 0
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            continue
        if evt.get("type") == "item.completed":
            item = evt.get("item") or {}
            if item.get("type") == "agent_message":
                text = (item.get("text") or "").strip()
        elif evt.get("type") == "turn.completed":
            usage = evt.get("usage") or {}
            input_tokens = int(usage.get("input_tokens") or 0)
            output_tokens = int(usage.get("output_tokens") or 0)
            reasoning_tokens = int(usage.get("reasoning_output_tokens") or 0)

    if not text:
        # Fallback: scan plain stdout for last non-empty line (some codex
        # versions emit human-readable output even with --json)
        for line in reversed(proc.stdout.splitlines()):
            s = line.strip()
            if s and not s.startswith("{") and not s.startswith("---"):
                text = s
                break

    if not text:
        raise RuntimeError(f"codex exec returned no agent_message; stdout tail: {proc.stdout[-300:]!r}")

    return CodexResponse(
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
    )
