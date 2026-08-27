from dataclasses import dataclass
from threading import Lock
from typing import Any, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.architectures import (
    MHAConfig,
    MultiHeadAttention,
    MultiHeadLatentAttention,
)

MAX_INPUT_TOKENS = 10
MAX_DECODED_TOKENS = 11

router = APIRouter(prefix="/api", tags=["attention"])


class RunRequest(BaseModel):
    text: str
    architecture: str = "mha"


class DecodeRequest(BaseModel):
    session_id: str
    new_token: Optional[str] = None


@dataclass
class AttentionSession:
    attention: Any
    state: Any


_sessions: dict[str, AttentionSession] = {}
_sessions_lock = Lock()


def build_attention(architecture: str) -> Any:
    variants = {
        "mha": lambda: MultiHeadAttention(MHAConfig(architecture="mha")),
        "mqa": lambda: MultiHeadAttention(
            MHAConfig(
                architecture="mqa",
                num_q_heads=2,
                num_kv_heads=1,
            )
        ),
        "gqa": lambda: MultiHeadAttention(
            MHAConfig(
                architecture="gqa",
                num_q_heads=4,
                num_kv_heads=2,
            )
        ),
        "rope": lambda: MultiHeadAttention(
            MHAConfig(
                architecture="rope",
                use_rope=True,
            )
        ),
        "mla": MultiHeadLatentAttention,
    }
    factory = variants.get(architecture.lower())
    if factory is None:
        raise ValueError(f"Unsupported attention architecture: {architecture}")
    return factory()


def run_attention(text: str, architecture: str = "mha") -> dict[str, Any]:
    tokens = text.split()
    if not tokens:
        raise ValueError("Please enter some tokens.")

    warnings: list[str] = []
    if len(tokens) > MAX_INPUT_TOKENS:
        tokens = tokens[:MAX_INPUT_TOKENS]
        warnings.append("MVP supports up to 10 tokens. Input was truncated.")

    attention = build_attention(architecture)
    result = attention.prefill(tokens)
    session_id = uuid4().hex
    with _sessions_lock:
        _sessions[session_id] = AttentionSession(attention, result.state)

    payload = result.to_dict()
    payload["session_id"] = session_id
    payload["warnings"] = warnings
    return payload


def decode_attention(
    session_id: str,
    new_token: Optional[str] = None,
) -> dict[str, Any]:
    with _sessions_lock:
        session = _sessions.get(session_id)
        if session is None:
            raise KeyError("Attention session was not found. Run prefill again.")
        if len(session.state.tokens) >= MAX_DECODED_TOKENS:
            raise ValueError("MVP decode supports up to 11 total tokens.")

        decoded_token = (
            new_token.strip()
            if new_token is not None
            else f"token_{len(session.state.tokens) + 1}"
        )
        if not decoded_token:
            raise ValueError("Decode token must not be empty.")

        result = session.attention.decode(session.state, decoded_token)
        session.state = result.state

    payload = result.to_dict()
    payload["session_id"] = session_id
    payload["warnings"] = []
    payload["decoded_token"] = decoded_token
    return payload


def reset_sessions() -> None:
    with _sessions_lock:
        _sessions.clear()


@router.post("/run")
def run(request: RunRequest) -> dict[str, Any]:
    try:
        return run_attention(request.text, request.architecture)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.post("/decode")
def decode(request: DecodeRequest) -> dict[str, Any]:
    try:
        return decode_attention(request.session_id, request.new_token)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=error.args[0]) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
