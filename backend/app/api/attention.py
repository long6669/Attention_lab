from collections import OrderedDict
from dataclasses import dataclass, field
from threading import Lock
from time import monotonic
from typing import Any, Optional
from uuid import uuid4

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.architectures import (
    CompressedSparseAttention,
    KimiDeltaAttention,
    MHAConfig,
    MultiHeadAttention,
    MultiHeadLatentAttention,
)
from app.memory import infer_axes, json_values, slice_memory
from app.metrics import build_execution_metrics

MAX_INPUT_TOKENS = 10
MAX_DECODED_TOKENS = 11
MAX_SESSIONS = 256
SESSION_TTL_SECONDS = 30 * 60

router = APIRouter(prefix="/api", tags=["attention"])


class RunRequest(BaseModel):
    text: str
    architecture: str = "mha"


class DecodeRequest(BaseModel):
    session_id: str
    new_token: Optional[str] = None


class MemorySliceRequest(BaseModel):
    session_id: str
    memory_id: str
    start: int = 0
    end: int = 8
    head: Optional[int] = None


@dataclass
class AttentionSession:
    attention: Any
    state: Any
    last_access: float = field(default_factory=monotonic)
    lock: Lock = field(default_factory=Lock)


_sessions: OrderedDict[str, AttentionSession] = OrderedDict()
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
        "kda": KimiDeltaAttention,
        "csa": lambda: CompressedSparseAttention(
            MHAConfig(
                architecture="csa",
                routing_top_k=2,
            )
        ),
        "hca": lambda: CompressedSparseAttention(
            MHAConfig(
                architecture="hca",
                routing_top_k=3,
            )
        ),
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
        _prune_expired_sessions()
        if len(_sessions) >= MAX_SESSIONS:
            _sessions.popitem(last=False)
        _sessions[session_id] = AttentionSession(attention, result.state)

    payload = result.to_dict()
    payload["session_id"] = session_id
    payload["warnings"] = warnings
    payload["metrics"] = build_execution_metrics(payload)
    return payload


def decode_attention(
    session_id: str,
    new_token: Optional[str] = None,
) -> dict[str, Any]:
    session = _get_session(session_id)
    with session.lock:
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
        session.last_access = monotonic()

    payload = result.to_dict()
    payload["session_id"] = session_id
    payload["warnings"] = []
    payload["decoded_token"] = decoded_token
    payload["metrics"] = build_execution_metrics(payload)
    return payload


def _get_session(session_id: str) -> AttentionSession:
    with _sessions_lock:
        _prune_expired_sessions()
        session = _sessions.get(session_id)
        if session is None:
            raise KeyError("Attention session was not found. Run prefill again.")
        session.last_access = monotonic()
        _sessions.move_to_end(session_id)
        return session


def _prune_expired_sessions(now: Optional[float] = None) -> None:
    current_time = monotonic() if now is None else now
    expired = [
        session_id
        for session_id, session in _sessions.items()
        if current_time - session.last_access >= SESSION_TTL_SECONDS
    ]
    for session_id in expired:
        del _sessions[session_id]


def reset_sessions() -> None:
    with _sessions_lock:
        _sessions.clear()


def read_memory_slice(request: MemorySliceRequest) -> dict[str, Any]:
    session = _get_session(request.session_id)
    with session.lock:
        value = getattr(session.state, request.memory_id, None)
        if not isinstance(value, np.ndarray):
            raise KeyError(f"Memory tensor was not found: {request.memory_id}")
        array = value.copy()
        session.last_access = monotonic()

    axes, growth_axis = infer_axes(array, request.memory_id)
    if request.start < 0 or request.end <= request.start:
        raise ValueError("Memory slice range must satisfy 0 <= start < end.")
    if growth_axis is not None and request.start >= array.shape[growth_axis]:
        raise ValueError("Memory slice start is outside the growth axis.")
    if request.head is not None and "head" in axes:
        head_axis = axes.index("head")
        if request.head < 0 or request.head >= array.shape[head_axis]:
            raise ValueError("Memory slice head is outside the head axis.")

    sliced = slice_memory(
        array,
        axes,
        growth_axis,
        request.start,
        request.end,
        request.head,
    )
    return {
        "id": request.memory_id,
        "shape": list(sliced.shape),
        "dtype": str(sliced.dtype),
        "numel": int(sliced.size),
        "bytes": int(sliced.nbytes),
        "axes": list(axes),
        "selection": {
            "start": request.start,
            "end": min(
                request.end,
                array.shape[growth_axis] if growth_axis is not None else 1,
            ),
            "head": request.head,
        },
        "values": json_values(sliced),
    }


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


@router.post("/memory/slice")
def memory_slice(request: MemorySliceRequest) -> dict[str, Any]:
    try:
        return read_memory_slice(request)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=error.args[0]) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
