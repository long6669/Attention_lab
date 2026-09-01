from math import prod
from typing import Any, Optional


def build_execution_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    """Build comparable, shape-based metrics from an executed graph."""
    graph = payload["graph"]
    tensors = payload["tensors"]
    parameter_ids = {
        parameter_id
        for node in graph["nodes"]
        for parameter_id in node["attrs"].get("parameter_ids", [])
    }
    estimated_flops = sum(
        _estimate_node_flops(node, tensors) for node in graph["nodes"]
    )
    parameter_count = sum(
        tensors[tensor_id]["numel"]
        for tensor_id in parameter_ids
        if tensor_id in tensors
    )
    activation_bytes = sum(
        tensor["bytes"]
        for tensor_id, tensor in tensors.items()
        if tensor_id not in parameter_ids
    )
    memory = payload["memory"]
    tokens = max(int(memory["tokens"]), 1)
    growth = "constant" if memory["cache_kind"] == "recurrent" else "linear"
    bytes_per_token = (
        0 if growth == "constant" else int(memory["total_bytes"]) // tokens
    )

    return {
        "estimated_flops": estimated_flops,
        "flops_basis": "shape_estimate",
        "parameter_count": parameter_count,
        "activation_bytes": activation_bytes,
        "graph_nodes": len(graph["nodes"]),
        "graph_edges": len(graph["edges"]),
        "trace_steps": len(payload["trace"]),
        "memory_growth": growth,
        "memory_bytes_per_token": bytes_per_token,
    }


def _estimate_node_flops(
    node: dict[str, Any],
    tensors: dict[str, dict[str, Any]],
) -> int:
    output = _first_tensor(node["outputs"], tensors)
    inputs = [
        tensors[tensor_id] for tensor_id in node["inputs"] if tensor_id in tensors
    ]
    if output is None:
        return 0

    output_numel = int(output["numel"])
    op = node["op"]
    if op in {"linear", "low_rank_compression"} and inputs:
        reduction = inputs[0]["shape"][-1]
        return 2 * output_numel * int(reduction)
    if op in {"matmul", "indexer"} and inputs:
        reduction = inputs[0]["shape"][-1]
        return 2 * output_numel * int(reduction)
    if op == "softmax":
        return 5 * output_numel
    if op == "rope":
        return 3 * output_numel
    if op in {"scale", "decay", "erase", "write"}:
        return output_numel
    if op == "scan" and len(inputs) >= 2:
        feature_size = inputs[0]["shape"][-1]
        return 2 * output_numel * int(feature_size)
    if op == "sequence_compression" and inputs:
        return int(prod(inputs[0]["shape"]))
    if op == "routing" and len(inputs) >= 2:
        return 2 * output_numel * int(inputs[-1]["shape"][-1])
    return 0


def _first_tensor(
    tensor_ids: list[str],
    tensors: dict[str, dict[str, Any]],
) -> Optional[dict[str, Any]]:
    if not tensor_ids:
        return None
    return tensors.get(tensor_ids[0])
