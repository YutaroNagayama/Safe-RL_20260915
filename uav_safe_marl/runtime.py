"""Cross-platform runtime configuration and reproducibility metadata."""

from __future__ import annotations

import os
import platform
import sys
from typing import Any


def configure_runtime(config: Any) -> None:
    """Apply execution-only Torch settings without changing exact-profile math."""
    try:
        import torch
    except ImportError:
        return
    performance = config.performance
    if performance.torch_num_threads:
        torch.set_num_threads(performance.torch_num_threads)
    try:
        torch.set_num_interop_threads(performance.torch_interop_threads)
    except RuntimeError:
        pass
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if performance.allow_tf32 else "highest")
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = bool(performance.allow_tf32)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = bool(performance.allow_tf32)
        torch.backends.cudnn.deterministic = bool(performance.deterministic_algorithms)
        torch.backends.cudnn.benchmark = not bool(performance.deterministic_algorithms)
    torch.use_deterministic_algorithms(bool(performance.deterministic_algorithms), warn_only=True)


def runtime_manifest(config: Any | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "python": sys.version,
        "logical_cpu_count": os.cpu_count(),
    }
    try:
        import numpy as np
        payload["numpy"] = np.__version__
    except ImportError:
        payload["numpy"] = None
    try:
        import gymnasium
        payload["gymnasium"] = gymnasium.__version__
    except ImportError:
        payload["gymnasium"] = None
    try:
        import torch
        payload.update({
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "torch_cuda": torch.version.cuda,
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
            "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        })
        if torch.cuda.is_available():
            payload["cuda_device_memory_bytes"] = [torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())]
    except ImportError:
        payload.update({"torch": None, "cuda_available": False, "mps_available": False})
    if config is not None:
        payload["requested_device"] = config.device
        payload["performance"] = config.performance.__dict__ if hasattr(config.performance, "__dict__") else {
            key: getattr(config.performance, key) for key in config.performance.__slots__
        }
    return payload
