from __future__ import annotations

import json
import subprocess
from functools import lru_cache
from pathlib import Path

from .models import RuntimeInfo

_PROBE = r"""
import json
import platform
import sys

info = {
    "available": True,
    "python": sys.executable,
    "python_version": platform.python_version(),
    "torch_version": None,
    "transformers_version": None,
    "cuda_available": None,
    "cuda_device_count": None,
    "npu_available": None,
    "error": None,
}

try:
    import torch
    info["torch_version"] = torch.__version__
    info["cuda_available"] = bool(torch.cuda.is_available())
    info["cuda_device_count"] = int(torch.cuda.device_count())
except Exception as exc:
    info["available"] = False
    info["error"] = "torch: " + str(exc)

try:
    import transformers
    info["transformers_version"] = transformers.__version__
except Exception:
    pass

try:
    import torch_npu
    info["npu_available"] = True
except Exception:
    info["npu_available"] = False

print(json.dumps(info))
"""


@lru_cache(maxsize=8)
def inspect_runtime(executable: Path) -> RuntimeInfo:
    try:
        result = subprocess.run(
            [str(executable), "-c", _PROBE],
            text=True,
            capture_output=True,
            timeout=10,
            check=True,
        )
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        return RuntimeInfo.model_validate(payload)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, IndexError) as exc:
        return RuntimeInfo(
            available=False,
            python=str(executable),
            error=str(exc),
        )
