from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


def _discover_cid_repo() -> Path:
    configured = os.environ.get("CID_FACTORY_CID_REPO")
    if configured:
        return Path(configured).expanduser().resolve()

    candidates = [
        Path.cwd().parent / "continuous-interaction-diffusion",
        Path(__file__).resolve().parents[3] / "continuous-interaction-diffusion",
        Path("/workspace/continuous-interaction-diffusion"),
    ]
    for candidate in candidates:
        if (candidate / "pyproject.toml").exists() and (candidate / "src/cid/cli.py").exists():
            return candidate.resolve()
    return candidates[0].resolve()

def _can_import_torch(executable: Path) -> bool:
    try:
        subprocess.run(
            [str(executable), "-c", "import torch"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return True


def _discover_cid_python(repo: Path) -> Path:
    configured = os.environ.get("CID_FACTORY_CID_PYTHON")
    if configured:
        return Path(configured).expanduser().absolute()

    candidates: list[Path] = [
        repo / ".venv/bin/python",
        repo / "venv/bin/python",
    ]
    torchrun = shutil.which("torchrun")
    if torchrun:
        torchrun_bin = Path(torchrun).absolute().parent
        candidates.extend([torchrun_bin / "python", torchrun_bin / "python3"])
    candidates.append(Path(sys.executable))
    for command in ("python3", "python"):
        found = shutil.which(command)
        if found:
            candidates.append(Path(found))

    seen: set[Path] = set()
    for candidate in candidates:
        executable = candidate.expanduser().absolute()
        if executable in seen or not executable.exists():
            continue
        seen.add(executable)
        if _can_import_torch(executable):
            return executable
    return Path(sys.executable).absolute()


@dataclass(frozen=True)
class Settings:
    cid_repo: Path
    state_dir: Path
    cid_python: Path

    @classmethod
    def load(cls) -> Settings:
        repo = _discover_cid_repo()
        state = Path(
            os.environ.get(
                "CID_FACTORY_STATE_DIR",
                Path.home() / ".local/share/cid-factory",
            )
        ).expanduser()
        return cls(
            cid_repo=repo,
            state_dir=state.resolve(),
            cid_python=_discover_cid_python(repo),
        )
