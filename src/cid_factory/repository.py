from __future__ import annotations

import subprocess
from pathlib import Path

from .models import RepositoryInfo


def _git(repo: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def inspect_repository(repo: Path) -> RepositoryInfo:
    available = (repo / "src/cid/cli.py").exists()
    if not available:
        return RepositoryInfo(available=False, path=str(repo))
    status = _git(repo, "status", "--porcelain")
    return RepositoryInfo(
        available=True,
        path=str(repo),
        branch=_git(repo, "branch", "--show-current"),
        head=_git(repo, "rev-parse", "--short=12", "HEAD"),
        dirty=bool(status),
        remote=_git(repo, "remote", "get-url", "origin"),
    )
