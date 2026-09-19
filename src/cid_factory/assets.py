from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .models import AssetKind, AssetRecord, AssetRoot

_STAGE_A_CHECKPOINT = re.compile(r"^stage-a-(?:latest|epoch-\d+|step-\d+)\.pt$")
_STAGE_B_CHECKPOINT = re.compile(r"^stage-b-(?:latest|epoch-\d+|step-\d+)(?:\.pt)?$")
_STAGE0_CHECKPOINT = re.compile(r"^checkpoint-\d+$")
_MANIFEST_NAMES = ("manifest.json", ".manifest.json", "reference-manifest")


def default_asset_roots(repo: Path) -> tuple[Path, ...]:
    configured = os.environ.get("CID_FACTORY_ASSET_ROOTS")
    if configured:
        raw = [Path(item).expanduser() for item in configured.split(os.pathsep) if item.strip()]
    else:
        raw = [
            repo / "data",
            repo / "runs",
            repo.parent / "runs",
            repo.parent / "models",
        ]

    roots: list[Path] = []
    seen: set[Path] = set()
    for path in raw:
        absolute = path.absolute()
        if absolute in seen:
            continue
        seen.add(absolute)
        roots.append(absolute)
    return tuple(roots)


def _safe_json(path: Path, *, max_bytes: int = 4 * 1024 * 1024) -> dict[str, Any]:
    try:
        if path.stat().st_size > max_bytes:
            return {}
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _manifest_summary(path: Path) -> dict[str, Any]:
    raw = _safe_json(path)
    keys = (
        "name",
        "format",
        "sha256",
        "examples",
        "transitions",
        "sequence_length",
        "thought_capacity_required",
        "max_trajectory_steps",
        "source_count",
    )
    return {key: raw[key] for key in keys if key in raw}


def _model_summary(path: Path) -> dict[str, Any]:
    raw = _safe_json(path / "config.json")
    keys = (
        "model_type",
        "architectures",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "vocab_size",
    )
    return {key: raw[key] for key in keys if key in raw}


def _checkpoint_stage(name: str) -> str | None:
    if _STAGE_A_CHECKPOINT.match(name):
        return "stage-a"
    if _STAGE_B_CHECKPOINT.match(name):
        return "stage-b"
    if _STAGE0_CHECKPOINT.match(name):
        return "stage0"
    return None


def _checkpoint_summary(path: Path) -> dict[str, Any]:
    details: dict[str, Any] = {}
    stage = _checkpoint_stage(path.name)
    if stage:
        details["stage"] = stage

    metadata = path / "metadata.json" if path.is_dir() else None
    if metadata is not None and metadata.is_file():
        raw = _safe_json(metadata)
        for key in (
            "step",
            "world_size",
            "dataset_sha256",
            "sequence_length",
            "completed_steps",
            "optimizer_steps",
            "epoch_progress",
        ):
            if key in raw:
                details[key] = raw[key]
    return details


def _dataset_summary(path: Path) -> dict[str, Any]:
    manifest = path.with_name(path.name.removesuffix(".jsonl") + ".manifest.json")
    if not manifest.is_file():
        return {}
    details = _manifest_summary(manifest)
    details["manifest"] = str(manifest)
    return details


def _is_model_dir(path: Path) -> bool:
    if not (path / "config.json").is_file():
        return False
    try:
        names = {entry.name for entry in path.iterdir()}
    except OSError:
        return False
    return any(
        name.endswith(".safetensors")
        or name.startswith("pytorch_model")
        or name == "model.safetensors.index.json"
        for name in names
    )


def _is_output_dir(path: Path) -> bool:
    return any(
        candidate.exists()
        for candidate in (
            path / "train_metrics.jsonl",
            path / "train_metrics.rank-0000.jsonl",
            path / "validation_metrics.jsonl",
        )
    )


def _is_manifest(path: Path) -> bool:
    name = path.name.lower()
    return (
        name.endswith(".manifest.json")
        or name == "manifest.json"
        or "reference-manifest" in name
    )


def _record(
    *,
    kind: AssetKind,
    path: Path,
    root: Path,
    details: dict[str, Any] | None = None,
) -> AssetRecord:
    try:
        stat = path.stat()
        modified_at = stat.st_mtime
        size_bytes = stat.st_size if path.is_file() else None
    except OSError:
        modified_at = None
        size_bytes = None
    return AssetRecord(
        kind=kind,
        name=path.name,
        path=str(path),
        root=str(root),
        modified_at=modified_at,
        size_bytes=size_bytes,
        is_symlink=path.is_symlink(),
        details=details or {},
    )


class AssetCatalog:
    def __init__(
        self,
        roots: Iterable[Path],
        *,
        max_depth: int = 5,
        scan_cap: int = 5000,
    ) -> None:
        self.roots = tuple(roots)
        self.max_depth = max_depth
        self.scan_cap = scan_cap

    def root_records(self) -> list[AssetRoot]:
        return [
            AssetRoot(path=str(root), label=root.name or str(root), available=root.is_dir())
            for root in self.roots
        ]

    def search(
        self,
        *,
        kind: AssetKind | None = None,
        query: str = "",
        limit: int = 200,
        extra_roots: Iterable[Path] = (),
    ) -> list[AssetRecord]:
        roots = self._combined_roots(extra_roots)
        needle = query.casefold().strip()
        records: list[AssetRecord] = []
        seen_records: set[tuple[AssetKind, str]] = set()
        visited = 0

        def append(record: AssetRecord) -> bool:
            key = (record.kind, record.path)
            if key in seen_records:
                return False
            seen_records.add(key)
            records.append(record)
            return len(records) >= limit

        for root in roots:
            if not root.is_dir():
                continue
            stack: list[tuple[Path, int]] = [(root, 0)]
            while stack and visited < self.scan_cap:
                current, depth = stack.pop()
                visited += 1
                if depth > self.max_depth:
                    continue

                directory_record, stop = self._classify_directory(current, root)
                if (
                    directory_record is not None
                    and self._matches(directory_record, kind=kind, needle=needle)
                    and append(directory_record)
                ):
                    return self._sorted(records)
                if stop:
                    continue

                try:
                    entries = list(os.scandir(current))
                except OSError:
                    continue

                for entry in entries:
                    path = Path(entry.path)
                    if entry.is_dir(follow_symlinks=False):
                        stack.append((path, depth + 1))
                        continue
                    file_record = self._classify_file(path, root)
                    if file_record is None:
                        continue
                    if self._matches(
                        file_record, kind=kind, needle=needle
                    ) and append(file_record):
                        return self._sorted(records)

        return self._sorted(records)

    def _combined_roots(self, extra_roots: Iterable[Path]) -> tuple[Path, ...]:
        roots: list[Path] = []
        seen: set[Path] = set()
        for root in (*self.roots, *extra_roots):
            absolute = root.expanduser().absolute()
            if absolute in seen:
                continue
            seen.add(absolute)
            roots.append(absolute)
        return tuple(roots)

    @staticmethod
    def _matches(record: AssetRecord, *, kind: AssetKind | None, needle: str) -> bool:
        if kind is not None and record.kind is not kind:
            return False
        if not needle:
            return True
        return needle in record.name.casefold() or needle in record.path.casefold()

    @staticmethod
    def _sorted(records: list[AssetRecord]) -> list[AssetRecord]:
        return sorted(
            records,
            key=lambda record: (record.modified_at or 0.0, record.path),
            reverse=True,
        )

    @staticmethod
    def _classify_directory(path: Path, root: Path) -> tuple[AssetRecord | None, bool]:
        stage = _checkpoint_stage(path.name)
        if stage in {"stage0", "stage-b"}:
            return (
                _record(
                    kind=AssetKind.CHECKPOINT,
                    path=path,
                    root=root,
                    details=_checkpoint_summary(path),
                ),
                True,
            )
        if _is_model_dir(path):
            return (
                _record(
                    kind=AssetKind.MODEL,
                    path=path,
                    root=root,
                    details=_model_summary(path),
                ),
                True,
            )
        if _is_output_dir(path):
            return (
                _record(kind=AssetKind.OUTPUT, path=path, root=root),
                False,
            )
        return None, False

    @staticmethod
    def _classify_file(path: Path, root: Path) -> AssetRecord | None:
        stage = _checkpoint_stage(path.name)
        if stage in {"stage-a", "stage-b"}:
            return _record(
                kind=AssetKind.CHECKPOINT,
                path=path,
                root=root,
                details=_checkpoint_summary(path),
            )
        if (
            path.name in {
                "train_metrics.jsonl",
                "validation_metrics.jsonl",
                "runtime_validation_metrics.jsonl",
            }
            or path.name.startswith("train_metrics.rank-")
        ):
            return None
        if path.suffix == ".jsonl":
            return _record(
                kind=AssetKind.DATASET,
                path=path,
                root=root,
                details=_dataset_summary(path),
            )
        if _is_manifest(path):
            return _record(
                kind=AssetKind.MANIFEST,
                path=path,
                root=root,
                details=_manifest_summary(path),
            )
        return None
