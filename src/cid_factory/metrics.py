from __future__ import annotations

import json
from pathlib import Path

from .models import MetricSeries, Stage


def _read_jsonl(path: Path, limit: int) -> list[dict[str, object]]:
    if not path.exists():
        return []
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                records.append(value)
    return records[-limit:]


def training_metrics(output_dir: Path, stage: Stage, limit: int = 500) -> MetricSeries:
    candidates = (
        [output_dir / "train_metrics.jsonl", output_dir / "train_metrics.rank-0000.jsonl"]
        if stage is not Stage.STAGE0
        else [output_dir / "train_metrics.rank-0000.jsonl"]
    )
    for path in candidates:
        records = _read_jsonl(path, limit)
        if records:
            return MetricSeries(records=records, source=str(path))
    return MetricSeries(records=[])


def validation_metrics(output_dir: Path, limit: int = 100) -> MetricSeries:
    path = output_dir / "validation_metrics.jsonl"
    return MetricSeries(
        records=_read_jsonl(path, limit),
        source=str(path) if path.exists() else None,
    )
