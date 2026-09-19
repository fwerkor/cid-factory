from __future__ import annotations

from pathlib import Path
from typing import Any

from .metrics import training_metrics, validation_metrics
from .models import RunComparisonEntry, RunRecord


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _validation_loss(record: dict[str, Any]) -> float | None:
    for key in ("mean_loss", "weighted_mean_loss", "loss", "raw_mean_loss"):
        value = _number(record.get(key))
        if value is not None:
            return value
    return None


def compare_run(record: RunRecord, *, limit: int = 1000) -> RunComparisonEntry:
    output = Path(record.output_dir).expanduser()
    training = training_metrics(output, record.stage, limit).records
    validation = validation_metrics(output, min(limit, 500)).records
    latest = training[-1] if training else {}

    latest_step_value = latest.get("optimizer_steps", latest.get("step"))
    latest_step = int(latest_step_value) if isinstance(latest_step_value, (int, float)) else None
    latest_loss = _number(latest.get("mean_loss", latest.get("loss")))
    latest_raw_loss = _number(latest.get("raw_mean_loss"))
    latest_lr = _number(latest.get("learning_rate", latest.get("lr")))
    elapsed = _number(latest.get("elapsed_seconds"))

    seen = _number(latest.get("windows_seen_in_epoch"))
    total = _number(latest.get("windows_total_in_epoch"))
    progress = None
    if seen is not None and total is not None and total > 0:
        progress = min(1.0, max(0.0, seen / total))

    validation_losses = [
        value
        for value in (_validation_loss(item) for item in validation)
        if value is not None
    ]

    return RunComparisonEntry(
        id=record.id,
        name=record.name,
        stage=record.stage,
        status=record.status,
        model=record.request.model,
        world_size=record.request.world_size,
        device=record.request.device,
        created_at=record.created_at,
        repo_head=record.repo_head,
        parameters=record.request.parameters,
        latest_step=latest_step,
        latest_loss=latest_loss,
        latest_raw_loss=latest_raw_loss,
        latest_learning_rate=latest_lr,
        elapsed_seconds=elapsed,
        progress_fraction=progress,
        validation_loss=validation_losses[-1] if validation_losses else None,
        best_validation_loss=min(validation_losses) if validation_losses else None,
        metrics=training,
    )
