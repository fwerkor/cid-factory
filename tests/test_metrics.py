import json
from pathlib import Path

from cid_factory.metrics import training_metrics, validation_metrics
from cid_factory.models import Stage


def test_training_metrics_reads_native_stage_a_file(tmp_path: Path) -> None:
    path = tmp_path / "train_metrics.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"optimizer_steps": 1, "mean_loss": 2.5}),
                "not-json",
                json.dumps({"optimizer_steps": 2, "mean_loss": 2.0}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    series = training_metrics(tmp_path, Stage.STAGE_A)

    assert series.source == str(path)
    assert [item["optimizer_steps"] for item in series.records] == [1, 2]


def test_stage0_reads_rank_zero_metrics(tmp_path: Path) -> None:
    path = tmp_path / "train_metrics.rank-0000.jsonl"
    path.write_text(json.dumps({"step": 10, "loss": 1.25}) + "\n", encoding="utf-8")

    series = training_metrics(tmp_path, Stage.STAGE0)

    assert series.records == [{"step": 10, "loss": 1.25}]


def test_validation_metrics_respects_limit(tmp_path: Path) -> None:
    path = tmp_path / "validation_metrics.jsonl"
    path.write_text(
        "".join(json.dumps({"epoch": index}) + "\n" for index in range(5)),
        encoding="utf-8",
    )

    series = validation_metrics(tmp_path, limit=2)

    assert [item["epoch"] for item in series.records] == [3, 4]
