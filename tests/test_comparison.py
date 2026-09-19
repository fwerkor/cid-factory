import json
from pathlib import Path

from cid_factory.comparison import compare_run
from cid_factory.models import CommandPreview, RunCreate, RunRecord, RunStatus, Stage


def test_compare_run_summarizes_native_metrics(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    (output / "train_metrics.jsonl").write_text(
        "\n".join(
            [
                json.dumps({
                    "optimizer_steps": 10,
                    "mean_loss": 2.0,
                    "raw_mean_loss": 2.2,
                    "learning_rate": 1e-4,
                    "elapsed_seconds": 5.0,
                    "windows_seen_in_epoch": 20,
                    "windows_total_in_epoch": 100,
                }),
                json.dumps({
                    "optimizer_steps": 20,
                    "mean_loss": 1.5,
                    "raw_mean_loss": 1.7,
                    "learning_rate": 8e-5,
                    "elapsed_seconds": 10.0,
                    "windows_seen_in_epoch": 50,
                    "windows_total_in_epoch": 100,
                }),
            ]
        ) + "\n",
        encoding="utf-8",
    )
    (output / "validation_metrics.jsonl").write_text(
        json.dumps({"mean_loss": 1.8}) + "\n" + json.dumps({"mean_loss": 1.6}) + "\n",
        encoding="utf-8",
    )

    request = RunCreate(
        name="run-a",
        stage=Stage.STAGE_A,
        model="model",
        data="/data/train.jsonl",
        output_dir=str(output),
        world_size=4,
        parameters={"learning_rate": 1e-4},
    )
    record = RunRecord(
        id="run-a",
        name="run-a",
        stage=Stage.STAGE_A,
        status=RunStatus.COMPLETED,
        created_at=1.0,
        output_dir=str(output),
        log_path="/tmp/run.log",
        repo_head="abc123",
        request=request,
        command=CommandPreview(argv=[], display="", cwd="/cid", environment={}),
    )

    result = compare_run(record)

    assert result.latest_step == 20
    assert result.latest_loss == 1.5
    assert result.latest_raw_loss == 1.7
    assert result.latest_learning_rate == 8e-5
    assert result.progress_fraction == 0.5
    assert result.validation_loss == 1.6
    assert result.best_validation_loss == 1.6
    assert result.repo_head == "abc123"
    assert result.parameters["learning_rate"] == 1e-4
