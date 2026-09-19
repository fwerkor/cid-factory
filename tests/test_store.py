from pathlib import Path

from cid_factory.models import CommandPreview, RunCreate, RunRecord, RunStatus, Stage
from cid_factory.store import RunStore


def make_record(run_id: str, created_at: float) -> RunRecord:
    request = RunCreate(
        name=run_id,
        stage=Stage.STAGE_A,
        model="model",
        data="/data/train.jsonl",
        output_dir=f"/runs/{run_id}",
    )
    return RunRecord(
        id=run_id,
        name=run_id,
        stage=Stage.STAGE_A,
        status=RunStatus.CREATED,
        created_at=created_at,
        output_dir=request.output_dir,
        log_path=f"/logs/{run_id}.log",
        request=request,
        command=CommandPreview(
            argv=["python", "-m", "cid.cli", "train"],
            display="python -m cid.cli train",
            cwd="/cid",
            environment={},
        ),
    )


def test_store_round_trip_and_update(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs.sqlite3")
    record = make_record("run-a", 1.0)
    store.put(record)

    loaded = store.get("run-a")
    assert loaded == record

    record.status = RunStatus.RUNNING
    store.put(record)
    assert store.get("run-a").status is RunStatus.RUNNING


def test_store_lists_newest_first(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs.sqlite3")
    store.put(make_record("older", 1.0))
    store.put(make_record("newer", 2.0))

    assert [record.id for record in store.list()] == ["newer", "older"]
