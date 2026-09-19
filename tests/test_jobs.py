import socket
from pathlib import Path

from cid_factory.jobs import JobManager
from cid_factory.models import CommandPreview, RunCreate, RunRecord, RunStatus, Stage
from cid_factory.store import RunStore


def test_stop_refuses_unverified_persisted_pid(tmp_path: Path, monkeypatch) -> None:
    store = RunStore(tmp_path / "runs.sqlite3")
    request = RunCreate(
        name="run-a",
        stage=Stage.STAGE_A,
        model="model",
        data="/data/train.jsonl",
        output_dir=str(tmp_path / "output"),
    )
    record = RunRecord(
        id="run-a",
        name="run-a",
        stage=Stage.STAGE_A,
        status=RunStatus.RUNNING,
        created_at=1.0,
        pid=4242,
        execution_host=socket.gethostname(),
        output_dir=request.output_dir,
        log_path=str(tmp_path / "run.log"),
        request=request,
        command=CommandPreview(
            argv=["python"],
            display="python",
            cwd="/cid",
            environment={},
        ),
    )
    store.put(record)

    monkeypatch.setattr("cid_factory.jobs._process_has_run_marker", lambda pid, run_id: False)

    def unexpected_kill(*args, **kwargs):
        raise AssertionError("unverified persisted PID must not be signalled")

    monkeypatch.setattr("cid_factory.jobs.os.killpg", unexpected_kill)
    manager = JobManager(
        Path("/cid"),
        tmp_path,
        store,
        python_executable=Path("/usr/bin/python3"),
    )

    stopped = manager.stop("run-a")

    assert stopped.status is RunStatus.UNKNOWN
