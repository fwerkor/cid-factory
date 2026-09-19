from __future__ import annotations

import os
import signal
import socket
import subprocess
import threading
import time
import uuid
from pathlib import Path

from .command import build_command
from .models import RunCreate, RunRecord, RunStatus
from .repository import inspect_repository
from .store import RunStore


def _process_has_run_marker(pid: int, run_id: str) -> bool:
    environ = Path(f"/proc/{pid}/environ")
    try:
        values = environ.read_bytes().split(b"\0")
    except OSError:
        return False
    marker = f"CID_FACTORY_RUN_ID={run_id}".encode()
    return marker in values


class JobManager:
    def __init__(
        self,
        repo: Path,
        state_dir: Path,
        store: RunStore,
        *,
        python_executable: Path,
    ) -> None:
        self.repo = repo
        self.state_dir = state_dir
        self.store = store
        self.python_executable = python_executable
        self._processes: dict[str, subprocess.Popen[str]] = {}
        self._lock = threading.RLock()

    def create(self, request: RunCreate, launch: bool = True) -> RunRecord:
        preview = build_command(
            request,
            self.repo,
            python_executable=self.python_executable,
        )
        repo_info = inspect_repository(self.repo)
        run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        log_dir = self.state_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        record = RunRecord(
            id=run_id,
            name=request.name,
            stage=request.stage,
            status=RunStatus.CREATED,
            created_at=time.time(),
            output_dir=request.output_dir,
            log_path=str(log_dir / f"{run_id}.log"),
            repo_head=repo_info.head,
            request=request,
            command=preview,
        )
        self.store.put(record)
        if launch:
            return self.start(run_id)
        return record

    def start(self, run_id: str) -> RunRecord:
        with self._lock:
            record = self._require(run_id)
            if record.status not in {RunStatus.CREATED, RunStatus.FAILED, RunStatus.STOPPED}:
                return record

            Path(record.output_dir).expanduser().mkdir(parents=True, exist_ok=True)
            log_path = Path(record.log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = log_path.open("a", encoding="utf-8")
            env = os.environ.copy()
            env.update(record.command.environment)
            env["CID_FACTORY_RUN_ID"] = record.id
            pythonpath = str(self.repo / "src")
            env["PYTHONPATH"] = (
                pythonpath
                if not env.get("PYTHONPATH")
                else pythonpath + os.pathsep + env["PYTHONPATH"]
            )
            process = subprocess.Popen(
                record.command.argv,
                cwd=record.command.cwd,
                env=env,
                text=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            record.status = RunStatus.RUNNING
            record.started_at = time.time()
            record.finished_at = None
            record.exit_code = None
            record.pid = process.pid
            record.execution_host = socket.gethostname()
            self._processes[run_id] = process
            self.store.put(record)
            threading.Thread(
                target=self._watch,
                args=(run_id, process, log_file),
                daemon=True,
            ).start()
            return record

    def _watch(
        self,
        run_id: str,
        process: subprocess.Popen[str],
        log_file,
    ) -> None:
        code = process.wait()
        log_file.close()
        with self._lock:
            record = self.store.get(run_id)
            if record is None:
                return
            record.exit_code = code
            record.finished_at = time.time()
            if record.status is RunStatus.STOPPING:
                record.status = RunStatus.STOPPED
            else:
                record.status = RunStatus.COMPLETED if code == 0 else RunStatus.FAILED
            self.store.put(record)
            self._processes.pop(run_id, None)

    def stop(self, run_id: str) -> RunRecord:
        with self._lock:
            record = self._require(run_id)
            if record.status is not RunStatus.RUNNING or record.pid is None:
                return record
            process = self._processes.get(run_id)
            owned = process is not None or self._owns_persisted_process(record)
            if not owned:
                record.status = RunStatus.UNKNOWN
                self.store.put(record)
                return record
            record.status = RunStatus.STOPPING
            self.store.put(record)
            try:
                os.killpg(record.pid, signal.SIGTERM)
            except ProcessLookupError:
                record.status = RunStatus.UNKNOWN
                self.store.put(record)
            return record

    def refresh(self, record: RunRecord) -> RunRecord:
        if record.status is not RunStatus.RUNNING or record.pid is None:
            return record
        with self._lock:
            process = self._processes.get(record.id)
            if process is not None:
                return record
            if not self._owns_persisted_process(record):
                record.status = RunStatus.UNKNOWN
                self.store.put(record)
        return record

    def get(self, run_id: str) -> RunRecord:
        return self.refresh(self._require(run_id))

    def list(self) -> list[RunRecord]:
        return [self.refresh(record) for record in self.store.list()]

    @staticmethod
    def _owns_persisted_process(record: RunRecord) -> bool:
        if record.pid is None or record.execution_host != socket.gethostname():
            return False
        return _process_has_run_marker(record.pid, record.id)

    def _require(self, run_id: str) -> RunRecord:
        record = self.store.get(run_id)
        if record is None:
            raise KeyError(run_id)
        return record
