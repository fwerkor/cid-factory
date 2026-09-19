from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

from .assets import AssetCatalog, default_asset_roots
from .command import build_command
from .comparison import compare_run
from .hardware import cuda_devices
from .jobs import JobManager
from .metrics import training_metrics, validation_metrics
from .models import (
    AssetKind,
    AssetRecord,
    AssetRoot,
    CommandPreview,
    RunComparisonEntry,
    RunCreate,
    RunRecord,
)
from .repository import inspect_repository
from .runtime import inspect_runtime
from .settings import Settings
from .store import RunStore
from .surfaces import STAGE_SURFACES

settings = Settings.load()
settings.state_dir.mkdir(parents=True, exist_ok=True)
store = RunStore(settings.state_dir / "factory.sqlite3")
jobs = JobManager(
    settings.cid_repo,
    settings.state_dir,
    store,
    python_executable=settings.cid_python,
)
assets = AssetCatalog(default_asset_roots(settings.cid_repo))

router = APIRouter(prefix="/api")


@router.get("/health")
def health() -> dict[str, object]:
    repo = inspect_repository(settings.cid_repo)
    return {"ok": repo.available, "repository": repo}


@router.get("/repository")
def repository():
    return inspect_repository(settings.cid_repo)


@router.get("/surfaces")
def surfaces():
    return STAGE_SURFACES


@router.get("/runtime")
def runtime():
    return inspect_runtime(settings.cid_python)


@router.get("/hardware")
def hardware():
    return {"devices": cuda_devices()}


@router.get("/assets/roots", response_model=list[AssetRoot])
def asset_roots():
    return assets.root_records()


@router.get("/assets", response_model=list[AssetRecord])
def list_assets(
    kind: AssetKind | None = None,
    query: str = "",
    limit: int = Query(default=200, ge=1, le=1000),
):
    extra_roots = [Path(record.output_dir).expanduser() for record in jobs.list()]
    return assets.search(
        kind=kind,
        query=query,
        limit=limit,
        extra_roots=extra_roots,
    )


@router.post("/runs/preview", response_model=CommandPreview)
def preview_run(request: RunCreate):
    repo = inspect_repository(settings.cid_repo)
    if not repo.available:
        raise HTTPException(status_code=503, detail="CID repository is not available")
    return build_command(
        request,
        settings.cid_repo,
        python_executable=settings.cid_python,
    )


@router.get("/runs", response_model=list[RunRecord])
def list_runs():
    return jobs.list()


@router.post("/runs", response_model=RunRecord)
def create_run(request: RunCreate, launch: bool = Query(default=True)):
    repo = inspect_repository(settings.cid_repo)
    if not repo.available:
        raise HTTPException(status_code=503, detail="CID repository is not available")
    try:
        return jobs.create(request, launch=launch)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/runs/compare", response_model=list[RunComparisonEntry])
def compare_runs(
    run_id: Annotated[list[str], Query()],
    limit: Annotated[int, Query(ge=10, le=5000)] = 1000,
):
    unique_ids = list(dict.fromkeys(run_id))
    if not 2 <= len(unique_ids) <= 4:
        raise HTTPException(status_code=422, detail="compare requires 2 to 4 unique runs")

    entries: list[RunComparisonEntry] = []
    for item in unique_ids:
        try:
            record = jobs.get(item)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"run not found: {item}") from exc
        entries.append(compare_run(record, limit=limit))
    return entries


@router.get("/runs/{run_id}", response_model=RunRecord)
def get_run(run_id: str):
    try:
        return jobs.get(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="run not found") from exc


@router.post("/runs/{run_id}/start", response_model=RunRecord)
def start_run(run_id: str):
    try:
        return jobs.start(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="run not found") from exc


@router.post("/runs/{run_id}/stop", response_model=RunRecord)
def stop_run(run_id: str):
    try:
        return jobs.stop(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="run not found") from exc


@router.get("/runs/{run_id}/metrics")
def get_metrics(run_id: str, limit: int = Query(default=500, ge=1, le=5000)):
    try:
        record = jobs.get(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="run not found") from exc
    output = Path(record.output_dir).expanduser()
    return {
        "training": training_metrics(output, record.stage, limit),
        "validation": validation_metrics(output, min(limit, 500)),
    }


@router.get("/runs/{run_id}/logs")
def get_logs(
    run_id: str,
    lines: int = Query(default=300, ge=1, le=5000),
):
    try:
        record = jobs.get(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="run not found") from exc
    path = Path(record.log_path)
    if not path.exists():
        return {"lines": [], "path": str(path)}
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return {"lines": content[-lines:], "path": str(path)}
