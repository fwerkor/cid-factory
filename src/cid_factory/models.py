from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class Stage(str, Enum):
    STAGE0 = "stage0"
    STAGE_A = "stage-a"
    STAGE_B = "stage-b"


class RunStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    STOPPING = "stopping"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"
    UNKNOWN = "unknown"


Primitive = str | int | float | bool | None


class RunCreate(BaseModel):
    name: str = Field(min_length=1, max_length=96)
    stage: Stage
    model: str = Field(min_length=1)
    data: str = Field(min_length=1)
    output_dir: str = Field(min_length=1)
    validation_data: str | None = None
    world_size: int = Field(default=1, ge=1, le=64)
    device: Literal["auto", "cuda", "npu", "cpu"] = "auto"
    dtype: Literal["bf16", "fp32"] = "bf16"
    resume: str | None = None
    init_cid_checkpoint: str | None = None
    visible_devices: list[int] | None = None
    parameters: dict[str, Primitive] = Field(default_factory=dict)
    extra_args: list[str] = Field(default_factory=list)

    @field_validator("extra_args")
    @classmethod
    def validate_extra_args(cls, value: list[str]) -> list[str]:
        if any("\x00" in arg for arg in value):
            raise ValueError("extra arguments cannot contain NUL bytes")
        return value


class CommandPreview(BaseModel):
    argv: list[str]
    display: str
    cwd: str
    environment: dict[str, str]
    warnings: list[str] = Field(default_factory=list)


class RunRecord(BaseModel):
    id: str
    name: str
    stage: Stage
    status: RunStatus
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    pid: int | None = None
    exit_code: int | None = None
    output_dir: str
    log_path: str
    repo_head: str | None = None
    request: RunCreate
    command: CommandPreview


class MetricSeries(BaseModel):
    records: list[dict[str, Any]]
    source: str | None = None


class HardwareDevice(BaseModel):
    index: int
    name: str
    kind: Literal["cuda", "npu"]
    memory_total_mb: float | None = None
    memory_used_mb: float | None = None
    utilization_percent: float | None = None
    temperature_c: float | None = None
    power_w: float | None = None


class RepositoryInfo(BaseModel):
    available: bool
    path: str
    branch: str | None = None
    head: str | None = None
    dirty: bool | None = None
    remote: str | None = None


class RuntimeInfo(BaseModel):
    available: bool
    python: str
    python_version: str | None = None
    torch_version: str | None = None
    transformers_version: str | None = None
    cuda_available: bool | None = None
    cuda_device_count: int | None = None
    npu_available: bool | None = None
    error: str | None = None
