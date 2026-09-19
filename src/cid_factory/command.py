from __future__ import annotations

import shlex
from pathlib import Path

from .models import CommandPreview, RunCreate, Stage

_FLAG_ALIASES = {
    "checkpoint_every": "checkpoint-every",
    "eval_every": "eval-every",
    "eval_batches": "eval-batches",
    "keep_checkpoints": "keep-checkpoints",
}


def _flag(name: str) -> str:
    return "--" + _FLAG_ALIASES.get(name, name.replace("_", "-"))


def _append_parameter(argv: list[str], key: str, value: object) -> None:
    flag = _flag(key)
    if value is None:
        return
    if isinstance(value, bool):
        argv.append(flag if value else "--no-" + flag.removeprefix("--"))
        return
    argv.extend([flag, str(value)])


def _launcher(
    python_executable: str | Path,
    world_size: int,
    module_or_script: list[str],
) -> list[str]:
    if world_size <= 1:
        return [str(python_executable), *module_or_script]
    return [
        str(python_executable),
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={world_size}",
        *module_or_script,
    ]


def build_command(
    request: RunCreate,
    repo: Path,
    python_executable: str | Path = "python3",
) -> CommandPreview:
    warnings: list[str] = []

    if request.visible_devices and len(request.visible_devices) != request.world_size:
        warnings.append("Selected device count differs from world size.")

    if request.stage is Stage.STAGE0:
        argv = [
            str(python_executable),
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={request.world_size}",
            str(repo / "scripts/train_diffusion_base.py"),
        ]
        argv.extend(
            [
                "--model",
                request.model,
                "--data-manifest",
                request.data,
                "--output-dir",
                request.output_dir,
            ]
        )
        if request.resume:
            argv.extend(["--resume", request.resume])
    else:
        command = "train" if request.stage is Stage.STAGE_A else "train-full"
        argv = _launcher(
            python_executable,
            request.world_size,
            ["-m", "cid.cli", command],
        )
        argv.extend(
            [
                "--data",
                request.data,
                "--output-dir",
                request.output_dir,
                "--model",
                request.model,
                "--device",
                request.device,
                "--dtype",
                request.dtype,
            ]
        )
        if request.validation_data:
            argv.extend(["--validation-data", request.validation_data])
        if request.resume:
            argv.extend(["--resume", request.resume])
        if request.stage is Stage.STAGE_B and request.init_cid_checkpoint:
            argv.extend(["--init-cid-checkpoint", request.init_cid_checkpoint])

    for key, value in request.parameters.items():
        _append_parameter(argv, key, value)
    argv.extend(request.extra_args)

    if request.stage is Stage.STAGE_B and request.device == "cuda" and request.world_size < 4:
        warnings.append(
            "The main CID repository rejects multi-GPU CUDA Stage B with fewer than 4 ranks."
        )
    if request.stage is Stage.STAGE_B and not request.resume and not request.init_cid_checkpoint:
        warnings.append("Fresh Stage B runs require a completed Stage A checkpoint.")
    if request.stage is Stage.STAGE_A and request.parameters.get("thought_capacity", 128) != 128:
        warnings.append("CID v1 Stage A currently requires thought capacity 128.")

    env: dict[str, str] = {}
    if request.visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in request.visible_devices)

    return CommandPreview(
        argv=argv,
        display=shlex.join(argv),
        cwd=str(repo),
        environment=env,
        warnings=warnings,
    )
