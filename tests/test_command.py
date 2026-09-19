from pathlib import Path

from cid_factory.command import build_command
from cid_factory.models import RunCreate, Stage


def test_stage_a_builds_main_repo_cli() -> None:
    request = RunCreate(
        name="a",
        stage=Stage.STAGE_A,
        model="model",
        data="/data/train.jsonl",
        validation_data="/data/val.jsonl",
        output_dir="/runs/a",
        world_size=4,
        device="cuda",
        parameters={"epochs": 3, "gradient_checkpointing": True},
    )
    preview = build_command(request, Path("/cid"), python_executable="/env/python")
    assert preview.argv[:5] == [
        "/env/python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=4",
    ]
    assert preview.argv[5:8] == ["-m", "cid.cli", "train"]
    assert "--validation-data" in preview.argv
    assert "--gradient-checkpointing" in preview.argv


def test_stage_b_warns_for_missing_init_checkpoint() -> None:
    request = RunCreate(
        name="b",
        stage=Stage.STAGE_B,
        model="model",
        data="/data/train.jsonl",
        output_dir="/runs/b",
        world_size=2,
        device="cuda",
    )
    preview = build_command(request, Path("/cid"), python_executable="/env/python")
    assert len(preview.warnings) == 2


def test_stage0_uses_main_repo_script() -> None:
    request = RunCreate(
        name="s0",
        stage=Stage.STAGE0,
        model="model",
        data="/data/manifest.json",
        output_dir="/runs/s0",
        parameters={"steps": 100},
    )
    preview = build_command(request, Path("/cid"), python_executable="/env/python")
    assert preview.argv[:5] == [
        "/env/python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=1",
    ]
    assert "/cid/scripts/train_diffusion_base.py" in preview.argv
    assert "--steps" in preview.argv


def test_single_rank_stage_a_uses_configured_python_directly() -> None:
    request = RunCreate(
        name="single",
        stage=Stage.STAGE_A,
        model="model",
        data="/data/train.jsonl",
        output_dir="/runs/single",
        world_size=1,
        device="cpu",
    )

    preview = build_command(request, Path("/cid"), python_executable="/env/python")

    assert preview.argv[:4] == ["/env/python", "-m", "cid.cli", "train"]
    assert "torch.distributed.run" not in preview.argv


def test_visible_devices_are_passed_as_environment_not_shell_text() -> None:
    request = RunCreate(
        name="gpu",
        stage=Stage.STAGE_A,
        model="model;touch /tmp/nope",
        data="/data/train.jsonl",
        output_dir="/runs/gpu",
        world_size=2,
        visible_devices=[3, 7],
    )

    preview = build_command(request, Path("/cid"), python_executable="/env/python")

    assert preview.environment == {"CUDA_VISIBLE_DEVICES": "3,7"}
    assert "model;touch /tmp/nope" in preview.argv
    assert "'model;touch /tmp/nope'" in preview.display
