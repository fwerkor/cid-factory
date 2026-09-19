import json
from pathlib import Path

from cid_factory.assets import AssetCatalog
from cid_factory.models import AssetKind


def test_asset_catalog_discovers_cid_assets(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    dataset = data / "train.jsonl"
    dataset.write_text('{"x": 1}\n', encoding="utf-8")
    (data / "train.manifest.json").write_text(
        json.dumps({"examples": 12, "transitions": 34, "sha256": "abc"}),
        encoding="utf-8",
    )

    checkpoint = tmp_path / "stage-a-epoch-0001.pt"
    checkpoint.write_bytes(b"checkpoint")

    stage_b = tmp_path / "stage-b-epoch-0001"
    stage_b.mkdir()
    (stage_b / "metadata.json").write_text(
        json.dumps({"world_size": 4, "dataset_sha256": "abc"}),
        encoding="utf-8",
    )

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"model_type": "test", "hidden_size": 1024}),
        encoding="utf-8",
    )
    (model / "model.safetensors").write_bytes(b"weights")

    output = tmp_path / "run-output"
    output.mkdir()
    (output / "train_metrics.jsonl").write_text("{}\n", encoding="utf-8")

    catalog = AssetCatalog([tmp_path], max_depth=3)
    assets = catalog.search(limit=100)

    kinds = {asset.kind for asset in assets}
    assert AssetKind.DATASET in kinds
    assert AssetKind.MANIFEST in kinds
    assert AssetKind.CHECKPOINT in kinds
    assert AssetKind.MODEL in kinds
    assert AssetKind.OUTPUT in kinds

    dataset_asset = next(asset for asset in assets if asset.path == str(dataset))
    assert dataset_asset.details["examples"] == 12
    assert dataset_asset.details["transitions"] == 34

    stage_b_asset = next(asset for asset in assets if asset.path == str(stage_b))
    assert stage_b_asset.details["stage"] == "stage-b"
    assert stage_b_asset.details["world_size"] == 4


def test_asset_catalog_filters_kind_and_query(tmp_path: Path) -> None:
    (tmp_path / "alpha.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "beta.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "train_metrics.jsonl").write_text("{}\n", encoding="utf-8")

    catalog = AssetCatalog([tmp_path])
    assets = catalog.search(kind=AssetKind.DATASET, query="beta")

    assert [asset.name for asset in assets] == ["beta.jsonl"]
    assert all(asset.name != "train_metrics.jsonl" for asset in catalog.search())


def test_asset_catalog_deduplicates_overlapping_roots(tmp_path: Path) -> None:
    child = tmp_path / "child"
    child.mkdir()
    dataset = child / "train.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")

    catalog = AssetCatalog([tmp_path])
    assets = catalog.search(extra_roots=[child])

    matching = [asset for asset in assets if asset.path == str(dataset)]
    assert len(matching) == 1
