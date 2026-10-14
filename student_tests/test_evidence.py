import json

import pytest

from lab.summarize import NAMES, load_evidence


def synthetic_evidence():
    """Small schema fixture, not a model result and never written to report/."""
    return {"score_version": 1, "split": "test", "coord": "norm1000", "n_images": 311,
            "missing_predictions": 0, "dataset_version": "test-only",
            "meta": {"trainable_params": 10, "peak_vram_gb": 1, "train_minutes": 1}}


def write_fixture(root, mutate=None):
    (root/"evidence").mkdir()
    for name in NAMES:
        item = synthetic_evidence()
        if mutate:
            mutate(name, item)
        (root/"evidence"/f"{name}.json").write_text(json.dumps(item))


def test_missing_experiments_cannot_generate_report(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_evidence(tmp_path)


def test_mixed_dataset_versions_are_rejected(tmp_path):
    write_fixture(tmp_path, lambda name, item: item.update(dataset_version="other") if name == "e3" else None)
    with pytest.raises(ValueError, match="dataset versions"):
        load_evidence(tmp_path)


@pytest.mark.parametrize("field,value", [("split", "val"), ("n_images", 40),
                                        ("missing_predictions", 1), ("coord", "pixel")])
def test_partial_or_wrong_split_evidence_is_rejected(tmp_path, field, value):
    write_fixture(tmp_path, lambda name, item: item.update({field: value}) if name == "e1" else None)
    with pytest.raises(ValueError, match="test evidence"):
        load_evidence(tmp_path)


def test_boolean_resource_metadata_is_not_a_valid_measurement(tmp_path):
    write_fixture(tmp_path, lambda name, item: item["meta"].update(train_minutes=True) if name == "e2" else None)
    with pytest.raises(ValueError, match="train_minutes"):
        load_evidence(tmp_path)
