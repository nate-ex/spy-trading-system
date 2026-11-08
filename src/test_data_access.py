"""
Tests for data_access.py's non-obvious logic (required purpose, logging
side effect) and split_dataset.py's split correctness. Synthetic data
only, all I/O redirected into tmp_path -- no real-data dependency, no
mutation of the real data/processed/ or logs/ directories.
"""

import json

import pandas as pd
import pytest

import data_access
import split_dataset


def _toy_feature_table() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    rows = []
    for d in dates:
        for m in range(3):
            rows.append({"time": d + pd.Timedelta(minutes=m), "date": d})
    df = pd.DataFrame(rows)
    df["split"] = "development"
    df.loc[df["date"] >= dates[8], "split"] = "held_out"
    return df


def test_split_dataset_writes_manifest_and_matching_row_counts(tmp_path, monkeypatch):
    src = tmp_path / "feature_table.parquet"
    _toy_feature_table().to_parquet(src, index=False)

    monkeypatch.setattr(split_dataset, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(split_dataset, "DEVELOPMENT_PATH", tmp_path / "development.parquet")
    monkeypatch.setattr(split_dataset, "HOLDOUT_PATH", tmp_path / "holdout_test.parquet")
    monkeypatch.setattr(split_dataset, "MANIFEST_PATH", tmp_path / "split_manifest.json")

    manifest = split_dataset.split_dataset(feature_table_path=src)

    dev = pd.read_parquet(split_dataset.DEVELOPMENT_PATH)
    held = pd.read_parquet(split_dataset.HOLDOUT_PATH)

    assert len(dev) == manifest["development"]["n_rows"]
    assert len(held) == manifest["holdout_test"]["n_rows"]
    assert len(dev) + len(held) == 30  # 10 days * 3 rows/day
    assert (dev["split"] == "development").all()
    assert (held["split"] == "held_out").all()
    assert split_dataset.MANIFEST_PATH.exists()
    with open(split_dataset.MANIFEST_PATH) as f:
        on_disk = json.load(f)
    assert on_disk["split_boundary_date"] == manifest["split_boundary_date"]


def test_split_dataset_raises_on_missing_split_column(tmp_path, monkeypatch):
    src = tmp_path / "feature_table.parquet"
    _toy_feature_table().drop(columns=["split"]).to_parquet(src, index=False)
    monkeypatch.setattr(split_dataset, "DEVELOPMENT_PATH", tmp_path / "development.parquet")
    monkeypatch.setattr(split_dataset, "HOLDOUT_PATH", tmp_path / "holdout_test.parquet")
    monkeypatch.setattr(split_dataset, "MANIFEST_PATH", tmp_path / "split_manifest.json")

    with pytest.raises(ValueError, match="split"):
        split_dataset.split_dataset(feature_table_path=src)


def _wire_data_access_to(tmp_path, monkeypatch):
    df = _toy_feature_table()
    dev_path = tmp_path / "development.parquet"
    held_path = tmp_path / "holdout_test.parquet"
    log_path = tmp_path / "holdout_access_log.md"
    df[df["split"] == "development"].to_parquet(dev_path, index=False)
    df[df["split"] == "held_out"].to_parquet(held_path, index=False)
    monkeypatch.setattr(data_access, "DEVELOPMENT_PATH", dev_path)
    monkeypatch.setattr(data_access, "HOLDOUT_PATH", held_path)
    monkeypatch.setattr(data_access, "HOLDOUT_ACCESS_LOG", log_path)
    return log_path


def test_load_development_returns_only_development_rows(tmp_path, monkeypatch):
    _wire_data_access_to(tmp_path, monkeypatch)
    out = data_access.load_development()
    assert (out["split"] == "development").all()


def test_load_holdout_test_requires_nonempty_purpose(tmp_path, monkeypatch):
    _wire_data_access_to(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="purpose"):
        data_access.load_holdout_test("")
    with pytest.raises(ValueError, match="purpose"):
        data_access.load_holdout_test("   ")


def test_load_holdout_test_logs_every_call(tmp_path, monkeypatch):
    log_path = _wire_data_access_to(tmp_path, monkeypatch)
    assert not log_path.exists()

    data_access.load_holdout_test(purpose="Q000 final evaluation")
    assert log_path.exists()
    contents = log_path.read_text()
    assert "Q000 final evaluation" in contents
    assert "test_data_access.py" in contents
    assert contents.count("`test_data_access.py`") == 1

    data_access.load_holdout_test(purpose="Q001 final evaluation")
    contents = log_path.read_text()
    assert contents.count("Q000 final evaluation") == 1
    assert "Q001 final evaluation" in contents
    assert contents.count("`test_data_access.py`") == 2
