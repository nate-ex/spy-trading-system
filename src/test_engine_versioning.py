"""Tests for engine_versioning.py -- the cache-staleness helper that ties
dashboard_demo_grid.parquet/backtest_signals_dev.parquet to the frozen
engine JSONs they were built from."""
from __future__ import annotations

import engine_versioning as ev


def test_is_stale_true_when_artifact_missing(tmp_path):
    assert ev.is_stale(tmp_path / "nope.parquet") is True


def test_is_stale_true_when_stamp_missing(tmp_path):
    artifact = tmp_path / "cache.parquet"
    artifact.write_text("data")
    assert ev.is_stale(artifact) is True  # exists, but never stamped


def test_write_stamp_then_not_stale(tmp_path, monkeypatch):
    artifact = tmp_path / "cache.parquet"
    artifact.write_text("data")
    ev.write_stamp(artifact)
    assert ev.is_stale(artifact) is False


def test_is_stale_true_after_engine_content_changes(tmp_path, monkeypatch):
    stage1 = tmp_path / "stage1_density_engine.json"
    touch = tmp_path / "touch_probability_engine.json"
    stage1.write_text('{"v": 1}')
    touch.write_text('{"v": 1}')
    monkeypatch.setattr(ev, "STAGE1_ENGINE_PATH", stage1)
    monkeypatch.setattr(ev, "TOUCH_ENGINE_PATH", touch)

    artifact = tmp_path / "cache.parquet"
    artifact.write_text("data")
    ev.write_stamp(artifact)
    assert ev.is_stale(artifact) is False

    # simulate a refit: engine JSON content changes
    stage1.write_text('{"v": 2}')
    assert ev.is_stale(artifact) is True


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
