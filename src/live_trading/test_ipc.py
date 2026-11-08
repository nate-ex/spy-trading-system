import json

import pytest

from live_trading import ipc


@pytest.fixture(autouse=True)
def _isolated_ipc_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(ipc, "IPC_DIR", tmp_path)
    monkeypatch.setattr(ipc, "COMMAND_PATH", tmp_path / "command.json")
    monkeypatch.setattr(ipc, "STATUS_PATH", tmp_path / "status.json")
    yield


def test_write_command_rejects_unknown_action():
    with pytest.raises(ValueError):
        ipc.write_command("explode")


def test_write_and_read_pending_command_roundtrip():
    written = ipc.write_command("deploy", config_name="my_strategy", mode="paper")
    pending = ipc.read_pending_command(last_processed_id=None)
    assert pending == written


def test_read_pending_command_returns_none_once_already_processed():
    written = ipc.write_command("pause")
    assert ipc.read_pending_command(last_processed_id=written["id"]) is None


def test_read_pending_command_missing_file_returns_none():
    assert ipc.read_pending_command(last_processed_id=None) is None


def test_read_pending_command_malformed_file_returns_none_not_raise():
    ipc.COMMAND_PATH.parent.mkdir(parents=True, exist_ok=True)
    ipc.COMMAND_PATH.write_text("{not valid json")
    assert ipc.read_pending_command(last_processed_id=None) is None


def test_kill_command_has_no_config_name_required():
    written = ipc.write_command("kill")
    assert written["action"] == "kill"
    assert written["config_name"] is None


def test_write_and_read_status_roundtrip():
    ipc.write_status(state="running", active_config="my_strategy", mode="paper",
                     positions=[{"symbol": "SPY", "qty": 10}], equity=100_000.0,
                     realized_pnl=50.0, unrealized_pnl=-10.0)
    status = ipc.read_status()
    assert status["state"] == "running"
    assert status["positions"] == [{"symbol": "SPY", "qty": 10}]
    assert status["equity"] == 100_000.0


def test_read_status_missing_file_returns_none():
    assert ipc.read_status() is None
