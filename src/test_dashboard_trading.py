"""Tests for dashboard_trading.py's graceful-degradation path -- a
LIVE-only machine (no development.parquet, e.g. a trading-floor Bloomberg
Terminal box provisioned without this project's historical research data)
must show an explanatory message, never an unhandled traceback."""

from __future__ import annotations

from unittest.mock import patch

import dashboard_trading as dt


class _FakeSessionState(dict):
    """Supports both `"x" in state` and `state.x = y`, matching the subset
    of st.session_state's dict/attribute dual interface _init_state uses."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class _FakeSt:
    """Minimal Streamlit stand-in -- just enough surface for
    render_trading_tab's early-return path to run without a real
    Streamlit script context."""

    def __init__(self):
        self.info_calls = []
        self.warning_calls = []
        self.session_state = _FakeSessionState()

    def subheader(self, *a, **k):
        pass

    def info(self, msg):
        self.info_calls.append(msg)

    def warning(self, msg):
        self.warning_calls.append(msg)


def test_render_trading_tab_shows_info_not_exception_when_development_data_missing():
    fake = _FakeSt()
    with patch("dashboard_trading.load_or_build_signals", side_effect=FileNotFoundError("nope")):
        dt.render_trading_tab(fake)  # must not raise
    assert len(fake.info_calls) == 1
    assert "development.parquet" in fake.info_calls[0]
    assert not fake.warning_calls  # returned before reaching the multiple-testing warning


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
