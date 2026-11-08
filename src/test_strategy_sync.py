"""Tests for strategy_sync.py -- mocks subprocess.run entirely (no real
git repo needed) so every branch (success, nothing-to-commit, commit
failure, push failure, git missing, timeout) is deterministic and fast."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import strategy_sync as ss


class _FakeCompleted:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _fake_run_factory(commit_result=None, push_result=None, add_raises=None):
    """Builds a fake subprocess.run that dispatches on the git subcommand
    (args[1]) so each test only needs to specify the branch it cares
    about; add always succeeds unless add_raises is given."""
    def _fake_run(args, **kwargs):
        if add_raises is not None and args[1] == "add":
            raise add_raises
        if args[1] == "add":
            return _FakeCompleted(0)
        if args[1] == "commit":
            return commit_result if commit_result is not None else _FakeCompleted(0)
        if args[1] == "push":
            return push_result if push_result is not None else _FakeCompleted(0)
        raise AssertionError(f"unexpected git subcommand: {args}")
    return _fake_run


def test_sync_dir_success(monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, "run", _fake_run_factory())
    ok, detail = ss.sync_dir(tmp_path, "save strategy X")
    assert ok is True and detail == "synced to GitHub"


def test_sync_dir_nothing_to_commit_fully_clean_repo(monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, "run", _fake_run_factory(
        commit_result=_FakeCompleted(1, stdout="nothing to commit, working tree clean")))
    ok, detail = ss.sync_dir(tmp_path, "save strategy X")
    assert ok is True and detail == "no changes to sync"


def test_sync_dir_nothing_to_commit_with_unrelated_dirty_files_elsewhere(monkeypatch, tmp_path):
    # Real git message when `git add --all -- dir_path` staged nothing
    # (dir_path itself unchanged) but OTHER files elsewhere in the repo
    # are still dirty -- distinct wording from the fully-clean-repo case
    # above, confirmed against a real repo while building this feature.
    monkeypatch.setattr(subprocess, "run", _fake_run_factory(
        commit_result=_FakeCompleted(1, stdout="Changes not staged for commit:\n\tmodified:   other_file.py\n\n"
                                              "no changes added to commit (use \"git add\" and/or \"git commit -a\")")))
    ok, detail = ss.sync_dir(tmp_path, "save strategy X")
    assert ok is True and detail == "no changes to sync"


def test_sync_dir_commit_failure_other_than_nothing_to_commit(monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, "run", _fake_run_factory(
        commit_result=_FakeCompleted(1, stderr="fatal: unable to auto-detect email address")))
    ok, detail = ss.sync_dir(tmp_path, "save strategy X")
    assert ok is False and "commit failed" in detail and "saved locally" in detail


def test_sync_dir_push_failure_still_reports_local_save_succeeded(monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, "run", _fake_run_factory(
        push_result=_FakeCompleted(1, stderr="! [rejected] main -> main (fetch first)")))
    ok, detail = ss.sync_dir(tmp_path, "save strategy X")
    assert ok is False and "push failed" in detail and "committed locally" in detail


def test_sync_dir_git_not_installed(monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, "run", _fake_run_factory(add_raises=FileNotFoundError()))
    ok, detail = ss.sync_dir(tmp_path, "save strategy X")
    assert ok is False and "git isn't available" in detail


def test_sync_dir_timeout(monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, "run", _fake_run_factory(
        add_raises=subprocess.TimeoutExpired(cmd="git add", timeout=20)))
    ok, detail = ss.sync_dir(tmp_path, "save strategy X")
    assert ok is False and "timed out" in detail


def test_sync_dir_never_raises_on_unexpected_error(monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, "run", _fake_run_factory(add_raises=RuntimeError("boom")))
    ok, detail = ss.sync_dir(tmp_path, "save strategy X")  # must not raise
    assert ok is False and "unexpected error" in detail


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
