"""Git sync for saved strategy configs (data/processed/backtests/configs/)
-- so a strategy saved on one machine (e.g. the researcher's dev box)
shows up for everyone else the next time they launch the dashboard (which
already runs `git pull` -- see scripts/launch_dashboard.bat), without
anyone needing to think about git themselves.

Deliberately narrow: every operation here touches ONLY `configs_dir`
(`backtest_storage.CONFIGS_DIR`), never a blanket `git add -A`, and NEVER
pulls/merges/rebases. Two reasons pulling is out of scope for this
module: (1) this dashboard often runs from the same clone someone is
actively developing in (see this session's history) -- an automatic pull
triggered by clicking "Save Strategy" could collide with in-progress,
uncommitted code changes in a way nobody asked for; (2) picking up
teammates' newly-saved strategies already happens for free on the next
`launch_dashboard.bat` run, which is the expected "showed up when he
launched it" moment the user described -- no need to reproduce that
inside a running session.

Every function here is best-effort and NEVER raises: a sync failure (no
git installed, no network, diverged remote, no configured git identity)
must never be allowed to undo the LOCAL save/delete that already
succeeded on disk. Callers show the user whatever this reports.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GIT_TIMEOUT_SECS = 20


def sync_dir(dir_path: Path, message: str) -> tuple[bool, str]:
    """Stage/commit/push everything under `dir_path` (adds, modifies, AND
    deletions -- `git add --all -- <path>` covers all three). Returns
    (ok, detail): ok=True with detail "no changes to sync" is the normal
    no-op case (re-saving something byte-identical to what's already
    committed); ok=False means the local save/delete still happened, it
    just isn't on GitHub yet.
    """
    try:
        subprocess.run(
            ["git", "add", "--all", "--", str(dir_path)],
            cwd=REPO_ROOT, check=True, capture_output=True, text=True, timeout=GIT_TIMEOUT_SECS,
        )
        commit = subprocess.run(
            ["git", "commit", "-m", message],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=GIT_TIMEOUT_SECS,
        )
        if commit.returncode != 0:
            combined = (commit.stdout + commit.stderr).lower()
            # Two distinct git messages both mean "this specific commit had
            # nothing staged to add" -- "nothing to commit, working tree
            # clean" when the WHOLE repo is clean, "no changes added to
            # commit" when `git add` staged nothing from `dir_path` but
            # OTHER unrelated files elsewhere in the repo are still dirty
            # (a real scenario: this dashboard can run from the same clone
            # someone is actively developing in -- confirmed live, not
            # just theorized, while testing this against the real repo).
            if "nothing to commit" in combined or "no changes added to commit" in combined:
                return True, "no changes to sync"
            return False, f"saved locally, but commit failed: {(commit.stderr or commit.stdout).strip()[:200]}"

        push = subprocess.run(
            ["git", "push", "origin", "main"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=GIT_TIMEOUT_SECS + 15,
        )
        if push.returncode != 0:
            return False, f"saved and committed locally, but push failed: {push.stderr.strip()[:200]}"
        return True, "synced to GitHub"
    except FileNotFoundError:
        return False, "saved locally, but git isn't available on this machine to sync"
    except subprocess.TimeoutExpired:
        return False, "saved locally, but git sync timed out (check network)"
    except Exception as exc:
        return False, f"saved locally, but sync hit an unexpected error: {exc}"
