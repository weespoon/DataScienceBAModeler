"""titanic_agent — an agent-run pipeline for the Kaggle Titanic competition.

Pipeline order (see CLAUDE.md):
    pull -> review -> split -> freeze -> explore -> tune -> ensemble

Two hard rules:
    1. Nothing under ``data/frozen/`` is modified after ``freeze`` has run.
    2. Every modeling run appends exactly one row to ``experiments/results.csv``.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "paths",
    "results_log",
    "pull",
    "review",
    "split",
    "freeze",
    "explore",
    "tune",
    "ensemble",
]
