"""titanic_agent — an agent-run pipeline for the Kaggle Titanic competition.

Pipeline order (see CLAUDE.md):
    pull -> review -> split -> freeze -> explore -> tune -> ensemble

``freeze`` refuses to run twice, so it cannot sit inside a loop: ``bootstrap``
covers steps 2-4 once, ``loop`` covers the repeatable steps 5-7.

Three hard rules:
    1. Nothing under ``data/frozen/`` is modified after ``freeze`` has run.
    2. The holdout is read once per frozen dataset, by ``ensemble`` only, and
       every read leaves a receipt.
    3. Every modeling run appends exactly one row to ``experiments/results.csv``.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    # pipeline steps, in order
    "pull",
    "review",
    "split",
    "freeze",
    "explore",
    "tune",
    "ensemble",
    # support
    "paths",
    "results_log",
    "receipts",
    "status",
]
