"""Smoke test: ``make audit`` — the honesty scoreboard.

Each test constructs a specific violation and asserts the audit goes red for
that reason. If an audit cannot fail, it is decoration; most of this file is
therefore about making it fail on purpose.

The one informational case matters too: rows that predate the hash chain must be
reported as *unverifiable*, not as tampered. An audit that cries wolf gets
ignored, which is the same outcome as not having one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from titanic_agent import audit, freeze, receipts, results_log, tune


@pytest.fixture
def ledger(frozen_dir: Path, log_path: Path) -> Path:
    """A small honest ledger against the fixture dataset."""
    tune.tune_all(("dummy", "logistic"), frozen_dir=frozen_dir, log_path=log_path, cv_folds=2)
    return log_path


def run(frozen_dir: Path, ledger: Path, receipts_path: Path | None = None):
    return audit.audit(
        frozen_dir=frozen_dir,
        log_path=ledger,
        receipts_path=receipts_path or (ledger.parent / "holdout_reads.json"),
    )


def named(report: audit.AuditReport, fragment: str) -> audit.Check:
    return next(c for c in report.checks if fragment in c.name)


# --------------------------------------------------------------------------- #
# the honest case
# --------------------------------------------------------------------------- #


def test_an_honest_run_passes(frozen_dir: Path, ledger: Path) -> None:
    report = run(frozen_dir, ledger)
    assert report.ok
    assert report.failures == []


def test_every_rule_is_checked(frozen_dir: Path, ledger: Path) -> None:
    names = " ".join(c.name for c in run(frozen_dir, ledger).checks)
    assert "rule 1" in names
    assert "rule 2" in names
    assert "rule 3" in names


def test_the_chain_is_reported_as_intact(frozen_dir: Path, ledger: Path) -> None:
    assert named(run(frozen_dir, ledger), "ledger chain").passed


# --------------------------------------------------------------------------- #
# Rule 1 — the failure injection
# --------------------------------------------------------------------------- #


def test_a_corrupted_frozen_file_fails_the_audit(writable_frozen_dir: Path, log_path: Path) -> None:
    # The injection the roadmap calls for: tamper with frozen data, confirm it
    # is caught rather than silently carried into every later score.
    tune.tune("dummy", frozen_dir=writable_frozen_dir, log_path=log_path, cv_folds=2)
    target = writable_frozen_dir / "train.csv"
    target.write_text(target.read_text(encoding="utf-8") + "\n999,1,3,X\n", encoding="utf-8")

    report = run(writable_frozen_dir, log_path)
    assert not report.ok
    assert named(report, "checksums").passed is False
    assert "suspect" in named(report, "checksums").detail


def test_a_corrupted_dataset_also_halts_the_loop(writable_frozen_dir: Path) -> None:
    # Detection is not enough on its own: the steps must refuse to run, or the
    # audit only tells you afterwards that the scores were worthless.
    target = writable_frozen_dir / "train.csv"
    target.write_text("broken\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        freeze.require_frozen(writable_frozen_dir)


def test_an_unfrozen_dataset_fails_immediately(tmp_path: Path, log_path: Path) -> None:
    results_log.ensure_log(log_path)
    report = run(tmp_path, log_path)
    assert not report.ok
    assert named(report, "dataset frozen").passed is False


# --------------------------------------------------------------------------- #
# Rule 3 — the ledger
# --------------------------------------------------------------------------- #


def test_a_deleted_row_fails_the_audit(frozen_dir: Path, ledger: Path) -> None:
    lines = ledger.read_text(encoding="utf-8").splitlines()
    ledger.write_text("\n".join(lines[:1] + lines[2:]) + "\n", encoding="utf-8")
    report = run(frozen_dir, ledger)
    assert not report.ok
    assert "deleted or edited" in named(report, "ledger chain").detail


def test_an_edited_score_fails_the_audit(frozen_dir: Path, log_path: Path) -> None:
    tune.tune_all(("dummy", "logistic"), frozen_dir=frozen_dir, log_path=log_path, cv_folds=2)
    text = log_path.read_text(encoding="utf-8")
    first_score = results_log.read_log(log_path).iloc[0]["cv_score"]
    log_path.write_text(text.replace(f"{float(first_score):.6f}", "0.999000"), encoding="utf-8")
    report = run(frozen_dir, log_path)
    assert not report.ok


def test_rows_predating_the_chain_are_informational_not_failures(
    frozen_dir: Path, log_path: Path
) -> None:
    # Unverifiable is not the same as proven bad.
    results_log.ensure_log(log_path)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("old-run,2026-01-01T00:00:00+00:00,tune,logistic,{},0.8,,,1,x,y,False,,,\n")
    report = run(frozen_dir, log_path)
    assert report.ok
    unchained = named(report, "unchained rows")
    assert unchained.informational and unchained.passed


# --------------------------------------------------------------------------- #
# Rule 2 — the holdout
# --------------------------------------------------------------------------- #


def test_two_receipts_fail_the_audit(frozen_dir: Path, ledger: Path, tmp_path: Path) -> None:
    manifest = freeze.load_manifest(frozen_dir)
    receipts_path = tmp_path / "holdout_reads.json"
    for run_id in ("first", "second"):
        receipts.record_holdout_read(
            frozen_sha256=manifest.dataset_sha256, run_id=run_id, path=receipts_path
        )
    report = run(frozen_dir, ledger, receipts_path)
    assert not report.ok
    assert "more than once" in named(report, "at most one holdout read").detail


def test_a_receipt_with_no_attestation_fails(
    frozen_dir: Path, ledger: Path, tmp_path: Path
) -> None:
    # The holdout was read, but no ledger row admits it.
    manifest = freeze.load_manifest(frozen_dir)
    receipts_path = tmp_path / "holdout_reads.json"
    receipts.record_holdout_read(
        frozen_sha256=manifest.dataset_sha256, run_id="unattested", path=receipts_path
    )
    report = run(frozen_dir, ledger, receipts_path)
    assert not report.ok
    assert "disagree" in named(report, "attestations match receipts").detail


def test_an_attestation_with_no_receipt_fails(
    frozen_dir: Path, log_path: Path, tmp_path: Path
) -> None:
    # A row claims it read the holdout, but no receipt exists. One of the two
    # records is false and the audit cannot tell which -- only that they differ.
    results_log.append_run(
        step="ensemble", model="vote", cv_score=0.9, holdout_read=True, path=log_path
    )
    report = run(frozen_dir, log_path, tmp_path / "none.json")
    assert not report.ok
    assert named(report, "attestations match receipts").passed is False


def test_a_consistent_single_read_passes(frozen_dir: Path, ledger: Path, tmp_path: Path) -> None:
    manifest = freeze.load_manifest(frozen_dir)
    receipts_path = tmp_path / "holdout_reads.json"
    receipts.record_holdout_read(
        frozen_sha256=manifest.dataset_sha256, run_id="final", path=receipts_path
    )
    results_log.append_run(
        step="ensemble", model="vote", cv_score=0.9, holdout_read=True, path=ledger
    )
    assert run(frozen_dir, ledger, receipts_path).ok


# --------------------------------------------------------------------------- #
# provenance and cli
# --------------------------------------------------------------------------- #


def test_dirty_provenance_is_informational(frozen_dir: Path, ledger: Path) -> None:
    report = run(frozen_dir, ledger)
    dirty = [c for c in report.checks if "uncommitted" in c.name]
    assert all(c.informational for c in dirty)


def test_foreign_fingerprints_are_flagged(frozen_dir: Path, ledger: Path) -> None:
    rows = results_log.read_log(ledger)
    rows.loc[0, "frozen_sha256"] = "f" * 64
    rows["params"] = rows["params"].map(json.dumps)
    rows.to_csv(ledger, index=False)
    report = run(frozen_dir, ledger)
    assert any("foreign datasets" in c.name for c in report.checks)


def test_main_exits_zero_when_clean(frozen_dir: Path, ledger: Path, capsys, monkeypatch) -> None:
    from titanic_agent import paths

    monkeypatch.setattr(paths, "RESULTS_LOG", ledger)
    monkeypatch.setattr(paths, "HOLDOUT_READS", ledger.parent / "holdout_reads.json")
    assert audit.main(["--frozen-dir", str(frozen_dir)]) == 0
    assert "No detectable violation" in capsys.readouterr().out


def test_main_exits_nonzero_on_violation(tmp_path: Path, capsys, monkeypatch) -> None:
    from titanic_agent import paths

    monkeypatch.setattr(paths, "RESULTS_LOG", tmp_path / "results.csv")
    assert audit.main(["--frozen-dir", str(tmp_path)]) == 1
    assert "violation" in capsys.readouterr().out


def test_the_audit_does_not_overclaim(frozen_dir: Path, ledger: Path, capsys, monkeypatch) -> None:
    # A passing audit means no *detectable* violation. Claiming it proves
    # honesty would be the same overclaiming this repo exists to study.
    from titanic_agent import paths

    monkeypatch.setattr(paths, "RESULTS_LOG", ledger)
    monkeypatch.setattr(paths, "HOLDOUT_READS", ledger.parent / "holdout_reads.json")
    audit.main(["--frozen-dir", str(frozen_dir)])
    out = capsys.readouterr().out
    assert "not the same as proof of honesty" in out
