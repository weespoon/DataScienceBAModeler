"""Smoke test: holdout read receipts — the Rule 2 mechanism.

The important case is :func:`test_a_second_read_still_leaves_a_second_receipt`.
``require_unread`` is supposed to make a second read impossible, but if one
happens anyway the receipt file must still show it. A mechanism that hides its
own failure is worse than no mechanism.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from titanic_agent import receipts

SHA_A = "a" * 64
SHA_B = "b" * 64


@pytest.fixture
def receipt_path(tmp_path: Path) -> Path:
    """Path to a throwaway receipt file (not yet created)."""
    return tmp_path / "experiments" / "holdout_reads.json"


def test_no_receipts_on_a_missing_file(receipt_path: Path) -> None:
    assert receipts.read_receipts(receipt_path) == []
    assert receipts.has_been_read(SHA_A, receipt_path) is False


def test_corrupt_receipt_file_reads_as_empty(tmp_path: Path) -> None:
    path = tmp_path / "holdout_reads.json"
    path.write_text("{not json", encoding="utf-8")
    assert receipts.read_receipts(path) == []


def test_require_unread_passes_when_nothing_was_read(receipt_path: Path) -> None:
    receipts.require_unread(SHA_A, receipt_path)  # must not raise


def test_recording_a_read_creates_a_receipt(receipt_path: Path) -> None:
    receipt = receipts.record_holdout_read(
        frozen_sha256=SHA_A,
        run_id="20261010T120000-abc123",
        score=0.8212,
        reason="final estimate",
        path=receipt_path,
    )
    assert receipt_path.is_file()
    assert receipt["frozen_sha256"] == SHA_A
    assert receipt["score"] == 0.8212
    assert receipt["timestamp_utc"].endswith("+00:00")
    assert receipts.has_been_read(SHA_A, receipt_path) is True


def test_require_unread_refuses_after_a_read(receipt_path: Path) -> None:
    receipts.record_holdout_read(frozen_sha256=SHA_A, run_id="run-1", score=0.82, path=receipt_path)
    with pytest.raises(RuntimeError, match="already read"):
        receipts.require_unread(SHA_A, receipt_path)


def test_the_refusal_names_the_earlier_run(receipt_path: Path) -> None:
    receipts.record_holdout_read(
        frozen_sha256=SHA_A, run_id="run-first", score=0.82, path=receipt_path
    )
    with pytest.raises(RuntimeError, match="run-first"):
        receipts.require_unread(SHA_A, receipt_path)


def test_receipts_are_scoped_to_one_frozen_dataset(receipt_path: Path) -> None:
    receipts.record_holdout_read(frozen_sha256=SHA_A, run_id="run-a", path=receipt_path)
    # A different frozen dataset gets its own single read.
    receipts.require_unread(SHA_B, receipt_path)
    assert receipts.has_been_read(SHA_B, receipt_path) is False


def test_a_second_read_still_leaves_a_second_receipt(receipt_path: Path) -> None:
    # require_unread() should have prevented this. If it is bypassed anyway, the
    # violation has to remain visible -- appending, never overwriting.
    receipts.record_holdout_read(frozen_sha256=SHA_A, run_id="run-1", path=receipt_path)
    receipts.record_holdout_read(frozen_sha256=SHA_A, run_id="run-2", path=receipt_path)
    found = receipts.receipts_for(SHA_A, receipt_path)
    assert len(found) == 2
    assert [r["run_id"] for r in found] == ["run-1", "run-2"]


def test_receipt_file_is_valid_json_a_human_can_read(receipt_path: Path) -> None:
    receipts.record_holdout_read(frozen_sha256=SHA_A, run_id="run-1", path=receipt_path)
    parsed = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert isinstance(parsed, list) and len(parsed) == 1


def test_reason_stays_on_one_line(receipt_path: Path) -> None:
    receipt = receipts.record_holdout_read(
        frozen_sha256=SHA_A, run_id="run-1", reason="line one\nline two", path=receipt_path
    )
    assert "\n" not in receipt["reason"]


def test_score_is_optional(receipt_path: Path) -> None:
    receipt = receipts.record_holdout_read(frozen_sha256=SHA_A, run_id="run-1", path=receipt_path)
    assert receipt["score"] is None
