# Pass 20261011-1535EDT

Written by `make report`. Every number below is read from the module that owns it — the ledger, the journal, the manifest, the receipts — so this page cannot disagree with them.

## The experiment

**`exp-008-decompose-champion`** — decompose champion

| | |
| --- | --- |
| verdict | **closed (verdict not stated in the summary)** |
| rows | 569 train / 143 val |
| best cv | 0.8325 |
| validation read | 0.8741 (1 read) |
| plan locked | locked, unchanged (`03bf15c6bf29`) |
| spend | {'wall_clock_s': 70.3} |

> Decomposed rf_kitchen. Replication: champion beats the old RF by +0.0129 on new rows (exp-007: +0.0264 selection, +0.0163 fresh seeds) -- the winner's curse, not a split interaction. rf_wcg alone ties the champion; no single change except title-median Age (-0.0047 if removed) is measurable. Winner without_max_features +0.0041 failed fresh-seed confirmation (-0.0032): champion holds. Val 0.8741 (cv 0.8325), within band; group-rate coverage grows with training size, so CV understates group models.

## The champion

`rf_kitchen` — crowned 2026-10-11T19:20:22+00:00 by exp-007-bakeoff, delta over the previous champion **+0.0264**.

- **Corrected:** rf_kitchen (exp-007-bakeoff).delta_vs_previous 0.026352 → 0.0163. The stored value is the best-of-20 SELECTION delta, which is biased upward by the winner choosing itself. The honest figure is the fresh-seed confirmation +0.0163 +/- 0.0053 (row 20261011T191821-b81d88, sigma corrected by amendment 20261011T235332-676fae); exp-008 then replicated it at +0.0129 +/- 0.0053 on rows the promotion never saw (row 20261011T193600-c52f68). bakeoff._apply records the confirmation delta for later crownings, so this applies to exp-007 only.

## Did the rules hold

`make audit`: **pass** (13 checks, 0 binding failures)

- [pass] `rule 1 · checksums match` — sha=83a4db7c5e7b over 4 files
- [pass] `rule 3 · ledger chain intact` — 133 chained, 19 predate the chain
- [note] `rule 3 · chain rebuilt` — 1 rebuild(s); most recent: schema migration: 15 -> 16 columns. Rows written before it cannot be verified against their original bytes — the chain proves nothing was removed *since* the rebuild
- [note] `rule 3 · unchained rows` — 19 row(s) written before the hash chain existed; they cannot be verified, which is not the same as being wrong
- [pass] `rule 2 · at most one holdout read` — 0 receipt(s) for this dataset
- [pass] `rule 2 · attestations match receipts` — 0 row(s) attest a read, 0 receipt(s) exist
- [pass] `rule 2 · one val read per experiment` — 6 val read(s) across 8 experiment(s)
- [pass] `rule 3 · locked plans unchanged` — 2 plan(s) still hash to what was journaled before scoring
- [pass] `rule 4 · feature provenance` — 46 derivable columns; every logged feature is one of them
- [pass] `rule 4 · external reads` — no module reads data from outside the sanctioned directories
- [pass] `rule 4 · score plausibility` — best cv 0.8401, best val 0.8741 — below the 0.90 implausibility line
- [pass] `rule 4 · stray data files` — only the expected files are present
- [note] `provenance · uncommitted code` — 152 of 152 row(s) were produced by a dirty tree and cannot be tied to committed code

`make leakage`: **pass** (4 structural guards)

- [pass] `feature provenance` — 46 derivable columns; every logged feature is one of them
- [pass] `external reads` — no module reads data from outside the sanctioned directories
- [pass] `score plausibility` — best cv 0.8401, best val 0.8741 — below the 0.90 implausibility line
- [pass] `stray data files` — only the expected files are present

## The final test

**Unread.** frozen/holdout.csv has never been scored for this dataset, so the one generalisation number is still available.

Reading it is a human's call and is deliberately absent from every `make` target. An autonomous pass does not take it.

## Decisions and refusals

- `2026-10-11T19:35:37+00:00` experiment: opened exp-008-decompose-champion: 569 train / 143 val (seed 20251018), caps {'wall_clock_s': 1800, 'loop_passes': 3}
- `2026-10-11T19:35:56+00:00` bakeoff: plan and predictions locked before scoring: experiments/exp-008-decompose-champion/plan.md sha256=03bf15c6bf293a0bfdb940800d72bce7ec353815999fe5caf23ac317aff093da
- `2026-10-11T19:35:57+00:00` bakeoff: exp-008-decompose-champion: 9 decompose arms vs champion rf_kitchen, 5x3 repeated stratified CV, paired. Declared before scoring: the winner is the arm with the largest delta_vs_champion; it replaces the champion only if (1) its selection delta >= 0.003, (2) its delta on 5 fresh fold seeds >= 0.003, and (3) nested CV of the whole pick-the-best procedure beats the champion (A - B > 0). The validation read scores the winner, whatever it is.
- `2026-10-11T19:36:18+00:00` bakeoff: winner without_max_features of 9; without_max_features vs champion, 5 fresh fold seeds: -0.0032 +/- 0.0046 (seed sd, -0.69 sigma; -1.8 passengers) (21s)
- `2026-10-11T19:37:24+00:00` bakeoff: HOLD without_max_features: PASS selection delta without_max_features vs champion: +0.0041 +/- 0.0024 (naive SE, +1.70 sigma; +2.3 passengers) vs bar 0.003; FAIL confirmation without_max_features vs champion, 5 fresh fold seeds: -0.0032 +/- 0.0046 (seed sd, -0.69 sigma; -1.8 passengers) vs bar 0.003; PASS nested A-B +0.0018 +/- 0.0018 vs 0
- `2026-10-11T19:37:34+00:00` evaluate: exp-008-decompose-champion validation: 0.8741 on 143 rows (random_forest); cv was 0.8325
- `2026-10-11T19:38:17+00:00` experiment: closed exp-008-decompose-champion: best_cv=0.832469, val=0.8741258741258742
- `2026-10-11T19:38:48+00:00` —: stopped: exp-007 follow-ups complete -- exp-008 decomposition closed (champion holds), replication measured, submission written from champion. Holdout NOT read, per user instruction.
- `2026-10-11T19:40:39+00:00` amendment: git_sha amendment for exp-008 rows: 20261011T194039-f8ade3

## Pipeline

| # | step | state |
| --- | --- | --- |
| 1 | pull | done |
| 2 | review | done |
| 3 | split | done |
| 4 | freeze | done |
| 5 | explore | done |
| 6 | tune | done |
| 7 | ensemble | done |

