# Pass 20261012-0909EDT

Written by `make report`. Every number below is read from the module that owns it — the ledger, the journal, the manifest, the receipts — so this page cannot disagree with them.

## The experiment

**`exp-009-simplify-champion-to-rf-wcg-title-median-age`** — simplify champion to rf_wcg + title-median Age

| | |
| --- | --- |
| verdict | **closed — verdict not stated in the summary** |
| rows | 569 train / 143 val |
| best cv | 0.8313 |
| validation read | 0.8531 (1 read) |
| plan locked | locked, unchanged (`6e15c1be3464`) |
| spend | {'wall_clock_s': 28.0} |

> Parsimony test: rf_simple (old RF + WCG rate + title-median Age; drops TitleFine, ticket size, all-member group rate, max_features 0.5) vs champion rf_kitchen, one challenger, no selection. Selection +0.0023 +/- 0.0047 (0.5 sigma), fresh seeds +0.0068 +/- 0.0046 (1.5 sigma), nested -0.0018 +/- 0.0058. Champion HELD under the declared rule (selection and nested checks failed); the simpler config ties it. Val 0.8531 for rf_simple (cv 0.8313), within band; permutation control at chance.

## The champion

`rf_kitchen` — crowned 2026-10-11T19:20:22+00:00 by exp-007-bakeoff, delta over the previous champion **+0.0264**.

- **Corrected:** rf_kitchen (exp-007-bakeoff).delta_vs_previous 0.026352 → 0.0163. The stored value is the best-of-20 SELECTION delta, which is biased upward by the winner choosing itself. The honest figure is the fresh-seed confirmation +0.0163 +/- 0.0053 (row 20261011T191821-b81d88, sigma corrected by amendment 20261011T235332-676fae); exp-008 then replicated it at +0.0129 +/- 0.0053 on rows the promotion never saw (row 20261011T193600-c52f68). bakeoff._apply records the confirmation delta for later crownings, so this applies to exp-007 only.

## Did the rules hold

`make audit`: **pass** (13 checks, 0 binding failures)

- [pass] `rule 1 · checksums match` — sha=83a4db7c5e7b over 4 files
- [pass] `rule 3 · ledger chain intact` — 139 chained, 19 predate the chain
- [note] `rule 3 · chain rebuilt` — 1 rebuild(s); most recent: schema migration: 15 -> 16 columns. Rows written before it cannot be verified against their original bytes — the chain proves nothing was removed *since* the rebuild
- [note] `rule 3 · unchained rows` — 19 row(s) written before the hash chain existed; they cannot be verified, which is not the same as being wrong
- [pass] `rule 2 · at most one holdout read` — 0 receipt(s) for this dataset
- [pass] `rule 2 · attestations match receipts` — 0 row(s) attest a read, 0 receipt(s) exist
- [pass] `rule 2 · one val read per experiment` — 7 val read(s) across 9 experiment(s)
- [pass] `rule 3 · locked plans unchanged` — 3 plan(s) still hash to what was journaled before scoring
- [pass] `rule 4 · feature provenance` — 46 derivable columns; every logged feature is one of them
- [pass] `rule 4 · external reads` — no module reads data from outside the sanctioned directories
- [pass] `rule 4 · score plausibility` — best cv 0.8401, best val 0.8741 — below the 0.90 implausibility line
- [pass] `rule 4 · stray data files` — only the expected files are present
- [note] `provenance · uncommitted code` — 152 of 158 row(s) were produced by a dirty tree and cannot be tied to committed code

`make leakage`: **pass** (4 structural guards)

- [pass] `feature provenance` — 46 derivable columns; every logged feature is one of them
- [pass] `external reads` — no module reads data from outside the sanctioned directories
- [pass] `score plausibility` — best cv 0.8401, best val 0.8741 — below the 0.90 implausibility line
- [pass] `stray data files` — only the expected files are present

## The final test

**Unread.** frozen/holdout.csv has never been scored for this dataset, so the one generalisation number is still available.

Reading it is a human's call and is deliberately absent from every `make` target. An autonomous pass does not take it.

## Decisions and refusals

- `2026-10-12T13:09:26+00:00` bakeoff: HOLD rf_simple: FAIL selection delta rf_simple vs champion: +0.0023 +/- 0.0047 (naive SE, +0.49 sigma; +1.3 passengers) vs bar 0.003; PASS confirmation rf_simple vs champion, 5 fresh fold seeds: +0.0068 +/- 0.0046 (naive SE; seed sd 0.0033 over 5, +1.46 sigma; +3.9 passengers) vs bar 0.003; FAIL nested A-B -0.0018 +/- 0.0058 vs 0

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

