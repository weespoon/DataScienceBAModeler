# exp-007 — result, scored against the locked predictions

`plan.md` is unchanged since its sha256 (`a388b3bb…`) was journaled before
scoring. Full table: `reports/05_bakeoff.md`.

## The decision-level predictions: 5 of 5

| prediction | p | outcome |
|---|---:|---|
| winner is group-aware | 75% | ✓ `rf_kitchen` |
| confirmation keeps it ≥ 0.003 | 65% | ✓ +0.0163 ± 0.0074 |
| nested A − B > 0 | 65% | ✓ +0.0246 ± 0.0080 |
| promotion | 50% | ✓ crowned |
| val within ±0.06 of cv | 90% | ✓ 0.8531 vs 0.8395 |

## The per-arm ranges: 10 of 19 hit, and the misses all lean one way

| arm | predicted | actual | |
|---|---|---:|---|
| rf_wcg | +0.005…+0.020 (≈ +0.010) | +0.0182 | ✓ (but nearly 2× the point) |
| rf_group / gb_group / et_group | +0.005…+0.020 | +0.0100 / +0.0129 / +0.0152 | ✓✓✓ |
| vote_group / stack_group | best member ± 0.005 | +0.0193 / +0.0211 | ✓✓ |
| rule_wcg | −0.015…+0.005 | **+0.0176** | ✗ |
| rule_gender | −0.04…−0.06 | −0.0328 | ✗ |
| rf_titlefine / rf_shallow / rf_age_by_title | ±0.004 | +0.0023 / −0.0012 / +0.0035 | ✓✓✓ |
| rf_maxfeat_half / rf_ticket_size | ±0.004 | **+0.0100** / +0.0041 | ✗✗ |
| gb_baseline | ±0.006 | +0.0017 | ✓ |
| vote_base | ±0.006 | **+0.0152** | ✗ |
| logit_rich | −0.012…0 | +0.0064 | ✗ |
| hgb_group | 0…+0.012 | +0.0070 | ✓ |
| svc_rich_wcg / logit_rich_wcg | 0…+0.012 | +0.0176 / +0.0158 | ✗✗ |

**Every one of the nine misses is a challenger doing better than predicted.**
Not one went the other way. That pattern is the finding. One explanation is
that I underrated the field. The other is that the champion was weak *on these
folds*. The same configuration scored 0.8360 on exp-006's rows and 0.8131 here,
and `rule_gender` losing only 0.033 to it is consistent with that.

That cuts against a claim in `plan.md`. It said the champion's params, tuned on
overlapping rows, "can only flatter it". In expectation that is true, but the
expectation is small. It does not stop a fixed configuration from being poorly
matched to a particular 569-row draw. Paired deltas remove the split drift that
moves *both* arms. They do not remove the interaction between a fixed config
and a split. So a delta_vs_champion carries some of the champion's own luck on
that draw. Recorded here; not yet measured.

## What was learned

1. **The woman-child-group rate is the lever.** It works as a single change
   (+0.0182, 4.4σ). The WCG hand rule, with no learning at all, ties the best
   learned single models. Most of what models add over the champion on this
   data is that one signal.
2. **The winner's curse is real and measured.** The selection delta was +0.0264;
   on fresh fold seeds it was +0.0163, a regression of 0.010.
3. **The nested procedure picked a different arm in every outer fold.** 4 of 5
   picks were WCG arms. The family is established; the best member of it is not.
4. **Leak controls held.** Permuted labels scored at chance, `make leakage` and
   `make audit` are clean, and the val read is inside its band.

## Next

- Decompose `rf_kitchen` against the new champion, each change as its own
  paired arm. Expect max_features and noise to explain most of its +0.008 over
  `rf_wcg`.
- Measure the config × split interaction above. Re-score the *previous*
  champion beside the new one in the next experiment; the `champion.jsonl`
  succession makes that free.
- Route the champion into `ensemble.py` so it can produce a submission. The
  group rates for test passengers come from their labelled companions.
