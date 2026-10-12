# Two runners, one surface

`v1-claude-code` is the tag for the single-runner state. The next runner is a
Claude Agent SDK script, then a raw Messages API tool-use loop. They live in
`agent/` on a branch and drive **this** repo — the same `make` targets, the same
frozen dataset, the same append-only ledger.

Not a fork. The toolbox is the shared thing; two copies of a hash-chained
ledger is how two records of the same events start to disagree.

## The contract

A runner may do exactly four things.

1. **Load the policy** from `titanic_agent.surface.policy_path()` and send it as
   the system prompt. Do not copy it.
2. **Send `titanic_agent.runner.PROMPT`.** Do not write your own. Two runners
   with two prompts is a comparison of prompts.
3. **Run make targets**, and nothing else. One tool, built from
   `surface.tool_schema()`; no shell tool. Call `surface.check(target,
   variables)` before invoking anything.
4. **Record the pass** with `runner.start()` and `runner.finish()`, filling a
   `runner.PassRecord`.

What a runner must not do: read the pass's result and act on it. Nothing
retries, nothing re-runs a pass whose verdict it disliked. `PassRecord` has no
verdict field for that reason, and a test keeps it that way.

## Why the enforcement moved into the package

| constraint | was | Agent SDK | raw API loop |
|---|---|---|---|
| allowed commands | `.claude/settings.json` | inherits | **nothing** |
| interpreter lockdown | `.claude/agent-settings.json` | inherits | **nothing** |
| Rule 1 — frozen data | the `PreToolUse` hook | inherits | **nothing** |
| Rule 2 — the final test | `.claude/settings.json` deny | inherits | **nothing** |

The Agent SDK is Claude Code as a library, so it inherits all four. A raw loop
inherits none — including Rule 2, the one measurement that cannot be taken
twice. So the decisions live in `titanic_agent.surface` and
`titanic_agent.guard`, and Claude Code's copies are *asserted against* them
rather than being the only copy.

Two notes for the SDK runner specifically:

- **Pass `setting_sources=["project"]` explicitly.** The SDK does load
  filesystem settings by default, so `.claude/settings.json` and the hook come
  along — but that is a default, and an SDK upgrade can change a default.
  Declaring it is the difference between inheriting Rule 1 and happening to.
- **Gate with a `PreToolUse` hook, not `can_use_tool`.** That callback fires
  only when the permission flow reaches a prompt, and `make agent` runs
  `acceptEdits` with a large allowlist, so almost nothing reaches one. A hook
  sees every call — the same mechanism `protect_frozen.py` already uses, which
  is why `guard.py` is importable.

## Why a target-name allowlist was not enough

`make champion PYTHON='echo PWNED'` used to run the caller's command instead of
the interpreter: a command-line assignment beats a Makefile `:=`, so every
allowlisted target was arbitrary execution. `Bash(make tune:*)` is a prefix
match, and the deny entries are literal substrings a glob walks past.

Closed in the Makefile — `override` on the interpreter paths, `$(filter)` on
enums, a pure-make numeric guard, and the environment for `NAME` and `SUMMARY`
— because that layer protects a human at a terminal too. `surface.check` is the
second layer: it refuses *and explains*, before make is invoked.

For a raw loop the right answer is stronger than either: give it
`surface.tool_schema()` and no shell. Then a compound command, `make -f`,
`make -C`, a `$(...)` expansion and `SHELL=` are things the model cannot
express, so there is nothing to filter.

## Confounds the shared surface does not remove

Worth writing down before any comparison is run, because each of these makes
"runner A did better" partly a statement about harnesses:

- **Context management.** Claude Code and the SDK compact; a raw loop does not,
  unless you build it. A pass that runs out of context is not a pass that
  reasoned worse.
- **Built-in tools.** The SDK ships Read/Write/Edit/Grep/Glob. A raw loop has
  whatever you define. The policy requires writing `plan.md` *before* scoring,
  so a raw loop needs a write tool — and the moment it has one,
  `guard.write_refusal` is the only thing standing between it and the ledger.
- **`CLAUDE.md` injection.** Automatic in Claude Code and the SDK; manual in a
  raw loop. Forgetting it is a different experiment, not a worse runner.
- **Skills, subagents, session resume.** Claude Code only.
- **Model and effort.** Pin both identically. An unpinned model makes every run
  incomparable to every other, including to itself a week later.

## What is still missing

- `pass-state`: a machine-readable snapshot for a runner to score a pass with.
  Deferred — exit codes already carry the two preflight gates (`make audit` and
  `make verify` return non-zero on a binding failure), and the honest shape is
  snapshot-before-and-after, never branch.
- A comparison table over `runner.read_records()`. Build it when there are two
  runners' records to put in it.
- `tests/test_permissions.py` hardcodes the Claude Code CLI's argv shape. That
  generalises when a second runner exists, not before.
