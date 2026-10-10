# conversations/

Transcripts of the sessions that built this repo, kept because the project's
subject *is* the agentic process. The commits record what changed; these record
why, what was tried and abandoned, and which mistakes were made on the way.

Chronological, named `NNN-YYYYMMDD-topic.md`.

## Redact before committing

**This repository is public.** A Claude Code export is a verbatim log, and
everything in it ships the moment it is pushed — including things that were
incidental to the work and never meant for an audience.

Run the redactor on every export before it goes in:

```bash
python scripts/redact.py ~/Downloads/<export>.txt -o conversations/NNN-topic.md
```

It replaces email addresses, credential-shaped strings and a configured list of
private terms with `[redacted]`, then prints a summary of what it changed and
refuses to write a file it could not fully clean.

Categories it handles:

- **Email addresses** — personal and work.
- **Employer and account names** — anything in `scripts/redact_terms.txt`, which
  is gitignored precisely so the list of things being hidden is not itself
  published.
- **Credential-shaped strings** — API token prefixes, bearer tokens, keys. The
  sessions discuss auth *mechanisms* (file paths, environment variable names),
  which is fine and useful; no token value was ever printed, and that should
  stay true.
- **Whole passages** — a conversation can be private without containing a single
  sensitive string. Employment, health, finances, opinions about named people.
  The redactor cannot find these. A human has to.

## The rule that matters

**Redaction is honest. Quiet rewriting is not.**

The difference is whether a reader can tell something was removed. `[redacted]`
and *"a passage about an unrelated personal matter was cut"* both preserve the
record's integrity. Silently editing a transcript so it reads better, or so past
reasoning looks sounder than it was, destroys the only thing these files are for.

If a cut changes what a session appears to have concluded, say so at the cut.

## What makes these worth keeping

The cold-start test in `docs/cold-start-001-result.md` found three defects
invisible to the session that wrote the code. The transcripts are the record of
*why* they were invisible — which decisions were made quickly, which were argued,
and which were never questioned at all.

That is the actual subject of this repo, and it is not recoverable from the diffs.
