# NVMe Base Spec Wiki — Schema (OKF v0.1 conformant)

This wiki is an OKF (Open Knowledge Format) Knowledge Bundle. Every non-reserved
`.md` file is a **Concept** with YAML frontmatter + markdown body. `index.md` and
`log.md` are reserved filenames with the meanings defined in OKF §6 / §7.
Reference: https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md

Paste this whole file into CLAUDE.md / AGENTS.md (or pass it as the system prompt,
see `run_ingest.py`) for the agent operating this bundle.

## Bundle root

```
wiki/
  index.md                  # okf_version: "0.1" declared here (root index.md
                             # is the only index.md allowed frontmatter)
  log.md

  architecture/
  commands/
    admin/
    fabrics/
    io/
  log-pages/
  data-structures/
    identify/
    other/
  status-codes/
  features/
  extended-capabilities/
  concepts/
    glossary.md
```

Max depth: 2 levels under `wiki/`. This is a house convention on top of OKF —
OKF itself doesn't mandate any particular directory shape (§3: "producers
organize concepts however makes sense").

## 1. Concept `type` values (frontmatter, REQUIRED per OKF §4.1)

OKF does not register types centrally — producers define their own. For this
bundle, use exactly one of:

| `type` | Used for | Lives under |
|---|---|---|
| `Command` | An NVMe command (opcode, Dword layout, behavior) | `commands/<set>/` |
| `Log Page` | A Get Log Page log type | `log-pages/` |
| `Data Structure` | Identify data structure or other struct (PRP, SGL, etc.) | `data-structures/` |
| `Status Code Table` | Grouped status/error value table | `status-codes/` |
| `Feature` | A Get/Set Features identifier | `features/` |
| `Extended Capability` | A §8.x optional capability area | `extended-capabilities/` |
| `Architecture Concept` | Theory of operation, queueing model, controller types | `architecture/` |

Glossary terms (§1.5 short definitions) do NOT get their own `type` or file —
they're entries inside `concepts/glossary.md` (see rule 9 below), consistent
with OKF's guidance that tags/small terms need not be separate concepts.

Consumers (including this agent, on future sessions) MUST tolerate unknown
`type` values gracefully per OKF §4.1 — if a new type is genuinely needed,
add it to this table rather than filing under an existing type that doesn't fit.

## 2. Frontmatter template

```yaml
---
type: Command                        # REQUIRED — one of the table above
title: Abort command                 # recommended
description: Aborts a previously submitted command by CID/SQID.  # one sentence
resource: null                       # no canonical URI for spec concepts; omit or null
tags: [admin-command, abort]         # optional, cross-cutting
timestamp: 2026-07-26T00:00:00Z      # ISO 8601, last meaningful edit

# Producer-defined extension keys (OKF §4.1 explicitly permits these):
source_sections: ["5.1", "5.1.1"]    # exact spec section numbers covered
source_pages: [158, 159]             # PDF page numbers
spec: base                           # base / nvme-of / nvme-mi / ...
spec_revision: "2.0a"
review_status: current               # current | superseded | contradicted | needs-review

# Structured correlation layer (house extension, NOT part of OKF core).
# OKF's own correlation mechanism is just prose links (section 5) — fine for
# a human/LLM reading a page, but not mechanically queryable. relates_to adds
# a parallel, greppable/lintable layer for tooling. It is supplementary: every
# entry here MUST also be reflected as a real prose link in the body somewhere
# (relates_to is not a substitute for writing the link, it's an index of it).
relates_to:
  - path: /extended-capabilities/reservations.md
    kind: used-by            # free text, but keep vocabulary small & reused:
                              # used-by | depends-on | returns-status-from |
                              # is-a | supersedes | conflicts-with
  - path: /status-codes/io-command-specific.md
    kind: returns-status-from
---
```

Only `type` is required (OKF §9 conformance rule). Everything else is
recommended-but-optional or a house extension — never block an ingest because
a field is missing. `relates_to` entries are optional per-page but, when
present, every `path` MUST resolve to a real or forward-referenced concept
(see filing rule 11 below on forward references) — the lint pass checks this
mechanically, without needing an LLM call.

## 3. Body conventions

Per OKF §4.2, these headings have conventional meaning — use them when they apply:

- `# Schema` — structured field/bit layout (e.g. a Command Dword table).
- `# Examples` — worked examples, fenced code blocks.
- `# Citations` — sources backing claims (see §6 below). Always the last
  section in the file.

For this bundle, also use (house convention, not part of OKF core):
- `# Behavior` — prose description of what the command/feature does.
- `# Status Values` — command-specific status codes, cross-linked to
  `status-codes/`.
- `# Revision History` — when a later spec revision changes this concept;
  do not silently overwrite, append here and set `review_status: needs-review`.

## 4. Filing rules (apply mechanically, in order; stop at first match)

1. **Command** (opcode + Command Dword layout) → `commands/<set>/<kebab-name>.md`,
   `<set>` in `{admin, fabrics, io}` matching spec sections 5/6/7.
2. **Get Log Page log type** → `log-pages/<kebab-name>.md`. The Get Log Page
   command page stays short and links out (`[SMART/Health Log](/log-pages/smart-health.md)`).
3. **Identify data structure** → `data-structures/identify/<kebab-name>.md`.
   The Identify command page stays short and links out.
4. **Other data structure** (PRP, SGL, list formats) → `data-structures/other/<kebab-name>.md`.
5. **Status/error value table** → append to the matching file under
   `status-codes/` (one file per table type — Generic, Admin-Specific,
   I/O-Specific, Fabrics-Specific, Media/Data Integrity, Path Related). Never
   one file per individual code.
6. **Get/Set Features identifier** → `features/<kebab-name>.md`.
7. **Extended Capability** (matches a section 8.x header) → `extended-capabilities/<kebab-name>.md`,
   even if the same topic also has a command or log-page entry — cross-link,
   don't duplicate.
8. **Architecture / theory of operation** (sections 2, 3) → `architecture/<kebab-name>.md`.
9. **Short glossary definition** (section 1.5, 1-2 sentences, no tables/diagrams
   of its own) → append as an anchored entry to `concepts/glossary.md`
   (`## term-name`). Do not create a standalone file.
10. **Anything else** → flag to the user; propose an addition to this list.

11. **Forward references are allowed and expected.** If a concept mentions
    something whose page doesn't exist yet (common during a large chunked
    ingest — see the schema's ingest workflow), still write both the prose
    link and the corresponding `relates_to` entry at the path that concept
    *will* have once filed (per rules 1-9 above), even though the file isn't
    there yet. This is valid per OKF §5.3 (a link may point to a
    not-yet-written concept) — do not skip the correlation just because the
    target doesn't exist yet. A later lint/repair pass reconciles these.

## 5. Cross-linking (OKF §5)

Use **bundle-relative absolute links** (leading `/`) — the OKF-recommended form,
stable across moves:

```markdown
See the [Identify Controller data structure](/data-structures/identify/identify-controller.md).
```

A link asserts a relationship; the kind of relationship is conveyed by
surrounding prose, not by the link syntax (OKF §5.3). Broken links are
tolerated, not errors — they may mark not-yet-written concepts.

## 6. Citations (OKF §8)

Every concept sourced from the spec ends with a numbered `# Citations` block:

```markdown
# Citations

[1] NVM Express Base Specification, revision 2.0a, section 5.1, p.158-159.
```

## 7. Index files (OKF §6)

`index.md` files have **no frontmatter** except the bundle-root `index.md`,
which MAY declare `okf_version: "0.1"`. Body format:

```markdown
---
okf_version: "0.1"
---

# Commands - Admin

* [Abort](commands/admin/abort.md) - Aborts a previously submitted command.
* [Get Log Page](commands/admin/get-log-page.md) - Retrieves a log page; see log-pages/.

# Extended Capabilities

* [Reservations](extended-capabilities/reservations.md) - Multi-host access control.
```

Update the relevant `index.md` (root + nearest subdirectory) on every ingest.
Descriptions should mirror the concept's frontmatter `description`.

## 8. Log file (OKF §7)

`log.md` - flat, date-grouped, newest first:

```markdown
# Directory Update Log

## 2026-07-26

* **Creation**: Added [Abort command](/commands/admin/abort.md) and
  [Abort Command Specific Status Values](/status-codes/admin-command-specific.md).
* **Update**: Linked [Reservations](/extended-capabilities/reservations.md) to
  the new [Reservation Acquire](/commands/io/reservation-acquire.md) command page.
```

## 9. Ingest workflow

1. Walk the source (JSON export or PDF page range) section by section.
2. Apply the filing rules (section 4) to each content block.
3. Check the nearest `index.md` for an existing concept before creating a new
   file - update in place rather than duplicating.
4. Write/update the concept file with full frontmatter (section 2) and citations
   (section 6).
5. Update `index.md` (root + subdirectory).
6. Append a dated entry to `log.md`.
7. On contradiction with an existing concept (e.g. a later spec revision):
   do not overwrite - add a `# Revision History` section, set
   `review_status: needs-review`, and surface it to the user.

## 10. Lint checklist

- Every `.md` file (except `index.md`/`log.md`) has a non-empty `type` field
  (OKF §9 conformance minimum).
- No orphan concepts (zero inbound links) under `commands/`, `log-pages/`,
  `data-structures/`, `extended-capabilities/`.
- Cross-cutting topics (Reservations, ANA, Sanitize) have their command,
  extended-capability, and log-page pages all linking each other.
- Any `concepts/glossary.md` entry referenced by 3+ concepts and grown past
  2-3 sentences → candidate to promote to its own file (revise rule 9).
- Any concept with `review_status: needs-review` → resolve or escalate.
- Every `relates_to` entry's `path` resolves to a file that actually exists
  (run mechanically — see `validate_relates_to()` in wiki_agent_deepagents.py;
  this check does NOT require an LLM call).
- Every `relates_to` entry has a matching prose link somewhere in the body
  (relates_to should never exist without the human-readable link it indexes).
- `kind` values stay within the small reused vocabulary above rather than
  drifting into ad hoc synonyms (e.g. don't mix "used-by" and "uses" for the
  same relationship direction across different pages).
