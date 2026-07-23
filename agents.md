# AGENTS.md — NVMe 2.0 Knowledge Wiki Schema

This file is the schema layer for this wiki, per the LLM-Wiki pattern (raw sources →
wiki → schema). Any agent operating on this repo — ingesting, answering queries,
or linting — must read this file first and follow it exactly. If a convention here
is ambiguous or missing for a case you hit, stop and ask rather than improvising,
then propose an addition to this file so the next run doesn't hit the same gap.

Note: version history for a concept is captured **inline, on its single canonical
page** (a "Version History" table), not as separate version-scoped files. See
§6 for why and how.

---

## 1. Layers

- **`raw/`** — the source NVMe 2.0 spec, already parsed to JSON. Immutable. Never
  edited by the agent. This is ground truth for verification.
- **wiki (everything outside `raw/` and `old_data/`)** — LLM-authored and
  LLM-maintained markdown, OKF-conformant. This is what agents and humans read.
- **`old_data/`** — archived snapshots of wiki pages taken immediately before a
  substantive overwrite. Not read during normal ingest/query. See §8.
- **This file (`AGENTS.md`)** — the schema. Edited deliberately and rarely, at
  higher review scrutiny than any single page edit.

---

## 2. Directory taxonomy

Concepts are filed by **what they are**, not by where they appear in the spec.
Two to three levels deep, maximum. If a folder exceeds ~30–40 files, split by a
natural sub-category (e.g. `commands/admin/`, `commands/io/`) — don't
pre-split speculatively.

```
nvme-wiki/
├── index.md
├── log.md
├── AGENTS.md
├── old_data/                      # mirrors wiki/ structure, timestamped snapshots
├── commands/
│   ├── index.md
│   ├── admin/
│   │   ├── index.md
│   │   └── <command-name>.md
│   └── io/
│       ├── index.md
│       └── <command-name>.md
├── data-structures/
│   ├── index.md
│   └── <structure-name>.md
├── registers/
│   ├── index.md
│   └── <register-name>.md
├── features/
│   ├── index.md
│   └── <feature-name>.md
├── error-codes/
│   ├── index.md
│   └── <status-code-name>.md
└── concepts/                      # cross-cutting ideas that don't fit above
    ├── index.md
    └── <concept-name>.md
```

If ingest surfaces a category that doesn't fit this taxonomy, propose a new
top-level folder here rather than shoving it into `concepts/` by default —
`concepts/` is for genuinely cross-cutting material, not a junk drawer.

---

## 3. Naming

- Filenames: lowercase, hyphenated, keyed by the **canonical entity name**, never
  by spec section/locator. `identify.md`, not `section-5-15.md`.
- One concept = one file. If a source passage covers three distinct entities,
  it produces three page edits/creations, not one.
- Names must be stable once assigned — renaming breaks every inbound link.
  If the spec renames something across versions, keep the original canonical
  name as the filename and note the rename in that page's Version History.

---

## 4. Frontmatter (OKF-conformant)

Every page requires at minimum:

```yaml
---
type: command | data-structure | register | feature | error-code | concept
title: Identify
description: One-line summary of what this is.
resource: raw/<pointer-to-source-json-node-or-section>
tags: [admin-command, discovery]
timestamp: 2026-07-22T10:00:00Z   # last substantive edit, not last touch
status: unverified | verified      # see §7
---
```

`resource` should point back into `raw/` (e.g. a JSON path or section id) so any
claim on the page is traceable to source.

---

## 5. Page body template

```markdown
# <Title>

<One-paragraph summary.>

## Details
<Main body — fields, behavior, semantics, as currently valid (latest version).>

## Related
- [Linked concept](/data-structures/foo.md)
- [Linked concept](/commands/admin/bar.md)

## Version History
| Version | Change |
|---------|--------|
| 1.0 | Introduced. |
| 1.4 | Extended with X. |
| 2.0 | Field Y deprecated — use Z instead. |
```

---

## 6. Versioning: single page, inline history (decided)

Every entity gets **one canonical page**, ever. It always describes current
(latest-version) behavior in the body, with deprecation/change notes inline
where relevant (`⚠ deprecated in 2.0` etc.), and a **Version History** table
at the bottom logging what changed and when.

Do not create `identify-v1.md`, `identify-v2.md`, etc. Reasons (for the record,
so this doesn't get re-litigated mid-project):
- Links from other pages to this entity stay valid forever — they never need
  to know which version to point at.
- "What changed between 1.4 and 2.0" is answered by reading one page, not by
  diffing files at query time.
- Most of an entity is unchanged across versions; version-scoping would
  duplicate all of that repeatedly.

---

## 7. Verification status

Every page carries `status: unverified` or `status: verified` in frontmatter.

- Pages land as `unverified` on first ingest.
- A page may only be promoted to `verified` after its claims are checked
  directly against the `raw/` spec JSON, paragraph/field by paragraph/field.
- An `unverified` page must not be cited as grounding for another page's
  claims. Treat this the same way an auditor treats circular evidence:
  unverified-citing-unverified is not allowed.
- Note contradictions or unresolved conflicts in `concepts/open-questions.md`
  (create it if it doesn't exist) rather than silently picking one version.

---

## 8. Editing existing pages: snapshot to `old_data/`

Before any **substantive** overwrite of an existing page:
1. Copy the current file, unmodified, to
   `old_data/<same-relative-path>__<ISO-timestamp>.md`.
2. Then edit the live page in place.

"Substantive" = a change to a fact, field, behavior, or the Version History
table. Pure formatting/typo/lint fixes do not require a snapshot — skip it for
those, or `old_data/` fills up with noise.

---

## 9. New page vs. edit-in-place

- **New page**: the source describes a distinct entity — a command, structure,
  register, feature, or error code — that doesn't already have a page under
  the taxonomy in §2.
- **Edit-in-place**: the source adds an attribute, clarification, or version
  delta to something that already has a page.
- Before creating a page, check that directory's `index.md` first (and grep
  the folder) to make sure it doesn't already exist under a slightly different
  name. If in doubt, prefer editing an existing near-match and note the
  alternate name, rather than creating a near-duplicate.

---

## 10. Cross-references

Use normal markdown relative links between pages. Folder placement gives a
concept one primary home; links carry the actual many-to-many relationships
(a command references several data structures; a data structure is used by
several commands). Don't try to encode relationships through folder nesting.

---

## 11. `index.md` (per directory, including root)

Each directory's `index.md` is a flat catalog of that directory's pages:

```markdown
# Admin Commands
- [Identify](identify.md) — retrieves controller/namespace identification data.
- [Create I/O Submission Queue](create-io-submission-queue.md) — ...
```

Updated on every ingest that adds or renames a page in that directory. The
root `index.md` links out to each category's own index rather than listing
every page directly.

---

## 12. `log.md`

Append-only. One entry per ingest/query/lint action, consistent prefix so it's
`grep`-able:

```markdown
## [2026-07-22] ingest | NVMe 2.0 spec §5 Admin Commands
Pages touched: commands/admin/identify.md (new), data-structures/identify-controller-data.md (new)
```

---

## 13. Workflows

**Ingest**: read source chunk from `raw/` → identify distinct entities →
for each, decide new-page vs. edit (§9) → snapshot if editing (§8) → write/update
page with frontmatter (§4), body (§5), Version History (§6) → update relevant
`index.md` files → append `log.md` entry → mark `status: unverified`.

**Query**: read root `index.md` → drill into relevant category index →
read the specific page(s) → answer with citations back to `resource:` paths.
Do not cite `unverified` pages as grounding for a new claim on another page.

**Lint** (run periodically): check for — contradictions between pages,
orphan pages with no inbound links, pages referenced but not yet created,
stale `unverified` pages that should be checked against `raw/`, missing
cross-references. Log findings to `concepts/open-questions.md`.
