"""
tools/pattern_cache.py

SQLite-backed cache that maps workbook structural fingerprints to validated
parser scripts. Replaces the previous Redis backend — no external server
needed, the DB is a single file on disk that persists automatically across
runs.

DB location: xlsx2json_agent/output/pattern_cache.db
             (created automatically on first use)

Data model
──────────
Table: patterns
  fp_hash      TEXT PRIMARY KEY   — 16-char sha256 of the structural fingerprint
  fingerprint  TEXT               — JSON of the human-readable fingerprint
  created_at   TEXT               — ISO timestamp of first insertion

Table: buckets
  id            TEXT PRIMARY KEY   — uuid4 short hex
  fp_hash       TEXT               — FK → patterns.fp_hash
  script        TEXT               — full Python source of the parser
  grammar_spec  TEXT               — JSON of the grammar spec
  validated     INTEGER            — 1 if validate_json passed, else 0
  success_count INTEGER            — how many files it has worked on
  fail_count    INTEGER            — how many times it failed validation
  created_at    TEXT               — ISO timestamp

Fingerprint
───────────
Derived from the raw inspect_excel output (before any LLM call) so we can
skip the schema sniffer entirely on a cache hit. Covers:
  - sorted sheet names
  - normalised header cell values (rows 1–2, stripped lowercase)
  - column count per sheet
  - unique row-type values seen in the first 4 columns (Group/Reg/…)
  - unique fill colours seen
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# DB lives next to the output directory, persists between runs
_DB_PATH = Path(__file__).resolve().parent.parent / "output" / "pattern_cache.db"


# ---------------------------------------------------------------------------
# DB bootstrap
# ---------------------------------------------------------------------------

def _db() -> sqlite3.Connection:
    """Open (and if needed create) the SQLite database."""
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(_DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")   # safe concurrent reads
    con.execute("PRAGMA foreign_keys=ON")
    con.executescript("""
        CREATE TABLE IF NOT EXISTS patterns (
            fp_hash     TEXT PRIMARY KEY,
            fingerprint TEXT NOT NULL,
            created_at  TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS buckets (
            id            TEXT PRIMARY KEY,
            fp_hash       TEXT NOT NULL REFERENCES patterns(fp_hash),
            script        TEXT NOT NULL,
            grammar_spec  TEXT NOT NULL DEFAULT '{}',
            validated     INTEGER NOT NULL DEFAULT 0,
            success_count INTEGER NOT NULL DEFAULT 0,
            fail_count    INTEGER NOT NULL DEFAULT 0,
            created_at    TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_buckets_fp ON buckets(fp_hash);
    """)
    con.commit()
    return con


# ---------------------------------------------------------------------------
# Fingerprinting — pure Python, no LLM
# ---------------------------------------------------------------------------

def _normalise(val: Any) -> str:
    if val is None:
        return ""
    return re.sub(r"\s+", " ", str(val).strip().lower())


def build_fingerprint(inspection_json: str) -> dict:
    """Derive a structural fingerprint from inspect_excel output.

    Returns a dict with human-readable keys and a `hash` field (16-char hex)
    used as the primary key in the DB.
    """
    data = json.loads(inspection_json) if isinstance(inspection_json, str) else inspection_json
    sheets = data.get("sheets", {})

    fp: dict[str, Any] = {
        "sheet_names": sorted(sheets.keys()),
        "sheets": {},
    }

    all_row_type_values: set[str] = set()
    all_fill_colours:    set[str] = set()

    for sheet_name, sheet in sheets.items():
        rows      = sheet.get("rows", [])
        col_count = sheet.get("columns_sampled", 0)

        header_vals: list[str] = []
        for row in rows[:2]:
            for v in (row.get("values") or []):
                n = _normalise(v)
                if n:
                    header_vals.append(n)

        for row in rows:
            vals = row.get("values") or []
            for v in vals[:4]:
                n = _normalise(v)
                if n and len(n) < 20:
                    all_row_type_values.add(n)
            fill = row.get("fill")
            if fill:
                all_fill_colours.add(fill)

        fp["sheets"][sheet_name] = {
            "col_count": col_count,
            "header_signature": sorted(set(header_vals)),
        }

    fp["row_type_values"] = sorted(all_row_type_values)
    fp["fill_colours"]    = sorted(all_fill_colours)

    canonical  = json.dumps(fp, sort_keys=True, ensure_ascii=False)
    fp["hash"] = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    return fp


# ---------------------------------------------------------------------------
# Bucket operations
# ---------------------------------------------------------------------------

def get_buckets(fp_hash: str) -> list[dict]:
    """Return all cached buckets for a fingerprint, best first
    (highest success_count, then newest)."""
    try:
        con = _db()
        rows = con.execute(
            """
            SELECT id, script, grammar_spec, validated,
                   success_count, fail_count, created_at
            FROM   buckets
            WHERE  fp_hash = ?
            ORDER  BY success_count DESC, created_at DESC
            """,
            (fp_hash,),
        ).fetchall()
        con.close()
        return [
            {
                "id":            r["id"],
                "script":        r["script"],
                "grammar_spec":  json.loads(r["grammar_spec"] or "{}"),
                "validated":     bool(r["validated"]),
                "success_count": r["success_count"],
                "fail_count":    r["fail_count"],
                "created_at":    r["created_at"],
            }
            for r in rows
        ]
    except sqlite3.Error as e:
        print(f"[cache] get_buckets error: {e}")
        return []


def push_bucket(
    fp_hash: str,
    script: str,
    grammar_spec: dict,
    validated: bool,
) -> str:
    """Insert a new bucket. Upserts the parent pattern row if needed."""
    bucket_id = uuid.uuid4().hex[:8]
    now       = datetime.now(timezone.utc).isoformat()
    try:
        con = _db()
        # Ensure parent pattern row exists
        con.execute(
            "INSERT OR IGNORE INTO patterns(fp_hash, fingerprint, created_at) VALUES (?,?,?)",
            (fp_hash, "{}", now),
        )
        con.execute(
            """
            INSERT INTO buckets
                (id, fp_hash, script, grammar_spec, validated,
                 success_count, fail_count, created_at)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            (
                bucket_id,
                fp_hash,
                script,
                json.dumps(grammar_spec, ensure_ascii=False),
                int(validated),
                1 if validated else 0,
                0 if validated else 1,
                now,
            ),
        )
        con.commit()
        con.close()
        print(f"[cache] stored new bucket {bucket_id} "
              f"under pattern {fp_hash} (validated={validated})")
    except sqlite3.Error as e:
        print(f"[cache] push_bucket error: {e}")
    return bucket_id


def mark_bucket_success(fp_hash: str, bucket_id: str) -> None:
    _update_counters(bucket_id, success_delta=1, fail_delta=0)


def mark_bucket_failure(fp_hash: str, bucket_id: str) -> None:
    _update_counters(bucket_id, success_delta=0, fail_delta=1)


def _update_counters(bucket_id: str, success_delta: int, fail_delta: int) -> None:
    try:
        con = _db()
        con.execute(
            """
            UPDATE buckets
            SET    success_count = success_count + ?,
                   fail_count    = fail_count    + ?,
                   validated     = CASE WHEN success_count + ? > 0 THEN 1 ELSE 0 END
            WHERE  id = ?
            """,
            (success_delta, fail_delta, success_delta, bucket_id),
        )
        con.commit()
        con.close()
    except sqlite3.Error as e:
        print(f"[cache] update_counters error: {e}")


# ---------------------------------------------------------------------------
# Pattern table pretty-print
# ---------------------------------------------------------------------------

def print_pattern_table() -> None:
    """Print a summary table of all stored patterns and their buckets."""
    try:
        con = _db()
        rows = con.execute(
            """
            SELECT p.fp_hash,
                   COUNT(b.id)           AS bucket_count,
                   MAX(b.success_count)  AS best_success,
                   MIN(b.created_at)     AS first_seen,
                   GROUP_CONCAT(b.grammar_spec, '||') AS specs
            FROM   patterns p
            LEFT   JOIN buckets b ON b.fp_hash = p.fp_hash
            GROUP  BY p.fp_hash
            ORDER  BY first_seen DESC
            """
        ).fetchall()
        con.close()
    except sqlite3.Error as e:
        print(f"[cache] cannot open DB at {_DB_PATH}: {e}")
        return

    if not rows:
        print(f"No patterns stored yet.  (DB: {_DB_PATH})")
        return

    print(f"\n  DB: {_DB_PATH}")
    print(f"\n{'─'*95}")
    print(f"  {'PATTERN HASH':<18} {'BUCKETS':<8} {'BEST SUCCESS':<14} {'SHEETS':<32} {'FIRST SEEN'}")
    print(f"{'─'*95}")
    for r in rows:
        sheets = ""
        if r["specs"]:
            first_spec_raw = r["specs"].split("||")[0]
            try:
                spec = json.loads(first_spec_raw)
                sheets = ", ".join(spec.get("sheets_to_parse", []))[:30]
            except (json.JSONDecodeError, AttributeError):
                pass
        print(f"  {r['fp_hash']:<18} {r['bucket_count'] or 0:<8} "
              f"{r['best_success'] or 0:<14} {sheets:<32} {(r['first_seen'] or '')[:10]}")
    print(f"{'─'*95}\n")


if __name__ == "__main__":
    print_pattern_table()
