"""
tools/pattern_cache.py

Redis-backed cache that maps workbook structural fingerprints to validated
parser scripts. One fingerprint key can hold multiple script "buckets" —
useful when the same sheet layout has had several working parsers over time
(e.g. after a format tweak that broke the first script).

Redis is expected at the port configured in REDIS_PORT (default 9834).

Data model
──────────
key   : "xlsx2json:pattern:{sha256[:16]}"
value : JSON-encoded list of bucket dicts, newest last:
  {
    "id":            str,          # uuid4 short
    "script":        str,          # full python source of the parser
    "grammar_spec":  dict,         # grammar spec from schema sniffer
    "validated":     bool,         # did validate_json return ok=True?
    "success_count": int,          # how many files it's worked on since stored
    "fail_count":    int,          # how many times it failed validation
    "created_at":    str           # ISO timestamp
  }

Fingerprint
───────────
Derived from the *raw inspect_excel output* (before any LLM call) so we
can skip the schema sniffer entirely on a cache hit. Covers:
  - sorted sheet names
  - normalised header cell values (row 1–2, stripped lowercase)
  - column count per sheet
  - unique row-type values seen (Group/Reg/…)
  - unique fill colours seen
"""
from __future__ import annotations

import hashlib
import json
import uuid
import re
from datetime import datetime, timezone
from typing import Any

import redis

REDIS_HOST = "localhost"
REDIS_PORT = 9834
REDIS_DB   = 0
KEY_PREFIX = "xlsx2json:pattern"
KEY_TTL    = None   # no expiry by default; set to seconds if you want auto-eviction


def _redis() -> redis.Redis:
    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        decode_responses=True,
        socket_connect_timeout=3,
    )


# ---------------------------------------------------------------------------
# Fingerprinting
# ---------------------------------------------------------------------------

def _normalise(val: Any) -> str:
    """Normalise a cell value to a stable lowercase string."""
    if val is None:
        return ""
    return re.sub(r"\s+", " ", str(val).strip().lower())


def build_fingerprint(inspection_json: str) -> dict:
    """Derive a structural fingerprint from inspect_excel output.

    Returns a dict with human-readable keys (for display/logging) and a
    `hash` field containing the 16-char hex digest used as the Redis key.
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
        rows = sheet.get("rows", [])
        col_count = sheet.get("columns_sampled", 0)

        # Collect header values from first 2 rows
        header_vals: list[str] = []
        for row in rows[:2]:
            for v in (row.get("values") or []):
                n = _normalise(v)
                if n:
                    header_vals.append(n)

        # Collect unique row-type-looking values (short strings in early columns)
        for row in rows:
            vals = row.get("values") or []
            for v in vals[:4]:   # row-type column is almost always within first 4 cols
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

    # Stable hash
    canonical = json.dumps(fp, sort_keys=True, ensure_ascii=False)
    fp["hash"] = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    return fp


def _redis_key(fp_hash: str) -> str:
    return f"{KEY_PREFIX}:{fp_hash}"


# ---------------------------------------------------------------------------
# Bucket operations
# ---------------------------------------------------------------------------

def get_buckets(fp_hash: str) -> list[dict]:
    """Return all cached buckets for a fingerprint hash, best first
    (sorted by success_count desc, then created_at desc)."""
    try:
        r = _redis()
        raw = r.get(_redis_key(fp_hash))
        if not raw:
            return []
        buckets = json.loads(raw)
        return sorted(buckets, key=lambda b: (-b.get("success_count", 0), b.get("created_at", "")), reverse=False)
    except (redis.RedisError, json.JSONDecodeError) as e:
        print(f"[cache] get_buckets error: {e}")
        return []


def push_bucket(
    fp_hash: str,
    script: str,
    grammar_spec: dict,
    validated: bool,
) -> str:
    """Add a new bucket under a fingerprint key. Returns the new bucket id."""
    bucket_id = uuid.uuid4().hex[:8]
    new_bucket = {
        "id":            bucket_id,
        "script":        script,
        "grammar_spec":  grammar_spec,
        "validated":     validated,
        "success_count": 1 if validated else 0,
        "fail_count":    0 if validated else 1,
        "created_at":    datetime.now(timezone.utc).isoformat(),
    }
    try:
        r = _redis()
        key = _redis_key(fp_hash)
        raw = r.get(key)
        buckets = json.loads(raw) if raw else []
        buckets.append(new_bucket)
        r.set(key, json.dumps(buckets, ensure_ascii=False))
        if KEY_TTL:
            r.expire(key, KEY_TTL)
        print(f"[cache] stored new bucket {bucket_id} under pattern {fp_hash} (validated={validated})")
    except redis.RedisError as e:
        print(f"[cache] push_bucket error: {e}")
    return bucket_id


def mark_bucket_success(fp_hash: str, bucket_id: str) -> None:
    """Increment success_count for a bucket after it produces valid output."""
    _update_bucket_counter(fp_hash, bucket_id, success_delta=1, fail_delta=0)


def mark_bucket_failure(fp_hash: str, bucket_id: str) -> None:
    """Increment fail_count when a cached bucket fails validation."""
    _update_bucket_counter(fp_hash, bucket_id, success_delta=0, fail_delta=1)


def _update_bucket_counter(fp_hash: str, bucket_id: str, success_delta: int, fail_delta: int) -> None:
    try:
        r = _redis()
        key = _redis_key(fp_hash)
        raw = r.get(key)
        if not raw:
            return
        buckets = json.loads(raw)
        for b in buckets:
            if b["id"] == bucket_id:
                b["success_count"] = b.get("success_count", 0) + success_delta
                b["fail_count"]    = b.get("fail_count", 0)    + fail_delta
                b["validated"]     = b["success_count"] > 0
                break
        r.set(key, json.dumps(buckets, ensure_ascii=False))
    except redis.RedisError as e:
        print(f"[cache] update_bucket_counter error: {e}")


# ---------------------------------------------------------------------------
# Pattern table pretty-print (for CLI inspection)
# ---------------------------------------------------------------------------

def print_pattern_table() -> None:
    """Print a summary table of all stored patterns and their buckets."""
    try:
        r = _redis()
        keys = r.keys(f"{KEY_PREFIX}:*")
    except redis.RedisError as e:
        print(f"[cache] cannot connect to Redis: {e}")
        return

    if not keys:
        print("No patterns stored yet.")
        return

    print(f"\n{'─'*90}")
    print(f"{'PATTERN HASH':<18} {'BUCKETS':<8} {'BEST SUCCESS':<14} {'SHEETS':<30} {'STORED'}")
    print(f"{'─'*90}")
    for key in sorted(keys):
        raw = r.get(key)
        if not raw:
            continue
        buckets = json.loads(raw)
        fp_hash = key.split(":")[-1]
        best_success = max((b.get("success_count", 0) for b in buckets), default=0)
        # Extract sheet names from first bucket's grammar spec
        sheets = ", ".join(
            (buckets[0].get("grammar_spec") or {}).get("sheets_to_parse", [])
        )[:28]
        created = min(b.get("created_at", "") for b in buckets)[:10]
        print(f"{fp_hash:<18} {len(buckets):<8} {best_success:<14} {sheets:<30} {created}")
    print(f"{'─'*90}\n")


if __name__ == "__main__":
    print_pattern_table()
