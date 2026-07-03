"""
tools/resolve_address.py

Resolves dynamic register address expressions such as:
    0x018 + 0x20 * N
    0x100 + N * 4
    BASE + 0x10*N

Strategy:
  1. Parse and normalise the expression to identify the variable (default N).
  2. Search the register description (and optionally the group description) for
     a numeric value or range that pins N  (e.g. "N = 0 to 3", "N=0,1,2,3",
     "for N in range 0..7", etc.)
  3. If found, evaluate the expression for every value of N and return the
     expanded list.
  4. If NOT found, print the expression to stdout and prompt the user
     interactively for a value or range ("0", "0-3", "0,1,2,3").
  5. Return a JSON object the agent can read:
       {
         "expression": "0x018+0x20*N",
         "variable": "N",
         "n_values": [0, 1, 2, 3],
         "n_source": "description" | "user",
         "addresses": ["0x018", "0x038", "0x058", "0x078"]
       }
"""
from __future__ import annotations
import json
import re


# ---------------------------------------------------------------------------
# Patterns that suggest "N = <something>" in natural language descriptions
# ---------------------------------------------------------------------------
_N_RANGE_PATTERNS = [
    # "N = 0 to 3", "N=0 to 7", "N = 0...7"
    r"[Nn]\s*[=:]\s*(\d+)\s*(?:to|\.\.\.?|~|–|-)\s*(\d+)",
    # "N = 0, 1, 2, 3"  (explicit list)
    r"[Nn]\s*[=:]\s*((?:\d+\s*,\s*)+\d+)",
    # "for N in {0,1,2,3}"
    r"for\s+[Nn]\s+in\s+\{((?:\d+\s*,\s*)+\d+)\}",
    # "where N is 0 to 3"
    r"where\s+[Nn]\s+is\s+(\d+)\s*(?:to|\.\.\.?|~|–|-)\s*(\d+)",
    # "N ranges from 0 to 3"
    r"[Nn]\s+ranges?\s+from\s+(\d+)\s+to\s+(\d+)",
]

_N_SINGLE_PATTERNS = [
    # "N = 2"
    r"[Nn]\s*[=:]\s*(\d+)\b",
    # "where N is 2"
    r"where\s+[Nn]\s+is\s+(\d+)\b",
]


def _find_n_in_text(text: str) -> list[int] | None:
    """Try to extract N values from a description string.
    Returns a list of ints or None if nothing could be parsed."""
    if not text:
        return None

    for pat in _N_RANGE_PATTERNS:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            groups = m.groups()
            # list pattern: "0, 1, 2, 3" -> split on commas
            if len(groups) == 1:
                vals = [int(v.strip()) for v in groups[0].split(",")]
                return sorted(set(vals))
            # range pattern: (start, end)
            start, end = int(groups[0]), int(groups[1])
            return list(range(start, end + 1))

    for pat in _N_SINGLE_PATTERNS:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return [int(m.group(1))]

    return None


def _parse_user_input(raw: str) -> list[int]:
    """Parse user-supplied N spec: '3', '0-3', '0,1,2,3'."""
    raw = raw.strip()
    # range: "0-3" or "0..3"
    range_m = re.match(r"^(\d+)\s*(?:-|\.\.\.?)\s*(\d+)$", raw)
    if range_m:
        return list(range(int(range_m.group(1)), int(range_m.group(2)) + 1))
    # comma list: "0,1,2,3"
    if "," in raw:
        return sorted(int(v.strip()) for v in raw.split(","))
    # single value
    return [int(raw)]


def _evaluate_expression(expr: str, variable: str, value: int) -> str:
    """Evaluate the address expression for a given variable value.
    Returns a '0x...' hex string."""
    # Replace the variable (case-insensitive, word boundary) with its int value
    filled = re.sub(rf"\b{re.escape(variable)}\b", str(value), expr, flags=re.IGNORECASE)
    # Evaluate — only allow safe characters: digits, hex prefix, +, -, *, /, (, ), spaces
    if re.search(r"[^0-9a-fA-FxX+\-*/() \t]", filled):
        raise ValueError(f"Unsafe characters in address expression after substitution: '{filled}'")
    result = eval(filled)  # noqa: S307 — sanitised above
    return hex(result)


def resolve_address(expression: str, description: str = "", variable: str = "N") -> str:
    """Resolve a dynamic register address expression that contains a variable
    (default 'N') into a list of concrete hex addresses.

    Steps:
      1. Try to find the variable's value or range in `description`.
      2. If not found, prompt the user interactively on the CLI.
      3. Evaluate the expression for each value and return a JSON object:
         {expression, variable, n_values, n_source, addresses}

    expression: e.g. "0x018+0x20*N" or "0x100+N*4"
    description: the register or group description text to search for N's value.
    variable: the variable name to substitute (default "N").
    """
    n_values = _find_n_in_text(description)
    n_source = "description"

    if n_values is None:
        print(f"\n[resolve_address] Dynamic address detected: {expression}")
        print(f"  Variable '{variable}' not found in description: {description!r}")
        print(f"  Enter a value or range for '{variable}'")
        print("  Examples: 0   |   0-3   |   0,1,2,3")
        raw = input(f"  {variable} = ").strip()
        n_values = _parse_user_input(raw)
        n_source = "user"

    addresses = [_evaluate_expression(expression, variable, v) for v in n_values]

    # Zero-padded suffixes: width driven by the largest N value so they sort correctly
    pad = len(str(max(n_values)))
    pad = max(pad, 2)  # minimum 2 digits → _00, _01, …
    suffixes = [f"_{v:0{pad}d}" for v in n_values]

    return json.dumps({
        "expression": expression,
        "variable": variable,
        "n_values": n_values,
        "n_source": n_source,
        "addresses": addresses,
        "name_suffixes": suffixes,   # e.g. ["_00", "_01", "_02", "_03"]
    })


if __name__ == "__main__":
    # Quick smoke test
    import sys
    expr = sys.argv[1] if len(sys.argv) > 1 else "0x018+0x20*N"
    desc = sys.argv[2] if len(sys.argv) > 2 else "N = 0 to 3"
    print(resolve_address(expr, desc))
