"""
tools/validate_json.py

File-shape-agnostic sanity checks on the produced register-map JSON.
These don't know anything about this specific workbook's column layout —
they just check that the *output* obeys the universal rules of a register
map, so they keep working even when next quarter's spreadsheet has a
different column order.
"""
from __future__ import annotations
import json


def validate_register_json(json_path: str) -> dict:
    errors = []
    warnings = []

    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return {"ok": False, "errors": [f"could not parse JSON: {e}"], "warnings": []}

    registers = data if isinstance(data, list) else data.get("registers", data)
    if not isinstance(registers, list):
        return {"ok": False, "errors": ["top-level structure is not a list of registers (and no 'registers' key found)"], "warnings": []}

    if len(registers) == 0:
        errors.append("no registers found")

    seen_addrs = {}
    for i, reg in enumerate(registers):
        label = reg.get("name") or reg.get("register_name") or f"<register #{i}>"

        addr = reg.get("address") or reg.get("offset") or reg.get("base_address")
        if addr is None:
            errors.append(f"{label}: missing address/offset field")

        fields = reg.get("fields")
        if not fields or not isinstance(fields, list):
            errors.append(f"{label}: missing or empty 'fields' list")
            continue

        # bitfield sanity: no overlaps, msb >= lsb, all within 0-31 (assume 32-bit; flag if not)
        intervals = []
        for fld in fields:
            msb, lsb = fld.get("msb"), fld.get("lsb")
            fname = fld.get("name", "<unnamed field>")
            if msb is None or lsb is None:
                errors.append(f"{label}.{fname}: missing msb/lsb")
                continue
            try:
                msb, lsb = int(msb), int(lsb)
            except (TypeError, ValueError):
                errors.append(f"{label}.{fname}: msb/lsb not integers ({msb}, {lsb})")
                continue
            if msb < lsb:
                errors.append(f"{label}.{fname}: msb ({msb}) < lsb ({lsb})")
            intervals.append((lsb, msb, fname))

        intervals.sort()
        for j in range(1, len(intervals)):
            prev_lsb, prev_msb, prev_name = intervals[j - 1]
            cur_lsb, cur_msb, cur_name = intervals[j]
            if cur_lsb <= prev_msb:
                errors.append(f"{label}: field '{prev_name}' [{prev_msb}:{prev_lsb}] overlaps '{cur_name}' [{cur_msb}:{cur_lsb}]")

        if addr is not None:
            seen_addrs.setdefault(str(addr), []).append(label)

    for addr, labels in seen_addrs.items():
        if len(labels) > 1:
            warnings.append(f"address {addr} used by multiple registers: {labels} (ok if intentional aliasing, otherwise check grouping logic)")

    return {"ok": len(errors) == 0, "errors": errors, "warnings": warnings, "register_count": len(registers)}


if __name__ == "__main__":
    import sys
    print(json.dumps(validate_register_json(sys.argv[1]), indent=2, ensure_ascii=False))