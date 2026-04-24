#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rdl_lint.py — RDL file validator for PeakRDL-Renode projects
=============================================================
Runs the same compile → elaborate → export pipeline as `peakrdl renode`,
but collects every warning, error and fatal message into a structured
report instead of dumping them to the terminal.

Usage
-----
    python rdl_lint.py <file.rdl> [options]

Options
    -N, --namespace  NAMESPACE   C# namespace  (required for export stage)
    -n, --name       NAME        Peripheral name (optional)
    --no-out-vars    REG [REG…]  Register names to suppress out-variables
    --all-public                 Make all fields public
    --output         PATH        Write report to this file (default: rdl_lint_report.txt)
    --skip-export                Only run compile + elaborate, skip C# export stage
    -v, --verbose                Also print the report to stdout

Exit codes
    0  No errors or fatal issues found
    1  One or more errors/fatals found
    2  Usage / argument error
    # Compile + elaborate only (no namespace needed)
python rdl_lint.py my_peripheral.rdl --skip-export

# Full pipeline including C# export validation
python rdl_lint.py my_peripheral.rdl -N MyNamespace

# With --no-out-vars (also validates that the register names actually exist)
python rdl_lint.py my_peripheral.rdl -N MyNamespace --no-out-vars ctrl status

# Print report to stdout AND save it
python rdl_lint.py my_peripheral.rdl -N MyNamespace -v --output my_report.txt
"""

import sys
import os
import argparse
import textwrap
import datetime
from typing import Optional, List

# ── systemrdl imports ────────────────────────────────────────────────────────
try:
    from systemrdl import RDLCompiler
    from systemrdl.messages import (
        MessagePrinter, Severity, RDLCompileError
    )
    from systemrdl.source_ref import DetailedFileSourceRef, FileSourceRef
    from systemrdl.node import AddrmapNode, RegNode
except ImportError:
    print("ERROR: 'systemrdl' is not installed.  Run:  pip install peakrdl",
          file=sys.stderr)
    sys.exit(2)

# ── peakrdl_renode import (optional – needed for export stage) ───────────────
try:
    from peakrdl_renode.cs_exporter import CSharpExporter
    _RENODE_AVAILABLE = True
except ImportError:
    _RENODE_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Message capture
# ─────────────────────────────────────────────────────────────────────────────

class _CapturingPrinter(MessagePrinter):
    """
    Replaces systemrdl's default stderr printer.
    Stores every message as a dict; never writes to the console.
    """

    def __init__(self, store: list):
        self._store = store

    def print_message(self, severity: Severity, text: str, src_ref) -> None:
        entry = {
            "severity":  severity,
            "text":      text,
            "file":      None,
            "line":      None,
            "line_text": None,
            "col_start": None,
            "col_end":   None,
        }
        if isinstance(src_ref, DetailedFileSourceRef):
            entry["file"]      = src_ref.path
            entry["line"]      = src_ref.line
            entry["line_text"] = src_ref.line_text.rstrip()
            entry["col_start"] = src_ref.line_selection[0]
            entry["col_end"]   = src_ref.line_selection[1]
        elif isinstance(src_ref, FileSourceRef):
            entry["file"] = src_ref.path

        self._store.append(entry)

    def emit_message(self, lines: List[str]) -> None:
        pass  # suppress all console output – we handle reporting ourselves


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline runner
# ─────────────────────────────────────────────────────────────────────────────

# Internal "pipeline aborted" messages added by systemrdl itself – they are
# redundant in a structured report because the underlying cause is already
# captured as a separate message.
_NOISE_PATTERNS = (
    "aborted due to previous errors",
    "Elaborate aborted",
    "Compile aborted",
    "Parse aborted",
)


def _is_noise(text: str) -> bool:
    return any(p.lower() in text.lower() for p in _NOISE_PATTERNS)


def run_pipeline(
    rdl_file: str,
    namespace: Optional[str],
    name: Optional[str],
    all_public: bool,
    no_out_vars: set,
    skip_export: bool,
) -> dict:
    """
    Execute compile → elaborate → export and return a structured result dict:

        {
          "rdl_file":    str,
          "compile_ok":  bool,
          "elaborate_ok": bool,
          "export_ok":   bool   (None if skipped),
          "messages":    [ { severity, text, file, line, line_text,
                             col_start, col_end }, … ]
        }
    """

    messages: list = []
    printer = _CapturingPrinter(messages)

    rdlc = RDLCompiler()
    rdlc.env.msg.printer    = printer
    rdlc.env.msg.min_verbosity = Severity.WARNING   # capture warnings + above

    compile_ok  = False
    elaborate_ok = False
    export_ok   = None  # None = not attempted

    # ── Phase 1 : compile ────────────────────────────────────────────────────
    try:
        rdlc.compile_file(rdl_file)
        compile_ok = True
    except RDLCompileError:
        pass  # messages already captured by the printer
    except FileNotFoundError:
        messages.append({
            "severity":  Severity.FATAL,
            "text":      f"File not found: {rdl_file}",
            "file":      rdl_file,
            "line":      None, "line_text": None,
            "col_start": None, "col_end":   None,
        })
        return _build_result(rdl_file, compile_ok, elaborate_ok, export_ok,
                             messages)
    except Exception as exc:
        messages.append({
            "severity":  Severity.FATAL,
            "text":      f"Unexpected error during compilation: {exc}",
            "file":      rdl_file,
            "line":      None, "line_text": None,
            "col_start": None, "col_end":   None,
        })

    # ── Phase 2 : elaborate ──────────────────────────────────────────────────
    if compile_ok:
        try:
            root = rdlc.elaborate()
            elaborate_ok = True
        except RDLCompileError:
            pass
        except Exception as exc:
            messages.append({
                "severity":  Severity.FATAL,
                "text":      f"Unexpected error during elaboration: {exc}",
                "file":      rdl_file,
                "line":      None, "line_text": None,
                "col_start": None, "col_end":   None,
            })

    # ── Phase 2b : validate --no-out-vars names ──────────────────────────────
    if elaborate_ok and no_out_vars:
        known = _collect_register_names(rdlc.elaborate() if False else root)
        for reg_name in sorted(no_out_vars):
            if reg_name not in known:
                messages.append({
                    "severity":  Severity.WARNING,
                    "text":      (
                        f"--no-out-vars: register instance '{reg_name}' was not "
                        f"found in the design.  Known registers: "
                        f"{', '.join(sorted(known)) or '(none)'}"
                    ),
                    "file":      None,
                    "line":      None, "line_text": None,
                    "col_start": None, "col_end":   None,
                })

    # ── Phase 3 : export ─────────────────────────────────────────────────────
    if elaborate_ok and not skip_export:
        if not _RENODE_AVAILABLE:
            messages.append({
                "severity":  Severity.WARNING,
                "text":      (
                    "peakrdl_renode is not installed – export stage skipped.  "
                    "Install it with: pip install <path-to-PeakRDL-renode>"
                ),
                "file":      None,
                "line":      None, "line_text": None,
                "col_start": None, "col_end":   None,
            })
            export_ok = None
        elif namespace is None:
            messages.append({
                "severity":  Severity.WARNING,
                "text":      (
                    "No --namespace supplied – export stage skipped.  "
                    "Pass -N NAMESPACE to also validate C# code generation."
                ),
                "file":      None,
                "line":      None, "line_text": None,
                "col_start": None, "col_end":   None,
            })
            export_ok = None
        else:
            export_ok = False
            try:
                CSharpExporter().export(
                    root,
                    path       = os.devnull,   # discard output – validation only
                    name       = name,
                    namespace  = namespace,
                    all_public = all_public,
                    no_out_vars = no_out_vars,
                )
                export_ok = True
            except RuntimeError as exc:
                messages.append({
                    "severity":  Severity.ERROR,
                    "text":      f"C# export error: {exc}",
                    "file":      rdl_file,
                    "line":      None, "line_text": None,
                    "col_start": None, "col_end":   None,
                })
            except Exception as exc:
                messages.append({
                    "severity":  Severity.ERROR,
                    "text":      f"Unexpected export error: {type(exc).__name__}: {exc}",
                    "file":      rdl_file,
                    "line":      None, "line_text": None,
                    "col_start": None, "col_end":   None,
                })

    return _build_result(rdl_file, compile_ok, elaborate_ok, export_ok,
                         messages)


def _build_result(rdl_file, compile_ok, elaborate_ok, export_ok, messages):
    # Strip internal "aborted" noise before returning
    filtered = [m for m in messages if not _is_noise(m["text"])]
    return {
        "rdl_file":    rdl_file,
        "compile_ok":  compile_ok,
        "elaborate_ok": elaborate_ok,
        "export_ok":   export_ok,
        "messages":    filtered,
    }


def _collect_register_names(top_node) -> set:
    """Walk the elaborated node tree and collect all register instance names."""
    names: set = set()

    def _walk(node):
        if isinstance(node, RegNode):
            names.add(node.inst_name)
        for child in node.children():
            _walk(child)

    _walk(top_node)
    return names


# ─────────────────────────────────────────────────────────────────────────────
# Report formatter
# ─────────────────────────────────────────────────────────────────────────────

_SEV_LABEL = {
    Severity.WARNING: "WARNING",
    Severity.ERROR:   "ERROR  ",
    Severity.FATAL:   "FATAL  ",
    Severity.INFO:    "INFO   ",
    Severity.DEBUG:   "DEBUG  ",
}

_STAGE_LABEL = {
    True:  "PASSED",
    False: "FAILED",
    None:  "SKIPPED",
}


def build_report(result: dict, no_out_vars: set, verbose_export: bool) -> str:
    lines = []
    sep  = "=" * 72
    sep2 = "-" * 72

    # ── Header ───────────────────────────────────────────────────────────────
    lines.append(sep)
    lines.append("  PeakRDL-Renode  RDL Lint Report")
    lines.append(f"  Generated : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  File      : {os.path.abspath(result['rdl_file'])}")
    lines.append(sep)

    # ── Pipeline summary ─────────────────────────────────────────────────────
    lines.append("")
    lines.append("Pipeline Stages")
    lines.append(sep2)
    lines.append(f"  1. Compile    : {_STAGE_LABEL[result['compile_ok']]}")
    lines.append(f"  2. Elaborate  : {_STAGE_LABEL[result['elaborate_ok']]}")
    exp_label = _STAGE_LABEL[result['export_ok']]
    lines.append(f"  3. CS Export  : {exp_label}")
    lines.append("")

    # ── Message counts ───────────────────────────────────────────────────────
    msgs = result["messages"]
    fatals   = [m for m in msgs if m["severity"] == Severity.FATAL]
    errors   = [m for m in msgs if m["severity"] == Severity.ERROR]
    warnings = [m for m in msgs if m["severity"] == Severity.WARNING]

    lines.append("Summary")
    lines.append(sep2)
    lines.append(f"  Fatals   : {len(fatals)}")
    lines.append(f"  Errors   : {len(errors)}")
    lines.append(f"  Warnings : {len(warnings)}")
    lines.append("")

    if not msgs:
        lines.append("  No issues found.  RDL file is valid.")
        lines.append("")
        lines.append(sep)
        return "\n".join(lines)

    # ── Detailed issues ──────────────────────────────────────────────────────
    def render_group(group_msgs, title):
        if not group_msgs:
            return
        lines.append(title)
        lines.append(sep2)
        for idx, m in enumerate(group_msgs, 1):
            sev_tag = _SEV_LABEL.get(m["severity"], m["severity"].name.ljust(7))

            # Location string
            if m["file"] and m["line"]:
                loc = f"{os.path.basename(m['file'])}  line {m['line']}"
            elif m["file"]:
                loc = os.path.basename(m["file"])
            else:
                loc = "no location"

            lines.append(f"  [{idx:>3}]  {sev_tag}  {loc}")

            # Wrap long message text
            wrapped = textwrap.wrap(m["text"], width=66,
                                    initial_indent="         ",
                                    subsequent_indent="         ")
            lines.extend(wrapped)

            # Source context snippet
            if m["line_text"] is not None:
                stripped = m["line_text"].strip()
                leading  = len(m["line_text"]) - len(m["line_text"].lstrip())
                c_start  = max(0, m["col_start"] - leading)
                c_end    = max(0, m["col_end"]   - leading)
                marker   = " " * c_start + "^" * max(1, c_end - c_start + 1)
                lines.append(f"         | {stripped}")
                lines.append(f"         | {marker}")

            lines.append("")

    render_group(fatals,   "Fatal Errors")
    render_group(errors,   "Errors")
    render_group(warnings, "Warnings")

    lines.append(sep)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        prog="rdl_lint",
        description=(
            "Validate a SystemRDL file through the full PeakRDL-Renode "
            "pipeline (compile → elaborate → C# export) and write a "
            "structured report."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples
            --------
              # Compile + elaborate only
              python rdl_lint.py my_peripheral.rdl --skip-export

              # Full pipeline with C# export validation
              python rdl_lint.py my_peripheral.rdl -N MyNamespace

              # Suppress out-vars for specific registers
              python rdl_lint.py my_peripheral.rdl -N MyNamespace \\
                  --no-out-vars ctrl status
        """),
    )

    parser.add_argument(
        "rdl_file",
        metavar="FILE.rdl",
        help="Path to the SystemRDL file to validate",
    )
    parser.add_argument(
        "-N", "--namespace",
        metavar="NAMESPACE",
        default=None,
        help="C# peripheral namespace (required to run the export stage)",
    )
    parser.add_argument(
        "-n", "--name",
        metavar="NAME",
        default=None,
        help="C# peripheral class name (defaults to top addrmap name)",
    )
    parser.add_argument(
        "--no-out-vars",
        metavar="REGISTER_NAME",
        nargs="+",
        default=[],
        help=(
            "Register instance names whose fields should omit out variable "
            "declarations in the generated C#.  "
            "Example: --no-out-vars ctrl status"
        ),
    )
    parser.add_argument(
        "--all-public",
        action="store_true",
        help="Make all generated fields public",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        default="rdl_lint_report.txt",
        help="Path for the report file (default: rdl_lint_report.txt)",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Only run compile + elaborate; skip the C# export stage",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Also print the report to stdout",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # ── Run pipeline ─────────────────────────────────────────────────────────
    result = run_pipeline(
        rdl_file    = args.rdl_file,
        namespace   = args.namespace,
        name        = args.name,
        all_public  = args.all_public,
        no_out_vars = set(args.no_out_vars),
        skip_export = args.skip_export,
    )

    # ── Build report ─────────────────────────────────────────────────────────
    report = build_report(
        result,
        no_out_vars    = set(args.no_out_vars),
        verbose_export = not args.skip_export,
    )

    # ── Write report file ────────────────────────────────────────────────────
    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write(report + "\n")

    # ── Console output ───────────────────────────────────────────────────────
    msgs = result["messages"]
    fatals  = sum(1 for m in msgs if m["severity"] == Severity.FATAL)
    errors  = sum(1 for m in msgs if m["severity"] == Severity.ERROR)
    warnings = sum(1 for m in msgs if m["severity"] == Severity.WARNING)

    if args.verbose:
        print(report)
    else:
        # Brief one-line summary to stdout
        if not msgs:
            print(f"OK  No issues found in '{args.rdl_file}'")
        else:
            parts = []
            if fatals:   parts.append(f"{fatals} fatal(s)")
            if errors:   parts.append(f"{errors} error(s)")
            if warnings: parts.append(f"{warnings} warning(s)")
            print(f"{'FAIL' if (fatals or errors) else 'WARN'}  "
                  f"{', '.join(parts)} — see '{args.output}'")

    # ── Exit code ────────────────────────────────────────────────────────────
    has_errors = any(
        m["severity"] in (Severity.ERROR, Severity.FATAL)
        for m in msgs
    )
    sys.exit(1 if has_errors else 0)


if __name__ == "__main__":
    main()
