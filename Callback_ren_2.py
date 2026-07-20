#!/usr/bin/env python3
"""
gen_renode_callbacks.py

Companion generator to PeakRDL-renode.

PeakRDL-renode (peakrdl renode ...) emits a *stub* C# partial class: registers
and fields with the right widths/offsets, but no behavior -- no
ValueProviderCallback / WriteCallback wiring.

This script compiles the same .rdl file with the systemrdl-compiler and walks
the elaborated register model to emit the *second* file: a partial class
(same class/namespace) whose Init() method wires up:

  - a ValueProviderCallback for every field whose `sw` access is `r` or `rw`
    (software can read it -> we need to supply the value)
  - a WriteCallback for every field whose `sw` access is `w` or `rw`
    (software can write it -> we need to capture/store the value)

The default callback bodies just read/write a backing `Value` on the field
itself (mirroring the example you supplied). That's a reasonable default for
plain storage registers; anything with real side effects (triggering an
action, clearing-on-read, etc.) you'll edit by hand afterwards -- this script
gets you the boilerplate wiring so you don't have to type it for every
field in a large peripheral.

Usage:
    python3 gen_renode_callbacks.py peripheral.rdl \\
        -n SMPH \\
        -N Antmicro.Renode.Peripherals.Example_Registers \\
        -o SMPH_callbacks.cs

Requires:
    pip install systemrdl-compiler
"""

import argparse
import re
import sys
from systemrdl import RDLCompiler, RDLCompileError
from systemrdl.node import AddrmapNode, RegNode, RegfileNode, MemNode, FieldNode


def pascal_case(name: str) -> str:
    """
    Convert an RDL instance name to the PascalCase property name
    PeakRDL-renode uses in the stub.

    Splits on underscores/non-alphanumerics and capitalizes each chunk,
    e.g. "register_name" -> "RegisterName". A single all-caps chunk like
    "REG0" reduces to "Reg0" (str.capitalize() behaviour), which matches
    the example stub exactly.
    """
    if not name:
        return name
    parts = re.split(r'[^0-9A-Za-z]+', name)
    return "".join(p[:1].upper() + p[1:].lower() for p in parts if p)


def sw_access(field: FieldNode) -> str:
    """Return the field's sw access as one of 'r', 'w', 'rw', 'na'."""
    access = field.get_property('sw')
    return access.name.lower() if hasattr(access, 'name') else str(access).lower()


def iter_regs(node):
    """
    Yield every RegNode under an addrmap/regfile/mem node, recursively.

    `mem { ... } external some_mem @ addr;` blocks are still walked into --
    the `external` qualifier only affects how PeakRDL-renode wires up bus
    access in the *stub* file, it doesn't change the fact that the
    registers inside still need callbacks wired here.
    """
    for child in node.children():
        if isinstance(child, RegNode):
            yield child
        elif isinstance(child, (AddrmapNode, RegfileNode, MemNode)):
            yield from iter_regs(child)


def build_field_block(field_ref: str, access: str, indent: str) -> str:
    """Build the C# block wiring callbacks for a single field."""
    lines = []

    if access in ('r', 'rw'):
        lines.append(f"{indent}{field_ref}.ValueProviderCallback += (_) =>")
        lines.append(f"{indent}{{")
        lines.append(f"{indent}    return {field_ref}.Value;")
        lines.append(f"{indent}}};")

    if access in ('w', 'rw'):
        if lines:
            lines.append("")
        lines.append(f"{indent}{field_ref}.WriteCallback += (oldval, newval) =>")
        lines.append(f"{indent}{{")
        lines.append(f"{indent}    {field_ref}.Value = (uint)newval;")
        lines.append(f"{indent}}};")

    return "\n".join(lines)


def build_array_field_block(local_var: str, field_ref_expr: str, access: str, indent: str) -> str:
    """
    Build the C# for a single field inside an array-register loop.

    A `for (var i = 0; ...)` header variable is shared storage across all
    iterations in C# (unlike `foreach`, this was never fixed by the C# 5
    closure change), so capturing it directly in a lambda would make every
    field's callback see only the final index once the loop finishes. To
    avoid that, we declare a fresh local *inside* the loop body -- such
    variables get a distinct instance per iteration and are safe to
    capture -- and reference that local from the callbacks instead of the
    loop index.
    """
    lines = [f"{indent}var {local_var} = {field_ref_expr};"]

    if access in ('r', 'rw'):
        lines.append(f"{indent}{local_var}.ValueProviderCallback += (_) =>")
        lines.append(f"{indent}{{")
        lines.append(f"{indent}    return {local_var}.Value;")
        lines.append(f"{indent}}};")

    if access in ('w', 'rw'):
        lines.append(f"{indent}{local_var}.WriteCallback += (oldval, newval) =>")
        lines.append(f"{indent}{{")
        lines.append(f"{indent}    {local_var}.Value = (uint)newval;")
        lines.append(f"{indent}}};")

    return "\n".join(lines)


def build_reg_block(reg: RegNode, base_indent: str) -> str:
    """
    Build the C# for a single register, whether scalar or an array.

    Array registers (declared like `register_name[231] @ 0x0 += 0x8;`,
    typically found inside a `mem { ... }` block) are wired inside a
    `for` loop over the array's dimensions instead of being unrolled.
    A local copy of the loop variable is used in each closure to avoid
    the classic C# "captured loop variable" bug.
    """
    reg_prop = pascal_case(reg.inst_name)

    if reg.is_array:
        dims = reg.array_dimensions  # e.g. [231] or [rows, cols] for multi-dim arrays
        idx_names = [f"i{d}" for d in range(len(dims))]

        header_lines = []
        indent = base_indent
        for depth, (dim, idx) in enumerate(zip(dims, idx_names)):
            header_lines.append(f"{indent}for (var {idx} = 0; {idx} < {dim}; {idx}++)")
            header_lines.append(f"{indent}{{")
            indent += "    "

        index_expr = "".join(f"[{idx}]" for idx in idx_names)
        body_indent = indent

        field_blocks = []
        for field in reg.fields():
            access = sw_access(field)
            if access == 'na':
                continue
            field_ref_expr = f"this.{reg_prop}{index_expr}.{field.inst_name}"
            local_var = f"field_{field.inst_name}"
            field_blocks.append(build_array_field_block(local_var, field_ref_expr, access, body_indent))

        if not field_blocks:
            return ""

        footer_lines = []
        for depth in range(len(dims)):
            indent = indent[:-4]
            footer_lines.append(f"{indent}}}")

        return "\n".join(header_lines) + "\n" + "\n\n".join(field_blocks) + "\n" + "\n".join(footer_lines)

    else:
        field_blocks = []
        for field in reg.fields():
            access = sw_access(field)
            if access == 'na':
                continue
            field_ref = f"this.{reg_prop}.{field.inst_name}"
            field_blocks.append(build_field_block(field_ref, access, base_indent))

        return "\n\n".join(field_blocks)


def build_cs_file(top_node: AddrmapNode, class_name: str, namespace: str, rdl_filename: str) -> str:
    indent = "            "  # 12 spaces, matches the sample's nesting depth
    body_parts = []

    regs = list(iter_regs(top_node))
    if not regs:
        print("Warning: no registers found in the elaborated design.", file=sys.stderr)

    for reg in regs:
        block = build_reg_block(reg, indent)
        if block:
            body_parts.append(block)

    body = "\n\n\n".join(body_parts)

    return f'''\
// Generated by gen_renode_callbacks.py from {rdl_filename}
// Companion callback file for the PeakRDL-renode stub -- do not regenerate
// over hand-written edits without diffing first.
using Antmicro.Renode.Time;
using Antmicro.Renode.Core;
using Antmicro.Renode.Core.Structure.Registers;
using Antmicro.Renode.Peripherals.Bus;
using Antmicro.Renode.Peripherals.Timers;
using Antmicro.Renode.Utilities;
using System.Linq;
using System.Collections.Generic;
using Antmicro.Renode.Logging;
using System.IO;
using System.Threading;
using System;

namespace {namespace} // This namespace name will be given by the user
{{
    public partial class {class_name} : IProvidesRegisterCollection<DoubleWordRegisterCollection>, IPeripheral, IDoubleWordPeripheral, INumberedGPIOOutput, IGPIOReceiver
    {{
        partial void Init()
        {{
            this.Log(LogLevel.Info, "Example peripheral constructor");
{body}
        }}
    }}
}}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rdl_file", help="Path to the .rdl input file")
    parser.add_argument("-n", "--name", required=True, dest="class_name",
                         help="Peripheral/class name (should match the -n passed to `peakrdl renode`)")
    parser.add_argument("-N", "--namespace", required=True,
                         help="C# namespace (should match the -N passed to `peakrdl renode`)")
    parser.add_argument("-o", "--output", required=True, help="Output .cs file path")
    parser.add_argument("--top", default=None, help="Top-level addrmap name, if the file defines more than one")
    args = parser.parse_args()

    rdlc = RDLCompiler()
    try:
        rdlc.compile_file(args.rdl_file)
        root = rdlc.elaborate(top_def_name=args.top) if args.top else rdlc.elaborate()
    except RDLCompileError:
        sys.exit(1)

    top_node = root.top

    cs_text = build_cs_file(top_node, args.class_name, args.namespace, args.rdl_file)

    with open(args.output, "w") as f:
        f.write(cs_text)

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
