#!/usr/bin/env python3
"""
gen_renode_callbacks.py

Companion generator to PeakRDL-renode.

PeakRDL-renode (peakrdl renode ...) emits a *stub* C# partial class:
registers/fields with the right widths/offsets, but no behavior -- no
ValueProviderCallback / WriteCallback wiring. This script compiles the same
.rdl with the systemrdl-compiler and emits the second file: a partial class
(same class/namespace) whose Init() method wires up:

  - a ValueProviderCallback for every field whose `sw` access is `r` or `rw`
  - a WriteCallback for every field whose `sw` access is `w` or `rw`

Both simply read/write a backing `.Value` on the field itself, mirroring
what you'd write by hand for plain storage registers. Anything with real
side effects still needs manual edits after generation.

NAMING -- verified against PeakRDL-renode's actual source
(github.com/renode/renode, tools/PeakRDL-renode), not guessed:
  - Field identifiers are `field.inst_name.upper()`.
  - Register properties are the pascal-cased instance name, prefixed by
    every ancestor `regfile` name (also pascal-cased), joined with `_`.
    E.g. a reg `fields` inside `regfile registers` becomes `Registers_Fields`.
  - Pascal-casing is done with the same `case-converter` PyPI package
    (`caseconverter.pascalcase`) PeakRDL-renode itself uses, so identifiers
    match exactly rather than approximating the convention.

SCOPE / KNOWN LIMITATION -- also verified against the real source, not
guessed:
  `mem { ... }` blocks (SystemRDL requires them to be `external`) are
  compiled by PeakRDL-renode into a *completely different* code path: a
  raw byte-array-backed struct with plain C# get/set properties that
  bit-pack directly into a `byte[]`. There is no IValueRegisterField /
  IFlagRegisterField, no `.Value`, and critically no
  ValueProviderCallback/WriteCallback anywhere in that generated code --
  the string "Callback" does not appear anywhere in the plugin's source
  for the mem code path. So there is nothing for a companion callback
  file to hook into for `mem`-based "register bank" arrays; custom
  behavior for those has to be implemented by overriding
  ReadDoubleWord/WriteDoubleWord (or working with the generated wrapper's
  plain properties directly) in your own code, not through this script.
  This script therefore skips `mem` blocks entirely and prints a warning
  if it finds one, rather than emitting code that won't compile.

  Likewise, plain `reg` arrays declared *outside* a `mem` block (directly
  under addrmap/regfile) aren't handled by PeakRDL-renode's array-aware
  code path either -- that only exists for the `mem` case -- so this
  script does not attempt to special-case them; if you have one, check
  that PeakRDL-renode's own stub output looks like what you expect before
  relying on this script's output for it.

Usage:
    python3 gen_renode_callbacks.py peripheral.rdl \\
        -n SMPH \\
        -N Antmicro.Renode.Peripherals.Example_Registers \\
        -o SMPH_callbacks.cs

Requires:
    pip install systemrdl-compiler case-converter
"""

import argparse
import sys
from systemrdl import RDLCompiler, RDLCompileError
from systemrdl.node import AddrmapNode, RegNode, RegfileNode, MemNode, FieldNode

try:
    import caseconverter
except ImportError:
    print("This script requires the 'case-converter' package (pip install case-converter) "
          "-- the same one PeakRDL-renode itself uses for identifier naming.", file=sys.stderr)
    raise


def reg_property_name(reg: RegNode, regfile_stack: list) -> str:
    """
    Reproduce PeakRDL-renode's `variable_name()`: pascal-case each ancestor
    regfile name plus the register's own name, joined with '_'.
    """
    parts = [caseconverter.pascalcase(name) for name in regfile_stack]
    parts.append(caseconverter.pascalcase(reg.inst_name))
    return "_".join(parts)


def field_property_name(field: FieldNode) -> str:
    """Reproduce PeakRDL-renode's field identifier: inst_name.upper()."""
    return field.inst_name.upper()


def field_value_type(field: FieldNode) -> str:
    """
    PeakRDL-renode generates 1-bit fields as IFlagRegisterField (`.Value`
    is bool) and any wider field as IValueRegisterField (`.Value` is uint)
    -- confirmed against real generated stubs. The callback signatures
    differ accordingly (Action<bool,bool> vs Action<uint,uint>), so the
    write-side cast has to match.
    """
    return 'bool' if field.width == 1 else 'uint'


def cast_expr(value_type: str, expr: str) -> str:
    return expr if value_type == 'bool' else f"(uint){expr}"


def sw_access(field: FieldNode) -> str:
    """Return the field's sw access as one of 'r', 'w', 'rw', 'na'."""
    access = field.get_property('sw')
    return access.name.lower() if hasattr(access, 'name') else str(access).lower()


def iter_regs(node, regfile_stack=None):
    """
    Yield (RegNode, regfile_stack) for every register under an
    addrmap/regfile, recursively. `mem` blocks are skipped -- see the
    module docstring for why -- with a warning printed to stderr.
    """
    if regfile_stack is None:
        regfile_stack = []

    for child in node.children():
        if isinstance(child, RegNode):
            yield child, regfile_stack
        elif isinstance(child, RegfileNode):
            yield from iter_regs(child, regfile_stack + [child.inst_name])
        elif isinstance(child, AddrmapNode):
            yield from iter_regs(child, regfile_stack)
        elif isinstance(child, MemNode):
            print(
                f"Warning: skipping mem block '{child.inst_name}' -- PeakRDL-renode compiles "
                f"mem contents to a plain byte-packed struct with no ValueProviderCallback/"
                f"WriteCallback hooks, so there's nothing here for a callback file to wire up. "
                f"See the module docstring for details.",
                file=sys.stderr,
            )


def build_field_block(field_ref: str, access: str, indent: str, value_type: str) -> str:
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
        lines.append(f"{indent}    {field_ref}.Value = {cast_expr(value_type, 'newval')};")
        lines.append(f"{indent}}};")

    return "\n".join(lines)


def build_reg_block(reg: RegNode, regfile_stack: list, base_indent: str) -> str:
    reg_prop = reg_property_name(reg, regfile_stack)

    field_blocks = []
    for field in reg.fields():
        access = sw_access(field)
        if access == 'na':
            continue
        field_ref = f"this.{reg_prop}.{field_property_name(field)}"
        value_type = field_value_type(field)
        field_blocks.append(build_field_block(field_ref, access, base_indent, value_type))

    return "\n\n".join(field_blocks)


def build_cs_file(top_node: AddrmapNode, class_name: str, namespace: str, rdl_filename: str) -> str:
    indent = "            "  # 12 spaces, matches the sample's nesting depth
    body_parts = []

    regs = list(iter_regs(top_node))
    if not regs:
        print("Warning: no callback-eligible registers found (mem blocks don't count -- see above).",
              file=sys.stderr)

    # Flag same-address register pairs: PeakRDL-renode merges a read-only +
    # write-only register sharing one address into a single struct with
    # different naming than either register alone. Replicating that merge
    # logic isn't done here, so warn rather than silently emit wrong names.
    by_addr = {}
    for reg, regfile_stack in regs:
        by_addr.setdefault(reg.absolute_address, []).append(reg.inst_name)
    for addr, names in by_addr.items():
        if len(names) > 1:
            print(
                f"Warning: registers {names} share address {hex(addr)}. PeakRDL-renode merges "
                f"same-address read-only/write-only register pairs into a single combined struct "
                f"with its own naming -- this script does not replicate that merge, so check the "
                f"real generated stub's property name for these before trusting this output.",
                file=sys.stderr,
            )

    for reg, regfile_stack in regs:
        block = build_reg_block(reg, regfile_stack, indent)
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
