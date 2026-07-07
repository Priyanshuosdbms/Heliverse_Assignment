"""
tools/run_parser.py

Executes the parser script the codegen sub-agent wrote, against the real
workbook, in a subprocess. Output is streamed live to the terminal as it
arrives (so you're never staring at a blank screen) while also being
captured and returned for the validation loop.
"""
from __future__ import annotations
import subprocess
import sys
import threading
from pathlib import Path

# ANSI for run_parser terminal output (no external deps)
_DIM   = "\033[2m"
_CYAN  = "\033[36m"
_RED   = "\033[31m"
_BOLD  = "\033[1m"
_R     = "\033[0m"


def _stream_pipe(pipe, label: str, store: list[str]) -> None:
    """Read a pipe line by line, print live, and accumulate into `store`."""
    prefix = f"{_DIM}  [{label}]{_R} "
    for raw_line in pipe:
        line = raw_line if isinstance(raw_line, str) else raw_line.decode("utf-8", errors="replace")
        store.append(line)
        # stderr lines are warnings/errors — highlight them
        colour = _RED if label == "stderr" else _CYAN
        print(f"{colour}{prefix}{line.rstrip()}{_R}", flush=True)
    pipe.close()


def run_generated_parser(
    script_path: str,
    xlsx_path: str,
    out_json_path: str,
    timeout: int = 600,
) -> dict:
    """Run the generated parser script and stream its output live.

    stdout/stderr are printed to the terminal as each line arrives so you
    can see progress on large workbooks. Both are also captured and
    returned (last 4 000 chars each) for the agent's validation loop.
    """
    print(f"\n{_BOLD}{_CYAN}  ▶  Running generated parser...{_R}")
    print(f"{_DIM}  {sys.executable} {script_path} {xlsx_path} {out_json_path}{_R}\n")

    proc = subprocess.Popen(
        [sys.executable, script_path, xlsx_path, out_json_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,          # line-buffered
    )

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    # Read both pipes concurrently so neither blocks the other
    t_out = threading.Thread(target=_stream_pipe, args=(proc.stdout, "stdout", stdout_lines), daemon=True)
    t_err = threading.Thread(target=_stream_pipe, args=(proc.stderr, "stderr", stderr_lines), daemon=True)
    t_out.start()
    t_err.start()

    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        t_out.join(timeout=2)
        t_err.join(timeout=2)
        print(f"\n{_RED}{_BOLD}  ✖  Parser timed out after {timeout}s.{_R}")
        return {
            "returncode": -1,
            "stdout": "".join(stdout_lines)[-4000:],
            "stderr": f"Timed out after {timeout}s\n" + "".join(stderr_lines)[-4000:],
            "output_exists": False,
        }

    t_out.join()
    t_err.join()

    stdout_str = "".join(stdout_lines)
    stderr_str = "".join(stderr_lines)

    status = "✔  Parser finished" if proc.returncode == 0 else "✖  Parser exited with error"
    colour = _CYAN if proc.returncode == 0 else _RED
    print(f"\n{colour}{_BOLD}  {status} (returncode={proc.returncode}){_R}")

    out = {
        "returncode": proc.returncode,
        "stdout": stdout_str[-4000:],
        "stderr": stderr_str[-4000:],
        "output_exists": Path(out_json_path).exists(),
    }
    if out["output_exists"]:
        out["output_preview"] = Path(out_json_path).read_text(encoding="utf-8")[:3000]
    return out

