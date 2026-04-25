#!/usr/bin/env python3
from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys
import tempfile


CHUNK_ALIGNMENT = 2 << 20


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def run(exe: pathlib.Path, args: list[str], env_overrides: dict[str, str] | None = None, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        [str(exe), *args],
        input=input_text,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def expect_success(proc: subprocess.CompletedProcess[str], expected_stdout: str) -> None:
    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()
    if proc.returncode != 0:
        fail(f"expected success, got {proc.returncode}, stderr={stderr!r}")
    if stdout != expected_stdout:
        fail(f"stdout mismatch: expected {expected_stdout!r}, got {stdout!r}")


def expect_failure(proc: subprocess.CompletedProcess[str]) -> None:
    if proc.returncode == 0:
        fail("expected failure exit code")


def make_boundary_file(path: pathlib.Path, separator: bytes) -> None:
    prefix = b"a" * (CHUNK_ALIGNMENT - len(separator))
    suffix = b"b\n" + b" " * (2 * CHUNK_ALIGNMENT)
    path.write_bytes(prefix + separator + suffix)


def maybe_compare_with_wc(exe: pathlib.Path, temp_dir: pathlib.Path) -> None:
    wc_path = shutil.which("wc")
    if wc_path is None:
        return

    first = temp_dir / "wc_compare_a.txt"
    second = temp_dir / "wc_compare_b.txt"
    first.write_text("a\tb\nПривет мир\n", encoding="utf-8", newline="")
    second.write_text("alpha beta\nz\n", encoding="utf-8", newline="")

    strict = run(exe, ["--strict", "-l", "-w", "-m", "-L", str(first)])
    expect_success(strict, strict.stdout.strip())
    wc_strict = subprocess.run([wc_path, "-l", "-w", "-m", "-L", str(first)], capture_output=True, text=True, check=False)
    if wc_strict.returncode != 0:
        fail(f"system wc failed: {wc_strict.stderr.strip()!r}")
    if strict.stdout.split(maxsplit=4)[:4] != wc_strict.stdout.split(maxsplit=4)[:4]:
        fail(f"strict output diverges from wc: ours={strict.stdout!r} wc={wc_strict.stdout!r}")

    classic = run(exe, ["--lines", "--words", "--bytes", str(first), str(second)])
    expect_success(classic, classic.stdout.strip())
    wc_classic = subprocess.run([wc_path, "-l", "-w", "-c", str(first), str(second)], capture_output=True, text=True, check=False)
    if wc_classic.returncode != 0:
        fail(f"system wc failed: {wc_classic.stderr.strip()!r}")
    if classic.stdout.split()[:9] != wc_classic.stdout.split()[:9]:
        fail(f"classic multi-file output diverges from wc: ours={classic.stdout!r} wc={wc_classic.stdout!r}")

    totals = run(exe, ["--lines", "--words", "--bytes", "--total=always", str(first), str(second)])
    expect_success(totals, totals.stdout.strip())
    if " total" not in totals.stdout:
        fail("expected explicit total row with --total=always")


def main() -> int:
    if len(sys.argv) != 2:
        fail("usage: run_tests.py <fastawc-binary>")
    exe = pathlib.Path(sys.argv[1]).resolve()
    if not exe.is_file():
        fail(f"binary not found: {exe}")

    with tempfile.TemporaryDirectory(prefix="fastawc-tests-") as temp_root:
        temp_dir = pathlib.Path(temp_root)

        compat = temp_dir / "compat.txt"
        compat.write_text("a\tb\nABCD\n", encoding="utf-8", newline="")
        expect_success(run(exe, ["-L", str(compat)]), f"4 {compat}")
        expect_success(run(exe, ["--strict", "-L", str(compat)]), f"9 {compat}")
        expect_success(run(exe, ["--mode=strict", "-L", str(compat)]), f"9 {compat}")

        expect_failure(run(exe, ["--badopt"]))

        stdin_bytes = 5 if os.name == "nt" else 4
        stdin_proc = run(exe, ["-l", "-w", "-c"], input_text="a b\n")
        expect_success(stdin_proc, f"1 2 {stdin_bytes}")

        dash_file = temp_dir / "-foo.txt"
        dash_file.write_text("x y\n", encoding="utf-8", newline="")
        expect_success(run(exe, ["--", str(dash_file)]), f"1 2 4 {dash_file}")

        env_parallel = {
            "FASTAWC_THREADS": "2",
            "FASTAWC_MIN_PARALLEL_MB": "1",
            "FASTAWC_BYTES_PER_WORKER_MB": "1",
            "FASTAWC_TARGET_CHUNK_MB": "2",
        }

        nbsp_file = temp_dir / "nbsp_boundary.txt"
        make_boundary_file(nbsp_file, "\u00A0".encode("utf-8"))
        expect_success(run(exe, ["--strict", "-w", str(nbsp_file)], env_parallel), f"2 {nbsp_file}")

        ideographic_space_file = temp_dir / "ideo_boundary.txt"
        make_boundary_file(ideographic_space_file, "\u3000".encode("utf-8"))
        expect_success(run(exe, ["--strict", "-w", str(ideographic_space_file)], env_parallel), f"2 {ideographic_space_file}")

        long_line = temp_dir / "long_line.txt"
        prefix = "a" * (CHUNK_ALIGNMENT - 1)
        suffix_spaces = " " * CHUNK_ALIGNMENT
        long_line.write_text(prefix + "\t表" + suffix_spaces + "\n", encoding="utf-8", newline="")
        expected_width = str((CHUNK_ALIGNMENT - 1) + 1 + 2 + CHUNK_ALIGNMENT)
        expect_success(run(exe, ["--strict", "-L", str(long_line)], env_parallel), f"{expected_width} {long_line}")

        devanagari = temp_dir / "devanagari.txt"
        devanagari.write_text("\u0915\u093f\n", encoding="utf-8", newline="")
        expect_success(run(exe, ["--strict", "-m", "-L", str(devanagari)]), f"3 1 {devanagari}")

        zwj = temp_dir / "zwj.txt"
        zwj.write_text("a\u200db\n", encoding="utf-8", newline="")
        expect_success(run(exe, ["--strict", "-L", str(zwj)]), f"2 {zwj}")
        expect_success(run(exe, ["--strict", "-L", str(zwj)], {"FASTAWC_REPEAT": "3"}), f"2 {zwj}")

        mahjong = temp_dir / "mahjong.txt"
        mahjong.write_text("\U0001F004\n", encoding="utf-8", newline="")
        expect_success(run(exe, ["--strict", "-m", "-L", str(mahjong)]), f"2 2 {mahjong}")

        emoji_mixed = temp_dir / "emoji_mixed.txt"
        emoji_mixed.write_text("a\U0001F600b\n", encoding="utf-8", newline="")
        expect_success(run(exe, ["--strict", "-m", "-L", str(emoji_mixed)]), f"4 4 {emoji_mixed}")

        cyrillic_combining = temp_dir / "cyrillic_combining.txt"
        cyrillic_combining.write_text("a\u0483b\n", encoding="utf-8", newline="")
        expect_success(run(exe, ["--strict", "-L", str(cyrillic_combining)]), f"2 {cyrillic_combining}")

        maybe_compare_with_wc(exe, temp_dir)

    print("fastawc regression tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
