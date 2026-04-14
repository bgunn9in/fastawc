#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_BACKENDS = ("auto", "scalar", "avx2")
DEFAULT_ARGS = ("-l", "-w", "-c", "-m", "-L")
DEFAULT_GENERATED_SIZE_MB = 256
DEFAULT_TEXT_LINE = "The quick brown fox jumps over the lazy dog 1234567890\n"
DEFAULT_UTF8_LINE = "Привет мир 12345\n"
SCENARIOS: dict[str, tuple[str, ...]] = {
    "full": DEFAULT_ARGS,
    "classic": ("-l", "-w", "-c"),
    "unicode": ("-m", "-L"),
    "bytes": ("-c",),
    "fast-full": DEFAULT_ARGS,
    "fast-classic": ("-l", "-w", "-c"),
    "strict-full": ("--strict", "-l", "-w", "-c", "-m", "-L"),
    "strict-classic": ("--strict", "-l", "-w", "-c"),
}


def resolve_binary(explicit: str | None) -> Path:
    if explicit is not None:
        binary = Path(explicit).expanduser()
        if binary.is_file():
            return binary.resolve()
        raise FileNotFoundError(f"binary not found: {binary}")

    candidates = (
        Path("build/Release/fastawc.exe"),
        Path("build/Release/fastawc"),
        Path("build/fastawc.exe"),
        Path("build/fastawc"),
        Path("Release/fastawc.exe"),
        Path("Release/fastawc"),
        Path("fastawc.exe"),
        Path("fastawc"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError("unable to locate fastawc binary; pass --binary")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark fastawc backends on a target file and report min/avg/max timings."
    )
    parser.add_argument("--binary", help="Path to the fastawc binary.")
    parser.add_argument(
        "--file",
        default="big.txt",
        help="Input file for benchmarking. Default: %(default)s",
    )
    parser.add_argument(
        "--generate-test-file",
        action="store_true",
        help="Generate the benchmark input file before running benchmarks.",
    )
    parser.add_argument(
        "--generate-size-mb",
        type=int,
        default=DEFAULT_GENERATED_SIZE_MB,
        help="Generated test file size in MiB. Default: %(default)s",
    )
    parser.add_argument(
        "--generate-profile",
        choices=("ascii", "mixed", "utf8"),
        default="ascii",
        help="Content profile for generated test data. Default: %(default)s",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of runs per backend. Default: %(default)s",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup runs per backend before measurements. Default: %(default)s",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=list(DEFAULT_BACKENDS),
        help="Backends to benchmark. Default: %(default)s",
    )
    parser.add_argument(
        "--interleave",
        action="store_true",
        help="Run backends round-robin to reduce cache/order bias.",
    )
    parser.add_argument(
        "--affinity",
        help="Pin benchmarked processes to logical CPUs, for example 0-7 or 0,2,4,6.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=tuple(SCENARIOS.keys()),
        help="Benchmark predefined argument groups instead of --args.",
    )
    parser.add_argument(
        "--args",
        nargs=argparse.REMAINDER,
        default=list(DEFAULT_ARGS),
        help="Arguments passed to fastawc before the input file. Default: -l -w -c -m -L",
    )
    return parser.parse_args()


def parse_cpu_set(spec: str) -> list[int]:
    cpus: set[int] = set()
    for part in spec.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            begin_text, end_text = token.split("-", 1)
            begin = int(begin_text)
            end = int(end_text)
            if begin < 0 or end < begin:
                raise ValueError(f"invalid CPU range: {token}")
            cpus.update(range(begin, end + 1))
        else:
            cpu = int(token)
            if cpu < 0:
                raise ValueError(f"invalid CPU index: {token}")
            cpus.add(cpu)

    if not cpus:
        raise ValueError("affinity set is empty")

    return sorted(cpus)


def set_process_affinity(pid: int, cpus: list[int]) -> None:
    if os.name == "nt":
        mask = 0
        for cpu in cpus:
            mask |= 1 << cpu

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        access = 0x0200 | 0x0400
        handle = kernel32.OpenProcess(access, False, pid)
        if not handle:
            raise OSError(ctypes.get_last_error(), f"OpenProcess failed for pid {pid}")
        try:
            if not kernel32.SetProcessAffinityMask(handle, ctypes.c_size_t(mask)):
                raise OSError(ctypes.get_last_error(), f"SetProcessAffinityMask failed for pid {pid}")
        finally:
            kernel32.CloseHandle(handle)
        return

    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(pid, cpus)
        return

    raise NotImplementedError("CPU affinity is not supported on this platform")


def run_once(binary: Path, input_file: Path, backend: str, extra_args: list[str], affinity: list[int] | None) -> float:
    command = [str(binary), *extra_args, str(input_file)]
    env = os.environ.copy()
    if backend == "auto":
        env.pop("FASTAWC_BACKEND", None)
    else:
        env["FASTAWC_BACKEND"] = backend

    started = time.perf_counter()
    proc = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
    )
    try:
        if affinity is not None:
            set_process_affinity(proc.pid, affinity)
        _, stderr = proc.communicate()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if proc.returncode != 0:
        message = (stderr or "").strip() or f"exit code {proc.returncode}"
        raise RuntimeError(f"{backend}: {message}")
    return elapsed_ms


def benchmark_backends(
    binary: Path,
    input_file: Path,
    backends: list[str],
    runs: int,
    warmup: int,
    extra_args: list[str],
    interleave: bool,
    affinity: list[int] | None,
) -> dict[str, list[float]]:
    results: dict[str, list[float]] = {backend: [] for backend in backends}

    if interleave:
        for round_index in range(warmup + runs):
            for backend in backends:
                elapsed_ms = run_once(binary, input_file, backend, extra_args, affinity)
                if round_index >= warmup:
                    results[backend].append(elapsed_ms)
    else:
        for backend in backends:
            for _ in range(warmup):
                run_once(binary, input_file, backend, extra_args, affinity)
            for _ in range(runs):
                results[backend].append(run_once(binary, input_file, backend, extra_args, affinity))

    return results


def print_results(results: dict[str, list[float]]) -> None:
    header = f"{'backend':<10} {'runs':>4} {'min ms':>12} {'avg ms':>12} {'max ms':>12}"
    print(header)
    print("-" * len(header))
    for backend, timings in results.items():
        print(
            f"{backend:<10} "
            f"{len(timings):>4} "
            f"{min(timings):>12.2f} "
            f"{statistics.fmean(timings):>12.2f} "
            f"{max(timings):>12.2f}"
        )


def resolve_scenarios(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    if args.scenarios:
        return [(name, list(SCENARIOS[name])) for name in args.scenarios]
    return [("custom", list(args.args))]


def build_profile_chunk(profile: str) -> bytes:
    if profile == "ascii":
        return DEFAULT_TEXT_LINE.encode("utf-8") * 4096
    if profile == "utf8":
        return DEFAULT_UTF8_LINE.encode("utf-8") * 4096
    if profile == "mixed":
        return (DEFAULT_TEXT_LINE * 3 + DEFAULT_UTF8_LINE).encode("utf-8") * 2048
    raise ValueError(f"unknown generate profile: {profile}")


def generate_test_file(path: Path, size_mb: int, profile: str) -> tuple[int, str]:
    if size_mb <= 0:
        raise ValueError("--generate-size-mb must be > 0")

    target_size = size_mb << 20
    chunk = build_profile_chunk(profile)
    written = 0

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        while written + len(chunk) <= target_size:
            f.write(chunk)
            written += len(chunk)
        if written < target_size:
            tail = chunk[:target_size - written]
            f.write(tail)
            written += len(tail)

    return written, profile


def main() -> int:
    args = parse_args()
    binary = resolve_binary(args.binary)
    input_file = Path(args.file).expanduser()
    if args.runs <= 0:
        raise ValueError("--runs must be > 0")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")

    affinity = parse_cpu_set(args.affinity) if args.affinity else None
    scenarios = resolve_scenarios(args)

    generated_info: tuple[int, str] | None = None
    if args.generate_test_file:
        generated_info = generate_test_file(input_file, args.generate_size_mb, args.generate_profile)
    elif not input_file.is_file():
        raise FileNotFoundError(
            f"input file not found: {input_file}; pass --generate-test-file to create it automatically"
        )

    print(f"binary : {binary}")
    print(f"file   : {input_file.resolve()}")
    if generated_info is not None:
        generated_bytes, generated_profile = generated_info
        print(f"generated: {generated_bytes} bytes ({generated_profile})")
    print(f"runs   : {args.runs}")
    print(f"warmup : {args.warmup}")
    print(f"order  : {'interleaved' if args.interleave else 'grouped'}")
    if affinity is not None:
        print(f"affinity: {','.join(str(cpu) for cpu in affinity)}")
    print()

    for index, (scenario_name, scenario_args) in enumerate(scenarios):
        if index != 0:
            print()
        print(f"scenario: {scenario_name}")
        print(f"args    : {' '.join(scenario_args)}")
        print()
        results = benchmark_backends(
            binary=binary,
            input_file=input_file,
            backends=args.backends,
            runs=args.runs,
            warmup=args.warmup,
            extra_args=scenario_args,
            interleave=args.interleave,
            affinity=affinity,
        )
        print_results(results)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
