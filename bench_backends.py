#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import csv
import dataclasses
import json
import os
import re
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
DEFAULT_WHITESPACE_LINE = "word\t \u00a0\u3000word  \t\u200dword\n"
DEFAULT_SHORT_LINE = "x\n"
DEFAULT_LONG_LINE = ("The quick brown fox jumps over the lazy dog 1234567890 " * 256) + "\n"
DEFAULT_CYRILLIC_LINE = "Быстрый счётчик строк и слов проверяет кириллицу 12345\n"
DEFAULT_CJK_LINE = "快速文本计数器处理中文和日本語の行 12345\n"
DEFAULT_EMOJI_LINE = "emoji 😀😃😄😁😆 text 🧪📚🚀 words\n"
DEFAULT_TABS_LINE = "col1\tcol2\tcol3\tcol4\tcol5\n"
DEFAULT_CONTROLS_LINE = "alpha\x01beta\x02gamma\x7fdelta\n"
DEFAULT_NOSPACES_LINE = ("abcdefghijklmnopqrstuvwxyz0123456789" * 32) + "\n"
DEFAULT_DENSE_NEWLINES_LINE = "a\nb\nc\nd\ne\nf\ng\nh\n"
SPEED_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s+MiB/s")
SCENARIOS: dict[str, tuple[str, ...]] = {
	"full": DEFAULT_ARGS,
	"classic": ("-l", "-w", "-c"),
	"lines": ("-l",),
	"unicode": ("-m", "-L"),
	"bytes": ("-c",),
	"fast-full": DEFAULT_ARGS,
	"fast-classic": ("-l", "-w", "-c"),
	"fast-lines": ("-l",),
	"strict-full": ("--strict", "-l", "-w", "-c", "-m", "-L"),
	"strict-classic": ("--strict", "-l", "-w", "-c"),
	"strict-lines": ("--strict", "-l"),
}


@dataclasses.dataclass(frozen=True)
class RunSample:
    elapsed_ms: float
    speed_mib_s: float | None


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
    parser.add_argument("--baseline-binary", help="Path to a baseline fastawc binary for comparison mode.")
    parser.add_argument(
        "--candidate-binary",
        help="Path to a candidate fastawc binary for comparison mode. Defaults to --binary or auto-detection.",
    )
    parser.add_argument(
        "--noise-pct",
        type=float,
        default=2.0,
        help="Delta threshold treated as noise in comparison mode. Default: %(default)s",
    )
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
        choices=(
            "ascii",
            "mixed",
            "utf8",
            "whitespace",
            "longlines",
            "shortlines",
            "cyrillic",
            "cjk",
            "emoji",
            "tabs",
            "controls",
            "nospaces",
            "dense-newlines",
        ),
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
    parser.add_argument(
        "--no-speed",
        action="store_true",
        help="Do not pass --speed to fastawc. By default benchmarked commands include --speed.",
    )
    parser.add_argument("--json-out", help="Write benchmark results as JSON.")
    parser.add_argument("--csv-out", help="Write benchmark results as CSV.")
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


def parse_speed(stdout: str) -> float | None:
    matches = SPEED_RE.findall(stdout)
    if not matches:
        return None
    return float(matches[-1])


def run_once(binary: Path, input_file: Path, backend: str, extra_args: list[str], affinity: list[int] | None) -> RunSample:
    command = [str(binary), *extra_args, str(input_file)]
    env = os.environ.copy()
    if backend == "auto":
        env.pop("FASTAWC_BACKEND", None)
    else:
        env["FASTAWC_BACKEND"] = backend

    started = time.perf_counter()
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
    )
    try:
        if affinity is not None:
            set_process_affinity(proc.pid, affinity)
        stdout, stderr = proc.communicate()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if proc.returncode != 0:
        message = (stderr or "").strip() or f"exit code {proc.returncode}"
        raise RuntimeError(f"{backend}: {message}")
    return RunSample(elapsed_ms=elapsed_ms, speed_mib_s=parse_speed(stdout or ""))


def benchmark_backends(
    binary: Path,
    input_file: Path,
    backends: list[str],
    runs: int,
    warmup: int,
    extra_args: list[str],
    interleave: bool,
    affinity: list[int] | None,
) -> dict[str, list[RunSample]]:
    results: dict[str, list[RunSample]] = {backend: [] for backend in backends}

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


def sample_timings(samples: list[RunSample]) -> list[float]:
    return [sample.elapsed_ms for sample in samples]


def sample_speeds(samples: list[RunSample]) -> list[float]:
    return [sample.speed_mib_s for sample in samples if sample.speed_mib_s is not None]


def print_results(results: dict[str, list[RunSample]]) -> None:
    has_speed = any(sample.speed_mib_s is not None for timings in results.values() for sample in timings)
    header = f"{'backend':<10} {'runs':>4} {'min ms':>12} {'avg ms':>12} {'max ms':>12}"
    if has_speed:
        header += f" {'avg MiB/s':>12}"
    print(header)
    print("-" * len(header))
    for backend, timings in results.items():
        elapsed = sample_timings(timings)
        speeds = sample_speeds(timings)
        row = (
            f"{backend:<10} "
            f"{len(timings):>4} "
            f"{min(elapsed):>12.2f} "
            f"{statistics.fmean(elapsed):>12.2f} "
            f"{max(elapsed):>12.2f}"
        )
        if has_speed:
            row += f" {statistics.fmean(speeds):>12.2f}" if speeds else f" {'':>12}"
        print(row)


def make_summary(results: dict[str, list[RunSample]]) -> dict[str, dict[str, float | int | None]]:
    summary: dict[str, dict[str, float | int | None]] = {}
    for backend, samples in results.items():
        elapsed = sample_timings(samples)
        speeds = sample_speeds(samples)
        summary[backend] = {
            "runs": len(samples),
            "min_ms": min(elapsed),
            "avg_ms": statistics.fmean(elapsed),
            "max_ms": max(elapsed),
            "avg_mib_s": statistics.fmean(speeds) if speeds else None,
        }
    return summary


def make_comparison_summary(
    baseline: dict[str, dict[str, float | int | None]],
    candidate: dict[str, dict[str, float | int | None]],
    noise_pct: float,
) -> dict[str, dict[str, float | str | None]]:
    comparison: dict[str, dict[str, float | str | None]] = {}
    for backend, baseline_metrics in baseline.items():
        candidate_metrics = candidate[backend]
        baseline_avg = float(baseline_metrics["avg_ms"])
        candidate_avg = float(candidate_metrics["avg_ms"])
        baseline_speed = baseline_metrics.get("avg_mib_s")
        candidate_speed = candidate_metrics.get("avg_mib_s")
        delta_ms = candidate_avg - baseline_avg
        delta_pct = (delta_ms / baseline_avg * 100.0) if baseline_avg != 0.0 else 0.0
        if abs(delta_pct) <= noise_pct:
            status = "noise"
        else:
            status = "faster" if delta_ms < 0.0 else "slower"
        comparison[backend] = {
            "baseline_avg_ms": baseline_avg,
            "candidate_avg_ms": candidate_avg,
            "delta_ms": delta_ms,
            "delta_pct": delta_pct,
            "status": status,
            "baseline_avg_mib_s": float(baseline_speed) if baseline_speed is not None else None,
            "candidate_avg_mib_s": float(candidate_speed) if candidate_speed is not None else None,
        }
    return comparison


def print_comparison(comparison: dict[str, dict[str, float | str | None]]) -> None:
    header = f"{'backend':<10} {'base avg':>12} {'cand avg':>12} {'delta ms':>12} {'delta %':>10} {'status':>8}"
    has_speed = any(metrics.get("baseline_avg_mib_s") is not None or metrics.get("candidate_avg_mib_s") is not None for metrics in comparison.values())
    if has_speed:
        header += f" {'base MiB/s':>12} {'cand MiB/s':>12}"
    print(header)
    print("-" * len(header))
    for backend, metrics in comparison.items():
        row = (
            f"{backend:<10} "
            f"{float(metrics['baseline_avg_ms']):>12.2f} "
            f"{float(metrics['candidate_avg_ms']):>12.2f} "
            f"{float(metrics['delta_ms']):>12.2f} "
            f"{float(metrics['delta_pct']):>9.2f}% "
            f"{str(metrics['status']):>8}"
        )
        if has_speed:
            base_speed = metrics.get("baseline_avg_mib_s")
            candidate_speed = metrics.get("candidate_avg_mib_s")
            row += (
                f" {float(base_speed):>12.2f}" if base_speed is not None else f" {'':>12}"
            )
            row += (
                f" {float(candidate_speed):>12.2f}" if candidate_speed is not None else f" {'':>12}"
            )
        print(row)


def write_json_report(
    output_path: Path,
    *,
    binary: object,
    input_file: Path,
    runs: int,
    warmup: int,
    interleave: bool,
    affinity: list[int] | None,
    scenarios: list[dict[str, object]],
) -> None:
    payload = {
        "binary": str(binary) if isinstance(binary, Path) else binary,
        "file": str(input_file),
        "runs": runs,
        "warmup": warmup,
        "order": "interleaved" if interleave else "grouped",
        "affinity": affinity,
        "scenarios": scenarios,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv_report(output_path: Path, scenarios: list[dict[str, object]]) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scenario",
            "backend",
            "binary_role",
            "runs",
            "min_ms",
            "avg_ms",
            "max_ms",
            "avg_mib_s",
            "delta_ms",
            "delta_pct",
            "status",
        ])
        for scenario in scenarios:
            scenario_name = scenario["name"]
            if "summary" in scenario:
                summary = scenario["summary"]
                for backend, metrics in summary.items():
                    writer.writerow([
                        scenario_name,
                        backend,
                        "single",
                        metrics["runs"],
                        f"{metrics['min_ms']:.2f}",
                        f"{metrics['avg_ms']:.2f}",
                        f"{metrics['max_ms']:.2f}",
                        f"{metrics['avg_mib_s']:.2f}" if metrics.get("avg_mib_s") is not None else "",
                        "",
                        "",
                        "",
                    ])
            else:
                baseline_summary = scenario["baseline_summary"]
                candidate_summary = scenario["candidate_summary"]
                comparison = scenario["comparison"]
                for role, summary in (("baseline", baseline_summary), ("candidate", candidate_summary)):
                    for backend, metrics in summary.items():
                        delta = comparison[backend] if role == "candidate" else None
                        writer.writerow([
                            scenario_name,
                            backend,
                            role,
                            metrics["runs"],
                            f"{metrics['min_ms']:.2f}",
                            f"{metrics['avg_ms']:.2f}",
                            f"{metrics['max_ms']:.2f}",
                            f"{metrics['avg_mib_s']:.2f}" if metrics.get("avg_mib_s") is not None else "",
                            f"{float(delta['delta_ms']):.2f}" if delta else "",
                            f"{float(delta['delta_pct']):.2f}" if delta else "",
                            delta["status"] if delta else "",
                        ])


def resolve_scenarios(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    if args.scenarios:
        return [(name, list(SCENARIOS[name])) for name in args.scenarios]
    return [("custom", list(args.args))]


def benchmark_args(args: argparse.Namespace, scenario_args: list[str]) -> list[str]:
    if args.no_speed or "--speed" in scenario_args:
        return scenario_args
    return ["--speed", *scenario_args]


def build_profile_chunk(profile: str) -> bytes:
    if profile == "ascii":
        return DEFAULT_TEXT_LINE.encode("utf-8") * 4096
    if profile == "utf8":
        return DEFAULT_UTF8_LINE.encode("utf-8") * 4096
    if profile == "mixed":
        return (DEFAULT_TEXT_LINE * 3 + DEFAULT_UTF8_LINE).encode("utf-8") * 2048
    if profile == "whitespace":
        return (DEFAULT_WHITESPACE_LINE * 4096).encode("utf-8")
    if profile == "longlines":
        return (DEFAULT_LONG_LINE * 128).encode("utf-8")
    if profile == "shortlines":
        return (DEFAULT_SHORT_LINE * (1 << 16)).encode("utf-8")
    if profile == "cyrillic":
        return (DEFAULT_CYRILLIC_LINE * 4096).encode("utf-8")
    if profile == "cjk":
        return (DEFAULT_CJK_LINE * 4096).encode("utf-8")
    if profile == "emoji":
        return (DEFAULT_EMOJI_LINE * 4096).encode("utf-8")
    if profile == "tabs":
        return (DEFAULT_TABS_LINE * 8192).encode("utf-8")
    if profile == "controls":
        return (DEFAULT_CONTROLS_LINE * 8192).encode("utf-8")
    if profile == "nospaces":
        return (DEFAULT_NOSPACES_LINE * 512).encode("utf-8")
    if profile == "dense-newlines":
        return (DEFAULT_DENSE_NEWLINES_LINE * (1 << 14)).encode("utf-8")
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
    comparison_mode = args.baseline_binary is not None
    if comparison_mode:
        baseline_binary = resolve_binary(args.baseline_binary)
        candidate_binary = resolve_binary(args.candidate_binary or args.binary)
        binary: Path | None = None
    else:
        baseline_binary = None
        candidate_binary = None
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

    if comparison_mode:
        print(f"baseline : {baseline_binary}")
        print(f"candidate: {candidate_binary}")
        print(f"noise pct: {args.noise_pct:.2f}")
    else:
        print(f"binary : {binary}")
    print(f"file   : {input_file.resolve()}")
    if generated_info is not None:
        generated_bytes, generated_profile = generated_info
        print(f"generated: {generated_bytes} bytes ({generated_profile})")
    print(f"runs   : {args.runs}")
    print(f"warmup : {args.warmup}")
    print(f"order  : {'interleaved' if args.interleave else 'grouped'}")
    print(f"speed  : {'off' if args.no_speed else 'on'}")
    if affinity is not None:
        print(f"affinity: {','.join(str(cpu) for cpu in affinity)}")
    print()

    scenario_reports: list[dict[str, object]] = []
    for index, (scenario_name, scenario_args) in enumerate(scenarios):
        command_args = benchmark_args(args, scenario_args)
        if index != 0:
            print()
        print(f"scenario: {scenario_name}")
        print(f"args    : {' '.join(command_args)}")
        print()
        if comparison_mode:
            assert baseline_binary is not None
            assert candidate_binary is not None
            baseline_results = benchmark_backends(
                binary=baseline_binary,
                input_file=input_file,
                backends=args.backends,
                runs=args.runs,
                warmup=args.warmup,
                extra_args=command_args,
                interleave=args.interleave,
                affinity=affinity,
            )
            candidate_results = benchmark_backends(
                binary=candidate_binary,
                input_file=input_file,
                backends=args.backends,
                runs=args.runs,
                warmup=args.warmup,
                extra_args=command_args,
                interleave=args.interleave,
                affinity=affinity,
            )
            baseline_summary = make_summary(baseline_results)
            candidate_summary = make_summary(candidate_results)
            comparison = make_comparison_summary(baseline_summary, candidate_summary, args.noise_pct)
            print_comparison(comparison)
            scenario_reports.append({
                "name": scenario_name,
                "args": command_args,
                "baseline_summary": baseline_summary,
                "candidate_summary": candidate_summary,
                "comparison": comparison,
            })
        else:
            assert binary is not None
            results = benchmark_backends(
                binary=binary,
                input_file=input_file,
                backends=args.backends,
                runs=args.runs,
                warmup=args.warmup,
                extra_args=command_args,
                interleave=args.interleave,
                affinity=affinity,
            )
            print_results(results)
            scenario_reports.append({
                "name": scenario_name,
                "args": command_args,
                "summary": make_summary(results),
            })

    if args.json_out:
        binary_report: object
        if comparison_mode:
            binary_report = {
                "baseline": str(baseline_binary),
                "candidate": str(candidate_binary),
            }
        else:
            binary_report = binary
        write_json_report(
            Path(args.json_out).expanduser(),
            binary=binary_report,
            input_file=input_file.resolve(),
            runs=args.runs,
            warmup=args.warmup,
            interleave=args.interleave,
            affinity=affinity,
            scenarios=scenario_reports,
        )
    if args.csv_out:
        write_csv_report(Path(args.csv_out).expanduser(), scenario_reports)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
