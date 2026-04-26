# fastawc
High-throughput `wc`-like utility focused on modern desktop x86-64 CPUs.

## Build
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

Optional build switches:
- `FASTAWC_ENABLE_LTO=ON|OFF`: enable link-time optimization for Release builds. Default: `ON`.
- `FASTAWC_ENABLE_AVX2_BACKEND=ON|OFF`: build the AVX2 backend. Default: `ON`.
- `FASTAWC_PGO_PHASE=off|generate|use`: optional profile-guided optimization phase. Default: `off`.

The build produces a single `fastawc` binary with:
- scalar backend for generic x86-64
- AVX2 backend built as a separate object target and selected at runtime when the CPU/OS support it
- Windows and POSIX file I/O paths with memory mapping for regular files and streaming fallback for stdin/pipes

### Optional PGO Build
PGO is a manual two-stage workflow and should be compared against the normal LTO build on the target machine before use.

Instrumented build:
```bash
cmake -S . -B build-pgo -DCMAKE_BUILD_TYPE=Release -DFASTAWC_PGO_PHASE=generate
cmake --build build-pgo --config Release
```

Train with representative inputs:
```bash
build-pgo/fastawc --strict -l -w -c -m -L big.txt
build-pgo/fastawc -l -w -c -m -L big.txt
build-pgo/fastawc -m -L big.txt
```

Optimized rebuild using the collected profile:
```bash
cmake -S . -B build-pgo -DCMAKE_BUILD_TYPE=Release -DFASTAWC_PGO_PHASE=use
cmake --build build-pgo --config Release
```

## Runtime Tuning
Automatic backend and parallelism selection is enabled by default.

Counting modes:
- `fast` is the default and prioritizes throughput
- `strict` is slower and aims to be closer to `wc` semantics for `-w`, `-m`, and `-L`

On POSIX systems, strict display-width counting initializes `LC_CTYPE` from the user's environment before calling `wcwidth()`, matching system `wc` locale behavior more closely. Windows uses the built-in Unicode width tables.

Optional environment overrides:
- `FASTAWC_BACKEND=scalar|avx2`
- `FASTAWC_AUTOTUNE=1`
- `FASTAWC_THREADS=<n>`
- `FASTAWC_MIN_PARALLEL_MB=<n>`
- `FASTAWC_BYTES_PER_WORKER_MB=<n>`
- `FASTAWC_TARGET_CHUNK_MB=<n>`
- `FASTAWC_NO_MMAP=1`
- `FASTAWC_STREAM_BUFFER_MB=<n>`
- `FASTAWC_WILLNEED=1`

`FASTAWC_AUTOTUNE=1` runs a short startup calibration for strict `-m/-L` workloads and picks the faster backend for the current CPU between `scalar` and `avx2`.
`FASTAWC_NO_MMAP=1` forces regular files through the streaming path for I/O comparison benchmarks.
`FASTAWC_STREAM_BUFFER_MB=<n>` changes the streaming read buffer size for stdin, pipes, and no-mmap file benchmarks.
`FASTAWC_WILLNEED=1` enables aggressive OS readahead hints where supported.

## CLI Options
- `-l`, `--lines`: print newline counts.
- `-w`, `--words`: print word counts.
- `-c`, `--bytes`: print byte counts.
- `-m`, `--chars`: print character counts.
- `-L`, `--max-line-length`: print maximum display width.
- `--fast`: use fast counting semantics. This is the default.
- `--strict`: use stricter `wc`-compatible counting for words, characters, and display width.
- `--mode=fast|strict`: select counting mode explicitly.
- `--total=auto|always|only|never`: control total row output.
- `--speed`: append processing throughput as `MiB/s`.
- `--help`: display usage.
- `--version`: print version information.

When `--speed` is enabled, throughput is computed from bytes processed by `fastawc` and elapsed time inside the process. For mapped regular files this uses the known file size without adding byte-count work to the scan path unless `-c` is also requested.

## Notes
- Target floor for the fast backend is AVX2-class CPUs such as Intel Core i7-6700.
- `big.7z` is sample benchmark data.
- `bench_backends.py` benchmarks `auto`, `scalar` and `avx2` backends on Windows and POSIX systems.

## Benchmarking
Basic run:
```bash
python bench_backends.py --file big.txt
```

`bench_backends.py` passes `--speed` to benchmarked `fastawc` runs by default and reports average `MiB/s` when the binary prints it. Use `--no-speed` when comparing against an older binary that does not support throughput output.

In benchmark reports, `avg ms` is wall-clock time measured by the Python harness around the subprocess. `avg MiB/s` is parsed from `fastawc --speed` output and reflects the tool's internal processing interval.

Strict compatibility run:
```bash
build/Release/fastawc --strict -l -w -c -m -L big.txt
```

Show processing throughput:
```bash
build/Release/fastawc --speed -l -w -c big.txt
```

Generate a benchmark file automatically:
```bash
python bench_backends.py --generate-test-file --generate-size-mb 256 --generate-profile ascii
```

Lower-noise run with warmup, round-robin backend order and CPU pinning:
```bash
python bench_backends.py --file big.txt --runs 10 --warmup 2 --interleave --affinity 0-7
```

Write machine-readable reports:
```bash
python bench_backends.py --file big.txt --scenarios strict-full --json-out bench.json --csv-out bench.csv
```

Compare two binaries:
```bash
python bench_backends.py --baseline-binary old/fastawc --candidate-binary build/Release/fastawc --file big.txt --scenarios classic strict-full --noise-pct 2
```

Split benchmark by workload shape:
```bash
python bench_backends.py --file big.txt --runs 5 --warmup 1 --interleave --affinity 0-7 --scenarios full classic unicode bytes
```

Predefined scenarios:
- `full`: `-l -w -c -m -L`
- `classic`: `-l -w -c`
- `lines`: `-l`
- `unicode`: `-m -L`
- `bytes`: `-c`
- `fast-full`: `-l -w -c -m -L`
- `fast-classic`: `-l -w -c`
- `fast-lines`: `-l`
- `strict-full`: `--strict -l -w -c -m -L`
- `strict-classic`: `--strict -l -w -c`
- `strict-lines`: `--strict -l`

Generated test data profiles:
- `ascii`: ASCII-heavy English text
- `mixed`: mostly ASCII with periodic UTF-8 lines
- `utf8`: UTF-8-heavy text
- `whitespace`: whitespace-pathological input with tabs and Unicode spaces
- `longlines`: long-line-heavy input for `-L` stress
- `shortlines`: many short lines for newline-density stress
- `cyrillic`: mostly 2-byte Cyrillic UTF-8 text
- `cjk`: mostly 3-byte CJK text with wide display-width characters
- `emoji`: 4-byte UTF-8-heavy text for supplementary-plane stress
- `tabs`: ASCII tab-heavy text for strict `-L`
- `controls`: ASCII control-heavy text for zero-width handling
- `nospaces`: long words with rare whitespace
- `dense-newlines`: very short lines for newline-density stress
