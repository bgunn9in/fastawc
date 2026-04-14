# TODO

## P0 Correctness And Semantics
- Fix strict-mode chunk boundary state for parallel mapped scans.
  Right now `initialPrevSpaceBit` in [sources/main.cpp](/E:/developing/fastawc/sources/main.cpp:218) is derived from `is_space_ascii(previous_byte)`. That is wrong for `--strict -w` when the previous code point is non-ASCII whitespace such as `NBSP`, `U+200x`, `U+3000`, or when the boundary sits after a multibyte code point. The strict path needs boundary carry state derived from the previous decoded code point, not the previous raw byte.
- Add explicit regression tests for strict-mode chunk boundaries.
  Cover UTF-8 whitespace across chunk boundaries, combining marks, wide characters, trailing incomplete UTF-8, and long lines spanning multiple chunks.
- Audit strict-mode display width on Windows.
  The hand-written `is_combining_mark` and `is_wide_codepoint` tables in [headers/engine_impl.h](/E:/developing/fastawc/headers/engine_impl.h) are intentionally incomplete. Verify which cases still diverge from `wcwidth`-style behavior and either expand the fast tables or clearly document the limits.

## P1 Biggest Speed Wins
- Rework strict-mode AVX2 so it stops falling back to scalar decoding for every non-ASCII 32/128-byte block.
  Current strict AVX2 in [headers/engine_impl.h](/E:/developing/fastawc/headers/engine_impl.h) mostly wins only on ASCII regions. The next real speedup is a vectorized UTF-8 structural scan that identifies ASCII-only runs, continuation masks, lead masks, and newline/tab masks before handing only the hard cases to scalar decode.
- Add an ASCII-run accelerator for strict mode at the chunk level, not only per 32-byte block.
  Strict mode already has a fast ASCII block path, but it still re-enters scalar often. Detect longer ASCII regions and process them with a specialized strict-ASCII kernel that handles `-w`, `-m`, and `-L` without per-byte UTF-8 state machinery.
- Split strict processing into three specialized families instead of one generic decoder.
  There should be separate strict kernels for:
  `-l -w -c`
  `-m -L`
  mixed full mode
  The current generic strict decoder pays for logic that some scan modes do not need.
- Reduce strict `-L` cost by making display-width handling staged.
  ASCII printable runs can stay purely vectorized. Only tabs, controls, combining marks, and wide code points should divert to the expensive exact path.

## P1 Backend And Runtime Selection
- Move backend selection from a static CPU decision to a workload-aware decision.
  Right now [sources/main.cpp](/E:/developing/fastawc/sources/main.cpp:583) uses a simple heuristic to force scalar for strict `-m/-L`. Push this logic into a proper runtime selector that considers:
  scan mode
  fast vs strict mode
  file size
  measured or configured crossover points
- Add per-mode runtime tuning defaults.
  `choose_runtime_config()` in [sources/runtime.cpp](/E:/developing/fastawc/sources/runtime.cpp) currently keys mostly off `scanMode` and backend ISA. It should also consider `ScanModeKind`, because strict mode has very different compute density from fast mode.
- Add optional startup micro-autotuning.
  A short opt-in calibration pass on a synthetic buffer could determine whether strict full mode is faster on scalar or AVX2 on the current CPU.

## P1 Parallel Execution
- Replace chunk-start ASCII whitespace bootstrap with richer carry state.
  Even after the correctness fix, the worker handoff should carry at least:
  previous decoded code point class
  whether a UTF-8 sequence is open
  current display-width state for line continuation
  That will make parallel strict mode both correct and easier to optimize.
- Consider a dedicated fixed-size work-stealing queue in `ThreadPool`.
  The current pool in [sources/thread_pool.cpp](/E:/developing/fastawc/sources/thread_pool.cpp) uses `deque + mutex + notify_all` per `parallel_for`. That is acceptable for large file chunks, but it is still heavier than necessary and makes small-task scaling worse.
- Add a "single large file" fast scheduler path.
  For the dominant workload, chunk descriptors can be precomputed and consumed from a lock-free index instead of pushing one task node per chunk.

## P2 Engine Structure
- Remove duplication between fast and strict scan engines.
  [headers/engine_impl.h](/E:/developing/fastawc/headers/engine_impl.h) now contains two largely parallel processor families. Refactor around reusable primitives:
  line counting
  ASCII whitespace transitions
  UTF-8 decode/state transitions
  display-width accumulation
  That will reduce maintenance risk and make future ISA work easier.
- Separate correctness-oriented decode utilities from hot AVX2 kernels.
  The file currently mixes parser-grade Unicode logic and SIMD kernels in one header. Split into smaller headers or namespaces so the hot path remains readable and easier to benchmark independently.
- Replace giant scan-mode switch tables with generated dispatch tables.
  The current manual `0x0 .. 0xF` processor switches are error-prone. Generate them once with constexpr tables for both fast and strict families.

## P2 SIMD Roadmap
- Add SSE4.2 backend for pre-AVX2 x86-64.
  That preserves a decent fast-mode implementation on machines that cannot run AVX2 but are still modern enough to benefit from SIMD.
- Investigate AVX-512 backend behind runtime dispatch.
  Especially for `fast classic` and ASCII-heavy `fast full`, AVX-512 may materially reduce instruction count and improve line/word throughput.
- Revisit whitespace classification.
  The current nibble-LUT approach is decent, but it may still be worth testing:
  packed compare blends
  alternative shuffle LUT layouts
  wider unroll factors
  instruction scheduling tuned for Intel Skylake-class cores

## P2 I/O And Memory Mapping
- Benchmark explicit large-page or huge-page mapping options where available.
  This may help on extremely large files, but only if the deployment environment allows it and the gains justify the complexity.
- Compare `FILE_FLAG_SEQUENTIAL_SCAN` / `madvise` variants with more aggressive readahead hints.
  This is workload-dependent and should be benchmark-driven.
- Add an option to disable mapping for apples-to-apples streaming benchmarks.
  That will make backend comparisons cleaner when investigating CPU-bound versus I/O-bound regimes.

## P3 Benchmarking And Validation
- Extend `bench_backends.py` with mode-aware summaries and CSV/JSON export.
  This will make it easier to compare `fast` vs `strict` regressions over time.
- Add a correctness test harness that diffs against system `wc` when available.
  This should cover:
  default mode
  `--strict`
  multiple files
  stdin
  long options
  total modes
- Add benchmark datasets that are deliberately different:
  ASCII-heavy prose
  UTF-8-heavy multilingual text
  whitespace-pathological input
  very long lines
  many short lines

## Suggested Execution Order
- Fix strict parallel chunk-boundary correctness first.
- Recover strict-mode performance with ASCII-run and vectorized UTF-8 structural scanning.
- Replace ad-hoc backend heuristics with workload-aware runtime selection.
- Refactor engine structure only after the hot-path behavior is stable and benchmarked.
