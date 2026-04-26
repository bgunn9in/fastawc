# TODO: Performance Roadmap

This document is intentionally focused on speed. Correctness work should be added here only when it blocks or validates a performance change.

## Measurement Rules

Every optimization below should be benchmarked before and after the change. Use at least:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

Recommended benchmark matrix:

```bash
python bench_backends.py --generate-test-file --generate-size-mb 256 --generate-profile ascii --file build/bench_ascii_256.txt --runs 10 --warmup 2 --interleave --scenarios bytes classic full strict-classic strict-full
python bench_backends.py --generate-test-file --generate-size-mb 256 --generate-profile mixed --file build/bench_mixed_256.txt --runs 10 --warmup 2 --interleave --scenarios classic full strict-classic strict-full unicode
python bench_backends.py --generate-test-file --generate-size-mb 256 --generate-profile utf8 --file build/bench_utf8_256.txt --runs 10 --warmup 2 --interleave --scenarios classic full strict-classic strict-full unicode
python bench_backends.py --generate-test-file --generate-size-mb 256 --generate-profile whitespace --file build/bench_ws_256.txt --runs 10 --warmup 2 --interleave --scenarios strict-classic strict-full
python bench_backends.py --generate-test-file --generate-size-mb 256 --generate-profile longlines --file build/bench_long_256.txt --runs 10 --warmup 2 --interleave --scenarios full strict-full unicode
python bench_backends.py --generate-test-file --generate-size-mb 256 --generate-profile shortlines --file build/bench_short_256.txt --runs 10 --warmup 2 --interleave --scenarios classic full strict-classic strict-full
```

For noisy machines, pin CPUs:

```bash
python bench_backends.py --file build/bench_ascii_256.txt --runs 15 --warmup 3 --interleave --affinity 0-7 --scenarios classic full strict-full
```

When comparing large changes, build a clean baseline from `HEAD~1` or a worktree and run the same benchmark command against both binaries.

Success criteria for merging a speed change:

- No regression in `ctest`.
- No measurable slowdown above noise on `ascii`, `mixed`, and `utf8` for the touched scenarios.
- Clear speedup in at least one named scenario, or a documented reason why the change is enabling infrastructure.
- Benchmark output saved or summarized in the commit/PR notes.

## P0: Benchmarking And Profiling Infrastructure

### [x] Add baseline comparison mode to `bench_backends.py`

Hypothesis: current measurements are useful but still manual. A built-in baseline/current comparator would make speed work less error-prone and reduce false positives.

Target files:

- `bench_backends.py`

Plan:

- Add `--baseline-binary` and `--candidate-binary`.
- Run the same scenario/backend matrix for both binaries.
- Print per-scenario deltas as percent and absolute milliseconds.
- Mark changes as `faster`, `slower`, or `noise` using a configurable threshold such as `--noise-pct 2.0`.
- Include deltas in JSON and CSV reports.

Validation:

- Compare current `build/Release/fastawc.exe` against itself and verify near-zero deltas.
- Compare a known older binary against the current optimized binary and verify the report exposes the strict Unicode width improvement.

Risk:

- Benchmark reports can become too complex. Keep the existing simple one-binary mode as the default.

Progress:

- Implemented `--baseline-binary`, `--candidate-binary`, and `--noise-pct`.
- Added comparison output with absolute and percent deltas.
- Extended JSON and CSV reports to include baseline/candidate summaries and delta status.

### [x] Add profile-guided benchmark modes

Hypothesis: the existing generated data profiles are good for broad checks, but they do not isolate several hot paths tightly enough.

Target files:

- `bench_backends.py`

New profiles to add:

- `cyrillic`: mostly 2-byte UTF-8 with spaces and newlines.
- `cjk`: mostly 3-byte wide characters with regular line breaks.
- `emoji`: 4-byte UTF-8-heavy data for strict fallback stress.
- `tabs`: ASCII text with frequent tabs for `-L`.
- `controls`: ASCII text with frequent zero-width controls.
- `nospaces`: long words with rare whitespace for word-transition stress.
- `dense-newlines`: very short lines for line-count and max-line merging stress.

Validation:

- Generated file sizes must match the requested MiB exactly.
- Add one smoke benchmark command for each new profile to README or benchmark docs.

Risk:

- Too many scenarios can slow routine benchmarking. Keep them opt-in.

Progress:

- Added `cyrillic`, `cjk`, `emoji`, `tabs`, `controls`, `nospaces`, and `dense-newlines` profiles.
- Documented the new generated profiles in README.

### [x] Add optional profiler-friendly execution mode

Hypothesis: profiling a subprocess loop adds noise. A mode that repeatedly scans the same mapped file in one process would make VTune, WPA, Linux `perf`, and sampling profilers more useful.

Target files:

- `sources/main.cpp`
- `bench_backends.py`

Plan:

- Add hidden or clearly experimental environment variable, for example `FASTAWC_REPEAT=<n>`.
- Process the same input repeatedly before printing one final result or suppressing output in benchmark mode.
- Ensure counts are not accidentally accumulated across iterations unless explicitly intended.

Validation:

- `FASTAWC_REPEAT=1` must match normal output.
- `FASTAWC_REPEAT=10` should be usable with `stdout` redirected and should not leak memory.

Risk:

- Hidden benchmark behavior must not affect normal CLI semantics.

Progress:

- Added `FASTAWC_REPEAT=<n>` to repeat regular-file processing in one process while keeping normal output shape.
- Kept stdin and pipe inputs single-pass because they cannot be rewound safely.
- Added regression coverage that repeated processing preserves visible counts.

### [x] Add throughput reporting to the CLI and benchmark reports

Hypothesis: benchmark runs should expose both elapsed subprocess time and the tool's own processing throughput so CPU/backend changes can be interpreted more directly.

Target files:

- `sources/main.cpp`
- `headers/types.h`
- `bench_backends.py`
- `tests/run_tests.py`
- `README.md`

Plan:

- Add `--speed` to print processing throughput in `MiB/s`.
- Keep normal output unchanged unless `--speed` is explicitly requested.
- Make `bench_backends.py` pass `--speed` by default.
- Parse throughput from benchmarked runs and include it in console, JSON, and CSV output.
- Keep `--no-speed` available for comparing against older binaries.

Validation:

- Add a regression test for `--speed` output format.
- Run C++ regression tests.
- Run benchmark smoke tests with default speed reporting and with `--no-speed`.

Risk:

- Throughput printing can distort small-file timings if it adds work to hot paths or forces extra stdout handling.

Progress:

- Added `--speed` CLI output as an extra field after selected count columns.
- For mapped regular files, throughput now uses the known mapped file size without forcing byte counting in the scan path unless `-c` is requested.
- `bench_backends.py` now enables `--speed` by default, reports `avg MiB/s`, and keeps `--no-speed` for compatibility.
- Comparison reports include baseline/candidate `MiB/s` plus absolute and percent throughput deltas.
- When `--no-speed` is used, `bench_backends.py` sends stdout to `DEVNULL` instead of reading a pipe.
- README documents `--speed`, benchmark throughput reporting, and the difference between Python wall-clock timing and internal `fastawc` throughput.

### [x] Refresh user-facing CLI documentation

Hypothesis: README had benchmark examples but no compact command-line reference, which made new options such as `--speed` easy to miss.

Target files:

- `README.md`

Plan:

- Add a concise list of counting, mode, total, speed, help, and version options.
- Document how `--speed` computes throughput.
- Keep detailed benchmark examples in the benchmarking section.

Validation:

- Review README for consistency with parser-supported options.

Risk:

- Documentation can drift if options change. Keep the section compact and option-focused.

Progress:

- Added a CLI Options section covering `-l/-w/-c/-m/-L`, `--fast`, `--strict`, `--mode`, `--total`, `--speed`, `--help`, and `--version`.
- Clarified that mapped regular-file throughput uses file size and does not add byte-count scan work unless `-c` is requested.

## P1: Strict AVX2 UTF-8 Hot Path

### Extend strict mixed-block fast path to valid 4-byte UTF-8

Hypothesis: `try_process_strict_short_utf8_mixed_block()` currently accepts common valid 2-byte and 3-byte sequences but rejects 4-byte sequences. Emoji-heavy and supplementary-plane input therefore falls back to scalar spans inside AVX2 blocks.

Target files:

- `headers/engine_impl.h`

Plan:

- Extend `try_process_strict_short_utf8_mixed_block()` to decode valid 4-byte sequences when they fit inside the 32-byte block.
- Validate lead byte range `0xF0..0xF4`.
- Enforce overlong and upper-bound constraints:
  - `F0` second byte must be `>= 0x90`.
  - `F4` second byte must be `<= 0x8F`.
  - all continuation bytes must be `0x80..0xBF`.
- Route decoded code points through `handle_codepoint()`, preserving strict `-m`, `-L`, and `-w` behavior.
- Keep block-boundary cases conservative and fall back to scalar if the sequence crosses the 32-byte boundary.

Validation:

- Add tests for `U+1F004`, a common emoji, and a 4-byte sequence split across a chunk boundary.
- Benchmark `utf8`, new `emoji`, and `mixed` profiles with `strict-full` and `unicode`.

Risk:

- Incorrect UTF-8 validity checks can change strict character counts. Prefer conservative fallback over accepting questionable sequences.

Progress:

- Investigated a direct 4-byte extension to the existing mixed-block fast path.
- Reverted the hot-path change after baseline comparison showed a strict-full AVX2 regression on the `emoji` profile.
- Kept regression coverage for mixed ASCII plus a 4-byte emoji.
- Next attempt should first add a cheap non-mutating prevalidation/structure mask so failed 32-byte attempts do not add a second pass over common mixed blocks.
- Added conservative structural prevalidation and 4-byte decoding for word-including strict AVX2 mixed blocks.
- Kept wordless `unicode` workloads on the previous fallback after benchmarks showed that 4-byte direct decoding did not help them reliably.
- Added a forced-parallel chunk-boundary regression for a 4-byte emoji under strict `-m -L`.
- Candidate outputs matched baseline on generated `mixed`, `utf8`, and `emoji` 128 MiB profiles for `strict-full`, `unicode`, and `strict-classic`.
- Rechecked the wordless `unicode` path after structural validation, dispatch cleanup, and scalar/autotune work.
- Broadened the structurally validated F0+ block shortcut to wordless strict char/max-line workloads as well.
- Candidate outputs matched the micro-baseline on generated `emoji`, `utf8`, `mixed`, `cjk`, and `ascii` 128 MiB profiles for `unicode`, `strict-full`, and `strict-classic`.
- Benchmark result summary against the pre-change micro-baseline:
  - `emoji unicode` improved by about 4.8% in subprocess wall-clock and about 5.3% in internal MiB/s, which is useful but still close to the configured 5% noise threshold.
  - `utf8`, `mixed`, `cjk`, and `ascii` neighboring checks stayed within noise.
  - Explicit scalar backend is unaffected by this AVX2-only change.

### [x] Vectorize UTF-8 structural classification before scalar fallback

Hypothesis: the current strict AVX2 path identifies non-ASCII bytes with `_mm256_movemask_epi8()` and then walks ASCII/non-ASCII spans. More structure can be extracted cheaply: lead-byte masks, continuation masks, and likely-valid 2/3/4-byte sequence starts.

Target files:

- `headers/engine_impl.h`

Plan:

- Build masks for:
  - continuation bytes,
  - 2-byte leads,
  - 3-byte leads,
  - 4-byte leads,
  - invalid lead bytes.
- Use shifted masks to identify self-contained valid sequences in a 32-byte block.
- Process ASCII subspans with the AVX2 ASCII block path where possible.
- Process valid non-ASCII sequences with a small decoded loop.
- Send only invalid or boundary-crossing fragments to the generic scalar decoder.

Validation:

- Compare against strict `wc` where available.
- Add malformed UTF-8 tests: overlong sequences, lone continuation bytes, truncated sequences, surrogate-range encodings.
- Benchmark `utf8`, `mixed`, `emoji`, and `whitespace` profiles.

Risk:

- This is a high-risk hot path. Implement behind a small helper and keep the existing scalar fallback intact.

Progress:

- Added unsigned byte-range mask helpers for AVX2 classification.
- Fixed the `>= F0` byte mask to use unsigned comparison semantics.
- Added structural masks for continuation bytes, 2-byte leads, 3-byte leads, 4-byte leads, invalid lead bytes, and invalid constrained second bytes.
- Added a conservative self-contained-block validator that rejects boundary-crossing and malformed sequences before the mutating decoder runs.
- Extended the strict short UTF-8 mixed-block decoder to handle valid 4-byte sequences after structural prevalidation.
- Initially allowing the old 2/3-byte decoder on ASCII+UTF-8 mixed blocks regressed `utf8 strict-full`; the path is now gated so the old 2/3-byte shortcut only handles fully non-ASCII blocks.
- The new 4-byte shortcut is currently enabled only for word-including strict workloads, where it showed a stable win on `emoji`.
- Regression checks versus clean baseline on 128 MiB generated profiles:
  - `mixed`: `strict-full`, `unicode`, and `strict-classic` stayed within noise.
  - `utf8`: `strict-full`, `unicode`, and `strict-classic` stayed within noise.
  - `emoji`: `strict-full` improved by about 9%, `strict-classic` improved by about 6%, and `unicode` stayed within noise.

### [x] Add a strict word-only AVX2 Unicode whitespace path

Hypothesis: strict `-l -w -c` does not need full Unicode display width or character width. It only needs newline count, byte count, ASCII word transitions, and a small set of Unicode whitespace code points. The current AVX2 mixed path still decodes too much for some non-ASCII input.

Target files:

- `headers/engine_impl.h`

Plan:

- Add a specialized strict AVX2 path for `CountWords=true`, `CountChars=false`, `CountMaxLine=false`.
- Detect UTF-8 encodings of Unicode whitespace directly:
  - `C2 85`, `C2 A0`
  - `E1 9A 80`
  - `E2 80 80..8A`, `E2 80 A8`, `E2 80 A9`, `E2 80 AF`
  - `E2 81 9F`, `E2 81 A0`
  - `E3 80 80`
- Treat all other valid and invalid non-ASCII sequences as non-space for word-transition purposes.
- Avoid calling `unicode_display_width()` and avoid full code-point decode unless needed for boundary correctness.

Validation:

- Add tests for every Unicode whitespace sequence above.
- Benchmark `strict-classic` on `mixed`, `utf8`, `whitespace`, and new `cyrillic` profiles.

Risk:

- Word-boundary state across chunks must remain correct. Reuse `ChunkBoundaryState` logic and add boundary tests for each Unicode whitespace length.

Progress:

- Added a conservative strict word-only AVX2 fast path for mixed blocks that contain non-ASCII bytes but no possible Unicode whitespace lead bytes.
- Blocks containing `C2`, `E1`, `E2`, or `E3` still fall back to the existing scalar strict word decoder, preserving Unicode whitespace handling for `U+0085`, `U+00A0`, `U+1680`, `U+2000..U+200A`, `U+2028`, `U+2029`, `U+202F`, `U+205F`, `U+2060`, and `U+3000`.
- Reused the existing AVX2 line/word block for the safe case where all non-ASCII bytes can be treated as non-space.
- Existing forced-parallel Unicode whitespace boundary tests pass.
- Candidate strict-classic outputs match baseline on generated `cyrillic`, `cjk`, `emoji`, and `mixed` 128 MiB profiles.
- On 128 MiB generated profiles versus clean baseline, strict-classic auto backend improved approximately:
  - `cyrillic`: 61.06 ms -> 30.20 ms.
  - `cjk`: 53.29 ms -> 41.94 ms.
  - `emoji`: 47.68 ms -> 24.42 ms.

## P1: Fast AVX2 ASCII And Classic Paths

### [x] Specialize `-c` byte-only mode even more aggressively

Hypothesis: `-c` on mapped files can be answered from file size without scanning. The current code already returns size when `scanMode == 0`, but byte counting is represented outside `scanMode`, so `-c` still goes through file processing paths unless `processor == nullptr`.

Target files:

- `sources/main.cpp`
- `headers/types.h`

Plan:

- Audit the current `make_scan_mode()` and `processor == nullptr` flow for `-c`.
- Ensure mapped regular files return byte count without allocating worker vectors or touching file pages.
- Ensure streaming input still reads the stream to count bytes.
- Consider making byte-only status explicit in a small helper such as `is_byte_only(options, scanMode)`.

Validation:

- Benchmark `bytes` on large mapped files.
- Test stdin byte counting still works.

Risk:

- Multiple-file totals and labels must stay unchanged.

Progress:

- Added a regular-file metadata fast path for byte-only mode before opening or memory-mapping the file.
- Kept stdin, pipes, and non-regular paths on the existing streaming fallback.

### [x] Optimize ASCII whitespace mask generation

Hypothesis: `mask_whitespace32()` uses two shuffle LUTs and constructs constants inside the helper. Compilers often hoist them, but generated code should be verified. A compare-based classifier or static constant load may be faster on some CPUs.

Target files:

- `headers/engine_impl.h`

Plan:

- Inspect MSVC and Clang/GCC assembly for `mask_whitespace32()`.
- Compare current nibble LUT against:
  - direct compares for `' '`, `'\t'..'\r'`,
  - packed range checks,
  - preloaded constants reused across blocks.
- Keep the fastest variant per compiler if differences are meaningful.

Validation:

- Benchmark `classic`, `full`, `strict-classic`, and `strict-full` on `ascii`, `shortlines`, and `nospaces`.

Risk:

- Some variants may be faster on one microarchitecture and slower on another. Prefer stable wins or compiler-specific branches only with evidence.

Progress:

- Replaced the AVX2 nibble-LUT ASCII whitespace classifier with a direct classifier:
  - equality compare for ASCII space,
  - unsigned range mask for `'\t'..'\r'`.
- Reused the unsigned byte-range helper added for UTF-8 structural classification.
- Built a micro-baseline from the already-validated current binary before this change and compared candidate against it.
- Candidate outputs matched micro-baseline on generated `ascii`, `mixed`, `tabs`, and `nospaces` 128 MiB profiles for `classic`, `full`, `strict-classic`, and `strict-full`.
- Benchmark result summary:
  - `ascii classic` improved by about 5%.
  - `tabs full` improved by about 6%.
  - `mixed`, `tabs`, and `nospaces` checked scenarios stayed within noise except wins already covered by the strict tab path.
  - No checked time regression exceeded the configured 5% threshold.

### [x] Add a newline-only AVX2 path for `-l`

Hypothesis: line-only scans do not need whitespace or UTF-8 lead masks. A dedicated loop can reduce instruction count and improve throughput on newline-heavy files.

Target files:

- `headers/engine_impl.h`

Plan:

- Add a specialized path for `CountLines=true`, all other scan bits false.
- Unroll by 128 or 256 bytes.
- Only perform newline compare, movemask, and popcount.
- Consider larger unroll if front-end overhead dominates.

Validation:

- Add a benchmark scenario for `lines`: `-l`.
- Benchmark `lines` on `ascii`, `shortlines`, and `dense-newlines`.

Risk:

- Dispatch table grows slightly. Keep the implementation small and isolated.

Progress:

- Audited the current AVX2 dispatch and confirmed `CountLines=true` with all other scan bits false already compiles down to a newline-only block path in both fast and strict modes.
- Added explicit `lines`, `fast-lines`, and `strict-lines` benchmark scenarios so this path is tracked.
- No new backend or dispatch-table expansion was needed.

## P1: Strict Display Width

### [x] Expand staged Windows display-width fast paths

Hypothesis: Windows strict `-L` now has fast returns for common fixed-width ranges, but many frequent scripts still fall through to binary-search tables.

Target files:

- `headers/engine_impl.h`
- `tests/run_tests.py`

Plan:

- Add fast returns for additional safe fixed-width ranges after verifying they do not include combining marks:
  - Greek and Coptic non-combining ranges.
  - Hebrew base letters excluding marks.
  - Arabic base letters excluding combining marks.
  - Hiragana and Katakana ranges already covered by wide table but can be returned early if safe.
- Keep zero-width ranges ahead of any broad range that contains combining marks.
- Add targeted regression tests for representative zero-width marks in any broadened script range.

Validation:

- Benchmark `strict-full` and `unicode` on new `cyrillic`, `cjk`, and mixed multilingual profiles.
- Compare outputs against system `wc` on POSIX when equivalent test data is available.

Risk:

- Unicode width tables are subtle. Only add ranges with high confidence and tests for exclusions.

Progress:

- Added Windows-only fixed-width fast returns for conservative Greek, Hebrew, Arabic, Arabic-Indic digit, Hiragana, and Katakana ranges.
- Kept combining marks and zero-width ranges out of those fast returns; the Hiragana/Katakana wide shortcut remains after `is_zero_width_codepoint()`.
- Added regression coverage for:
  - Greek base letters plus a combining acute accent,
  - Hebrew base letters plus a niqqud mark,
  - Arabic base letters plus a kasra mark,
  - Hiragana and Katakana wide display columns.
- Candidate outputs matched the pre-change micro-baseline on generated Greek, Hebrew, Arabic, Kana, Cyrillic, CJK, and mixed 128 MiB files for `--strict -m -L` and `--strict -l -w -c -m -L`.
- Benchmark result summary against the pre-change micro-baseline:
  - `--strict -m -L` improved by about 57-60% on generated Greek, Hebrew, and Arabic profiles.
  - `--strict -m -L` improved by about 15% on generated Kana.
  - Generated Cyrillic and CJK `strict-full` and `unicode` stayed within noise.
  - A first mixed-profile `strict-full` run showed a regression, but a 12-run repeat of `strict-full`, `strict-classic`, and `unicode` stayed within noise.

### [x] Avoid `wcwidth()` calls on POSIX for common ranges

Hypothesis: POSIX builds call `wcwidth()` for every non-ASCII code point in strict `-L`. Fast local checks for common ranges could mirror the Windows staged path and avoid libc call overhead.

Target files:

- `headers/engine_impl.h`
- `sources/main.cpp`
- `tests/run_tests.py`
- `README.md`

Plan:

- Initialize POSIX `LC_CTYPE` from the process environment before calling `wcwidth()`.
- Before calling `wcwidth()`, return known widths for safe common ranges:
  - Cyrillic base letters excluding combining marks.
  - Hebrew base letters excluding marks.
  - Arabic base letters and Arabic-Indic digits excluding combining marks.
  - Hiragana and Katakana ranges that do not include combining marks.
  - CJK Unified Ideographs.
  - Hangul syllables.
- Keep behavior conservative for ambiguous-width characters by falling back to `wcwidth()`.

Validation:

- Run POSIX tests against system `wc`.
- Benchmark `strict-full` and `unicode` on Linux/macOS if available.

Risk:

- Locale-dependent `wcwidth()` behavior may differ for ambiguous-width characters. Do not shortcut ambiguous ranges or Latin-1 until that behavior is explicitly specified.

Progress:

- Attempted a conservative POSIX shortcut for Cyrillic, Hebrew, Arabic, CJK, Hangul, Hiragana, and Katakana ranges.
- Reverted the shortcut before keeping it because WSL regression testing exposed an existing locale dependency: `wcwidth()` returns width 0 for Devanagari in the current test process, so adding local shortcuts would silently change POSIX behavior for only selected scripts.
- Before this item is implemented, decide and test the POSIX locale policy explicitly:
  - initialize a UTF-8 locale in the program before `wcwidth()`, or
  - preserve current process-locale behavior and avoid local width shortcuts that bypass libc.
- Keep this item open until POSIX Unicode width semantics are made explicit and benchmarked against that policy.
- Chose the POSIX policy: initialize `LC_CTYPE` from the user's environment at startup so `wcwidth()` follows the active locale like system `wc`.
- Updated the Devanagari regression test to account for platform display-width semantics: Windows keeps the built-in width-table expectation, POSIX follows `wcwidth()` and matches `wc -m -L` under WSL.
- Reapplied a conservative POSIX pre-`wcwidth()` shortcut after the locale policy was explicit.
- Candidate outputs matched the post-locale POSIX micro-baseline on generated Cyrillic, Hebrew, Arabic, Kana, CJK, and mixed 128 MiB files for `--strict -m -L` and `--strict -l -w -c -m -L`.
- WSL regression tests passed after the locale/test update.
- Benchmark result summary against the post-locale POSIX micro-baseline:
  - `--strict -m -L` improved by about 14.5% on generated Kana and CJK profiles.
  - `--strict -m -L` improved by about 5.4% on generated Arabic.
  - `--strict -m -L` on generated Cyrillic, Hebrew, and mixed profiles stayed within noise.
  - `strict-full` on generated Cyrillic, Arabic, Kana, CJK, and mixed profiles stayed within noise.

### [x] Vectorize ASCII tab handling for strict `-L`

Hypothesis: strict ASCII blocks with tabs fall back to scalar because tab width depends on current column. Files with periodic tabs pay a high cost even when everything else is ASCII.

Target files:

- `headers/engine_impl.h`

Plan:

- For ASCII blocks with tabs but no newlines, compute the number of printable columns between tab positions using masks and update `currentLineLength` segment by segment.
- For blocks with both tabs and newlines, process segments between tabs/newlines using bit operations instead of per-byte scalar loops.
- Keep scalar fallback for dense or unusual control characters.

Validation:

- Add a `tabs` benchmark profile.
- Add tests for tabs near 8-column boundaries and across chunk boundaries.

Risk:

- Tab width is stateful. This optimization must carefully preserve current line length across blocks and chunks.

Progress:

- Added an AVX2 ASCII display-width helper for strict blocks that contain tabs.
- Kept line, word, and character counting vectorized for ASCII tab blocks.
- Replaced per-byte scalar fallback with a mask-driven update that iterates only tab/newline positions and counts printable spans with popcount.
- Preserved existing zero-width ASCII control handling and tab expansion based on the carried display column.
- Added regression coverage for tabs around 8-column boundaries.
- Candidate outputs matched baseline on generated `tabs`, `ascii`, `utf8`, and `mixed` 128 MiB profiles for `strict-full`, `unicode`, and `full`.
- On the generated `tabs` 128 MiB profile, `strict-full` improved by about 41%; `unicode` and fast `full` stayed within noise.

## P1: Parallelism And Chunking

### [x] Eliminate per-file heap allocations in mapped parallel processing

Hypothesis: `process_mapped_data()` allocates `results`, `chunkStarts`, and `chunkEnds` vectors for every parallel file. With at most 8 workers, fixed-size stack arrays can reduce overhead for medium files and many-file workloads.

Target files:

- `sources/main.cpp`
- `headers/types.h`

Plan:

- Replace worker-count-sized vectors with `std::array<..., kMaxParallelWorkers>` or a small fixed structure.
- Expose the max worker constant from one header or keep a local maximum that matches runtime.
- Keep `ChunkResult` aligned to avoid false sharing.

Validation:

- Benchmark many medium files and one large file.
- Run regression tests with forced low chunk sizes using `FASTAWC_*` env vars.

Risk:

- The max worker value must not diverge between runtime and stack array sizing.

Progress:

- Moved `kMaxParallelWorkers` into shared types so runtime and mapped processing use one limit.
- Replaced per-file worker vectors with fixed-size stack arrays for chunk results and bounds.
- Kept `ChunkResult` alignment intact.

### [x] Tune chunk sizing by measured backend throughput

Hypothesis: current chunk thresholds are static heuristics. Strict scalar, strict AVX2, fast AVX2, and byte-only workloads have different optimal parallelism.

Target files:

- `sources/runtime.cpp`
- `bench_backends.py`

Plan:

- Run a grid benchmark over:
  - worker count,
  - target chunk size,
  - minimum bytes per worker,
  - file profile,
  - scan scenario.
- Update defaults from data instead of intuition.
- Consider separate defaults for:
  - fast classic,
  - fast full,
  - strict classic,
  - strict full,
  - unicode only,
  - line only.

Validation:

- Save benchmark CSV/JSON and summarize chosen thresholds.
- Ensure small files do not regress from thread overhead.

Risk:

- Optimal thresholds are machine-dependent. Defaults should be conservative, with env overrides kept intact.

Progress:

- Ran a large-file grid on `big.txt` and found that larger chunks appeared faster across sampled scenarios.
- Checked the same larger-chunk settings on `mixed` 128 MiB and rejected them as default tuning because they regressed medium-file `full` and strict workloads.
- Root-caused the large-file result to uneven chunk slicing when the selected worker count is capped by `maxWorkers`: early chunks used `targetChunkSize` and the final chunk received most of the file.
- Kept the runtime thresholds unchanged for now; fixed the capped-worker chunk distribution separately and will retune thresholds after broader profile data.
- Re-ran tuning after the balanced-worker, strict tab, and whitespace-mask changes.
- Rejected broad `32/24/24` and `48/32/48` overrides because they still regressed `full` and strict-heavy scenarios on 128 MiB generated profiles.
- Found a stable win for the narrow fast `-m -L` workload on AVX2 by using `minParallel=32 MiB`, `bytesPerWorker=24 MiB`, and `targetChunk=24 MiB`.
- Applied that tuning only when `scanKind == fast`, backend is AVX2, and `scanMode == kScanChars | kScanMaxLine`.
- Candidate outputs matched micro-baseline on generated `ascii`, `mixed`, `utf8`, and `emoji` 128 MiB profiles for `unicode`, `full`, `strict-full`, and `strict-classic`.
- Benchmark result summary against the pre-tuning micro-baseline:
  - `unicode` improved by about 14-15% on `ascii`, `mixed`, `utf8`, and `emoji`.
  - Neighboring `full`, `strict-full`, and `strict-classic` checks stayed within noise after sequential rechecks.

### [x] Balance capped-worker chunk distribution

Hypothesis: once `choose_worker_count()` has selected a final worker count, mapped processing should split the file approximately evenly across those workers. The previous planner still used `targetChunkSize` for early chunks, so very large files could create several small chunks and one huge tail chunk.

Target files:

- `sources/main.cpp`

Plan:

- In `process_mapped_data()`, keep `targetChunkSize` for deciding how many workers to use.
- After the worker count is fixed, assign each worker an even share of the remaining file.
- Preserve existing chunk alignment and UTF-8 boundary adjustment.
- Keep fixed stack arrays and thread-pool scheduling unchanged.

Validation:

- Build and run `ctest`.
- Compare candidate against a clean baseline on `big.txt` for `bytes`, `classic`, `full`, `strict-classic`, `strict-full`, and `unicode`.
- Recheck `ascii`, `mixed`, and `utf8` 128 MiB generated profiles so medium files do not regress.
- Compare forced `scalar` mode on a generated profile.

Risk:

- More even splitting can increase cross-chunk merge and boundary work on medium files. If regressions appear, gate the even-split path by file size.

Progress:

- Updated mapped chunk planning to split the remaining file evenly across the already-selected worker count.
- Preserved the existing alignment and UTF-8 boundary correction logic.
- `ctest` passes after the change.
- On `big.txt` versus clean baseline, current auto backend changed approximately:
  - `bytes`: 4.12 ms -> 4.00 ms.
  - `classic`: 1327.97 ms -> 530.58 ms.
  - `full`: 1882.68 ms -> 582.90 ms.
  - `strict-classic`: 3501.94 ms -> 786.44 ms.
  - `strict-full`: 5093.90 ms -> 1215.18 ms.
  - `unicode`: 1629.13 ms -> 563.74 ms.
- On `mixed` 128 MiB auto backend, all checked scenarios stayed within the noise threshold.
- Sequential rechecks cleared initially suspicious `ascii`/`utf8` 128 MiB slowdown readings that came from running benchmark jobs in parallel.
- Forced `scalar` on `mixed` 128 MiB stayed within noise for most scenarios and improved `strict-full`.

### [x] Revisit strict-heavy worker cap after chunk balancing

Hypothesis: the old strict-heavy cap of 6 workers limited tail imbalance and merge overhead under the previous chunk planner. After even chunk distribution, allowing the normal `kMaxParallelWorkers` limit should improve large strict `-m/-L` workloads and may also help medium files.

Target files:

- `sources/runtime.cpp`

Plan:

- Remove the special strict-heavy cap to 6 workers.
- Keep the global `kMaxParallelWorkers` limit at 8.
- Keep scalar and AVX2 support only; do not add any new backend-specific path.

Validation:

- Build and run `ctest`.
- Compare `strict-full` and `unicode` on `big.txt`.
- Recheck 128 MiB `ascii`, `mixed`, and `utf8` generated profiles.
- Recheck forced `scalar` on a generated profile.

Risk:

- Some machines may hit memory bandwidth limits earlier than 8 workers. Keep `FASTAWC_THREADS=<n>` available for runtime override.

Progress:

- Removed the strict-heavy 6-worker cap; runtime still clamps all work to `kMaxParallelWorkers == 8`.
- `ctest` passes after the change.
- On `big.txt`, `strict-full` improved further to about 999 ms in the comparison run, versus about 1215 ms with balanced chunks but the old cap.
- On 128 MiB `mixed`, `ascii`, and `utf8` strict-full checks, candidate was faster than baseline in the measured runs.
- Forced `scalar` on `mixed` 128 MiB improved `strict-full` and kept `unicode` within noise.

### [x] Audit worker-local chunk claims larger than one index

Hypothesis: `ThreadPool` uses a `fetch_add(1)` scheduler. For many small chunks, atomic traffic can matter. Claiming small batches may reduce overhead.

Target files:

- `sources/thread_pool.cpp`
- `headers/thread_pool.h`

Plan:

- Add a batch size parameter or compute one from `taskCount`.
- Each worker claims `N` adjacent task indices with one atomic operation.
- Keep behavior identical for small `taskCount`.

Validation:

- Force small chunks with environment variables and benchmark large files.
- Benchmark many medium files.

Risk:

- Larger claims can reduce load balance if chunks have uneven cost. Use small batches and measure.

Progress:

- Audited current mapped parallel execution and confirmed `taskCount` is capped by `kMaxParallelWorkers == 8`.
- After balanced chunk distribution, there are no many-small-task mapped workloads in the current design.
- Closed this as not worth implementing now; batching would add scheduler complexity without a measured bottleneck.

## P2: I/O And Memory Mapping

### [x] Add an option to benchmark streaming versus memory mapping

Hypothesis: mapped I/O is best for many regular-file cases, but streaming may be competitive or more predictable on some storage and OS combinations. A controlled switch would make CPU/backend benchmarking cleaner.

Target files:

- `sources/platform.cpp`
- `sources/main.cpp`
- `README.md`
- `bench_backends.py`

Plan:

- Add `FASTAWC_NO_MMAP=1` or a documented command-line option.
- Force regular files through the streaming path.
- Keep mapped mode as default.

Validation:

- Benchmark mapped vs streaming on large cached files and cold-ish files.
- Verify stdin and pipes are unaffected.

Risk:

- Extra options can clutter user-facing CLI. Prefer environment variable unless this becomes a common feature.

Progress:

- Added `FASTAWC_NO_MMAP=1` to force regular files through the streaming path.
- Kept memory mapping as the default.
- Documented the environment variable in README.

### [x] Test OS-specific readahead and prefetch hints

Hypothesis: the current code uses `FILE_FLAG_SEQUENTIAL_SCAN`, `posix_fadvise`, and `madvise(MADV_SEQUENTIAL)`, but more aggressive hints may help very large files.

Target files:

- `sources/platform.cpp`

Ideas:

- Windows `PrefetchVirtualMemory` after mapping.
- POSIX `posix_fadvise(..., POSIX_FADV_WILLNEED)` for large files.
- Conditional `madvise(MADV_WILLNEED)` for mapped files.

Validation:

- Benchmark large files bigger than RAM-cache comfort zones if possible.
- Separate cached-file CPU benchmarks from cold-file I/O benchmarks.

Risk:

- Readahead hints can hurt on some systems by evicting useful cache. Keep any aggressive hint opt-in unless broadly proven.

Progress:

- Added opt-in `FASTAWC_WILLNEED=1`.
- Windows mapped files now call `PrefetchVirtualMemory` when requested.
- POSIX mapped/streaming file paths now request `POSIX_FADV_WILLNEED` and `MADV_WILLNEED` where available.
- Default behavior remains unchanged.

### [x] Increase or tune streaming buffer size

Hypothesis: `kStreamBufferSize` is 8 MiB. Larger buffers may reduce syscall overhead for pipes/files; smaller buffers may improve cache locality for strict processing.

Target files:

- `headers/types.h`
- `sources/main.cpp`

Plan:

- Add `FASTAWC_STREAM_BUFFER_MB`.
- Benchmark 1, 4, 8, 16, 32, and 64 MiB buffer sizes.
- Test both file fallback and stdin/pipe workloads.

Validation:

- Pipe input tests must still pass.
- Benchmark streaming mode after adding a no-mmap option.

Risk:

- Oversized buffers waste memory in many-file or constrained environments.

Progress:

- Added `FASTAWC_STREAM_BUFFER_MB=<n>` for streaming, stdin, pipe, and `FASTAWC_NO_MMAP=1` benchmarks.
- Kept the default at 8 MiB.
- Clamped oversized values to a 1024 MiB ceiling to avoid accidental huge allocations.

## P2: Backend Scope And Build Options

### [x] Keep backend support limited to scalar and AVX2

Decision: this project supports exactly two execution backends:

- `scalar`: the portable baseline for generic x86-64 builds and non-AVX2 runtime fallback.
- `avx2`: the only SIMD backend, built as a separate object target and selected at runtime when CPU and OS support are available.

Out of scope:

- No SSE, SSE2, SSE4.2, SSSE3, or other 128-bit SIMD backend.
- No AVX-512 backend.
- No additional ISA-specific backend unless project scope is explicitly changed later.

Hypothesis: keeping the backend matrix small will make performance work faster and safer. The highest-value wins are inside the existing AVX2 and scalar paths, not in maintaining more dispatch targets.

Target files:

- `CMakeLists.txt`
- `sources/runtime.cpp`
- `headers/engine.h`
- `headers/engine_impl.h`
- `README.md`

Plan:

- Keep `FASTAWC_ENABLE_AVX2_BACKEND` as the only optional ISA backend switch.
- Keep runtime backend overrides limited to `FASTAWC_BACKEND=scalar|avx2`.
- Remove or reject any future TODO item that proposes SSE, AVX-512, or additional backend families.
- Make sure README and benchmark docs describe only scalar and AVX2.
- If backend-specific tuning is needed, express it as scalar-vs-AVX2 tuning, not a new backend.

Validation:

- Build with AVX2 enabled and verify both `scalar` and `avx2` benchmark override modes work.
- Build with `-DFASTAWC_ENABLE_AVX2_BACKEND=OFF` and verify scalar-only builds still work.
- Confirm invalid `FASTAWC_BACKEND` values fall back safely.

Risk:

- Some older non-AVX2 machines may leave performance on the table. This is accepted to keep maintenance focused.

Progress:

- Removed SSE and AVX-512 work from the roadmap.
- Confirmed future backend work should stay limited to scalar and AVX2.
- Verified `-DFASTAWC_ENABLE_AVX2_BACKEND=OFF` scalar-only Release build passes regression tests.

### [x] Add PGO build workflow

Hypothesis: this code is branch-heavy in strict mode and dispatch-heavy at startup. Profile-guided optimization may improve layout and branch prediction.

Target files:

- `CMakeLists.txt`
- `README.md`
- optional helper script

Plan:

- Add documented MSVC and Clang/GCC PGO build steps.
- Train on representative benchmark profiles: `ascii`, `mixed`, `utf8`, `whitespace`, `longlines`, `shortlines`.
- Compare PGO against LTO-only builds.

Validation:

- Benchmark all standard scenarios.
- Ensure PGO artifacts do not get committed.

Risk:

- PGO can overfit to training data. Keep it optional.

Progress:

- Added optional `FASTAWC_PGO_PHASE=off|generate|use` CMake support.
- Kept the default as `off`, so normal Release/LTO builds are unchanged unless PGO is explicitly requested.
- Added MSVC link options for instrumented and profile-use PGO phases.
- Added GNU and Clang compile/link options for generate/use phases.
- Documented the two-stage PGO workflow in README.
- Built a WSL/GCC instrumented PGO binary, trained it on 64 MiB generated `ascii`, `mixed`, `utf8`, `whitespace`, `longlines`, and `shortlines` profiles, rebuilt with `-fprofile-use`, and passed regression tests.
- Compared the WSL/GCC PGO binary against the LTO-only WSL binary on 128 MiB `ascii`, `mixed`, and `utf8` profiles for `classic`, `full`, `strict-full`, and `unicode`.
- Result: all checked PGO deltas stayed within the configured 5% noise band. PGO is supported but should remain optional and target-machine validated before being used for release artifacts.

## P2: Runtime Selection And Autotuning

### [x] Extend autotune beyond strict `-m/-L`

Hypothesis: `FASTAWC_AUTOTUNE=1` currently targets strict char/max-line workloads. Some machines or workloads may prefer scalar or AVX2 differently for strict classic, Unicode-heavy data, or very small files.

Target files:

- `sources/main.cpp`
- `tests/run_tests.py`

Plan:

- Add calibration patterns for:
  - ASCII classic,
  - mixed strict classic,
  - UTF-8 strict full,
  - tab-heavy strict `-L`.
- Keep autotune opt-in.
- Cache chosen result per process and scan mode.

Validation:

- Confirm startup overhead remains acceptable.
- Compare selected backend against explicit `FASTAWC_BACKEND=scalar|avx2`.

Risk:

- Startup calibration can dominate small inputs. Never enable by default unless calibration becomes very cheap.

Progress:

- Kept autotune opt-in behind `FASTAWC_AUTOTUNE=1`.
- Added workload-specific calibration buffers:
  - ASCII classic,
  - mixed strict classic,
  - UTF-8 strict full,
  - tab-heavy strict `-L`.
- Reduced individual calibration buffers to 2 MiB and made them branch-local so only the needed pattern is initialized.
- Extended autotune to all scan modes instead of only strict `-m/-L`.
- Cached selected backend per process for each `scanKind + scanMode` pair.
- Added regression smoke coverage for `FASTAWC_AUTOTUNE=1` in fast classic, strict classic, and strict `-L`.
- Benchmarked `FASTAWC_AUTOTUNE=1` against explicit `scalar` and `avx2` on 128 MiB generated `ascii`, `mixed`, and `utf8` profiles.
- Found that static calibration alone cannot choose optimally for `strict-full`: ASCII/mixed prefer AVX2, while high non-ASCII UTF-8/CJK/emoji profiles prefer scalar. This is handled by the prefix-sampling item below.
- Startup calibration remains visible in subprocess wall-clock timings, so autotune stays opt-in.

### [x] Sample file profile for backend and chunk decisions

Hypothesis: a small prefix sample can estimate ASCII ratio, newline density, tab density, and non-ASCII shape. Runtime could choose backend/chunk settings more accurately than static scan-mode heuristics.

Target files:

- `sources/main.cpp`

Plan:

- Sample the first 64 KiB or 1 MiB of mapped regular files.
- Compute:
  - ASCII byte ratio,
  - newline density,
  - tab/control density,
  - bytes `>= 0xF0`,
  - approximate UTF-8 lead density.
- Use sample only for mapped regular files above a size threshold.
- Feed the profile into runtime config and backend selection.

Validation:

- Benchmark mixed workloads where static heuristics are weak.
- Verify overhead is invisible for large files and skipped for small files.

Risk:

- Sampling touches pages before the main scan and may hurt cold I/O. Make decisions conservative.

Progress:

- Added a mapped-file prefix sampler for `FASTAWC_AUTOTUNE=1`.
- The sampler reads up to the first 1 MiB and counts sampled bytes, non-ASCII bytes, bytes `>= 0xF0`, tabs, and newlines.
- Sampling is skipped for stdin, streaming inputs, files below 16 MiB, forced `FASTAWC_BACKEND`, and non-strict workloads.
- For strict char/max-line workloads, Unicode-heavy samples with more than 50% non-ASCII bytes choose scalar directly instead of paying calibration and then selecting AVX2.
- Moved backend/config/processor selection into per-file processing so mapped-file samples can affect backend choice without changing default non-autotuned behavior.
- Avoided startup calibration when creating the thread pool; the pool now uses ordinary backend selection for worker-count sizing.
- Validation:
  - Windows and WSL regression tests passed.
  - With `FASTAWC_AUTOTUNE=1`, generated `utf8 strict-full` now matches scalar timing and avoids the previous AVX2 mis-selection.
  - Generated `ascii` and `mixed strict-full` still choose AVX2 by internal throughput.
  - Default `auto` without `FASTAWC_AUTOTUNE` matches explicit AVX2 within noise on generated `ascii` and `mixed` profiles.

## P3: Engine Structure For Future Speed Work

### [x] Split hot engine code into focused headers

Hypothesis: `headers/engine_impl.h` mixes Unicode tables, scalar decoders, AVX2 kernels, and dispatch. This makes hot-path optimization harder and increases accidental compile-time or codegen coupling.

Target files:

- `headers/engine_impl.h`
- new headers under `headers/`, for example:
  - `engine_unicode.h`
  - `engine_scalar_impl.h`
  - `engine_avx2_impl.h`
  - `engine_dispatch.h`

Plan:

- Split without behavior changes first.
- Keep force-inline helpers visible where needed.
- Avoid introducing virtual dispatch or runtime abstractions in hot loops.

Validation:

- Binary output should remain equivalent.
- Benchmark before and after to catch codegen regressions.

Risk:

- Refactors can change inlining and code layout. Do this separately from functional optimizations.

Progress:

- Split Unicode whitespace, display-width, Windows width tables, and POSIX common width shortcuts from `headers/engine_impl.h` into `headers/engine_unicode.h`.
- Kept the new header included inside the backend implementation namespace so function names, force-inline behavior, and backend-specific compilation remain unchanged.
- Left scalar kernels, AVX2 kernels, and dispatch tables in `engine_impl.h` for separate future refactors; this avoids mixing several code-layout changes in one step.
- Windows and WSL regression tests passed after the split.
- Candidate outputs matched the pre-refactor root binary on generated `ascii`, `utf8`, and `mixed` 128 MiB files for `classic`, `strict-full`, and `unicode`.
- Benchmark comparison against the pre-refactor root binary on generated `ascii` and `utf8` 128 MiB files stayed within the configured 5% noise band.

### [x] Generate dispatch tables mechanically

Hypothesis: dispatch tables are currently explicit and correct, but future backends and specialized paths will make manual tables easier to break.

Target files:

- `headers/engine_impl.h`
- optional small script or constexpr generator

Plan:

- Keep current runtime table shape.
- Generate processor entries from scan-mode bit combinations.
- Support special-case overrides, such as line-only or byte-only processors.

Validation:

- Add compile-time static assertions for table size and null entry.
- Run all regression tests.

Risk:

- Over-engineering dispatch can make code harder to read. Keep generated logic simple.

Progress:

- Replaced manually written function-pointer dispatch entries with constexpr helpers that derive template arguments from scan-mode bits.
- Kept the same 16-entry table shape and the same `select_fast_processor()` / `select_strict_processor()` API.
- Added compile-time checks for the null entry and full-mode entry.
- Windows and WSL regression tests passed after the dispatch change.
- Explicit scalar benchmark comparison against the pre-refactor root binary on generated `ascii` and `utf8` 128 MiB profiles stayed within the configured 5% noise band.

## Suggested Execution Order

1. Consider splitting scalar and AVX2 implementation sections only after another clean benchmark checkpoint.
