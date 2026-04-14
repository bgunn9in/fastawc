#include "engine.h"
#include "platform.h"
#include "thread_pool.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace fastawc {

namespace {

constexpr size_t kChunkAlignment = 2u << 20;
constexpr const char* kVersionString = "fastawc 0.1.0";

struct alignas(64) ChunkResult {
	Counts counts{};
	uint64_t prefixLineLength = 0;
	uint64_t suffixLineLength = 0;
	bool sawNewline = false;
};

struct ParseResult {
	Options options{};
	std::string error;
	bool showHelp = false;
	bool showVersion = false;
};

struct OutputRow {
	Counts counts{};
	std::string label;
	bool showLabel = false;
};

FASTAWC_FORCEINLINE bool is_space_ascii(const uint8_t c) noexcept {
	return c == ' ' || static_cast<unsigned>(c - '\t') < 5u;
}

FASTAWC_FORCEINLINE bool is_utf8_continuation(const uint8_t c) noexcept {
	return (c & 0xC0u) == 0x80u;
}

unsigned choose_worker_count(const size_t size, const RuntimeConfig& config, const ThreadPool& pool) noexcept {
	const unsigned maxWorkers = std::min(config.maxWorkers, pool.worker_count());
	if (maxWorkers <= 1 || size < config.minParallelFileSize) {
		return 1;
	}

	unsigned workers = static_cast<unsigned>((size + config.targetChunkSize - 1) / config.targetChunkSize);
	workers = std::max(workers, static_cast<unsigned>(size / config.minBytesPerWorker));
	workers = std::min(workers, maxWorkers);
	return std::max(workers, 1u);
}

size_t align_chunk_end(
	const size_t start,
	const size_t proposed,
	const size_t size,
	const unsigned chunksLeft,
	const uint8_t* const data,
	const bool utf8Sensitive) noexcept
{
	if (chunksLeft <= 1 || proposed >= size) {
		return size;
	}

	size_t aligned = proposed & ~(kChunkAlignment - 1);
	if (aligned <= start) {
		aligned = proposed;
	}

	if (utf8Sensitive) {
		while (aligned > start && aligned < size && is_utf8_continuation(data[aligned])) {
			--aligned;
		}
		if (aligned <= start) {
			aligned = proposed;
		}
	}

	const size_t minRemaining = static_cast<size_t>(chunksLeft - 1) * kChunkAlignment;
	if (minRemaining < size && aligned > size - minRemaining) {
		aligned = size - minRemaining;
	}

	return std::min(std::max(aligned, start + 1), size);
}

ChunkResult process_memory_chunk(
	const uint8_t* const data,
	const size_t size,
	const uint32_t initialPrevSpaceBit,
	const ScanModeKind scanKind,
	const uint32_t scanMode,
	const ScanProcessor processor,
	const bool countBytes,
	const bool countMaxLine)
{
	ChunkResult result{};
	ScanState state{};
	state.prevSpaceBit = initialPrevSpaceBit;

	if (countBytes) {
		result.counts.byteCount = size;
	}

	if (processor != nullptr && size != 0) {
		processor(data, size, result.counts, state);
	}

	finalize_scan_state(scanKind, scanMode, result.counts, state);

	if (countMaxLine) {
		if (state.currentLineLength > result.counts.maxLineLength) {
			result.counts.maxLineLength = state.currentLineLength;
		}
		result.sawNewline = state.sawNewline;
		result.suffixLineLength = state.currentLineLength;
		result.prefixLineLength = state.sawNewline ? state.prefixLineLength : state.currentLineLength;
	}

	return result;
}

void merge_chunk_results(const std::vector<ChunkResult>& chunks, Counts& out, const bool countMaxLine) {
	uint64_t carriedLineLength = 0;

	for (const ChunkResult& chunk : chunks) {
		out.lineCount += chunk.counts.lineCount;
		out.wordCount += chunk.counts.wordCount;
		out.byteCount += chunk.counts.byteCount;
		out.charCount += chunk.counts.charCount;
		out.maxLineLength = std::max(out.maxLineLength, chunk.counts.maxLineLength);

		if (countMaxLine) {
			if (chunk.sawNewline) {
				out.maxLineLength = std::max(out.maxLineLength, carriedLineLength + chunk.prefixLineLength);
				carriedLineLength = chunk.suffixLineLength;
			}
			else {
				carriedLineLength += chunk.suffixLineLength;
			}
		}
	}

	if (countMaxLine) {
		out.maxLineLength = std::max(out.maxLineLength, carriedLineLength);
	}
}

void process_mapped_data(
	const uint8_t* const data,
	const size_t size,
	const ScanModeKind scanKind,
	const uint32_t scanMode,
	const ScanProcessor processor,
	const bool countBytes,
	const bool countMaxLine,
	const RuntimeConfig& config,
	ThreadPool& pool,
	Counts& out)
{
	if (size == 0) {
		return;
	}

	if (processor == nullptr) {
		if (countBytes) {
			out.byteCount = size;
		}
		return;
	}

	const unsigned workerCount = choose_worker_count(size, config, pool);
	if (workerCount <= 1) {
		out = process_memory_chunk(data, size, 1u, scanKind, scanMode, processor, countBytes, countMaxLine).counts;
		return;
	}

	struct JobContext {
		const uint8_t* data = nullptr;
		size_t size = 0;
		unsigned workerCount = 0;
		ScanModeKind scanKind = ScanModeKind::fast;
		uint32_t scanMode = 0;
		ScanProcessor processor = nullptr;
		bool countBytes = false;
		bool countMaxLine = false;
		std::vector<ChunkResult>* results = nullptr;
	};

	const bool utf8Sensitive = (scanMode & (kScanWords | kScanChars | kScanMaxLine)) != 0;
	std::vector<ChunkResult> results(workerCount);
	std::vector<size_t> chunkStarts(workerCount);
	std::vector<size_t> chunkEnds(workerCount);
	JobContext context{ data, size, workerCount, scanKind, scanMode, processor, countBytes, countMaxLine, &results };

	size_t chunkStart = 0;
	for (unsigned index = 0; index < workerCount; ++index) {
		const unsigned chunksLeft = workerCount - index;
		const size_t remaining = size - chunkStart;
		const size_t evenShare = remaining / chunksLeft;
		const size_t targetSize = std::min(config.targetChunkSize, evenShare + static_cast<size_t>(remaining % chunksLeft != 0));
		const size_t chunkEnd = align_chunk_end(chunkStart, chunkStart + targetSize, size, chunksLeft, data, utf8Sensitive);
		chunkStarts[index] = chunkStart;
		chunkEnds[index] = chunkEnd;
		chunkStart = chunkEnd;
	}

	pool.parallel_for(workerCount, [&context, &chunkStarts, &chunkEnds](const unsigned index) noexcept {
		const size_t chunkStart = chunkStarts[index];
		const size_t chunkEnd = chunkEnds[index];
		const uint32_t initialPrevSpaceBit =
			(chunkStart == 0) ? 1u : static_cast<uint32_t>(is_space_ascii(context.data[chunkStart - 1]));

		(*context.results)[index] = process_memory_chunk(
			context.data + chunkStart,
			chunkEnd - chunkStart,
			initialPrevSpaceBit,
			context.scanKind,
			context.scanMode,
			context.processor,
			context.countBytes,
			context.countMaxLine);
	});

	merge_chunk_results(results, out, countMaxLine);
}

void process_stream(
	FileSource& source,
	const ScanModeKind scanKind,
	const uint32_t scanMode,
	const ScanProcessor processor,
	const bool countBytes,
	const bool countMaxLine,
	Counts& out)
{
	auto buffer = std::make_unique<uint8_t[]>(kStreamBufferSize);

	if (processor == nullptr) {
		if (!countBytes) {
			return;
		}

		for (;;) {
			const size_t readBytes = read_stream(source, buffer.get(), kStreamBufferSize);
			if (readBytes == 0) {
				return;
			}
			out.byteCount += readBytes;
		}
	}

	ScanState state{};
	for (;;) {
		const size_t readBytes = read_stream(source, buffer.get(), kStreamBufferSize);
		if (readBytes == 0) {
			break;
		}

		if (countBytes) {
			out.byteCount += readBytes;
		}

		processor(buffer.get(), readBytes, out, state);
	}

	finalize_scan_state(scanKind, scanMode, out, state);

	if (countMaxLine && state.currentLineLength > out.maxLineLength) {
		out.maxLineLength = state.currentLineLength;
	}
}

bool process_path(
	const std::string& path,
	const ScanModeKind scanKind,
	const uint32_t scanMode,
	const ScanProcessor processor,
	const bool countBytes,
	const bool countMaxLine,
	const RuntimeConfig& config,
	ThreadPool& pool,
	Counts& out)
{
	if (path == "-") {
		FileSource source;
		open_stdin(source);
		process_stream(source, scanKind, scanMode, processor, countBytes, countMaxLine, out);
		return true;
	}

	FileSource source;
	std::string error;
	if (!open_regular_file(path, source, error)) {
		std::fprintf(stderr, "fastawc: %s: %s\n", path.c_str(), error.c_str());
		return false;
	}

	if (source.is_mapped()) {
		process_mapped_data(source.data, source.size, scanKind, scanMode, processor, countBytes, countMaxLine, config, pool, out);
	}
	else {
		process_stream(source, scanKind, scanMode, processor, countBytes, countMaxLine, out);
	}

	return true;
}

bool parse_total_mode(const std::string_view value, TotalMode& out) noexcept {
	if (value == "auto") {
		out = TotalMode::auto_mode;
		return true;
	}
	if (value == "always") {
		out = TotalMode::always;
		return true;
	}
	if (value == "only") {
		out = TotalMode::only;
		return true;
	}
	if (value == "never") {
		out = TotalMode::never;
		return true;
	}
	return false;
}

bool parse_scan_kind(const std::string_view value, ScanModeKind& out) noexcept {
	if (value == "fast") {
		out = ScanModeKind::fast;
		return true;
	}
	if (value == "strict") {
		out = ScanModeKind::strict;
		return true;
	}
	return false;
}

ParseResult parse_options(int argc, char** argv) {
	ParseResult result;
	bool endOfOptions = false;

	for (int i = 1; i < argc; ++i) {
		const std::string_view arg{ argv[i] };
		if (endOfOptions || arg.empty() || arg == "-" || arg[0] != '-') {
			result.options.files.emplace_back(arg);
			continue;
		}

		if (arg == "--") {
			endOfOptions = true;
			continue;
		}

		if (arg.rfind("--", 0) == 0) {
			const size_t eqPos = arg.find('=');
			const std::string_view option = arg.substr(2, eqPos == std::string_view::npos ? std::string_view::npos : eqPos - 2);
			const std::string_view value = eqPos == std::string_view::npos ? std::string_view{} : arg.substr(eqPos + 1);

			if (option == "help") {
				result.showHelp = true;
				return result;
			}
			if (option == "version") {
				result.showVersion = true;
				return result;
			}
			if (option == "lines") {
				result.options.optLines = true;
				continue;
			}
			if (option == "words") {
				result.options.optWords = true;
				continue;
			}
			if (option == "bytes") {
				result.options.optBytes = true;
				continue;
			}
			if (option == "chars") {
				result.options.optChars = true;
				continue;
			}
			if (option == "max-line-length") {
				result.options.optMaxLine = true;
				continue;
			}
			if (option == "fast") {
				result.options.scanKind = ScanModeKind::fast;
				continue;
			}
			if (option == "strict") {
				result.options.scanKind = ScanModeKind::strict;
				continue;
			}
			if (option == "mode") {
				std::string_view modeValue = value;
				if (modeValue.empty()) {
					if (i + 1 >= argc) {
						result.error = "fastawc: option '--mode' requires an argument";
						return result;
					}
					modeValue = argv[++i];
				}
				if (!parse_scan_kind(modeValue, result.options.scanKind)) {
					result.error = "fastawc: invalid argument for '--mode': " + std::string(modeValue);
					return result;
				}
				continue;
			}
			if (option == "total") {
				std::string_view totalValue = value;
				if (totalValue.empty()) {
					if (i + 1 >= argc) {
						result.error = "fastawc: option '--total' requires an argument";
						return result;
					}
					totalValue = argv[++i];
				}
				if (!parse_total_mode(totalValue, result.options.totalMode)) {
					result.error = "fastawc: invalid argument for '--total': " + std::string(totalValue);
					return result;
				}
				continue;
			}

			result.error = "fastawc: unrecognized option '" + std::string(arg) + "'";
			return result;
		}

		for (size_t j = 1; j < arg.size(); ++j) {
			switch (arg[j]) {
			case 'l': result.options.optLines = true; break;
			case 'w': result.options.optWords = true; break;
				case 'c': result.options.optBytes = true; break;
				case 'm': result.options.optChars = true; break;
				case 'L': result.options.optMaxLine = true; break;
			default:
				result.error = "fastawc: invalid option -- '" + std::string(1, static_cast<char>(arg[j])) + "'";
				return result;
			}
		}
	}

	if (!result.options.optLines &&
		!result.options.optWords &&
		!result.options.optBytes &&
		!result.options.optChars &&
		!result.options.optMaxLine) {
		result.options.optLines = true;
		result.options.optWords = true;
		result.options.optBytes = true;
	}

	if (result.options.files.empty()) {
		result.options.implicitStdin = true;
		result.options.files.emplace_back("-");
	}

	return result;
}

uint32_t count_active_columns(const Options& options) noexcept {
	return static_cast<uint32_t>(options.optLines) +
		static_cast<uint32_t>(options.optWords) +
		static_cast<uint32_t>(options.optBytes) +
		static_cast<uint32_t>(options.optChars) +
		static_cast<uint32_t>(options.optMaxLine);
}

size_t count_digits(const uint64_t value) noexcept {
	size_t digits = 1;
	uint64_t x = value;
	while (x >= 10) {
		x /= 10;
		++digits;
	}
	return digits;
}

size_t compute_field_width(const std::vector<OutputRow>& rows, const Options& options) noexcept {
	const uint32_t activeColumns = count_active_columns(options);
	if (rows.empty()) {
		return 1;
	}
	if (rows.size() == 1 && activeColumns == 1 && !rows.front().showLabel) {
		return 1;
	}

	size_t width = 1;
	for (const OutputRow& row : rows) {
		if (options.optLines) {
			width = std::max(width, count_digits(row.counts.lineCount));
		}
		if (options.optWords) {
			width = std::max(width, count_digits(row.counts.wordCount));
		}
		if (options.optBytes) {
			width = std::max(width, count_digits(row.counts.byteCount));
		}
		if (options.optChars) {
			width = std::max(width, count_digits(row.counts.charCount));
		}
		if (options.optMaxLine) {
			width = std::max(width, count_digits(row.counts.maxLineLength));
		}
	}
	return width;
}

void print_value(const uint64_t value, const size_t width, bool& firstField) {
	if (firstField) {
		std::printf("%*llu", static_cast<int>(width), static_cast<unsigned long long>(value));
		firstField = false;
	}
	else {
		std::printf(" %*llu", static_cast<int>(width), static_cast<unsigned long long>(value));
	}
}

void print_counts(const OutputRow& row, const Options& options, const size_t fieldWidth) {
	bool firstField = true;

	if (options.optLines) {
		print_value(row.counts.lineCount, fieldWidth, firstField);
	}
	if (options.optWords) {
		print_value(row.counts.wordCount, fieldWidth, firstField);
	}
	if (options.optBytes) {
		print_value(row.counts.byteCount, fieldWidth, firstField);
	}
	if (options.optChars) {
		print_value(row.counts.charCount, fieldWidth, firstField);
	}
	if (options.optMaxLine) {
		print_value(row.counts.maxLineLength, fieldWidth, firstField);
	}
	if (row.showLabel) {
		std::printf(" %s", row.label.c_str());
	}
	std::fputc('\n', stdout);
}

Counts accumulate_total(const std::vector<OutputRow>& rows) noexcept {
	Counts total{};
	for (const OutputRow& row : rows) {
		total.lineCount += row.counts.lineCount;
		total.wordCount += row.counts.wordCount;
		total.byteCount += row.counts.byteCount;
		total.charCount += row.counts.charCount;
		total.maxLineLength = std::max(total.maxLineLength, row.counts.maxLineLength);
	}
	return total;
}

void print_help() {
	std::puts("Usage: fastawc [OPTION]... [FILE]...");
	std::puts("Print newline, word, byte, character, and maximum display width counts.");
	std::puts("");
	std::puts("  -l, --lines            print the newline counts");
	std::puts("  -w, --words            print the word counts");
	std::puts("  -c, --bytes            print the byte counts");
	std::puts("  -m, --chars            print the character counts");
	std::puts("  -L, --max-line-length  print the maximum display width");
	std::puts("      --fast             use aggressive fast counting semantics");
	std::puts("      --strict           use stricter wc-compatible counting semantics");
	std::puts("      --mode=MODE        select fast or strict counting mode");
	std::puts("      --total=WHEN       auto, always, only, never");
	std::puts("      --help             display this help and exit");
	std::puts("      --version          output version information and exit");
}

const Backend& choose_execution_backend(const Options& options, const uint32_t scanMode) noexcept {
	const Backend& backend = select_backend();
	if (options.scanKind == ScanModeKind::strict && (scanMode & (kScanChars | kScanMaxLine)) != 0) {
		return scalar_backend();
	}
	return backend;
}

} // namespace

} // namespace fastawc

int main(int argc, char** argv) {
	using namespace fastawc;

	const ParseResult parsed = parse_options(argc, argv);
	if (!parsed.error.empty()) {
		std::fprintf(stderr, "%s\n", parsed.error.c_str());
		return 1;
	}
	if (parsed.showHelp) {
		print_help();
		return 0;
	}
	if (parsed.showVersion) {
		std::puts(kVersionString);
		return 0;
	}

	const Options& options = parsed.options;
	const uint32_t scanMode = make_scan_mode(options);
	const Backend& backend = choose_execution_backend(options, scanMode);
	const RuntimeConfig config = choose_runtime_config(backend, scanMode);
	ThreadPool pool(config.maxWorkers);
	const ScanProcessor processor =
		options.scanKind == ScanModeKind::strict
		? backend.selectStrictProcessor(scanMode)
		: backend.selectFastProcessor(scanMode);

	std::vector<OutputRow> fileRows;
	fileRows.reserve(options.files.size());
	bool hadErrors = false;

	for (const std::string& path : options.files) {
		Counts current{};
		if (!process_path(path, options.scanKind, scanMode, processor, options.optBytes, options.optMaxLine, config, pool, current)) {
			hadErrors = true;
			continue;
		}

		OutputRow row;
		row.counts = current;
		row.label = path == "-" ? std::string("-") : path;
		row.showLabel = options.totalMode != TotalMode::only && !(options.implicitStdin && path == "-");
		fileRows.push_back(std::move(row));
	}

	std::vector<OutputRow> rowsToPrint;
	rowsToPrint.reserve(fileRows.size() + 1);

	if (options.totalMode != TotalMode::only) {
		rowsToPrint = fileRows;
	}

	const bool needTotal =
		options.totalMode == TotalMode::always ||
		options.totalMode == TotalMode::only ||
		(options.totalMode == TotalMode::auto_mode && fileRows.size() > 1);

	if (needTotal) {
		OutputRow totalRow;
		totalRow.counts = accumulate_total(fileRows);
		totalRow.label = "total";
		totalRow.showLabel = options.totalMode != TotalMode::only;
		rowsToPrint.push_back(std::move(totalRow));
	}

	const size_t fieldWidth = compute_field_width(rowsToPrint, options);
	for (const OutputRow& row : rowsToPrint) {
		print_counts(row, options, fieldWidth);
	}

	return hadErrors ? 1 : 0;
}
