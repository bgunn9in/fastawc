#include "engine.h"
#include "platform.h"
#include "thread_pool.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <system_error>
#include <string>
#include <string_view>
#include <vector>

namespace fastawc {

namespace {

constexpr size_t kChunkAlignment = 2u << 20;
constexpr size_t kMaxStreamBufferSize = 1024ull << 20;
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

FASTAWC_FORCEINLINE bool is_unicode_whitespace(const uint32_t codePoint) noexcept {
	if (codePoint <= 0x7Fu) {
		return codePoint == ' ' || static_cast<unsigned>(codePoint - '\t') < 5u;
	}

	switch (codePoint) {
	case 0x0085u:
	case 0x00A0u:
	case 0x1680u:
	case 0x2028u:
	case 0x2029u:
	case 0x202Fu:
	case 0x205Fu:
	case 0x2060u:
	case 0x3000u:
		return true;
	default:
		break;
	}

	return codePoint >= 0x2000u && codePoint <= 0x200Au;
}

FASTAWC_FORCEINLINE BoundaryCodePointClass classify_boundary_codepoint(
	const uint32_t codePoint,
	const bool valid,
	const ScanModeKind scanKind) noexcept
{
	if (!valid) {
		return BoundaryCodePointClass::invalid;
	}
	if (codePoint <= 0x7Fu) {
		return is_space_ascii(static_cast<uint8_t>(codePoint))
			? BoundaryCodePointClass::ascii_space
			: BoundaryCodePointClass::ascii_nonspace;
	}
	if (scanKind == ScanModeKind::strict) {
		return is_unicode_whitespace(codePoint)
			? BoundaryCodePointClass::unicode_space
			: BoundaryCodePointClass::unicode_nonspace;
	}
	return BoundaryCodePointClass::unicode_nonspace;
}

bool decode_utf8_suffix_codepoint(
	const uint8_t* const data,
	const size_t begin,
	const size_t end,
	uint32_t& codePoint) noexcept
{
	if (begin >= end) {
		return false;
	}

	const uint8_t lead = data[begin];
	const size_t length = end - begin;
	if (lead < 0x80u) {
		if (length != 1) {
			return false;
		}
		codePoint = lead;
		return true;
	}

	uint32_t value = 0;
	uint32_t minCodePoint = 0;
	if (lead >= 0xC2u && lead <= 0xDFu) {
		if (length != 2) {
			return false;
		}
		value = lead & 0x1Fu;
		minCodePoint = 0x80u;
	}
	else if (lead >= 0xE0u && lead <= 0xEFu) {
		if (length != 3) {
			return false;
		}
		value = lead & 0x0Fu;
		minCodePoint = 0x800u;
	}
	else if (lead >= 0xF0u && lead <= 0xF4u) {
		if (length != 4) {
			return false;
		}
		value = lead & 0x07u;
		minCodePoint = 0x10000u;
	}
	else {
		return false;
	}

	for (size_t i = begin + 1; i < end; ++i) {
		if (!is_utf8_continuation(data[i])) {
			return false;
		}
		value = (value << 6) | static_cast<uint32_t>(data[i] & 0x3Fu);
	}

	if (value < minCodePoint || value > 0x10FFFFu || (value >= 0xD800u && value <= 0xDFFFu)) {
		return false;
	}

	codePoint = value;
	return true;
}

ChunkBoundaryState compute_chunk_boundary_state(
	const uint8_t* const data,
	const size_t chunkStart,
	const ScanModeKind scanKind) noexcept
{
	ChunkBoundaryState boundary{};
	if (chunkStart == 0) {
		boundary.prevCodePoint = ' ';
		boundary.prevCodePointClass = BoundaryCodePointClass::ascii_space;
		boundary.prevCodePointValid = true;
		return boundary;
	}

	if (scanKind != ScanModeKind::strict) {
		const uint32_t codePoint = data[chunkStart - 1];
		boundary.prevCodePoint = codePoint;
		boundary.prevCodePointClass = classify_boundary_codepoint(codePoint, true, scanKind);
		boundary.prevCodePointValid = true;
		boundary.prevSpaceBit = static_cast<uint32_t>(boundary.prevCodePointClass == BoundaryCodePointClass::ascii_space);
		return boundary;
	}

	size_t codePointStart = chunkStart - 1;
	unsigned backedUp = 0;
	while (codePointStart > 0 && backedUp < 3 && is_utf8_continuation(data[codePointStart])) {
		--codePointStart;
		++backedUp;
	}

	uint32_t codePoint = 0;
	if (decode_utf8_suffix_codepoint(data, codePointStart, chunkStart, codePoint)) {
		boundary.prevCodePoint = codePoint;
		boundary.prevCodePointClass = classify_boundary_codepoint(codePoint, true, scanKind);
		boundary.prevCodePointValid = true;
		boundary.prevSpaceBit =
			boundary.prevCodePointClass == BoundaryCodePointClass::ascii_space ||
			boundary.prevCodePointClass == BoundaryCodePointClass::unicode_space;
		return boundary;
	}

	boundary.prevCodePointClass = BoundaryCodePointClass::invalid;
	boundary.prevSpaceBit = 0u;
	return boundary;
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
	const ChunkBoundaryState& initialBoundary,
	const ScanModeKind scanKind,
	const uint32_t scanMode,
	const ScanProcessor processor,
	const bool countBytes,
	const bool countMaxLine)
{
	ChunkResult result{};
	ScanState state{};
	state.prevSpaceBit = initialBoundary.prevSpaceBit;

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

void merge_chunk_results(const ChunkResult* const chunks, const unsigned chunkCount, Counts& out, const bool countMaxLine) {
	uint64_t carriedLineLength = 0;

	for (unsigned index = 0; index < chunkCount; ++index) {
		const ChunkResult& chunk = chunks[index];
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

size_t choose_stream_buffer_size() noexcept {
	const char* const raw = std::getenv("FASTAWC_STREAM_BUFFER_MB");
	if (raw == nullptr || *raw == '\0') {
		return kStreamBufferSize;
	}

	char* end = nullptr;
	const unsigned long long value = std::strtoull(raw, &end, 10);
	if (end == raw || *end != '\0' || value == 0) {
		return kStreamBufferSize;
	}

	constexpr size_t kMaxStreamBufferMb = kMaxStreamBufferSize >> 20;
	const size_t clamped = static_cast<size_t>(std::min<unsigned long long>(value, kMaxStreamBufferMb));
	return clamped << 20;
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
		out = process_memory_chunk(data, size, ChunkBoundaryState{}, scanKind, scanMode, processor, countBytes, countMaxLine).counts;
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
		ChunkResult* results = nullptr;
	};

	const bool utf8Sensitive = (scanMode & (kScanWords | kScanChars | kScanMaxLine)) != 0;
	std::array<ChunkResult, kMaxParallelWorkers> results{};
	std::array<size_t, kMaxParallelWorkers> chunkStarts{};
	std::array<size_t, kMaxParallelWorkers> chunkEnds{};
	JobContext context{ data, size, workerCount, scanKind, scanMode, processor, countBytes, countMaxLine, results.data() };

	size_t chunkStart = 0;
	for (unsigned index = 0; index < workerCount; ++index) {
		const unsigned chunksLeft = workerCount - index;
		const size_t remaining = size - chunkStart;
		const size_t evenShare = remaining / chunksLeft;
		const size_t targetSize = evenShare + static_cast<size_t>(remaining % chunksLeft != 0);
		const size_t chunkEnd = align_chunk_end(chunkStart, chunkStart + targetSize, size, chunksLeft, data, utf8Sensitive);
		chunkStarts[index] = chunkStart;
		chunkEnds[index] = chunkEnd;
		chunkStart = chunkEnd;
	}

	pool.parallel_for(workerCount, [&context, &chunkStarts, &chunkEnds](const unsigned index) noexcept {
		const size_t chunkStart = chunkStarts[index];
		const size_t chunkEnd = chunkEnds[index];
		const ChunkBoundaryState initialBoundary = compute_chunk_boundary_state(context.data, chunkStart, context.scanKind);

		context.results[index] = process_memory_chunk(
			context.data + chunkStart,
			chunkEnd - chunkStart,
			initialBoundary,
			context.scanKind,
			context.scanMode,
			context.processor,
			context.countBytes,
			context.countMaxLine);
	});

	merge_chunk_results(results.data(), workerCount, out, countMaxLine);
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
	const size_t bufferSize = choose_stream_buffer_size();
	auto buffer = std::make_unique<uint8_t[]>(bufferSize);

	if (processor == nullptr) {
		if (!countBytes) {
			return;
		}

		for (;;) {
			const size_t readBytes = read_stream(source, buffer.get(), bufferSize);
			if (readBytes == 0) {
				return;
			}
			out.byteCount += readBytes;
		}
	}

	ScanState state{};
	for (;;) {
		const size_t readBytes = read_stream(source, buffer.get(), bufferSize);
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

bool try_count_regular_file_bytes(const std::string& path, Counts& out) {
	std::error_code error;
	const std::filesystem::path filePath(path);
	const bool regular = std::filesystem::is_regular_file(filePath, error);
	if (error || !regular) {
		return false;
	}

	const uintmax_t size = std::filesystem::file_size(filePath, error);
	if (error || size > std::numeric_limits<uint64_t>::max()) {
		return false;
	}

	out.byteCount = static_cast<uint64_t>(size);
	return true;
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

	if (processor == nullptr && countBytes && try_count_regular_file_bytes(path, out)) {
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

bool env_flag_enabled(const char* const name) noexcept {
	const char* const raw = std::getenv(name);
	if (raw == nullptr || *raw == '\0') {
		return false;
	}

	const std::string_view value{ raw };
	return value != "0" && value != "false" && value != "FALSE" && value != "False";
}

unsigned parse_repeat_count() noexcept {
	const char* const raw = std::getenv("FASTAWC_REPEAT");
	if (raw == nullptr || *raw == '\0') {
		return 1;
	}

	char* end = nullptr;
	const unsigned long value = std::strtoul(raw, &end, 10);
	if (end == raw || *end != '\0' || value == 0) {
		return 1;
	}
	return static_cast<unsigned>(std::min<unsigned long>(value, 1000000ul));
}

std::vector<uint8_t> make_autotune_buffer() {
	static constexpr std::string_view kPattern =
		"The Project Gutenberg eBook of War and Peace\r\n"
		"ASCII text with smart quotes \xE2\x80\x9Cquoted\xE2\x80\x9D words and accents \xC3\xA9.\r\n"
		"Tabs\tand wide chars \xE8\xA1\xA8 and combining e\xCC\x81.\r\n";

	std::vector<uint8_t> buffer;
	buffer.reserve(4u << 20);
	while (buffer.size() < (4u << 20)) {
		buffer.insert(buffer.end(), kPattern.begin(), kPattern.end());
	}
	buffer.resize(4u << 20);
	return buffer;
}

uint64_t measure_backend_processor(
	const Backend& backend,
	const ScanModeKind scanKind,
	const uint32_t scanMode,
	const uint8_t* const data,
	const size_t size) noexcept
{
	const ScanProcessor processor =
		scanKind == ScanModeKind::strict
		? backend.selectStrictProcessor(scanMode)
		: backend.selectFastProcessor(scanMode);
	if (processor == nullptr) {
		return UINT64_MAX;
	}

	uint64_t best = UINT64_MAX;
	for (unsigned iteration = 0; iteration < 3; ++iteration) {
		Counts counts{};
		ScanState state{};
		const auto started = std::chrono::steady_clock::now();
		processor(data, size, counts, state);
		finalize_scan_state(scanKind, scanMode, counts, state);
		const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
			std::chrono::steady_clock::now() - started).count();
		best = std::min<uint64_t>(best, static_cast<uint64_t>(elapsed));
	}

	return best;
}

const Backend& autotune_backend_for_workload(const ScanModeKind scanKind, const uint32_t scanMode) {
	static const std::vector<uint8_t> buffer = make_autotune_buffer();
	const Backend& scalar = scalar_backend();
#if defined(FASTAWC_HAS_AVX2_BACKEND) && FASTAWC_HAS_AVX2_BACKEND
	if (!cpu_supports_avx2()) {
		return scalar;
	}
	const Backend& avx2 = avx2_backend();
	const uint64_t scalarTime = measure_backend_processor(scalar, scanKind, scanMode, buffer.data(), buffer.size());
	const uint64_t avx2Time = measure_backend_processor(avx2, scanKind, scanMode, buffer.data(), buffer.size());
	return avx2Time < scalarTime ? avx2 : scalar;
#else
	(void)scanKind;
	(void)scanMode;
	return scalar;
#endif
}

const Backend& choose_execution_backend(const Options& options, const uint32_t scanMode) {
	if (std::getenv("FASTAWC_BACKEND") != nullptr) {
		return select_backend();
	}
	if (env_flag_enabled("FASTAWC_AUTOTUNE") &&
		options.scanKind == ScanModeKind::strict &&
		(scanMode & (kScanChars | kScanMaxLine)) != 0) {
		return autotune_backend_for_workload(options.scanKind, scanMode);
	}
	return select_backend();
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
	const RuntimeConfig config = choose_runtime_config(backend, options.scanKind, scanMode);
	ThreadPool pool(config.maxWorkers);
	const unsigned repeatCount = parse_repeat_count();
	const ScanProcessor processor =
		options.scanKind == ScanModeKind::strict
		? backend.selectStrictProcessor(scanMode)
		: backend.selectFastProcessor(scanMode);

	std::vector<OutputRow> fileRows;
	fileRows.reserve(options.files.size());
	bool hadErrors = false;

	for (const std::string& path : options.files) {
		Counts current{};
		bool processed = false;
		const unsigned iterations = path == "-" ? 1u : repeatCount;
		for (unsigned iteration = 0; iteration < iterations; ++iteration) {
			Counts iterationCounts{};
			if (!process_path(path, options.scanKind, scanMode, processor, options.optBytes, options.optMaxLine, config, pool, iterationCounts)) {
				hadErrors = true;
				processed = false;
				break;
			}
			current = iterationCounts;
			processed = true;
		}
		if (!processed) {
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
