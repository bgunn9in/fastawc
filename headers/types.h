#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#define FASTAWC_FORCEINLINE __forceinline
#else
#define FASTAWC_FORCEINLINE inline __attribute__((always_inline))
#endif

namespace fastawc {

enum class TotalMode : uint8_t {
	auto_mode = 0,
	always,
	only,
	never
};

enum class ScanModeKind : uint8_t {
	fast = 0,
	strict
};

struct Counts {
	uint64_t lineCount = 0;
	uint64_t wordCount = 0;
	uint64_t byteCount = 0;
	uint64_t charCount = 0;
	uint64_t maxLineLength = 0;
};

struct Options {
	bool optLines = false;
	bool optWords = false;
	bool optBytes = false;
	bool optChars = false;
	bool optMaxLine = false;
	bool implicitStdin = false;
	ScanModeKind scanKind = ScanModeKind::fast;
	TotalMode totalMode = TotalMode::auto_mode;
	std::vector<std::string> files;
};

struct ScanState {
	uint32_t prevSpaceBit = 1;
	uint64_t currentLineLength = 0;
	uint64_t prefixLineLength = 0;
	uint32_t utf8CodePoint = 0;
	uint32_t utf8MinCodePoint = 0;
	uint8_t utf8Expected = 0;
	uint8_t utf8Seen = 0;
	bool sawNewline = false;
};

struct RuntimeConfig {
	unsigned maxWorkers = 1;
	size_t minParallelFileSize = std::numeric_limits<size_t>::max();
	size_t minBytesPerWorker = std::numeric_limits<size_t>::max();
	size_t targetChunkSize = std::numeric_limits<size_t>::max();
};

constexpr uint32_t kScanLines = 1u << 0;
constexpr uint32_t kScanWords = 1u << 1;
constexpr uint32_t kScanChars = 1u << 2;
constexpr uint32_t kScanMaxLine = 1u << 3;

constexpr size_t kStreamBufferSize = 8u << 20;

FASTAWC_FORCEINLINE constexpr uint32_t make_scan_mode(const Options& options) noexcept {
	return
		(static_cast<uint32_t>(options.optLines) << 0) |
		(static_cast<uint32_t>(options.optWords) << 1) |
		(static_cast<uint32_t>(options.optChars) << 2) |
		(static_cast<uint32_t>(options.optMaxLine) << 3);
}

} // namespace fastawc
