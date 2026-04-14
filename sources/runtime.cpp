#include "engine.h"

#include <algorithm>
#include <bit>
#include <cstdlib>
#include <string_view>
#include <thread>

#if defined(_M_X64) || defined(_M_IX86)
#include <intrin.h>
#elif defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif

namespace fastawc {

namespace {

constexpr unsigned kMaxParallelWorkers = 8;

bool equals_ascii_ci(const std::string_view lhs, const std::string_view rhs) noexcept {
	if (lhs.size() != rhs.size()) {
		return false;
	}

	for (size_t i = 0; i < lhs.size(); ++i) {
		char a = lhs[i];
		char b = rhs[i];
		if (a >= 'A' && a <= 'Z') {
			a = static_cast<char>(a - 'A' + 'a');
		}
		if (b >= 'A' && b <= 'Z') {
			b = static_cast<char>(b - 'A' + 'a');
		}
		if (a != b) {
			return false;
		}
	}

	return true;
}

size_t parse_env_size_mb(const char* const name, const size_t fallback) noexcept {
	const char* const raw = std::getenv(name);
	if (raw == nullptr || *raw == '\0') {
		return fallback;
	}

	char* end = nullptr;
	const unsigned long long value = std::strtoull(raw, &end, 10);
	if (end == raw || *end != '\0' || value == 0) {
		return fallback;
	}

	return static_cast<size_t>(value) << 20;
}

unsigned parse_env_uint(const char* const name, const unsigned fallback) noexcept {
	const char* const raw = std::getenv(name);
	if (raw == nullptr || *raw == '\0') {
		return fallback;
	}

	char* end = nullptr;
	const unsigned long value = std::strtoul(raw, &end, 10);
	if (end == raw || *end != '\0' || value == 0) {
		return fallback;
	}

	return static_cast<unsigned>(value);
}

} // namespace

bool cpu_supports_avx2() noexcept {
#if !(defined(FASTAWC_HAS_AVX2_BACKEND) && FASTAWC_HAS_AVX2_BACKEND)
	return false;
#elif defined(_M_X64) || defined(_M_IX86)
	int regs[4] = {};
	__cpuidex(regs, 1, 0);
	const bool osxsave = (regs[2] & (1 << 27)) != 0;
	const bool avx = (regs[2] & (1 << 28)) != 0;
	if (!osxsave || !avx) {
		return false;
	}

	const unsigned long long xcr0 = _xgetbv(0);
	if ((xcr0 & 0x6u) != 0x6u) {
		return false;
	}

	__cpuidex(regs, 7, 0);
	return (regs[1] & (1 << 5)) != 0;
#elif defined(__x86_64__) || defined(__i386__)
	unsigned eax = 0;
	unsigned ebx = 0;
	unsigned ecx = 0;
	unsigned edx = 0;
	if (__get_cpuid_max(0, nullptr) == 0) {
		return false;
	}

	__cpuid_count(1, 0, eax, ebx, ecx, edx);
	const bool osxsave = (ecx & bit_OSXSAVE) != 0;
	const bool avx = (ecx & bit_AVX) != 0;
	if (!osxsave || !avx) {
		return false;
	}

	unsigned xcr0Low = 0;
	unsigned xcr0High = 0;
	asm volatile("xgetbv" : "=a"(xcr0Low), "=d"(xcr0High) : "c"(0));
	if ((xcr0Low & 0x6u) != 0x6u) {
		return false;
	}

	__cpuid_count(7, 0, eax, ebx, ecx, edx);
	return (ebx & bit_AVX2) != 0;
#else
	return false;
#endif
}

const Backend& select_backend() noexcept {
	const char* const forced = std::getenv("FASTAWC_BACKEND");
	if (forced != nullptr) {
		const std::string_view requested{ forced };
		if (equals_ascii_ci(requested, "scalar")) {
			return scalar_backend();
		}
#if defined(FASTAWC_HAS_AVX2_BACKEND) && FASTAWC_HAS_AVX2_BACKEND
		if (equals_ascii_ci(requested, "avx2") && cpu_supports_avx2()) {
			return avx2_backend();
		}
#endif
	}

#if defined(FASTAWC_HAS_AVX2_BACKEND) && FASTAWC_HAS_AVX2_BACKEND
	if (cpu_supports_avx2()) {
		return avx2_backend();
	}
#endif

	return scalar_backend();
}

RuntimeConfig choose_runtime_config(const Backend& backend, const ScanModeKind scanKind, const uint32_t scanMode) noexcept {
	RuntimeConfig config{};
	if (scanMode == 0) {
		config.maxWorkers = 1;
		return config;
	}

	unsigned hardwareWorkers = std::thread::hardware_concurrency();
	if (hardwareWorkers == 0) {
		hardwareWorkers = 1;
	}

	hardwareWorkers = std::min(hardwareWorkers, kMaxParallelWorkers);
	hardwareWorkers = parse_env_uint("FASTAWC_THREADS", hardwareWorkers);
	hardwareWorkers = std::max(1u, std::min(hardwareWorkers, kMaxParallelWorkers));

	const unsigned workBits = std::popcount(scanMode);
	const bool expensive =
		(scanMode & kScanWords) != 0 ||
		(scanMode & kScanChars) != 0 ||
		(scanMode & kScanMaxLine) != 0;
	const bool strictHeavy = scanKind == ScanModeKind::strict && (scanMode & (kScanChars | kScanMaxLine)) != 0;
	const bool strictClassic = scanKind == ScanModeKind::strict && !strictHeavy;

	size_t minParallelFileSize = backend.isAvx2 ? (expensive ? (64ull << 20) : (96ull << 20)) : (expensive ? (96ull << 20) : (128ull << 20));
	size_t minBytesPerWorker = backend.isAvx2 ? (expensive ? (48ull << 20) : (64ull << 20)) : (expensive ? (64ull << 20) : (96ull << 20));
	size_t targetChunkSize = backend.isAvx2 ? (expensive ? (32ull << 20) : (48ull << 20)) : (expensive ? (48ull << 20) : (64ull << 20));

	if (strictHeavy) {
		minParallelFileSize = backend.isAvx2 ? (32ull << 20) : (24ull << 20);
		minBytesPerWorker = backend.isAvx2 ? (24ull << 20) : (16ull << 20);
		targetChunkSize = backend.isAvx2 ? (24ull << 20) : (16ull << 20);
		hardwareWorkers = std::min(hardwareWorkers, 6u);
	}
	else if (strictClassic) {
		minParallelFileSize = backend.isAvx2 ? (32ull << 20) : (48ull << 20);
		minBytesPerWorker = backend.isAvx2 ? (24ull << 20) : (32ull << 20);
		targetChunkSize = backend.isAvx2 ? (24ull << 20) : (32ull << 20);
	}

	if (hardwareWorkers <= 2) {
		minParallelFileSize *= 2;
		minBytesPerWorker *= 2;
		targetChunkSize *= 2;
	}
	else if (hardwareWorkers >= 8 && workBits >= 3) {
		minParallelFileSize /= 2;
		minBytesPerWorker = std::max<size_t>(minBytesPerWorker / 2, 16ull << 20);
		targetChunkSize = std::max<size_t>(targetChunkSize / 2, 16ull << 20);
	}

	config.maxWorkers = hardwareWorkers;
	config.minParallelFileSize = parse_env_size_mb("FASTAWC_MIN_PARALLEL_MB", minParallelFileSize);
	config.minBytesPerWorker = parse_env_size_mb("FASTAWC_BYTES_PER_WORKER_MB", minBytesPerWorker);
	config.targetChunkSize = parse_env_size_mb("FASTAWC_TARGET_CHUNK_MB", targetChunkSize);
	return config;
}

void finalize_scan_state(const ScanModeKind scanKind, const uint32_t scanMode, Counts& counts, ScanState& state) noexcept {
	if (scanKind != ScanModeKind::strict || scanMode == 0 || state.utf8Expected == 0) {
		return;
	}

	state.utf8CodePoint = 0;
	state.utf8MinCodePoint = 0;
	state.utf8Expected = 0;
	state.utf8Seen = 0;

	if ((scanMode & kScanWords) != 0) {
		counts.wordCount += static_cast<uint64_t>(state.prevSpaceBit);
		state.prevSpaceBit = 0;
	}
}

} // namespace fastawc
