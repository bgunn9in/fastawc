#pragma once

#include "types.h"

namespace fastawc {

using ScanProcessor = void(*)(const uint8_t*, size_t, Counts&, ScanState&) noexcept;

struct Backend {
	const char* name = "scalar";
	bool isAvx2 = false;
	ScanProcessor (*selectFastProcessor)(uint32_t scanMode) noexcept = nullptr;
	ScanProcessor (*selectStrictProcessor)(uint32_t scanMode) noexcept = nullptr;
};

const Backend& scalar_backend() noexcept;

#if defined(FASTAWC_HAS_AVX2_BACKEND) && FASTAWC_HAS_AVX2_BACKEND
const Backend& avx2_backend() noexcept;
#endif

bool cpu_supports_avx2() noexcept;
const Backend& select_backend() noexcept;
RuntimeConfig choose_runtime_config(const Backend& backend, uint32_t scanMode) noexcept;
void finalize_scan_state(ScanModeKind scanKind, uint32_t scanMode, Counts& counts, ScanState& state) noexcept;

} // namespace fastawc
