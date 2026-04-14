#include <immintrin.h>

#include "engine.h"

#define FASTAWC_BACKEND_NAMESPACE avx2_backend_impl
#define FASTAWC_BACKEND_NAME "avx2"
#define FASTAWC_BACKEND_ENABLE_AVX2 1
#include "engine_impl.h"
#undef FASTAWC_BACKEND_ENABLE_AVX2
#undef FASTAWC_BACKEND_NAME
#undef FASTAWC_BACKEND_NAMESPACE

namespace fastawc {

const Backend& avx2_backend() noexcept {
	return avx2_backend_impl::backend_instance();
}

} // namespace fastawc
