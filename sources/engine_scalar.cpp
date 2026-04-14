#include "engine.h"

#define FASTAWC_BACKEND_NAMESPACE scalar_backend_impl
#define FASTAWC_BACKEND_NAME "scalar"
#define FASTAWC_BACKEND_ENABLE_AVX2 0
#include "engine_impl.h"
#undef FASTAWC_BACKEND_ENABLE_AVX2
#undef FASTAWC_BACKEND_NAME
#undef FASTAWC_BACKEND_NAMESPACE

namespace fastawc {

const Backend& scalar_backend() noexcept {
	return scalar_backend_impl::backend_instance();
}

} // namespace fastawc
