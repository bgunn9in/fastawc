#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace fastawc {

class ThreadPool {
public:
	explicit ThreadPool(unsigned workerCount);
	ThreadPool(const ThreadPool&) = delete;
	ThreadPool& operator=(const ThreadPool&) = delete;
	~ThreadPool();

	unsigned worker_count() const noexcept;

	template<class Fn>
	void parallel_for(const unsigned taskCount, Fn&& fn) {
		using Functor = std::decay_t<Fn>;
		struct Context {
			Functor* fn;
		} context{ &fn };

		auto thunk = [](void* const opaque, const unsigned index) noexcept {
			auto* const ctx = static_cast<Context*>(opaque);
			(*ctx->fn)(index);
		};

		parallel_for_impl(taskCount, &context, thunk);
	}

private:
	void worker_loop();
	void parallel_for_impl(unsigned taskCount, void* context, void(*fn)(void*, unsigned) noexcept);
	void run_batch_worker(void* context, void(*fn)(void*, unsigned) noexcept, unsigned taskCount) noexcept;

	std::vector<std::thread> workers_;
	std::mutex mutex_;
	std::condition_variable workAvailable_;
	std::condition_variable workFinished_;
	void(*batchFn_)(void*, unsigned) noexcept = nullptr;
	void* batchContext_ = nullptr;
	unsigned batchTaskCount_ = 0;
	unsigned generation_ = 0;
	std::atomic<unsigned> nextIndex_{ 0 };
	std::atomic<unsigned> activeWorkers_{ 0 };
	bool stop_ = false;
};

} // namespace fastawc
