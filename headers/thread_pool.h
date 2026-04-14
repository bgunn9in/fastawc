#pragma once

#include <condition_variable>
#include <cstdint>
#include <deque>
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
	struct Task {
		void(*fn)(void*, unsigned) noexcept = nullptr;
		void* context = nullptr;
		unsigned index = 0;
	};

	void worker_loop();
	void parallel_for_impl(unsigned taskCount, void* context, void(*fn)(void*, unsigned) noexcept);

	std::vector<std::thread> workers_;
	std::deque<Task> tasks_;
	std::mutex mutex_;
	std::condition_variable workAvailable_;
	std::condition_variable workFinished_;
	unsigned pending_ = 0;
	bool stop_ = false;
};

} // namespace fastawc
