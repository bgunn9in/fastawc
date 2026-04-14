#include "thread_pool.h"

namespace fastawc {

ThreadPool::ThreadPool(const unsigned workerCount) {
	const unsigned backgroundWorkers = workerCount > 0 ? workerCount - 1 : 0;
	workers_.reserve(backgroundWorkers);
	for (unsigned i = 0; i < backgroundWorkers; ++i) {
		workers_.emplace_back([this]() { worker_loop(); });
	}
}

ThreadPool::~ThreadPool() {
	{
		std::lock_guard lock(mutex_);
		stop_ = true;
	}
	workAvailable_.notify_all();
	for (std::thread& worker : workers_) {
		worker.join();
	}
}

unsigned ThreadPool::worker_count() const noexcept {
	return static_cast<unsigned>(workers_.size()) + 1u;
}

void ThreadPool::run_batch_worker(
	void* const context,
	void(*const fn)(void*, unsigned) noexcept,
	const unsigned taskCount) noexcept
{
	for (;;) {
		const unsigned index = nextIndex_.fetch_add(1, std::memory_order_relaxed);
		if (index >= taskCount) {
			break;
		}
		fn(context, index);
	}

	if (activeWorkers_.fetch_sub(1, std::memory_order_acq_rel) == 1u) {
		std::lock_guard lock(mutex_);
		workFinished_.notify_one();
	}
}

void ThreadPool::worker_loop() {
	unsigned seenGeneration = 0;
	for (;;) {
		void(*fn)(void*, unsigned) noexcept = nullptr;
		void* context = nullptr;
		unsigned taskCount = 0;
		{
			std::unique_lock lock(mutex_);
			workAvailable_.wait(lock, [this, &seenGeneration]() noexcept { return stop_ || generation_ != seenGeneration; });
			if (stop_) {
				return;
			}
			seenGeneration = generation_;
			fn = batchFn_;
			context = batchContext_;
			taskCount = batchTaskCount_;
		}

		run_batch_worker(context, fn, taskCount);
	}
}

void ThreadPool::parallel_for_impl(
	const unsigned taskCount,
	void* const context,
	void(*const fn)(void*, unsigned) noexcept)
{
	if (taskCount == 0 || fn == nullptr) {
		return;
	}
	if (taskCount == 1 || workers_.empty()) {
		for (unsigned index = 0; index < taskCount; ++index) {
			fn(context, index);
		}
		return;
	}

	{
		std::lock_guard lock(mutex_);
		batchFn_ = fn;
		batchContext_ = context;
		batchTaskCount_ = taskCount;
		nextIndex_.store(0, std::memory_order_relaxed);
		activeWorkers_.store(static_cast<unsigned>(workers_.size()) + 1u, std::memory_order_release);
		++generation_;
	}

	workAvailable_.notify_all();
	run_batch_worker(context, fn, taskCount);

	std::unique_lock lock(mutex_);
	workFinished_.wait(lock, [this]() noexcept { return activeWorkers_.load(std::memory_order_acquire) == 0u; });
}

} // namespace fastawc
