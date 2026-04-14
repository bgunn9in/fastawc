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

void ThreadPool::worker_loop() {
	for (;;) {
		Task task{};
		{
			std::unique_lock lock(mutex_);
			workAvailable_.wait(lock, [this]() noexcept { return stop_ || !tasks_.empty(); });
			if (stop_ && tasks_.empty()) {
				return;
			}

			task = tasks_.front();
			tasks_.pop_front();
		}

		task.fn(task.context, task.index);

		{
			std::lock_guard lock(mutex_);
			if (--pending_ == 0) {
				workFinished_.notify_one();
			}
		}
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
		pending_ = taskCount - 1;
		for (unsigned index = 1; index < taskCount; ++index) {
			tasks_.push_back(Task{ fn, context, index });
		}
	}

	workAvailable_.notify_all();
	fn(context, 0);

	std::unique_lock lock(mutex_);
	workFinished_.wait(lock, [this]() noexcept { return pending_ == 0; });
}

} // namespace fastawc
