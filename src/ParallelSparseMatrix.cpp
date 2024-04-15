#include <atomic>
#include <fstream>
#include <sstream>
#include <ctime>

#include "ParallelSparseMatrix.hpp"
#include "ThreadPool.hpp"

void addTasksAndWait(ThreadPool& pool, size_t n_, std::function<void(size_t)> task) {
    const size_t batch_size = 10000;
    size_t batch_count = (n_ + batch_size - 1) / batch_size;

    std::condition_variable cv;
    size_t tasks_completed = 0;
    std::mutex mutex;

    for (size_t batch_num = 0; batch_num < batch_count; batch_num++) {
        pool.enqueue([&cv, &tasks_completed, &mutex, batch_count, batch_num, n_, task] {
            size_t start = batch_num * batch_size;
            size_t end = std::min(start + batch_size, n_);
            for (size_t i = start; i < end; i++) {
                task(i);
            }

            {
                std::unique_lock<std::mutex> lock(mutex);
                if (++tasks_completed == batch_count) {
                    cv.notify_one();
                }
            }
        });
    }

    {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&] {
            return tasks_completed == batch_count;
        });
    }
}

ParallelSparseMatrix::ParallelSparseMatrix(const std::string& file_path) : SparseMatrixBase(file_path) {}

std::vector<uint32_t> ParallelSparseMatrix::reduce(bool run_twist) {
    if (run_twist) {
        runTwist();
    }

    std::atomic<bool> need_widen_buffer = false;
    ThreadPool pool;

    std::vector<uint32_t> row_index_buffer(row_index_.size(), 0);
    std::vector<uint32_t> to_add(n_, n_);
    std::vector<uint32_t> inverse_low(n_, n_);
    while (true) {
        inverse_low.assign(n_, n_);
        bool is_over = true;
        for (uint32_t i = 0; i < n_; i++) {
            uint32_t cur_low = getLow(i);
            if (cur_low == n_) {
                continue;
            }
            uint32_t cur_inverse_low = inverse_low[cur_low];

            if (cur_inverse_low == n_) {
                inverse_low[cur_low] = i;
            } else {
                to_add[i] = cur_inverse_low;
            }
            if (to_add[i] != n_) {
                is_over = false;
            }
        }
        if (is_over) {
            break;
        }

        if (need_widen_buffer.load()) {
            widenBuffer(row_index_buffer);
        }
        need_widen_buffer.store(false);

        addTasksAndWait(pool, n_, [&](size_t i) {
            if (to_add[i] != n_) {
                addColumn(i, to_add[i], row_index_buffer);
                if (!need_widen_buffer.load() && !enoughSizeForIteration(i)) {
                    need_widen_buffer.store(true);
                }
                to_add[i] = n_;
            }
        });
    }

    return getLowArray();
}
