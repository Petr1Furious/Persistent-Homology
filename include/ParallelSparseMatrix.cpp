#include "ParallelSparseMatrix.hpp"

#include <atomic>
#include <ctime>
#include <fstream>
#include <sstream>

#include "ThreadPool.hpp"

void addTasksAndWait(ThreadPool& pool, size_t n_,
                     std::function<void(size_t)> task) {
    const size_t batch_size = 10000;
    size_t batch_count = (n_ + batch_size - 1) / batch_size;

    std::condition_variable cv;
    size_t tasks_completed = 0;
    std::mutex mutex;

    for (size_t batch_num = 0; batch_num < batch_count; batch_num++) {
        pool.enqueue(
            [&cv, &tasks_completed, &mutex, batch_count, batch_num, n_, task] {
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
        cv.wait(lock, [&] { return tasks_completed == batch_count; });
    }
}

ParallelSparseMatrix::ParallelSparseMatrix(const std::string& file_path)
    : SparseMatrixBase(file_path) {}

std::vector<uint32_t> ParallelSparseMatrix::reduce(bool run_twist) {
    if (run_twist) {
        runTwist();
    }

    std::atomic<bool> need_widen_buffer = false;
    ThreadPool pool;

    std::vector<uint32_t> row_index_buffer(row_index_.size(), 0);
    std::vector<uint32_t> to_add(n_, n_);
    std::vector<std::atomic<uint32_t>> inverse_low(n_);
    for (auto& i : inverse_low) {
        i.store(n_);
    }
    while (true) {
        addTasksAndWait(pool, n_, [&](size_t i) {
            uint32_t cur_low = getLow(i);
            if (cur_low != n_) {
                while (true) {
                    uint32_t cur_value = inverse_low[cur_low].load();
                    if (i > cur_value) {
                        break;
                    }
                    if (inverse_low[cur_low].compare_exchange_weak(cur_value,
                                                                   i)) {
                        break;
                    }
                }
            }
        });

        addTasksAndWait(pool, n_, [&](size_t i) {
            uint32_t cur_low = getLow(i);
            if (cur_low != n_ && inverse_low[cur_low].load() != i) {
                to_add[i] = inverse_low[cur_low].load();
            }

            if (!need_widen_buffer.load() && !enoughSizeForIteration(i, to_add[i])) {
                need_widen_buffer.store(true, std::memory_order_relaxed);
            }
        });

        bool is_over = true;
        for (size_t i = 0; i < n_; i++) {
            if (to_add[i] != n_) {
                is_over = false;
                break;
            }
        }
        if (is_over) {
            break;
        }

        if (need_widen_buffer.load()) {
            std::vector<uint32_t> new_col_start(n_);

            uint32_t cur_col_start = 0;
            for (size_t i = 0; i < n_; i++) {
                new_col_start[i] = cur_col_start;

                uint32_t len = col_end_[i] - col_start_[i];
                cur_col_start += widen_coef_ * len;
                if (len != 0) {
                    cur_col_start += (i == n_ - 1 ? (uint32_t)row_index_.size()
                                                  : col_start_[i + 1]) -
                                     col_end_[i];
                }
            }

            uint32_t new_size = cur_col_start;

            row_index_buffer.resize(new_size, 0);

            addTasksAndWait(pool, n_, [&](size_t i) {
                int start = col_start_[i];
                int end = col_end_[i];
                int new_start = new_col_start[i];

                std::copy(row_index_.begin() + start, row_index_.begin() + end,
                          row_index_buffer.begin() + new_start);

                col_start_[i] = new_start;
                col_end_[i] = new_start + (end - start);
            });

            row_index_.resize(new_size, 0);
            std::swap(row_index_, row_index_buffer);
        }
        need_widen_buffer.store(false);

        addTasksAndWait(pool, n_, [&](size_t i) {
            if (to_add[i] != n_) {
                addColumn(i, to_add[i], row_index_buffer);
                to_add[i] = n_;
            }

            inverse_low[i].store(n_);
        });
    }

    return getLowArray();
}
