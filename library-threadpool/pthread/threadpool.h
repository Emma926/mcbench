/**
From PyTorch:

Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

From Caffe2:

Copyright (c) 2016-present, Facebook Inc. All rights reserved.

All contributions by Facebook:
Copyright (c) 2016 Facebook Inc.
 
All contributions by Google:
Copyright (c) 2015 Google Inc.
All rights reserved.
 
All contributions by Yangqing Jia:
Copyright (c) 2015 Yangqing Jia
All rights reserved.
 
All contributions from Caffe:
Copyright(c) 2013, 2014, 2015, the respective contributors
All rights reserved.
 
All other contributions:
Copyright(c) 2015, 2016 the respective contributors
All rights reserved.
 
Caffe2 uses a copyright model similar to Caffe: each contributor holds
copyright over their contributions to Caffe2. The project versioning records
all such contribution and copyright details. If a contributor wants to further
mark their specific copyright on a particular contribution, they should
indicate their copyright solely in the commit message of the change when it is
committed.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
   and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
**/

// original file:
// https://github.com/pytorch/pytorch/blob/6b748567478ccf88fbce31e7e29eab8c6e9e832e/c10/core/thread_pool.h

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>
#include <atomic>
#include <future>
#include <iostream>


class TaskThreadPoolBase {
 public:
  virtual void run(const std::function<void()>& func) = 0;

  virtual size_t size() const = 0;

  /**
   * The number of available (i.e. idle) threads in this thread pool.
   */
  virtual size_t numAvailable() const = 0;

  /**
   * Check if the current thread is from the thread pool.
   */
  virtual bool inThreadPool() const = 0;

  virtual ~TaskThreadPoolBase() noexcept {}

  static size_t defaultNumThreads() {
    return std::thread::hardware_concurrency();
  }
};

class ThreadPool : public TaskThreadPoolBase {
 protected:
  struct task_element_t {
    bool run_with_id;
    const std::function<void()> no_id;
    const std::function<void(std::size_t)> with_id;

    explicit task_element_t(const std::function<void()>& f)
        : run_with_id(false), no_id(f), with_id(nullptr) {}
    explicit task_element_t(const std::function<void(std::size_t)>& f)
        : run_with_id(true), no_id(nullptr), with_id(f) {}
  };

  std::queue<task_element_t> tasks_;
  std::vector<std::thread> threads_;
  std::mutex mutex_;
  std::condition_variable condition_;
  std::condition_variable completed_;
  std::atomic_bool running_;
  bool complete_;
  std::size_t available_;
  std::size_t total_;
  int numa_node_id_;

 public:
  ThreadPool() = delete;

  explicit ThreadPool(
      int pool_size,
      int numa_node_id = -1,
      std::function<void()> init_thread = nullptr);

  ~ThreadPool();

  size_t size() const override;

  size_t numAvailable() const override;

  bool inThreadPool() const override;

  void run(const std::function<void()>& func) override;

  template <typename Task>
  void runTaskWithID(Task task) {
    std::unique_lock<std::mutex> lock(mutex_);

    // Set task and signal condition variable so that a worker thread will
    // wake up and use the task.
    tasks_.push(
        task_element_t(static_cast<std::function<void(std::size_t)>>(task)));
    complete_ = false;
    condition_.notify_one();
  }

  /// @brief Wait for queue to be empty
  void waitWorkComplete();

 private:
  // @brief Entry point for pool threads.
  void main_loop(std::size_t index);
};
