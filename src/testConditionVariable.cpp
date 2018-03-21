//
// Created by dhding on 3/21/18.
//

#include <iostream>
#include "../util/Monitor.h"
#include "../extern/boost/threadpool.hpp"
#include <thread>
#include <mutex>
#include <condition_variable>

using namespace boost::threadpool;

const int it = 100;

class Semaphore {
public:
    Semaphore(int count_ = 0)
            : count(count_) {}

    inline void notify() {
        std::unique_lock<std::mutex> lock(mtx);
        count++;
        cv.notify_one();
    }

    inline void wait() {
        std::unique_lock<std::mutex> lock(mtx);

        while (count == 0) {
            cv.wait(lock);
        }
        count--;
    }

private:
    std::mutex mtx;
    std::condition_variable cv;
    int count;
};

void single_threaded(const unsigned long long count) {

    unsigned long long total = 0;
    for (unsigned long long i = 0; i < count; i++) {
        total += i;
    }

    std::cout << "i=" << total << std::endl;

    for (unsigned long long i = 0; i < count; i++) {
        total -= i;
    }

    std::cout << "i=" << total << std::endl;

}

void multi_threaded(const unsigned long long count, const int num_of_thread) {

    // we do not need to create new threads but reuse existed threads in the pool
    // this saves the cost for frequently creating new threads
    pool *thread_pool = new pool(num_of_thread);

    unsigned long long workload = count / num_of_thread + ((count % num_of_thread == 0) ? 0 : 1);
    std::vector<unsigned long long> individal_total(num_of_thread, 0);

    for (int thread_index = 0; thread_index < num_of_thread; thread_index++) {

        thread_pool->schedule(std::bind([&](const int thread_index) {

            unsigned long long start = workload * thread_index;
            unsigned long long end = std::min(workload + start, count);
            for (int i = 0; i < it; ++i) {
                for (unsigned long long i = start; i < end; i++) {
                    individal_total[thread_index] += i;
                }
            }

        }, thread_index));
    }

    // wait until all threads finish
    thread_pool->wait();

    unsigned long long total = 0;
    for (auto v:individal_total) {
        total += v;
    }

    std::cout << "i=" << total << std::endl;

    // reuse threads in the pool
    individal_total.resize(num_of_thread, 0);

    for (int thread_index = 0; thread_index < num_of_thread; thread_index++) {

        thread_pool->schedule(std::bind([&](const int thread_index) {

            unsigned long long start = workload * thread_index;
            unsigned long long end = std::min(workload + start, count);
            for (int i = 0; i < it; ++i) {
                for (unsigned long long i = start; i < end; i++) {
                    individal_total[thread_index] -= i;
                }
            }

        }, thread_index));
    }

    // wait until all threads finish
    thread_pool->wait();

    total = 0;
    for (auto v:individal_total) {
        total += v;
    }

    std::cout << "i=" << total << std::endl;

    // delete thread pool in the end
    delete thread_pool;
}

void cv(const unsigned long long count, const int num_of_thread) {
    pool *thread_pool = new pool(num_of_thread);
    std::condition_variable cv;
    int done = 0;
    std::mutex mtx;//mutex for done..

    unsigned long long workload = count / num_of_thread + ((count % num_of_thread == 0) ? 0 : 1);
    std::vector<unsigned long long> individal_total(num_of_thread, 0);

    for (int thread_index = 0; thread_index < num_of_thread; thread_index++) {

        thread_pool->schedule(std::bind([&](const int thread_index) {

            unsigned long long start = workload * thread_index;
            unsigned long long end = std::min(workload + start, count);

            for (int i = 0; i < it; ++i) {
                for (unsigned long long i = start; i < end; i++) {
                    individal_total[thread_index] += i;
                }
                {
                    std::lock_guard<std::mutex> lock(mtx);
                    done++;
                    cv.notify_all();
                    if (done == num_of_thread)continue;
                }
                std::unique_lock<std::mutex> l(mtx);
                while (done < num_of_thread) {
                    cv.wait(l);
                }
            }
        }, thread_index));
    }

    // wait until all threads finish
    thread_pool->wait();

    unsigned long long total = 0;
    for (auto v:individal_total) {
        total += v;
    }

    std::cout << "i=" << total << std::endl;

    // reuse threads in the pool
    individal_total.resize(num_of_thread, 0);

    for (int thread_index = 0; thread_index < num_of_thread; thread_index++) {

        thread_pool->schedule(std::bind([&](const int thread_index) {

            unsigned long long start = workload * thread_index;
            unsigned long long end = std::min(workload + start, count);

            for (int i = 0; i < it; ++i) {
                for (unsigned long long i = start; i < end; i++) {
                    individal_total[thread_index] -= i;
                }
                {
                    std::lock_guard<std::mutex> lock(mtx);
                    done++;
                    cv.notify_all();
                    if (done == num_of_thread)continue;
                }
                std::unique_lock<std::mutex> l(mtx);
                while (done < num_of_thread) {
                    cv.wait(l);
                }
            }

        }, thread_index));
    }

    // wait until all threads finish
    thread_pool->wait();

    total = 0;
    for (auto v:individal_total) {
        total += v;
    }

    std::cout << "i=" << total << std::endl;

    // delete thread pool in the end
    delete thread_pool;
}


void wait_test(const unsigned long long count, const int num_of_thread) {

    // we do not need to create new threads but reuse existed threads in the pool
    // this saves the cost for frequently creating new threads
    pool *thread_pool = new pool(num_of_thread);

    unsigned long long workload = count / num_of_thread + ((count % num_of_thread == 0) ? 0 : 1);
    std::vector<unsigned long long> individal_total(num_of_thread, 0);

    for (int i = 0; i < it; ++i) {
        for (int thread_index = 0; thread_index < num_of_thread; thread_index++) {

            thread_pool->schedule(std::bind([&](const int thread_index) {

                unsigned long long start = workload * thread_index;
                unsigned long long end = std::min(workload + start, count);
                for (int i = 0; i < 10; ++i) {
                    for (unsigned long long i = start; i < end; i++) {
                        individal_total[thread_index] += i;
                    }
                }

            }, thread_index));
        }

        // wait until all threads finish
        thread_pool->wait();
    }
    unsigned long long total = 0;
    for (auto v:individal_total) {
        total += v;
    }

    std::cout << "i=" << total << std::endl;

    // reuse threads in the pool
    individal_total.resize(num_of_thread, 0);

    for (int i = 0; i < it; ++i) {
        for (int thread_index = 0; thread_index < num_of_thread; thread_index++) {

            thread_pool->schedule(std::bind([&](const int thread_index) {

                unsigned long long start = workload * thread_index;
                unsigned long long end = std::min(workload + start, count);

                for (unsigned long long i = start; i < end; i++) {
                    individal_total[thread_index] -= i;
                }

            }, thread_index));
        }

        // wait until all threads finish
        thread_pool->wait();
    }

    total = 0;
    for (auto v:individal_total) {
        total += v;
    }

    std::cout << "i=" << total << std::endl;

    // delete thread pool in the end
    delete thread_pool;
}

int main() {

    unsigned long long count = 1000000001;

    Monitor timer;

    timer.start();
    single_threaded(count);
    timer.stop();

    std::cout << "time for single-threaded test: " << timer.getElapsedTime() << " secs" << std::endl;

    int num_of_thread = 4;

    timer.start();
    multi_threaded(count, num_of_thread);
    timer.stop();

    // In optimal situation, it should be 1/num_of_thread fraction of the time for single-threaded test.
    // But there are additional cost and the time should be between 1/num_of_thread and 1/(num_of_thread-1) fraction of the time for single-threaded test.
    std::cout << "time for multi-threaded test: " << timer.getElapsedTime() << " secs" << std::endl;

    timer.start();
    cv(count, num_of_thread);
    timer.stop();

    // In optimal situation, it should be 1/num_of_thread fraction of the time for single-threaded test.
    // But there are additional cost and the time should be between 1/num_of_thread and 1/(num_of_thread-1) fraction of the time for single-threaded test.
    std::cout << "time for condition variable test: " << timer.getElapsedTime() << " secs" << std::endl;

    timer.start();
    wait_test(count, num_of_thread);
    timer.stop();

    // In optimal situation, it should be 1/num_of_thread fraction of the time for single-threaded test.
    // But there are additional cost and the time should be between 1/num_of_thread and 1/(num_of_thread-1) fraction of the time for single-threaded test.
    std::cout << "time for threadpool wait test: " << timer.getElapsedTime() << " secs" << std::endl;

    return 0;
}