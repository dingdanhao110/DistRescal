#include "../util/Calculator.h"
#include "../util/Base.h"
#include "../struct/Scheduler2.h"
#include "../struct/Sample.h"
#include "../extern/boost/threadpool.hpp"

int main(int argc, char **argv) {
    auto computation_thread_pool = new pool(4);
    Scheduler2 scheduler(15, 100);
    std::mutex mutex_scheduler;
    for (int i = 0; i < 1000000; ++i) {
        Sample sample;
        sample.n_sub = rand() % 1499;
        sample.n_obj = rand() % 1499;
        sample.p_sub = rand() % 1499;
        sample.p_obj = sample.n_obj;
        scheduler.insert_sample(sample, i);
    }

    while (!scheduler.finished()) {
        computation_thread_pool->wait(3);
        array<int, 3> block;
        set<int> to_update;
        {
            std::lock_guard<std::mutex> lock(mutex_scheduler);
            scheduler.schedule_next(block, to_update);
        }
        if(to_update.size()) {
            {
                std::lock_guard<std::mutex> lock(mutex_scheduler);
                scheduler.report_finish(block);
            }
        }
        cerr << "Pending blocks: " << scheduler.pending_blocks.size() << endl;
        //cout<<"???\n";
    }
    return 0;
}