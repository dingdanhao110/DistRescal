#include "../util/Calculator.h"
#include "../util/Base.h"
#include "../struct/Scheduler2.h"
#include "../struct/Sample.h"
#include "../extern/boost/threadpool.hpp"

int main(int argc, char **argv) {
    auto computation_thread_pool = new pool(4);
    Scheduler2 scheduler(15, 40943);
    std::mutex mutex_scheduler;
    for (int i = 0; i < 1000000; ++i) {
        Sample sample;
        sample.n_sub = rand() % (15*40943-1);
        sample.n_obj = rand() % (15*40943-1);
        sample.p_sub = rand() % (15*40943-1);
        sample.p_obj = sample.n_obj;
        scheduler.insert_sample(sample, i);
    }

    while (!scheduler.finished()) {
        computation_thread_pool->wait(3);
        array<int, 3> locks;
        set<int> to_update;
        {
            std::lock_guard<std::mutex> lock(mutex_scheduler);
            scheduler.schedule_next(locks, to_update);
        }
        if(to_update.size()) {
            computation_thread_pool->schedule(std::bind([&](array<int,3> locks,set<int> to_update) {
                cerr<<"Thread "<<std::this_thread::get_id()<<": Work assigned!\n";
                //Call update functions
                for (auto &sample:to_update) {
                    //assume update
                }
                {
                    std::lock_guard<std::mutex>l(mutex_scheduler);
                    scheduler.report_finish(locks);
                }
                cerr<<"Thread "<<std::this_thread::get_id()<<": Work done!\n";
            },move(locks),move(to_update)));
        }
        else{
            cerr << "Pending blocks: " << scheduler.pending_blocks.size() << endl;
            cerr << "Active threads: " << computation_thread_pool->active() << endl;
            int i=0;
            for(int l:scheduler.lock_status){
                i+=l;
            }
            cerr << "Current locks: " << i << endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        cerr << "Pending blocks: " << scheduler.pending_blocks.size() << endl;
        //cout<<"???\n";
    }
    return 0;
}