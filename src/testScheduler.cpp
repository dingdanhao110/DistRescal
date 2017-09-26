#include "../util/Calculator.h"
#include "../util/Base.h"
#include "../struct/Scheduler.h"
#include "../struct/Sample.h"

int main(int argc, char **argv) {
    Scheduler scheduler(20, 10);
    for (int i = 0; i < 10000; ++i) {
        Sample sample;
        sample.n_sub = rand() % 199;
        sample.n_obj = rand() % 199;
        sample.p_sub = rand() % 199;
        sample.p_obj = sample.n_obj;
        scheduler.insert_sample(sample, i);
    }
    while (!scheduler.finished()) {
        array<int, 3> block;
        set<int> to_update;
        scheduler.schedule_next(block, to_update);
        cerr << "Current count: " << scheduler.count << endl;
        scheduler.report_finish(block);
        cout<<"???\n";
    }
    return 0;
}