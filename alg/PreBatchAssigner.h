//
// Created by dhding on 9/30/17.
//

#ifndef DISTRESCAL_PREBATCH_H
#define DISTRESCAL_PREBATCH_H

#include "../util/Base.h"
#include "../struct/Sample.h"
#include <queue>
#include "../struct/Samples.h"

using namespace std;

class PreBatch_assigner {
private:
    vector<set<int>> buckets;
    const Samples& samples;
    vector<vector<vector<queue<int>>>>& plan;
    //<iteration,threadid,batch,samples>

    std::vector<int> indices;

public:


    explicit PreBatch_assigner(int n, const Samples& s,
                               vector<vector<vector<queue<int>>>>& p) :
            buckets(n),
            samples(s),
            plan(p),
            indices(samples.num_of_train)
    {
        std::iota(std::begin(indices), std::end(indices), 0);
    }

    void assign_for_iteration(int it);

    inline bool is_intersect(Sample& sample,const set<int>& entities)const;
};

bool PreBatch_assigner::is_intersect(Sample &sample, const set<int> &entities) const {
    if (entities.find(sample.n_obj) != entities.end()) {
        //found
        return true;
    }
    if (entities.find(sample.n_sub) != entities.end()) {
        //found
        return true;
    }
    if (entities.find(sample.p_obj) != entities.end()) {
        //found
        return true;
    }
    if (entities.find(sample.p_sub) != entities.end()) {
        //found
        return true;
    }
    return false;
}

void PreBatch_assigner::assign_for_iteration(int it) {
    //random shuffle indices..
    std::random_shuffle(indices.begin(), indices.end());

    int batch = 0;//current batch #
    queue<int> current_queue;
    queue<int> next_batch;
    vector<int> bucket_size(buckets.size());

    for(int i: indices){
        current_queue.push(i);
    }

    while(true){
        if(current_queue.empty()){
            //next batch
            ++batch;
            swap(current_queue,next_batch);
            if(current_queue.empty()){
                //finished
                break;
            }
            for(auto& i:bucket_size){
                i=0;
            }
        }

        //fetch sample
        int index = current_queue.front();
        current_queue.pop();
        Sample sample=samples.get_sample(it,index);

        int to_insert = -1;//bucket to insert!
        for (int i = 0; i < buckets.size(); ++i) {
            if (is_intersect(sample,buckets[i])) {
                if(to_insert<0) {
                    to_insert = i;
                }
                else{
                    //CASE 1: can be assign to two buckets at the same time;
                    //postpone to next batch
                    next_batch.push(index);
                    continue;
                }
            }
        }

        //CASE 2: Can only be assign to one bucket
        if (to_insert >= 0) {
            //buckets[to_insert].insert(sample);

            plan[it][to_insert][batch].push(index);

            buckets[to_insert].insert(sample.p_obj);
            buckets[to_insert].insert(sample.n_obj);
            buckets[to_insert].insert(sample.p_sub);
            buckets[to_insert].insert(sample.n_sub);

            ++bucket_size[to_insert];
            continue;
        }

        //CASE 3: can assign to all buckets
        //Greedy assign, try to balance load.
        int min_size = std::numeric_limits<int>::max();
        to_insert = -1;
        for (int i = 0;i < buckets.size();++i) {
            if (bucket_size[i] < min_size) {
                min_size = bucket_size[i];
                to_insert = i;
            }
        }
        plan[it][to_insert][batch].push(index);
        buckets[to_insert].insert(sample.p_obj);
        buckets[to_insert].insert(sample.n_obj);
        buckets[to_insert].insert(sample.p_sub);
        buckets[to_insert].insert(sample.n_sub);

        ++bucket_size[to_insert];
    }
}

#endif //DISTRESCAL_PREBATCH_H
