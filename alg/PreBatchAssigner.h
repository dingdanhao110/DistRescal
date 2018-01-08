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
    vector<vector<vector<vector<int>>>>& plan;
    //<iteration,threadid,batch,samples>

    std::vector<int> indices;

public:


    explicit PreBatch_assigner(int n, const Samples& s,
                               vector<vector<vector<vector<int>>>>& p) :
            buckets(n),
            samples(s),
            plan(p),
            indices(samples.num_of_train)
    {
        std::iota(std::begin(indices), std::end(indices), 0);
    }

    void assign_for_iteration(int it);

    inline bool is_intersect(Sample& sample,const set<int>& entities)const;

    void clean_up(){
        for(auto& its:plan){
            for(auto& thrds:its){
                thrds.resize(0);
            }
        }
    }

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
    vector<int> sample_count(buckets.size());

    for(int i: indices){
        current_queue.push(i);
    }
    for(int i=0;i<buckets.size();++i){
        sample_count[i]=0;
        plan[it][i].push_back(vector<int>(0));
    }

    //int p=0;
    while(true){
        if(current_queue.empty()){
            //cout<<std::this_thread::get_id()<<": batch: "<<batch<<" sample: "<<p<<endl;
            //cout<<std::this_thread::get_id()<<": next batch: "<<batch+1<<" samples: "<<next_batch.size()<<endl;
            //p=0;

            //next batch
            ++batch;
            swap(current_queue,next_batch);
            if(current_queue.empty()){
                //finished
                break;
            }

            for(int i=0;i<buckets.size();++i){
                sample_count[i]=0;
                plan[it][i].push_back(vector<int>(0));
                buckets[i].clear();
            }
        }

        //++p;
        //fetch sample
        int index = current_queue.front();
        current_queue.pop();
        Sample sample=samples.get_sample(it,index);

        int to_insert = -1;//bucket to insert!
        bool continue_flag=false;
        for (int i = 0; i < buckets.size(); ++i) {
            if (is_intersect(sample,buckets[i])) {
                if(to_insert<0) {
                    to_insert = i;
                }
                else{
                    //CASE 1: can be assign to two buckets at the same time;
                    //postpone to next batch
                    next_batch.push(index);
                    continue_flag=true;
                    continue;
                }
            }
        }
        if(continue_flag)continue;

        //CASE 2: Can only be assign to one bucket
        if (to_insert >= 0) {
            //buckets[to_insert].insert(sample);

            plan[it][to_insert][batch].push_back(index);

            buckets[to_insert].insert(sample.p_obj);
            buckets[to_insert].insert(sample.n_obj);
            buckets[to_insert].insert(sample.p_sub);
            buckets[to_insert].insert(sample.n_sub);

            ++sample_count[to_insert];
            continue;
        }

        //CASE 3: can assign to all buckets
        //Greedy assign, try to balance load.
        int min_size = std::numeric_limits<int>::max();
        to_insert = -1;
        for (int i = 0;i < buckets.size();++i) {
            if (sample_count[i] < min_size) {
                min_size = sample_count[i];
                to_insert = i;
            }
        }
        plan[it][to_insert][batch].push_back(index);
        buckets[to_insert].insert(sample.p_obj);
        buckets[to_insert].insert(sample.n_obj);
        buckets[to_insert].insert(sample.p_sub);
        buckets[to_insert].insert(sample.n_sub);

        ++sample_count[to_insert];
    }
}

#endif //DISTRESCAL_PREBATCH_H
