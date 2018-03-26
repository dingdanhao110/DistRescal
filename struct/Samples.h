//
// Created by dhding on 1/5/18.
//

#ifndef DISTRESCAL_SAMPLES_H
#define DISTRESCAL_SAMPLES_H

#include "../util/Base.h"
#include "Sample.h"
#include "../util/Parameter.h"
#include "../util/Data.h"
#include "../alg/Sampler.h"
#include <vector>


using namespace std;

class Samples{
private:
    Data* data;
    vector<vector<pair<int,int>>> sample_points;//[it][ind]: the ind-th sample in it iteration
    Parameter* parameter;
    //int it_start=0;//precomputed samples:[it_start,it_start+num_of_its)

public:
    int num_of_train;
    explicit Samples(Data* d, Parameter* p)
            :sample_points(p->num_of_pre_its,
                           vector<pair<int,int>>(d->num_of_training_triples)),
             data(d)
    {
        parameter = p;
        num_of_train=d->num_of_training_triples;
    }

    /*
     * Generate samples using the threadpool provided
     * */
    void gen_samples(pool* threadpool){
        threadpool->wait();
        int workload = data->num_of_training_triples / parameter->num_of_thread;

        for (int it = 0; it < parameter->num_of_pre_its; ++it) {
            for (int thread_index = 0; thread_index < parameter->num_of_thread; thread_index++) {

                threadpool->schedule(std::bind([&](const int thread_index) {
                    std::mt19937 *generator = new std::mt19937(
                            clock() + std::hash<std::thread::id>()(std::this_thread::get_id()));

                    int start = thread_index * workload;
                    int end = std::min(start + workload, data->num_of_training_triples);

                    for (int n = start; n < end; n++) {
                        sample_points[it][n] = Sampler::random_sample_multithreaded(*data, n, generator);
                    }

                }, thread_index));
            }
            threadpool->wait();
        }

    }

    inline Sample get_sample(int it,int index)const{
        Sample s;
        s.relation_id=data->training_triples[index].relation;
        s.p_sub=data->training_triples[index].subject;
        s.p_obj=data->training_triples[index].object;
        s.n_sub=sample_points[it][index].first;
        s.n_obj=sample_points[it][index].second;
        return s;
    }

};



#endif //DISTRESCAL_SAMPLES_H
