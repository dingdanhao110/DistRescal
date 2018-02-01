//
// Created by dhding on 1/23/18.
//

#ifndef DISTRESCAL_PREBATCH_OPTIMIZER_H
#define DISTRESCAL_PREBATCH_OPTIMIZER_H


#include <fstream>
#include "../util/Base.h"
#include "../util/RandomUtil.h"
#include "../util/Monitor.h"
#include "../util/FileUtil.h"
#include "../util/CompareUtil.h"
#include "../util/EvaluationUtil.h"
#include "../util/Data.h"
#include "../util/Calculator.h"
#include "../util/Parameter.h"
#include "../alg/Optimizer.h"
#include "../alg/PreBatchAssigner_full.h"
#include "../struct/SHeap.h"
#include "splitEntity.h"

using namespace EvaluationUtil;
using namespace FileUtil;
using namespace Calculator;

/**
 * Margin based RESCAL_NOLOCK
 */
template<typename OptimizerType>
class PREBATCH_FULL_OPTIMIZER {
public:
    /**
     * Start to train
     */
    void train() {

        RandomUtil::init_seed();
        initialize();

        Monitor timer;

        std::vector<int> indices(data->num_of_training_triples);
        std::iota(std::begin(indices), std::end(indices), 0);

//        if (parameter.print_log_header) {
//            print_log(get_log_header());
//        }

        int workload = data->num_of_training_triples / parameter->num_of_thread;

        value_type total_time = 0.0;
        value_type assigner_time=0.0;
        int max_round = 1+ (parameter->epoch-1) / parameter->num_of_pre_its;

        vector<vector<vector<vector<int>>>> plan(parameter->num_of_pre_its,
                                                std::vector<vector<vector<int>>>(parameter->num_of_thread,
                                                                           std::vector<vector<int>>(0,std::vector<int>(0)) ));
        Samples samples(data,parameter);
        //vector<PreBatch_assigner> assigners(parameter->num_of_thread,PreBatch_assigner(parameter->num_of_thread,samples,plan));

        //std::ofstream fout("round.txt");

        unordered_set<int>freq_entities;
        unordered_set<int>freq_relations;
        for(int round=0;round<max_round;++round){
            cout<<"Round "<<round<<": "<<endl;
            cout<<"Preassign starts\n";
            std::fill(statistics.begin(),statistics.end(),0);

            int start_epoch = 1+round*parameter->num_of_pre_its;
            int end_epoch = std::min(start_epoch + parameter->num_of_pre_its, parameter->epoch+1);

            timer.start();
            samples.gen_samples(computation_thread_pool);

            cout<<"Pivot\n";
            int wl = (end_epoch-start_epoch-1)/parameter->num_of_thread+1;
            //cout<<start_epoch<<" "<<end_epoch<<" "<<wl<<endl;

            //clean up the plan table
            for(auto& its:plan){
                for(auto& thrds:its){
                    thrds.resize(0);
                }
            }

            for(int thread_index = 0; thread_index < parameter->num_of_thread; thread_index++){

                computation_thread_pool->schedule(std::bind([&](const int thread_index) {
                    int start = thread_index * wl;
                    int end = std::min(start+wl, end_epoch-start_epoch);
                    //cout<<start<<" "<<end<<endl;
                    PreBatch_assigner_full assigner(parameter->num_of_thread,samples,plan,
                                                    parameter,freq_entities,
                                                    freq_relations);
                    //assigner.clean_up();
                    for (int n = start; n < end; n++) {
                        assigner.assign_for_iteration(n);
                    }
                }, thread_index));
            }
            computation_thread_pool->wait();

            timer.stop();
            cout<<"Preassign ends\n";
            assigner_time+=timer.getElapsedTime();
            total_time+=timer.getElapsedTime();
            cout<<"Pre-pocessing time: "<<timer.getElapsedTime()<<endl;

            for(int epoch=start_epoch;epoch<end_epoch;++epoch){
                count=0;
                violations = 0;
                //Sample all training data
                timer.start();

                int current_epoch=epoch-start_epoch;
                const int max_batch=plan[current_epoch][0].size();
                if(!max_batch){cerr<<"Error at Epoch:"<<epoch<<endl;exit(-1);}

                //prebatch assigner
                for(int  batch=0;batch<max_batch;++batch) {
                    for (int thread_index = 0; thread_index < parameter->num_of_thread; thread_index++) {

                        computation_thread_pool->schedule(std::bind([&](const int thread_index) {

                            const vector<int>& queue = plan[current_epoch][thread_index][batch];
                            for(int index:queue) {
                                Sample sample = samples.get_sample(current_epoch, index);
                                this->update(sample);
                            }
                        }, thread_index));
                    }

                    computation_thread_pool->wait();
                }

                timer.stop();

                total_time += timer.getElapsedTime();
                cout << "------------------------" << endl;
                cout << "epoch " << epoch<< endl;
                cout << "Total tuples: " << count<< endl;

                cout << "training time " << timer.getElapsedTime() << " secs" << endl;

                cout << "violations: " << violations << endl;



                loss = 0;
                if(parameter->show_loss) {

                    timer.start();

                    loss = cal_loss();

                    cout << "loss: " << loss << endl;

                    timer.stop();

                    cout << "time for computing loss: " << timer.getElapsedTime() << " secs" << endl;
                }

                if (epoch % parameter->print_epoch == 0) {

                    timer.start();

                    eval(epoch);

                    timer.stop();

                    cout << "time for evaluation: " << timer.getElapsedTime() << " secs" << endl;

                }

                if (epoch % parameter->output_epoch == 0) {
                    output(epoch);
                }

                cout << "------------------------" << endl;

            }

            vector<pair<int,int>> heap_vec(0);
            vector<pair<int,int>> heap_rel_vec(0);
            heap_vec.reserve(statistics.size());
            heap_rel_vec.reserve(statistics.size());

            for(int i=0;i<statistics.size();++i){
                heap_vec.emplace_back(make_pair(i,statistics[i]));
            }
            for(int i=0;i<rel_statistics.size();++i){
                heap_rel_vec.emplace_back(make_pair(i,rel_statistics[i]));
            }
            cout<<"round "<<round<<" statistics:\n";

            sort(heap_vec.begin(),heap_vec.end(),comparator_bigger_than());
            sort(heap_rel_vec.begin(),heap_rel_vec.end(),comparator_bigger_than());
//            for(const auto& pair:heap_vec){
//                cout<<"("<<pair.first<<","<<pair.second<<") ";
//                //fout<<pair.first<<" "<<pair.second<<" ";
//            }
//            cout<<endl;
            //fout<<endl;

            int split=split_entity(heap_vec,*parameter,freq_entities);
            cout<<"Split at "<<split<<"th entity!! # of samples passed margin: "<<heap_vec[split].second<<endl;
            cout<<"Stat at "<<20<<"th entity!! # of samples passed margin: "<<heap_vec[20].second<<endl;
            split=split_relation(heap_rel_vec,*parameter,freq_relations);
            cout<<"Split at "<<split<<"th relation!! # of samples passed margin: "<<heap_rel_vec[split].second<<endl;
            cout<<"Stat at "<<20<<"th relation!! # of samples passed margin: "<<heap_rel_vec[20].second<<endl;

        }

        cout << "Total Training Time: " << total_time << " secs" << endl;
    }
protected:
    // Functor object of update function
    OptimizerType update_grad;

    pool *computation_thread_pool = nullptr;
    Parameter *parameter= nullptr;
    Data *data= nullptr;
    int current_epoch=0;
    int violations=0;
    int count=0;
    value_type loss=0;
    //int block_size;
    vector<int> statistics;
    vector<int> rel_statistics;
    value_type *embedA;//DenseMatrix, UNSAFE!
    value_type *embedR;//vector<DenseMatrix>, UNSAFE!
    value_type *embedA_G;//DenseMatrix, UNSAFE!
    value_type *embedR_G;//vector<DenseMatrix>, UNSAFE!

    value_type cal_loss() {
        return cal_loss_single_thread(parameter, data, embedA, embedR);
    }

    void init_G(const int D) {
        embedA_G = new value_type[data->num_of_entity * D];
        std::fill(embedA_G, embedA_G + data->num_of_entity * D, 0);

        embedR_G = new value_type[data->num_of_relation * D * D];
        std::fill(embedR_G, embedR_G + data->num_of_relation * D * D, 0);
    }

    void initialize() {

        this->current_epoch = 1;

        embedA = new value_type[data->num_of_entity * parameter->dimension];
        embedR = new value_type [data->num_of_relation * parameter->dimension * parameter->dimension];

        init_G(parameter->dimension);

        value_type bnd = sqrt(6) / sqrt(data->num_of_entity + parameter->dimension);

        for (int row = 0; row < data->num_of_entity; row++) {
            for (int col = 0; col < parameter->dimension; col++) {
                embedA[row * parameter->dimension + col] = RandomUtil::uniform_real(-bnd, bnd);
            }
        }

        bnd = sqrt(6) / sqrt(parameter->dimension + parameter->dimension);

        for (int R_i = 0; R_i < data->num_of_relation; R_i++) {
            value_type *sub_R = embedR + R_i * parameter->dimension * parameter->dimension;
            for (int row = 0; row < parameter->dimension; row++) {
                for (int col = 0; col < parameter->dimension; col++) {
                    sub_R[row * parameter->dimension + col] = RandomUtil::uniform_real(-bnd, bnd);
                }
            }
        }
    }

    virtual void update(const Sample &sample)=0;

    virtual void eval(const int epoch)=0;

    void output(const int epoch) {}

public:
    explicit PREBATCH_FULL_OPTIMIZER<OptimizerType>(Parameter &parameter, Data &data):
            statistics(data.num_of_entity,0),
            rel_statistics(data.num_of_relation,0)
    {
        this->parameter=&parameter;
        this->data=&data;
        computation_thread_pool = new pool(parameter.num_of_thread);
        //block_size=this->data->num_of_entity/(parameter.num_of_thread*3+3)+1;
    }

    ~PREBATCH_FULL_OPTIMIZER() {
        delete[] embedA;
        delete[] embedR;
        delete[] embedA_G;
        delete[] embedR_G;
        delete computation_thread_pool;
    }
};

#endif //DISTRESCAL_PREBATCH_OPTIMIZER_H
