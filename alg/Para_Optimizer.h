//
// Created by dhding on 3/23/18.
//

#ifndef DISTRESCAL_PARA_OPTIMIZER_H
#define DISTRESCAL_PARA_OPTIMIZER_H

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
#include "../util/Sync.h"
#include "splitEntity.h"

using namespace EvaluationUtil;
using namespace FileUtil;
using namespace Calculator;

/**
 * Margin based RESCAL_NOLOCK
 */
//template<typename OptimizerType>
class ParallelOptimizer {
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
        value_type assigner_time = 0.0;
        int max_round = 1 + (parameter->epoch - 1) / parameter->num_of_pre_its;

//        vector<vector<vector<vector<int>>>> plan(parameter->num_of_pre_its,
//                                                 std::vector<vector<vector<int>>>(parameter->num_of_thread,
//                                                                                  std::vector<vector<int>>(0,
//                                                                                                           std::vector<int>(
//                                                                                                                   0))));
        Samples samples(data, parameter);

        vector<pair<int, int>> thread_wl(parameter->num_of_thread);
        for (int thread_index = 0; thread_index < parameter->num_of_thread; thread_index++) {
            thread_wl[thread_index].first = thread_index * (parameter->dimension / parameter->num_of_thread);
            thread_wl[thread_index].second = min((thread_index + 1) * (parameter->dimension / parameter->num_of_thread),
                                                 parameter->dimension);
        }

        Barrier barrier(parameter->num_of_thread);

        for (int round = 0; round < max_round; ++round) {
            cout << "Round " << round << ": " << endl;
            cout << "Preassign starts\n";
//            std::fill(statistics.begin(),statistics.end(),0);

            int start_epoch = 1 + round * parameter->num_of_pre_its;
            int end_epoch = std::min(start_epoch + parameter->num_of_pre_its, parameter->epoch + 1);

            timer.start();
            samples.gen_samples(computation_thread_pool);

            cout << "Pivot\n";
            int wl = (end_epoch - start_epoch - 1) / parameter->num_of_thread + 1;

            timer.stop();
            cout << "Preassign ends\n";
            assigner_time += timer.getElapsedTime();
            total_time += timer.getElapsedTime();
            cout << "Pre-pocessing time: " << timer.getElapsedTime() << endl;

            for (int epoch = start_epoch; epoch < end_epoch; ++epoch) {
                violations = 0;
                //Sample all training data
                int current_epoch = epoch - start_epoch;
                timer.start();
                std::random_shuffle(indices.begin(), indices.end());
                for (int n:indices) {
                    //Check margin.
//                    cout<<"margin?\n";
                    Sample sample = samples.get_sample(current_epoch, n);

                    if (parameter->margin_on) {
                        if (pass_margin(sample)) {
                            ++violations;
                        } else continue;
                    }
//                    cout<<"margin passed\n";

                    for (int thread_index = 0; thread_index < parameter->num_of_thread; thread_index++) {
                        computation_thread_pool->schedule(std::bind([&](const int thread_index, const Sample &sample) {
//                                cout<<std::this_thread::get_id()<<" job assigned\n";
                            update_by_col(sample, thread_wl[thread_index].first,
                                          thread_wl[thread_index].second, barrier);
                        }, thread_index, sample));
                    }
//                    computation_thread_pool->wait();
                }


                timer.stop();

                total_time += timer.getElapsedTime();

                cout << "------------------------" << endl;
                cout << "epoch " << epoch << endl;
//                cout << "Total tuples: " << count<< endl;
                cout << "training time " << timer.getElapsedTime() << " secs" << endl;
                cout << "violations: " << violations << endl;


                loss = 0;
                if (parameter->show_loss) {

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
        }

        cout << "Total Training Time: " << total_time << " secs" << endl;
    }

protected:

    class adagrad_by_col {
    public:
        void operator()(value_type *factors, value_type *gradient, value_type *pre_gradient_square,
                        const int d, Parameter *parameter, int s_col, int e_col) {

            value_type scale = parameter->step_size;

            for (int i = s_col; i < e_col; i++) {
                pre_gradient_square[i] += gradient[i] * gradient[i];
                factors[i] += scale * gradient[i] / sqrt(pre_gradient_square[i] + min_not_zero_value);
            }
        }
    } update_grad;

    // Functor object of update function
//    OptimizerType update_grad;

    pool *computation_thread_pool = nullptr;
    Parameter *parameter = nullptr;
    Data *data = nullptr;
    int current_epoch = 0;
    int violations = 0;
    value_type loss = 0;
    value_type *embedA;//DenseMatrix, UNSAFE!
    value_type *embedR;//vector<DenseMatrix>, UNSAFE!
    value_type *embedA_G;//DenseMatrix, UNSAFE!
    value_type *embedR_G;//vector<DenseMatrix>, UNSAFE!


    virtual value_type cal_loss() {
        return cal_loss_single_thread(parameter, data, embedA, embedR);
    }

    virtual void init_G(const int D) {
        embedA_G = new value_type[data->num_of_entity * D];
        std::fill(embedA_G, embedA_G + data->num_of_entity * D, 0);

        embedR_G = new value_type[data->num_of_relation * D * D];
        std::fill(embedR_G, embedR_G + data->num_of_relation * D * D, 0);
    }

    virtual void initialize() {

        this->current_epoch = 1;

        embedA = new value_type[data->num_of_entity * parameter->dimension];
        embedR = new value_type[data->num_of_relation * parameter->dimension * parameter->dimension];

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

    //NOTICE: this function will call barrier.Sync()
    virtual void update_by_col(const Sample &sample, int s_col, int e_col, Barrier &barrier)=0;

    virtual void eval(const int epoch)=0;

    void output(const int epoch) {}

    virtual bool pass_margin(const Sample &sample)=0;// violate=true;


public:
    explicit ParallelOptimizer(Parameter &para, Data &data)
//            :
//            statistics(data.num_of_entity,0),
//            rel_statistics(data.num_of_relation,0),
//            violation_vec(parameter.num_of_thread)
    {
        this->parameter = &para;
        this->data = &data;
        computation_thread_pool = new pool(para.num_of_thread);
        //block_size=this->data->num_of_entity/(parameter.num_of_thread*3+3)+1;
    }

    ~ParallelOptimizer() {
        delete[] embedA;
        delete[] embedR;
        delete[] embedA_G;
        delete[] embedR_G;
        delete computation_thread_pool;
    }
};

#endif //DISTRESCAL_PARA_OPTIMIZER_H
