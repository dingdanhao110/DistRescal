//
// Created by dhding on 1/27/18.
//

#ifndef DISTRESCAL_NAIVE_OPTIMIZER_H
#define DISTRESCAL_NAIVE_OPTIMIZER_H

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
#include "../alg/PreBatchAssigner.h"
#include "../struct/SHeap.h"
#include "splitEntity.h"

using namespace EvaluationUtil;
using namespace FileUtil;
using namespace Calculator;

/**
 * Margin based RESCAL_NOLOCK
 */
template<typename OptimizerType>
class NAIVE_OPTIMIZER {
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

        for (int epoch = current_epoch; epoch <= parameter->epoch; epoch++) {

            violations = 0;
            loss = 0;

            timer.start();

            std::random_shuffle(indices.begin(), indices.end());

            for (int thread_index = 0; thread_index < parameter->num_of_thread; thread_index++) {

                computation_thread_pool->schedule(std::bind([&](const int thread_index) {

                    int start = thread_index * workload;
                    int end = std::min(start + workload, data->num_of_training_triples);
                    Sample sample;
                    for (int n = start; n < end; n++) {
                        if (parameter->num_of_thread == 1) {
                            Sampler::random_sample(*data, sample, indices[n]);
                        } else {
                            Sampler::random_sample_multithreaded(*data, sample, indices[n]);
                        }
                        //cout<<"Thread "<<std::this_thread::get_id()<<": Work assigned"<<endl;
                        update(sample);
                    }

                }, thread_index));
            }

            computation_thread_pool->wait();

            timer.stop();

            total_time += timer.getElapsedTime();

            cout << "------------------------" << endl;

            cout << "epoch " << epoch << ", time " << timer.getElapsedTime() << " secs" << endl;

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

        }

        cout << "Total Training Time: " << total_time << " secs" << endl;
    }

protected:
    // Functor object of update function
    OptimizerType update_grad;

    pool *computation_thread_pool = nullptr;
    Parameter *parameter = nullptr;
    Data *data = nullptr;
    int current_epoch = 0;
    int violations = 0;
    value_type loss = 0;
    //int block_size;
    vector<int> statistics;

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

    virtual void update(const Sample &sample)=0;

    virtual void eval(const int epoch)=0;

    void output(const int epoch) {}

public:
    explicit NAIVE_OPTIMIZER<OptimizerType>(Parameter
    &parameter,
    Data &data
    ):
    statistics(data
    .num_of_entity,0) {
        this->parameter = &parameter;
        this->data = &data;
        computation_thread_pool = new pool(parameter.num_of_thread);
        //block_size=this->data->num_of_entity/(parameter.num_of_thread*3+3)+1;
    }

    ~NAIVE_OPTIMIZER() {
        delete[] embedA;
        delete[] embedR;
        delete[] embedA_G;
        delete[] embedR_G;
        delete computation_thread_pool;
    }
};

#endif //DISTRESCAL_NAIVE_OPTIMIZER_H
