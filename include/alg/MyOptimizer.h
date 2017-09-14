//
// Created by dhding on 9/13/17.
//

#ifndef DISTRESCAL_MYOPTIMIZER_H
#define DISTRESCAL_MYOPTIMIZER_H

#include "alg/Sampler.h"
#include "util/Base.h"
#include "util/Parameter.h"
#include "util/Data.h"
#include "util/Monitor.h"
#include "boost/threadpool.hpp"

using namespace boost::threadpool;


/**
 * Stochastic Gradient Descent
 * @param factors
 * @param gradient
 * @param pre_gradient_square: Square of previous gradient (not used)
 * @param d: length of factors/gradient/pre_gradient_square
 * @param weight: scaling value
 */
class sgd {
public:
    void operator()(
    value_type *factors, value_type
    *gradient,
    value_type *pre_gradient_square,
    const int d,
     const Parameter& parameter,const value_type weight = 1.0
    ) {

        value_type scale = parameter.step_size * weight;


        for (int i = 0; i < d; i++) {
            factors[i] += scale * gradient[i];
        }

    }
};

/**
 * AdaGrad: Adaptive Gradient Algorithm
 * @param factors
 * @param gradient
 * @param pre_gradient_square: Square of previous gradient
 * @param d: length of factors/gradient/pre_gradient_square
 * @param weight: scaling value
 */
class adagrad {
public:
    void operator()(value_type *factors, value_type *gradient, value_type *pre_gradient_square,
             const int d, const Parameter& parameter, const value_type weight = 1.0) {

    value_type scale = parameter.step_size * weight;


    for (int i = 0; i < d; i++) {
        pre_gradient_square[i] += gradient[i] * gradient[i];
        factors[i] += scale * gradient[i] / sqrt(pre_gradient_square[i] + min_not_zero_value);
    }

    }
};

/**
 * AdaDelta: An Adaptive Learning Rate Method
 * @param factors
 * @param gradient
 * @param pre_gradient_square: Square of previous gradient
 * @param d: length of factors/gradient/pre_gradient_square
 * @param weight: scaling value
 */
class adadelta{
public:
    void operator() (value_type *factors, value_type *gradient, value_type *pre_gradient_square,
              const int d,  const Parameter& parameter,const value_type weight = 1.0) {

    value_type scale = parameter.step_size * weight;


    for (int i = 0; i < d; i++) {
        pre_gradient_square[i] = parameter.Rho * pre_gradient_square[i] + (1 - parameter.Rho) * gradient[i] * gradient[i];
        factors[i] += scale * gradient[i] / sqrt(pre_gradient_square[i] + min_not_zero_value);
    }

    }
};

template <typename OptimizerType=sgd>
class MyOptimizer {

private:

protected:

    // Functor object of update function
    OptimizerType update_grad;

    //Vec min_not_zeros; // initialize its size and values in inherited class

    const Parameter & parameter;
    const Data & data;
    int current_epoch;
    int violations;
    value_type loss;

    /**
     * Update function
     * @param sample
     * @param weight
     */
    virtual void update(const Sample &sample, const value_type weight = 1.0) = 0;

    /**
     * Initialization function
     * Initialize factors, pre_gradient_square, min_none_zeros...
     */
    virtual void initialize() = 0;

    /**
     * Evaluation function
     * @param epoch
     * @return
     */
    virtual string eval(const int epoch) = 0;

    /**
     * Output function after training
     * @param epoch
     */
    virtual void output(const int epoch) = 0;

    /**
     * Header for CSV file
     * @return
     */
    virtual string get_log_header() = 0;

    /**
     * Compute loss value
     * @return
     */
    virtual value_type cal_loss() {
        return 0;
    }

    void print_log(const string log_content) {
        ofstream log(parameter.log_path.c_str(), std::ofstream::out | std::ofstream::app);
        log << log_content << endl;
        log.close();
    }

public:

    explicit MyOptimizer (const Parameter &parameter, const Data &data) : parameter(parameter), data(data) {
    }

    /**
     * Start to train
     */
    void train() {

        initialize();



        Monitor timer;

        std::vector<int> indices(data.num_of_training_triples);
        std::iota(std::begin(indices), std::end(indices), 0);

//        if (parameter.print_log_header) {
//            print_log(get_log_header());
//        }

        int workload = data.num_of_training_triples / parameter.num_of_thread;

        value_type total_time = 0.0;

        for (int epoch = current_epoch; epoch <= parameter.epoch; epoch++) {

            violations = 0;
            loss = 0;

            timer.start();

            std::random_shuffle(indices.begin(), indices.end());

            pool *thread_pool = new pool(parameter.num_of_thread);
            std::function<void(int)> compute_func = [&](int thread_index) -> void {
                int start = thread_index * workload;
                int end = std::min(start + workload, data.num_of_training_triples);
                Sample sample;
                for (int n = start; n < end; n++) {
                    if(parameter.num_of_thread == 1){
                        Sampler::random_sample(data, sample, indices[n]);
                    } else {
                        Sampler::random_sample_multithreaded(data, sample, indices[n]);
                    }
                    //cout<<"Thread "<<std::this_thread::get_id()<<": Work assigned"<<endl;
                    update(sample);
                }
            };

            for(int thread_index = 0; thread_index < parameter.num_of_thread; thread_index++) {
                thread_pool->schedule(std::bind(compute_func, thread_index));
            }

            // wait until all threads finish
            thread_pool->wait();
            timer.stop();

            total_time += timer.getElapsedTime();

            cout << "epoch " << epoch << ", time " << timer.getElapsedTime() << " secs" << endl;

            cout << "violations: " << violations << endl;

//            if(parameter.eval_train) {
//                loss = cal_loss();
//
//                if (loss != 0) {
//                    cout << "loss: " << loss << endl;
//                }
//            }
//
//            if (epoch % parameter.print_epoch == 0) {
//
//                timer.start();
//
//                string log = eval(epoch);
//
//                timer.stop();
//
//                cout << "evaluation time: " << timer.getElapsedTime() << " secs" << endl;
//
//                print_log(log);
//            }
//
//            if (epoch % parameter.output_epoch == 0) {
//                output(epoch);
//            }
        }

        cout << "Total Training Time: " << total_time << " secs" << endl;
    }

};
#endif //DISTRESCAL_MYOPTIMIZER_H
