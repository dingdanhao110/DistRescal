#ifndef MARGINBASEDOPTIMIZER_H
#define MARGINBASEDOPTIMIZER_H

#include "alg/Sampler.h"
#include "util/Base.h"
#include "util/Parameter.h"
#include "util/Data.h"
#include "util/Monitor.h"

/**
 * Stochastic Gradient Descent
 * @param factors
 * @param gradient
 * @param pre_gradient_square: Square of previous gradient (not used)
 * @param d: length of factors/gradient/pre_gradient_square
 */
class sgd {
public:
    void operator()(value_type *factors, value_type *gradient, value_type *pre_gradient_square, const int d,
                    Parameter *parameter) {

        value_type scale = parameter->step_size;

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
 */
class adagrad {
public:
    void operator()(value_type *factors, value_type *gradient, value_type *pre_gradient_square,
                    const int d, Parameter *parameter) {

        value_type scale = parameter->step_size;

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
class adadelta {
public:
    void operator()(value_type *factors, value_type *gradient, value_type *pre_gradient_square,
                    const int d, Parameter *parameter) {

        value_type scale = parameter->step_size;

        for (int i = 0; i < d; i++) {
            pre_gradient_square[i] =
                    parameter->Rho * pre_gradient_square[i] + (1 - parameter->Rho) * gradient[i] * gradient[i];
            factors[i] += scale * gradient[i] / sqrt(pre_gradient_square[i] + min_not_zero_value);
        }
    }
};

template<typename OptimizerType=sgd>
class MarginBasedOptimizer {

private:

protected:

    // Functor object of update function
    OptimizerType update_grad;

    pool *computation_thread_pool = nullptr;
    Parameter *parameter = nullptr;
    Data *data = nullptr;
    int current_epoch;
    int violations;
    value_type loss;

    /**
     * Update function
     * @param sample
     * @param weight
     */
    virtual void update(const Sample &sample) = 0;

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
    virtual void eval(const int epoch) = 0;

    /**
     * Output function after training
     * @param epoch
     */
    virtual void output(const int epoch) = 0;

    /**
     * Compute loss value
     * @return
     */
    virtual value_type cal_loss() {
        return 0;
    }

public:

    explicit MarginBasedOptimizer(Parameter &parameter, Data &data) : parameter(&parameter), data(&data) {
        computation_thread_pool = new pool(parameter.num_of_thread);
    }

    ~MarginBasedOptimizer() {
        delete computation_thread_pool;
    }

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
                        if(parameter->num_of_thread == 1){
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

        cout << "Total Training Time: " << total_time << " secs" << endl;
    }

};

#endif //MARGINBASEDOPTIMIZER_H
