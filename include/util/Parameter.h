#ifndef PARAMETER_H
#define PARAMETER_H

#include "util/Base.h"

class Parameter {

public:

    int num_of_thread = 1;
    int num_of_eval_thread = 1;

    string train_data_path;
    string test_data_path;
    string valid_data_path;

    int epoch;
    value_type step_size;

    int print_epoch;
    int output_epoch;
    bool show_loss;  // true: show loss during training
    int hit_rate_topk;
    value_type margin;
    value_type Rho;
    value_type lambdaA; // regularization weight
    value_type lambdaR; // regularization weight
    int dimension;

    string optimization;

    string output_path;

    string get_all() {

        stringstream ss;

        ss << "number of threads: " << num_of_thread << endl;
        ss << "number of evaluation threads: " << num_of_eval_thread << endl;
        ss << "dimension: " << dimension << endl;
        ss << "lambdaA: " << lambdaA << endl;
        ss << "lambdaR: " << lambdaR << endl;
        ss << "margin: " << margin << endl;
        ss << "step_size: " << step_size << endl;
        ss << "optimization: " << optimization << endl;
        if (optimization == "adadelta") {
            ss << "Rho: " << Rho << endl;
        }
        ss << "show_loss: " << show_loss << endl;
        ss << "hit_rate_topk: " << hit_rate_topk << endl;
        ss << "epoch: " << epoch << endl;
        ss << "print_epoch: " << print_epoch << endl;
        ss << "output_epoch: " << output_epoch << endl;
        ss << "train data: " << train_data_path << endl;
        ss << "testing data: " << test_data_path << endl;
        ss << "validation data: " << valid_data_path << endl;
        ss << "output_path: " << output_path << endl;

        return ss.str();
    }
};

#endif //PARAMETER_H
