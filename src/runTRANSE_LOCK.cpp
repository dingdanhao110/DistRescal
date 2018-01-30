//
// Created by dhding on 1/26/18.
//
#include "../util/Base.h"
#include "../util/FileUtil.h"
#include "../util/Data.h"
#include "../util/Parameter.h"
#include "../alg/TransE_Lock.h"

void print_info(Parameter &parameter, Data &data){
    cout << "------------------------" << endl;
    cout << parameter.get_all();
    cout << data.get_info();
    cout << "------------------------" << endl;
}

int main(int argc, char **argv) {

    Parameter parameter;

    po::options_description desc("Allowed options");
    desc.add_options()
            ("help", "produce help message")
            ("opt", po::value<string>(&(parameter.optimization))->default_value("AdaGrad"), "optimization method, i.e., SGD, AdaGrad or AdaDelta")
            ("d", po::value<int>(&(parameter.dimension))->default_value(200), "number of dimensions")
            ("show_loss", po::value<bool>(&(parameter.show_loss))->default_value(0), "whether the program shows loss values during training")
            ("lambdaA", po::value<value_type>(&(parameter.lambdaA))->default_value(0), "regularization weight for entity")
            ("lambdaR", po::value<value_type>(&(parameter.lambdaR))->default_value(0), "regularization weight for relation")
            ("step_size", po::value<value_type>(&(parameter.step_size))->default_value(0.01), "step size")
            ("margin", po::value<value_type>(&(parameter.margin))->default_value(0.2), "margin")
            ("margin_on", po::value<bool>(&(parameter.margin_on))->default_value(1), "whether use margin update")
            ("pre_its", po::value<int>(&(parameter.num_of_pre_its))->default_value(-1), "number of precomputed batch assignment")
            ("thre_freq", po::value<value_type>(&(parameter.threshold_freq))->default_value(0.5), "threshold for frequent entities")
            ("est_entity_coeff", po::value<value_type>(&(parameter.est_entity_coeff))->default_value(1), "coefficient for real_size+sample_size")
            ("epoch", po::value<int>(&(parameter.epoch))->default_value(2000), "maximum training epoch")
            ("hit_rate_topk", po::value<int>(&(parameter.hit_rate_topk))->default_value(10), "hit rate@k")
            ("rho", po::value<value_type>(&(parameter.Rho))->default_value(0.9), "parameter for AdaDelta")
            ("n", po::value<int>(&(parameter.num_of_thread))->default_value(-1), "number of threads. -1: automatically set")
            ("n_e", po::value<int>(&(parameter.num_of_eval_thread))->default_value(-1), "number of threads for evaluation. -1: automatically set")
            ("p_epoch", po::value<int>(&(parameter.print_epoch))->default_value(2000), "print statistics every p_epoch")
            ("o_epoch", po::value<int>(&(parameter.output_epoch))->default_value(2000), "output A and R every o_epoch")
            ("t_path", po::value<string>(&(parameter.train_data_path))->default_value("../data/FB15k/freebase_mtr100_mte100-train.txt"), "path to training file")
            ("v_path", po::value<string>(&(parameter.valid_data_path))->default_value("../data/FB15k/freebase_mtr100_mte100-valid.txt"), "path to validation file")
            ("e_path", po::value<string>(&(parameter.test_data_path))->default_value("../data/FB15k/freebase_mtr100_mte100-test.txt"), "path to testing file")
            ("o_path", po::value<string>(&(parameter.output_path))->default_value("./output"), "path to output file");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        cout << desc << endl;
        return 0;
    } else if (parameter.train_data_path == "" || parameter.test_data_path == "" || parameter.valid_data_path == "") {
        cout << "Please specify path to training, validation and testing files" << endl << endl;
        cout << desc << endl;
        return 0;
    } else {

        std::transform(parameter.optimization.begin(), parameter.optimization.end(), parameter.optimization.begin(),
                       ::tolower);

        if (parameter.optimization != "sgd" && parameter.optimization != "adagrad" &&
            parameter.optimization != "adadelta") {
            cout << "Unrecognized optimization method. opt should be SGD, AdaGrad or AdaDelta" << endl << endl;
            cout << desc << endl;
            return 0;
        }
    }

    if (parameter.num_of_thread == -1) {
        int num_of_thread = std::thread::hardware_concurrency();
        parameter.num_of_thread = (num_of_thread == 0) ? 1 : num_of_thread;
    }

    if (parameter.num_of_eval_thread == -1) {
        int num_of_thread = std::thread::hardware_concurrency();
        parameter.num_of_eval_thread = (num_of_thread == 0) ? 1 : num_of_thread;
    }

    if (parameter.num_of_pre_its == -1) {
        int num_of_thread = std::thread::hardware_concurrency();
        parameter.num_of_pre_its = (num_of_thread == 0) ? 1 : num_of_thread;
    }

    Data data;
    data.prepare_data(parameter.train_data_path, parameter.valid_data_path, parameter.test_data_path);
//  data.output_decoder(parameter.output_path);

    print_info(parameter, data);

    if (parameter.optimization == "sgd") {
        TRANSE_LOCK<sgd> rescal_lock(parameter, data);
        rescal_lock.train();
    } else if (parameter.optimization == "adagrad") {
        TRANSE_LOCK<adagrad> rescal_lock(parameter, data);
        rescal_lock.train();
    } else if (parameter.optimization == "adadelta") {
        TRANSE_LOCK<adadelta> rescal_lock(parameter, data);
        rescal_lock.train();
    } else {
        cerr << "recognized method: " << parameter.optimization << endl;
        exit(1);
    }
    return 0;
}
