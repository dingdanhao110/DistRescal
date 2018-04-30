//
// Created by dhding on 4/30/18.
//
#include "mpi.h"
#include <cstdlib>

#include "../util/Base.h"
#include "../util/FileUtil.h"
#include "../util/Data.h"
#include "../util/Parameter.h"

void print_info(Parameter &parameter, Data &data) {
    cout << "------------------------" << endl;
//    cout << parameter.get_all();
    cout << data.get_info();
    cout << "------------------------" << endl;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    cout << argc << endl;
    for (int i = 0; i < argc; ++i) {
        cout << argv[i] << endl;
    }

    Parameter parameter;

    po::options_description desc("Allowed options");
    desc.add_options()
            ("help", "produce help message")
            ("n", po::value<int>(&(parameter.num_of_thread))->default_value(8),
             "number of threads. -1: automatically set")
            ("t_path",
             po::value<string>(&(parameter.train_data_path))->default_value("../data/WN18/wordnet-mlj12-train.txt"),
             "path to training file")
            ("v_path",
             po::value<string>(&(parameter.valid_data_path))->default_value("../data/WN18/wordnet-mlj12-valid.txt"),
             "path to validation file")
            ("e_path",
             po::value<string>(&(parameter.test_data_path))->default_value("../data/WN18/wordnet-mlj12-test.txt"),
             "path to testing file")
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

    }

    if (parameter.num_of_thread == -1) {
        int num_of_thread = std::thread::hardware_concurrency();
        parameter.num_of_thread = (num_of_thread == 0) ? 1 : num_of_thread;
    }

    if (parameter.num_of_eval_thread == -1) {
        int num_of_thread = std::thread::hardware_concurrency();
        parameter.num_of_eval_thread = (num_of_thread == 0) ? 1 : num_of_thread;
    }




    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    // Print off a hello world message
    printf("Hello world from processor %s, rank %d"
                   " out of %d processors\n",
           processor_name, world_rank, world_size);

    // Finalize the MPI environment.
    MPI_Finalize();


    return 0;
}