//
// Created by dhding on 4/30/18.
//
#include "mpi.h"
//#include "metis.h"
#include "parmetis.h"
#include <cstdlib>
#include <string>

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
    MPI_Init(NULL, NULL);
//    cout << argc << endl;
//    for (int i = 0; i < argc; ++i) {
//        cout << argv[i] << endl;
//    }

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

    Data data;
    data.prepare_data(parameter.train_data_path, parameter.valid_data_path, parameter.test_data_path);
//  data.output_decoder(parameter.output_path);

//    print_info(parameter, data);

    MPI_Comm comm_world = MPI_COMM_WORLD;
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
//    printf("Hello world from processor %s, rank %d"
//                   " out of %d processors\n",
//           processor_name, world_rank, world_size);

    string logfile = "log";
    logfile += to_string(world_rank) + ".txt";
    fstream fout(logfile.c_str());
    fout << "Hello world from rank" << world_rank << " out of " << world_size << " processors\n";

    //Step 0: generate graph and save according to world rank...

    //first partition the training set into k equal workloads
    int workload = data.num_of_training_triples / world_size;
    int start = world_rank * workload;
    int end = std::min(data.num_of_training_triples, start + workload);

    idx_t *vtxdist = new idx_t[world_size + 1];
    for (int i = 0; i < world_size; ++i) {
        vtxdist[i] = i * workload;
    }
    vtxdist[world_size] = data.num_of_training_triples;

    //transform training sample into undirected graph
    const int n = data.num_of_training_triples;
    int64_t m = 0;//number of total edges induced by [start,end) entities

    //need to calculate m first;
    for (int i = start; i < end; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i != j) {
                //judge whether i and j conflict with each other..
                if (data.training_triples[i].relation == data.training_triples[j].relation ||
                    data.training_triples[i].object == data.training_triples[j].subject ||
                    data.training_triples[i].object == data.training_triples[j].object ||
                    data.training_triples[i].subject == data.training_triples[j].subject ||
                    data.training_triples[i].subject == data.training_triples[j].object) {
                    ++m;
                }
            }
        }
    }
//    cout << "machine " << world_rank << ": " << m << endl;
    fout << "machine " << world_rank << ": " << m << endl;
    //Step 1: save to METIS style
    //Transform undirected graph into METIS input format
    idx_t *xadj = new idx_t[end - start + 1];
    idx_t *adjncy = new idx_t[m];

    int64_t count = 0;
    for (int i = start; i < end; ++i) {
        xadj[i] = count;
        for (int j = 0; j < n; ++j) {
            if (i != j) {
                //judge whether i and j conflict with each other..
                if (data.training_triples[i].relation == data.training_triples[j].relation ||
                    data.training_triples[i].object == data.training_triples[j].subject ||
                    data.training_triples[i].object == data.training_triples[j].object ||
                    data.training_triples[i].subject == data.training_triples[j].subject ||
                    data.training_triples[i].subject == data.training_triples[j].object) {
                    //conflict
                    adjncy[count] = j;
                    ++count;
                }
            }
        }
    }
    xadj[end - start] = count;
//    cout<<"m="<<m<<" count="<<count<<endl;

    //Step 2: call parMetis
    //Run metis algorithm
    idx_t nvtxs = n;//The number of vertices in the graph.
    idx_t ncon = 1;//The number of balancing constraints. It should be at least 1.
    idx_t nparts = parameter.num_of_thread;
    idx_t objval = 0;
    idx_t *part = new idx_t[end - start + 1];



    idx_t options[METIS_NOPTIONS];
    options[METIS_OPTION_PTYPE] = METIS_PTYPE_KWAY;
    options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;
    options[METIS_OPTION_CTYPE] = METIS_CTYPE_SHEM;
    options[METIS_OPTION_IPTYPE] = METIS_IPTYPE_GROW;
    options[METIS_OPTION_RTYPE] = METIS_RTYPE_FM;
    options[METIS_OPTION_NCUTS] = 1;
    options[METIS_OPTION_NSEPS] = 1;
    options[METIS_OPTION_NUMBERING] = 0;
    options[METIS_OPTION_MINCONN] = 1;
    options[METIS_OPTION_NO2HOP] = 0;
    options[METIS_OPTION_CONTIG] = 1;
    options[METIS_OPTION_CCORDER] = 1;
    options[METIS_OPTION_PFACTOR] = 100;
    options[METIS_OPTION_UFACTOR] = 100;
    options[METIS_OPTION_DBGLVL] = METIS_DBG_INFO | METIS_DBG_TIME;


    int result = ParMETIS_V3_PartKway(vtxdist, &nvtxs, &ncon, xadj, adjncy,
                                      NULL /*vwgt*/, NULL /*vsize*/, NULL /*adjwgt*/, &nparts, NULL /*tpwgts*/,
                                      NULL /*ubvec*/, options, &objval, part, &comm_world);

//    METIS_SetDefaultOptions(options);

    switch (result) {
        case METIS_OK:
            fout << "the function returned normally!\n";
            break;
        case METIS_ERROR_INPUT:
            fout << "an input error\n";
            exit(-1);
        case METIS_ERROR_MEMORY:
            fout << "could not allocate the required memory \n";
            exit(-1);
        case METIS_ERROR:
            fout << "Other errors..\n";
            exit(-1);
    }

    //Step 3: save partition result to local file
    //(/*id*/ threadID)
    //dump partition to file
    string dump_file_str = string("Partition_") + to_string(world_rank) + ".txt";
    fstream fout2(dump_file_str.c_str());
    for (int i = 0; i < end - start; ++i) {
        fout2 << part[i] << endl;
    }
    fout2 << endl;
    fout2.close();
    fout.close();



    // Finalize the MPI environment.
    MPI_Finalize();


    return 0;
}