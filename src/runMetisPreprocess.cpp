//
// Created by dhding on 4/17/18.
//

#include "../util/Base.h"
#include "../util/FileUtil.h"
#include "../util/Data.h"
#include "../util/Parameter.h"
#include "metis.h"

string dump_file_str = "../data/part_dump.txt";

void print_info(Parameter &parameter, Data &data) {
    cout << "------------------------" << endl;
//    cout << parameter.get_all();
    cout << data.get_info();
    cout << "------------------------" << endl;
}

int main(int argc, char **argv) {

    Parameter parameter;

    po::options_description desc("Allowed options");
    desc.add_options()
            ("help", "produce help message")
            ("n", po::value<int>(&(parameter.num_of_thread))->default_value(1),
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

    print_info(parameter, data);

    //first transform training sample into undirected graph
    const int n = data.num_of_training_triples;
    int64_t m = 0;//number of total edges. Same edge counted twice.

    //need to calculate m first;
    for (int i = 0; i < n; ++i) {
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
//    cout<<m<<endl;

    //Transform undirected graph into METIS input format
    idx_t *xadj = new idx_t[n + 1];
    idx_t *adjncy = new idx_t[m];

    int64_t count = 0;
    for (int i = 0; i < n; ++i) {
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
    xadj[n] = count;
//    cout<<"m="<<m<<" count="<<count<<endl;

    //Run metis algorithm
    idx_t nvtxs = n;//The number of vertices in the graph.
    idx_t ncon = 1;//The number of balancing constraints. It should be at least 1.
    idx_t nparts = parameter.num_of_thread;
    idx_t objval = 0;
    idx_t *part = new idx_t[nvtxs];

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

    int result = METIS_PartGraphKway(&nvtxs, &ncon, xadj, adjncy,
                                     NULL /*vwgt*/, NULL /*vsize*/, NULL /*adjwgt*/, &nparts, NULL /*tpwgts*/,
                                     NULL /*ubvec*/, options, &objval, part);

//    METIS_SetDefaultOptions(options);

    switch (result) {
        case METIS_OK:
            cout << "the function returned normally!\n";
            break;
        case METIS_ERROR_INPUT:
            cerr << "an input error\n";
            exit(-1);
        case METIS_ERROR_MEMORY:
            cerr << "could not allocate the required memory \n";
            exit(-1);
        case METIS_ERROR:
            cerr << "Other errors..\n";
            exit(-1);
    }

    //dump partition to file
    fstream fout(dump_file_str.c_str());
    for (int i = 0; i < nvtxs; ++i) {
        fout << part[i] << " ";
    }
    fout << endl;
    fout.close();

    //Load the scheduler from partition results.
    vector<vector<int>> bins(parameter.num_of_thread);
    for (int i = 0; i < nvtxs; ++i) {
        bins[part[i]].emplace_back(i);
    }


    //TODO: test number of contested locks when using partition algorithms


    delete[] xadj;
    delete[] adjncy;
    delete[] part;
    return 0;
}