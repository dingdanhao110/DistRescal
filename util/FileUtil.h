#ifndef FILEUTIL_H
#define FILEUTIL_H

#include "Base.h"
#include "../struct/Triple.h"
#include <fstream>

using std::fstream;
using std::ifstream;
using std::ofstream;


namespace FileUtil {

    void read_triple_data(string data_path, vector<Triple<string> > &triples, int &num_of_triple) {

        ifstream data_file(data_path.c_str());
        if(!data_file.good()){
            cerr << "cannot find file: " << data_path << endl;
            exit(1);
        }

        string line;

        num_of_triple = 0;

        while (getline(data_file, line)) {
            boost::trim(line);
            if (line.length() == 0) {
                continue;
            }
            vector<string> par;
            boost::split(par, line, boost::is_any_of("\t"));

            boost::trim(par[0]);
            boost::trim(par[1]);
            boost::trim(par[2]);

            triples.push_back(Triple<string>(par[0], par[1], par[2]));

            num_of_triple++;
        }

        data_file.close();

    }

    void output_matrix(value_type *data, const int row_num, const int col_num, const string file_name,
                       const string output_folder) {

        boost::filesystem::create_directories(output_folder);

        string data_file_name = output_folder + "/" + file_name;

        ofstream output(data_file_name.c_str());
        for (int row = 0; row < row_num; row++) {
            value_type *row_vec = data + row * col_num;
            for (int col = 0; col < col_num; col++) {
                output << row_vec[col] << " ";
            }
            output << endl;
        }
        output.close();
    }

}

#endif //FILEUTIL_H
