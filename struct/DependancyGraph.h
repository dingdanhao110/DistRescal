//
// Created by dhding on 4/17/18.
//

#ifndef DISTRESCAL_DEPENDANCYGRAPH_H
#define DISTRESCAL_DEPENDANCYGRAPH_H

#include "../util/Base.h"
#include "../util/Data.h"
#include "metis.h"

class Vertex {
public:
    int id;
    set<int> neighbors;
};

class DepGraph {
    vector<Vertex> vertices;
    Data &data;

    void build_graph() {

    }

public:
    explicit DepGraph(Data &d) :
            data(d) {
        vertices.reserve(d.num_of_training_triples);
        build_graph();
    }

    void dump_into(string train_file) {}
};


#endif //DISTRESCAL_DEPENDANCYGRAPH_H
