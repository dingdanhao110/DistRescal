//
// Created by dhding on 1/28/18.
//

#ifndef DISTRESCAL_TRANSE_PREBATCH_FULL_H
#define DISTRESCAL_TRANSE_PREBATCH_FULL_H

#include "Prebatch_full_Optimizer.h"

template<typename OptimizerType>
class TRANSE_PREBATCH_FULL : virtual public PREBATCH_FULL_OPTIMIZER<OptimizerType> {
    using PREBATCH_FULL_OPTIMIZER<OptimizerType>::embedA_G;
    using PREBATCH_FULL_OPTIMIZER<OptimizerType>::embedA;
    using PREBATCH_FULL_OPTIMIZER<OptimizerType>::embedR_G;
    using PREBATCH_FULL_OPTIMIZER<OptimizerType>::embedR;
    using PREBATCH_FULL_OPTIMIZER<OptimizerType>::data;
    using PREBATCH_FULL_OPTIMIZER<OptimizerType>::parameter;
    using PREBATCH_FULL_OPTIMIZER<OptimizerType>::update_grad;
    using PREBATCH_FULL_OPTIMIZER<OptimizerType>::statistics;
    using PREBATCH_FULL_OPTIMIZER<OptimizerType>::rel_statistics;
public:
    explicit TRANSE_PREBATCH_FULL(Parameter &parameter, Data &data) :
            PREBATCH_FULL_OPTIMIZER<OptimizerType>(parameter, data) {}

private:

    void eval(const int epoch) {

        hit_rate testing_measure = eval_hit_rate_TransE(parameter, data, embedA, embedR);

        string prefix = "testing data >>> ";

        print_hit_rate(prefix, parameter->hit_rate_topk, testing_measure);
    }

    inline value_type cal_transe_score(int sub_id, int obj_id, int rel_id) {
        value_type sum = 0;
        if (parameter->L1_flag) {
            for (int i = 0; i < parameter->dimension; i++) {
                sum += abs(embedA[obj_id * parameter->dimension + i] - embedA[sub_id * parameter->dimension + i] - embedR[rel_id * parameter->dimension + i]);
            }
        } else {
            for (int i = 0; i < parameter->dimension; i++) {
                sum += sqr(embedA[obj_id * parameter->dimension + i] - embedA[sub_id * parameter->dimension + i] - embedR[rel_id * parameter->dimension + i]);
            }
        }
        return sum;
    }

    void init_G(const int D) {
        embedA_G = new value_type[data->num_of_entity * D];
        std::fill(embedA_G, embedA_G + data->num_of_entity * D, 0);

        embedR_G = new value_type[data->num_of_relation * D];
        std::fill(embedR_G, embedR_G + data->num_of_relation * D, 0);
    }

    void initialize() {

        this->current_epoch = 1;

        embedA = new value_type[data->num_of_entity * parameter->dimension];
        embedR = new value_type[data->num_of_relation * parameter->dimension];

        init_G(parameter->dimension);

        value_type bnd = sqrt(6) / sqrt(data->num_of_entity + parameter->dimension);

        for (int row = 0; row < data->num_of_entity; row++) {
            for (int col = 0; col < parameter->dimension; col++) {
                embedA[row * parameter->dimension + col] = RandomUtil::uniform_real(-bnd, bnd);
            }
        }

        bnd = sqrt(6) / sqrt(data->num_of_relation + parameter->dimension);

        for (int row = 0; row < data->num_of_relation; row++) {
            for (int col = 0; col < parameter->dimension; col++) {
                embedR[row * parameter->dimension + col] = RandomUtil::uniform_real(-bnd, bnd);
            }
        }
    }

    bool update(const Sample &sample) {

        bool subject_replace = (sample.n_sub != sample.p_sub); // true: subject is replaced, false: object is replaced.

        value_type positive_score = cal_transe_score(sample.p_sub, sample.p_obj, sample.relation_id);
        value_type negative_score = cal_transe_score(sample.n_sub, sample.n_obj, sample.relation_id);

        value_type margin = positive_score + parameter->margin - negative_score;

        if (parameter->margin_on) {
            if (margin > 0) {
                //violations++;
            } else {
                return false;
            }
        }

        ++statistics[sample.n_obj];
        ++statistics[sample.n_sub];
        ++statistics[sample.p_obj];
        ++statistics[sample.p_sub];
        ++rel_statistics[sample.relation_id];

        value_type *p_sub_vec = embedA + sample.p_sub * parameter->dimension;
        value_type *p_obj_vec = embedA + sample.p_obj * parameter->dimension;

        value_type *n_sub_vec = embedA + sample.n_sub * parameter->dimension;
        value_type *n_obj_vec = embedA + sample.n_obj * parameter->dimension;

        value_type *rel_vec = embedR + sample.relation_id * parameter->dimension;

        //Vec x = 2 * (row(transeA, sample.p_obj) - row(transeA, sample.p_sub) - row(transeR, sample.relation_id));
        value_type* x=new value_type[parameter->dimension];
        for(int i=0;i<parameter->dimension;++i){
            x[i] = 2 * (embedA[sample.p_obj*parameter->dimension+i]-embedA[sample.p_sub*parameter->dimension+i]-embedR[sample.relation_id*parameter->dimension+i]);
        }

        if (parameter->L1_flag) {
            for (int i = 0; i < parameter->dimension; i++) {
                if (x[i] > 0) {
                    x[i] = 1;
                } else {
                    x[i] = -1;
                }
            }
        }

        update_grad(rel_vec, x,
                    embedR_G + sample.relation_id * parameter->dimension, parameter->dimension,parameter);

        update_grad(p_sub_vec, x, embedA_G + sample.p_sub * parameter->dimension,
                    parameter->dimension,parameter);

        for(int i=0;i<parameter->dimension;++i) {
            x[i] = -x[i];
        }

        update_grad(p_obj_vec, x, embedA_G + sample.p_obj * parameter->dimension,
                    parameter->dimension,parameter);

        //x = 2 * (row(transeA, sample.n_obj) - row(transeA, sample.n_sub) - row(transeR, sample.relation_id));
        for(int i=0;i<parameter->dimension;++i){
            x[i] = 2 * (embedA[sample.n_obj*parameter->dimension+i]-embedA[sample.n_sub*parameter->dimension+i]-embedR[sample.relation_id*parameter->dimension+i]);
        }

        if (parameter->L1_flag) {
            for (int i = 0; i < parameter->dimension; i++) {
                if (x[i] > 0) {
                    x[i] = 1;
                } else {
                    x[i] = -1;
                }
            }
        }


        if (subject_replace) {

            update_grad(p_obj_vec, x,
                        embedA_G + sample.p_obj * parameter->dimension, parameter->dimension,parameter);

            for(int i=0;i<parameter->dimension;++i) {
                x[i] = -x[i];
            }

            update_grad(n_sub_vec, x,
                        embedA_G + sample.n_sub * parameter->dimension, parameter->dimension,parameter);

        } else {

            update_grad(n_obj_vec, x,
                        embedA_G + sample.n_obj * parameter->dimension, parameter->dimension,parameter);
            for(int i=0;i<parameter->dimension;++i) {
                x[i] = -x[i];
            }
            update_grad(p_sub_vec, x,
                        embedA_G + sample.p_sub * parameter->dimension, parameter->dimension,parameter);
        }
        update_grad(rel_vec, x,
                    embedR_G + sample.relation_id * parameter->dimension, parameter->dimension,parameter);


        normalizeOne(rel_vec, parameter->dimension);
        normalizeOne(p_sub_vec, parameter->dimension);
        normalizeOne(p_obj_vec, parameter->dimension);

        if (subject_replace) {
            normalizeOne(n_sub_vec, parameter->dimension);
        } else {
            normalizeOne(n_obj_vec, parameter->dimension);
        }

        delete[] x;
        return true;
    }

    void output(const int epoch) {

        string output_path = parameter->output_path + "/" + to_string(epoch);

        output_matrix(embedA, data->num_of_entity, parameter->dimension, "A_" + to_string(epoch) + ".dat", output_path);
        output_matrix(embedR, data->num_of_relation, parameter->dimension, "R_" + to_string(epoch) + ".dat", output_path);

        if(parameter->optimization=="adagrad" || parameter->optimization=="adadelta"){
            output_matrix(embedA_G, data->num_of_entity, parameter->dimension, "A_G_" + to_string(epoch) + ".dat", output_path);
            output_matrix(embedR_G, data->num_of_relation, parameter->dimension, "R_G_" + to_string(epoch) + ".dat", output_path);
        }

    }
};


#endif //DISTRESCAL_TRANSE_PREBATCH_FULL_H
