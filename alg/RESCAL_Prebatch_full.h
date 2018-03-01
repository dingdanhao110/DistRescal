//
// Created by dhding on 1/28/18.
//

#ifndef DISTRESCAL_TRANSE_PREBATCH_FULL_H
#define DISTRESCAL_TRANSE_PREBATCH_FULL_H

#include "Prebatch_full_Optimizer.h"

template<typename OptimizerType>
class RESCAL_PREBATCH_FULL : virtual public PREBATCH_FULL_OPTIMIZER<OptimizerType> {
    using PREBATCH_FULL_OPTIMIZER<OptimizerType>::embedA_G;
    using PREBATCH_FULL_OPTIMIZER<OptimizerType>::embedA;
    using PREBATCH_FULL_OPTIMIZER<OptimizerType>::embedR_G;
    using PREBATCH_FULL_OPTIMIZER<OptimizerType>::embedR;
    using PREBATCH_FULL_OPTIMIZER<OptimizerType>::data;
    using PREBATCH_FULL_OPTIMIZER<OptimizerType>::parameter;
    using PREBATCH_FULL_OPTIMIZER<OptimizerType>::violations;
    using PREBATCH_FULL_OPTIMIZER<OptimizerType>::update_grad;
    using PREBATCH_FULL_OPTIMIZER<OptimizerType>::statistics;
    using PREBATCH_FULL_OPTIMIZER<OptimizerType>::rel_statistics;
public:
    explicit RESCAL_PREBATCH_FULL(Parameter &parameter, Data &data) :
            PREBATCH_FULL_OPTIMIZER<OptimizerType>(parameter, data) {}

private:

    void eval(const int epoch) {

        hit_rate testing_measure = eval_hit_rate(parameter, data, embedA, embedR);

        string prefix = "testing data >>> ";

        print_hit_rate(prefix, parameter->hit_rate_topk, testing_measure);
    }

    void init_G(const int D) {
        embedA_G = new value_type[data->num_of_entity * D];
        std::fill(embedA_G, embedA_G + data->num_of_entity * D, 0);

        embedR_G = new value_type[data->num_of_relation * D * D];
        std::fill(embedR_G, embedR_G + data->num_of_relation * D * D, 0);
    }

    void initialize() {

        this->current_epoch = 1;

        embedA = new value_type[data->num_of_entity * parameter->dimension];
        embedR = new value_type [data->num_of_relation * parameter->dimension * parameter->dimension];

        init_G(parameter->dimension);

        value_type bnd = sqrt(6) / sqrt(data->num_of_entity + parameter->dimension);

        for (int row = 0; row < data->num_of_entity; row++) {
            for (int col = 0; col < parameter->dimension; col++) {
                embedA[row * parameter->dimension + col] = RandomUtil::uniform_real(-bnd, bnd);
            }
        }

        bnd = sqrt(6) / sqrt(parameter->dimension + parameter->dimension);

        for (int R_i = 0; R_i < data->num_of_relation; R_i++) {
            value_type *sub_R = embedR + R_i * parameter->dimension * parameter->dimension;
            for (int row = 0; row < parameter->dimension; row++) {
                for (int col = 0; col < parameter->dimension; col++) {
                    sub_R[row * parameter->dimension + col] = RandomUtil::uniform_real(-bnd, bnd);
                }
            }
        }
    }

    bool update(const Sample &sample) {
        //cout<<sample.relation_id<<" "<<sample.p_obj<<" "<<sample.p_sub<<" "<<sample.n_obj<<" "<<sample.n_sub<<endl;
        value_type positive_score = cal_rescal_score(sample.relation_id, sample.p_sub, sample.p_obj, embedA, embedR, parameter);
        value_type negative_score = cal_rescal_score(sample.relation_id, sample.n_sub, sample.n_obj, embedA, embedR, parameter);
        if (parameter->margin_on) {
            if (positive_score - negative_score >= parameter->margin) {
                return false;
            }
        }

        value_type p_pre = 1;
        value_type n_pre = 1;

        ++statistics[sample.n_obj];
        ++statistics[sample.n_sub];
        ++statistics[sample.p_obj];
        ++statistics[sample.p_sub];
        ++rel_statistics[sample.relation_id];
        //DenseMatrix grad4R(parameter.rescal_D, parameter.rescal_D);
        value_type *grad4R = new value_type[parameter->dimension * parameter->dimension];
        unordered_map<int, value_type *> grad4A_map;

        // Step 1: compute gradient descent
        update_4_R(sample, grad4R, p_pre, n_pre);
        update_4_A(sample, grad4A_map, p_pre, n_pre);

        // Step 2: do the update
        update_grad(embedR + sample.relation_id * parameter->dimension * parameter->dimension, grad4R,
                    embedR_G + sample.relation_id * parameter->dimension * parameter->dimension,
                    parameter->dimension * parameter->dimension, parameter);

        value_type *A_grad = new value_type[parameter->dimension];
        for (auto ptr = grad4A_map.begin(); ptr != grad4A_map.end(); ptr++) {

            //Vec A_grad = ptr->second - parameter.lambdaA * row(rescalA, ptr->first);
            //TODO: DOUBLE CHECK
            for (int i = 0; i < parameter->dimension; ++i) {
                A_grad[i] = ptr->second[i] - parameter->lambdaA * embedA[ptr->first * parameter->dimension + i];
            }

            update_grad(embedA + parameter->dimension * ptr->first, A_grad,
                        embedA_G + parameter->dimension * ptr->first,
                        parameter->dimension, parameter);
        }
        delete[] A_grad;//cout<<"Free A-grd\n";
        delete[] grad4R;//cout<<"Free grad4R\n";

        for(auto pair:grad4A_map){
            delete[] pair.second;
        }
        //cout<<"Free grad4A_map\n";
        //cout<<"Exiting update\n";
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

    void update_4_A(const Sample &sample, unordered_map<int, value_type*> &grad4A_map, const value_type p_pre,
                    const value_type n_pre) {
        //cout<<"Entering update4A\n";
        //DenseMatrix &R_k = rescalR[sample.relation_id];
        value_type * R_k = embedR + sample.relation_id * parameter->dimension * parameter->dimension;

        //Vec p_tmp1 = prod(R_k, row(rescalA, sample.p_obj));
        value_type *p_tmp1 = new value_type[parameter->dimension];
        for (int i = 0; i < parameter->dimension; ++i) {
            p_tmp1[i] = 0;
            for (int j = 0; j < parameter->dimension; ++j) {
                p_tmp1[i] += R_k[i * parameter->dimension + j] * embedA[sample.p_obj * parameter->dimension + j];
            }
        }

        //Vec p_tmp2 = prod(row(rescalA, sample.p_sub), R_k);
        value_type *p_tmp2 = new value_type[parameter->dimension];
        for (int i = 0; i < parameter->dimension; ++i) {
            p_tmp2[i] = 0;
            for (int j = 0; j < parameter->dimension; ++j) {
                p_tmp2[i] += embedA[sample.p_sub * parameter->dimension + j] * R_k[j * parameter->dimension + i];
            }
        }

        //Vec n_tmp1 = prod(R_k, row(rescalA, sample.n_obj));
        value_type *n_tmp1 = new value_type[parameter->dimension];
        for (int i = 0; i < parameter->dimension; ++i) {
            n_tmp1[i] = 0;
            for (int j = 0; j < parameter->dimension; ++j) {
                n_tmp1[i] += R_k[i * parameter->dimension + j] * embedA[sample.n_obj * parameter->dimension + j];
            }
        }

        //Vec n_tmp2 = prod(row(rescalA, sample.n_sub), R_k);
        value_type *n_tmp2 = new value_type[parameter->dimension];
        for (int i = 0; i < parameter->dimension; ++i) {
            n_tmp2[i] = 0;
            for (int j = 0; j < parameter->dimension; ++j) {
                n_tmp2[i] += embedA[sample.n_sub * parameter->dimension + j] * R_k[j * parameter->dimension + i];
            }
        }

        //grad4A_map[sample.p_sub] = p_tmp1;
        grad4A_map[sample.p_sub] = new value_type[parameter->dimension];
        for (int i = 0; i < parameter->dimension; ++i) {
            grad4A_map[sample.p_sub][i] = p_tmp1[i];
        }

        auto ptr = grad4A_map.find(sample.p_obj);
        if (ptr == grad4A_map.end()) {
            //grad4A_map[sample.p_obj] = p_pre * p_tmp2;
            grad4A_map[sample.p_obj] = new value_type[parameter->dimension];
            for (int i = 0; i < parameter->dimension; ++i) {
                grad4A_map[sample.p_obj][i] = p_tmp2[i] * p_pre;
            }
        } else {
            //grad4A_map[sample.p_obj] += p_pre * p_tmp2;
            for (int i = 0; i < parameter->dimension; ++i) {
                grad4A_map[sample.p_obj][i] += p_pre * p_tmp2[i];
            }
        }

        ptr = grad4A_map.find(sample.n_sub);
        if (ptr == grad4A_map.end()) {
            //grad4A_map[sample.n_sub] = n_pre * (-n_tmp1);
            grad4A_map[sample.n_sub] = new value_type[parameter->dimension];
            for (int i = 0; i < parameter->dimension; ++i) {
                grad4A_map[sample.n_sub][i] = n_pre * (-n_tmp1[i]);
            }
        } else {
            //grad4A_map[sample.n_sub] += n_pre * (-n_tmp1);
            for (int i = 0; i < parameter->dimension; ++i) {
                grad4A_map[sample.n_sub][i] += n_pre * (-n_tmp1[i]);
            }
        }

        ptr = grad4A_map.find(sample.n_obj);
        if (ptr == grad4A_map.end()) {
            //grad4A_map[sample.n_obj] = n_pre * (-n_tmp2);
            grad4A_map[sample.n_obj] = new value_type[parameter->dimension];
            for (int i = 0; i < parameter->dimension; ++i) {
                grad4A_map[sample.n_obj][i] = n_pre * (-n_tmp2[i]);
            }
        } else {
            //grad4A_map[sample.n_obj] += n_pre * (-n_tmp2);
            for (int i = 0; i < parameter->dimension; ++i) {
                grad4A_map[sample.n_obj][i] += n_pre * (-n_tmp2[i]);
            }
        }

        delete [] p_tmp1;
        delete [] p_tmp2;
        delete [] n_tmp1;
        delete [] n_tmp2;
        //cout<<"Exiting update4A\n";
    }

    void update_4_R(const Sample &sample, value_type *grad4R, const value_type p_pre, const value_type n_pre) {
        //cout<<"Entering update4R\n";
        value_type *p_sub = embedA + sample.p_sub * parameter->dimension;
        value_type *p_obj = embedA + sample.p_obj * parameter->dimension;

        value_type *n_sub = embedA + sample.n_sub * parameter->dimension;
        value_type *n_obj = embedA + sample.n_obj * parameter->dimension;


        //grad4R.clear();
        std::fill(grad4R, grad4R + parameter->dimension * parameter->dimension, 0);

//        for (int i = 0; i < parameter->rescal_D; i++) {
//            for (int j = 0; j < parameter->rescal_D; j++) {
//                grad4R(i, j) += p_pre * p_sub[i] * p_obj[j] - n_pre * n_sub[i] * n_obj[j];
//            }
//        }
//
        for (int i = 0; i < parameter->dimension; i++) {
            for (int j = 0; j < parameter->dimension; j++) {
                grad4R[i * parameter->dimension + j] += p_pre * p_sub[i] * p_obj[j] - n_pre * n_sub[i] * n_obj[j];
            }
        }

//        grad4R += - parameter->lambdaR * rescalR[sample.relation_id];
        value_type *R_k = embedR + sample.relation_id * parameter->dimension * parameter->dimension;

        for (int i = 0; i < parameter->dimension; i++) {
            for (int j = 0; j < parameter->dimension; j++) {
                grad4R[i * parameter->dimension + j] +=
                        -parameter->lambdaR * R_k[i * parameter->dimension + j];
            }
        }
        //TODO: Double check.
        //cout<<"Exiting update4R\n";
    }
};


#endif //DISTRESCAL_TRANSE_PREBATCH_FULL_H
