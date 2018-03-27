//
// Created by dhding on 3/23/18.
//

#ifndef DISTRESCAL_RESCAL_PARA_H
#define DISTRESCAL_RESCAL_PARA_H

#include "../util/Base.h"
#include "../util/RandomUtil.h"
#include "../util/Monitor.h"
#include "../util/FileUtil.h"
#include "../util/CompareUtil.h"
#include "../util/EvaluationUtil.h"
#include "../util/Data.h"
#include "../util/Calculator.h"
#include "../util/Parameter.h"
#include "../alg/Para_Optimizer.h"

using namespace EvaluationUtil;
using namespace FileUtil;
using namespace Calculator;

/**
 * Margin based RESCAL
 */
//template<typename OptimizerType>
class RESCAL_PARA : virtual public ParallelOptimizer {

protected:
    using ParallelOptimizer::data;
    using ParallelOptimizer::parameter;
    using ParallelOptimizer::current_epoch;
    using ParallelOptimizer::violations;
    using ParallelOptimizer::update_grad;
    using ParallelOptimizer::embedA;
    using ParallelOptimizer::embedR;
    using ParallelOptimizer::embedA_G;
    using ParallelOptimizer::embedR_G;

    value_type cal_loss() {
        return cal_loss_single_thread(parameter, data, embedA, embedR);
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
        embedR = new value_type[data->num_of_relation * parameter->dimension * parameter->dimension];

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

    void update_by_col(const Sample &sample, int s_col, int e_col, Barrier &barrier) {

        value_type p_pre = 1;
        value_type n_pre = 1;

        //DenseMatrix grad4R(parameter.rescal_D, parameter.rescal_D);
        value_type *grad4R = new value_type[parameter->dimension * parameter->dimension];
        unordered_map<int, value_type *> grad4A_map;

        // Step 1: compute gradient descent
        update_4_R(sample, grad4R, p_pre, n_pre, s_col, e_col);
        update_4_A(sample, grad4A_map, p_pre, n_pre, s_col, e_col);

        // Step 1.5: sync all thread before doing update
        barrier.Sync();

        // Step 2: do the update
        update_grad(embedR + sample.relation_id * parameter->dimension * parameter->dimension, grad4R,
                    embedR_G + sample.relation_id * parameter->dimension * parameter->dimension,
                    parameter->dimension * parameter->dimension, parameter, s_col * parameter->dimension,
                    e_col * parameter->dimension);

        value_type *A_grad = new value_type[parameter->dimension];
        for (auto ptr = grad4A_map.begin(); ptr != grad4A_map.end(); ptr++) {

            //Vec A_grad = ptr->second - parameter.lambdaA * row(rescalA, ptr->first);
            //TODO: DOUBLE CHECK
            for (int i = s_col; i < e_col; ++i) {
                A_grad[i] = ptr->second[i] - parameter->lambdaA * embedA[ptr->first * parameter->dimension + i];
            }

            update_grad(embedA + parameter->dimension * ptr->first, A_grad,
                        embedA_G + parameter->dimension * ptr->first,
                        parameter->dimension, parameter, s_col, e_col);
        }
        delete[] A_grad;//cout<<"Free A-grd\n";
        delete[] grad4R;//cout<<"Free grad4R\n";

        for (auto pair:grad4A_map) {
            delete[] pair.second;
        }
        //cout<<"Free grad4A_map\n";
        //cout<<"Exiting update\n";
    }


    void eval(const int epoch) {

        hit_rate testing_measure = eval_hit_rate(parameter, data, embedA, embedR);

        string prefix = "testing data >>> ";

        print_hit_rate(prefix, parameter->hit_rate_topk, testing_measure);
    }

    void output(const int epoch) {}

    virtual bool pass_margin(const Sample &sample) {
        // violate=true;

        value_type positive_score = cal_rescal_score(sample.relation_id, sample.p_sub, sample.p_obj, embedA, embedR,
                                                     parameter);
        value_type negative_score = cal_rescal_score(sample.relation_id, sample.n_sub, sample.n_obj, embedA, embedR,
                                                     parameter);
        if (positive_score - negative_score >= parameter->margin) {
            return false;
        } else return true;
    }

protected:

    void update_4_A(const Sample &sample, unordered_map<int, value_type *> &grad4A_map, const value_type p_pre,
                    const value_type n_pre, int s_col, int e_col) {
        //cout<<"Entering update4A\n";
        //DenseMatrix &R_k = rescalR[sample.relation_id];
        value_type *R_k = embedR + sample.relation_id * parameter->dimension * parameter->dimension;

        //Vec p_tmp1 = prod(R_k, row(rescalA, sample.p_obj));
        value_type *p_tmp1 = new value_type[parameter->dimension];
        std::fill(p_tmp1, p_tmp1 + parameter->dimension, 0);

        for (int i = s_col; i < e_col; ++i) {
            for (int j = 0; j < parameter->dimension; ++j) {
                p_tmp1[i] += R_k[i * parameter->dimension + j] * embedA[sample.p_obj * parameter->dimension + j];
            }
        }

        //Vec p_tmp2 = prod(row(rescalA, sample.p_sub), R_k);
        value_type *p_tmp2 = new value_type[parameter->dimension];
        std::fill(p_tmp2, p_tmp2 + parameter->dimension, 0);
        for (int i = s_col; i < e_col; ++i) {
            for (int j = 0; j < parameter->dimension; ++j) {
                p_tmp2[i] += embedA[sample.p_sub * parameter->dimension + j] * R_k[j * parameter->dimension + i];
            }
        }

        //Vec n_tmp1 = prod(R_k, row(rescalA, sample.n_obj));
        value_type *n_tmp1 = new value_type[parameter->dimension];
        std::fill(n_tmp1, n_tmp1 + parameter->dimension, 0);
        for (int i = s_col; i < e_col; ++i) {
            for (int j = 0; j < parameter->dimension; ++j) {
                n_tmp1[i] += R_k[i * parameter->dimension + j] * embedA[sample.n_obj * parameter->dimension + j];
            }
        }

        //Vec n_tmp2 = prod(row(rescalA, sample.n_sub), R_k);
        value_type *n_tmp2 = new value_type[parameter->dimension];
        std::fill(n_tmp2, n_tmp2 + parameter->dimension, 0);
        for (int i = s_col; i < e_col; ++i) {
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

        delete[] p_tmp1;
        delete[] p_tmp2;
        delete[] n_tmp1;
        delete[] n_tmp2;
        //cout<<"Exiting update4A\n";
    }

    void update_4_R(const Sample &sample, value_type *grad4R, const value_type p_pre, const value_type n_pre, int s_col,
                    int e_col) {
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
        for (int i = s_col; i < e_col; ++i) {
            for (int j = 0; j < parameter->dimension; j++) {
                grad4R[i * parameter->dimension + j] += p_pre * p_sub[i] * p_obj[j] - n_pre * n_sub[i] * n_obj[j];
            }
        }

//        grad4R += - parameter->lambdaR * rescalR[sample.relation_id];
        value_type *R_k = embedR + sample.relation_id * parameter->dimension * parameter->dimension;

        for (int i = s_col; i < e_col; ++i) {
            for (int j = 0; j < parameter->dimension; j++) {
                grad4R[i * parameter->dimension + j] +=
                        -parameter->lambdaR * R_k[i * parameter->dimension + j];
            }
        }
        //cout<<"Exiting update4R\n";
    }

public:
    explicit RESCAL_PARA(Parameter &parameter, Data &data) : ParallelOptimizer(
            parameter, data) {}

    ~RESCAL_PARA() {
        delete[] embedA;
        delete[] embedR;
        delete[] embedA_G;
        delete[] embedR_G;
    }
};

#endif //DISTRESCAL_RESCAL_PARA_H
