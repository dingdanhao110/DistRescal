//
// Created by dhding on 10/23/17.
//

#ifndef DISTRESCAL_RESCAL_POS_H
#define DISTRESCAL_RESCAL_POS_H

#include "../util/Base.h"
#include "../util/RandomUtil.h"
#include "../util/Monitor.h"
#include "../util/FileUtil.h"
#include "../util/CompareUtil.h"
#include "../util/EvaluationUtil.h"
#include "../util/Data.h"
#include "../util/Calculator.h"
#include "../util/Parameter.h"
#include "../alg/MarginBasedOptimizer.h"

using namespace EvaluationUtil;
using namespace FileUtil;
using namespace Calculator;

/**
 * Margin based RESCAL
 */
template<typename OptimizerType>
class RESCAL_POS : virtual public MarginBasedOptimizer<OptimizerType> {

protected:
    using MarginBasedOptimizer<OptimizerType>::data;
    using MarginBasedOptimizer<OptimizerType>::parameter;
    using MarginBasedOptimizer<OptimizerType>::current_epoch;
    using MarginBasedOptimizer<OptimizerType>::violations;
    using MarginBasedOptimizer<OptimizerType>::update_grad;

    value_type *rescalA;//DenseMatrix, UNSAFE!
    value_type *rescalR;//vector<DenseMatrix>, UNSAFE!
    value_type *rescalA_G;//DenseMatrix, UNSAFE!
    value_type *rescalR_G;//vector<DenseMatrix>, UNSAFE!

    value_type cal_loss() {
        return cal_loss_single_thread(parameter, data, rescalA, rescalR);
    }

    void init_G(const int D) {
        rescalA_G = new value_type[data->num_of_entity * D];
        std::fill(rescalA_G, rescalA_G + data->num_of_entity * D, 0);

        rescalR_G = new value_type[data->num_of_relation * D * D];
        std::fill(rescalR_G, rescalR_G + data->num_of_relation * D * D, 0);
    }

    void initialize() {

        this->current_epoch = 1;

        rescalA = new value_type[data->num_of_entity * parameter->dimension];
        rescalR = new value_type [data->num_of_relation * parameter->dimension * parameter->dimension];

        init_G(parameter->dimension);

        value_type bnd = sqrt(6) / sqrt(data->num_of_entity + parameter->dimension);

        for (int row = 0; row < data->num_of_entity; row++) {
            for (int col = 0; col < parameter->dimension; col++) {
                rescalA[row * parameter->dimension + col] = RandomUtil::uniform_real(-bnd, bnd);
            }
        }

        bnd = sqrt(6) / sqrt(parameter->dimension + parameter->dimension);

        for (int R_i = 0; R_i < data->num_of_relation; R_i++) {
            value_type *sub_R = rescalR + R_i * parameter->dimension * parameter->dimension;
            for (int row = 0; row < parameter->dimension; row++) {
                for (int col = 0; col < parameter->dimension; col++) {
                    sub_R[row * parameter->dimension + col] = RandomUtil::uniform_real(-bnd, bnd);
                }
            }
        }
    }

    void update(const Sample &sample) {
        value_type p_pre = 1;

        //DenseMatrix grad4R(parameter.rescal_D, parameter.rescal_D);
        value_type *grad4R = new value_type[parameter->dimension * parameter->dimension];
        unordered_map<int, value_type *> grad4A_map;

        // Step 1: compute gradient descent
        update_4_R(sample, grad4R, p_pre);
        update_4_A(sample, grad4A_map, p_pre);

        // Step 2: do the update
        update_grad(rescalR + sample.relation_id * parameter->dimension * parameter->dimension, grad4R,
                    rescalR_G + sample.relation_id * parameter->dimension * parameter->dimension,
                    parameter->dimension * parameter->dimension, parameter);

        value_type *A_grad = new value_type[parameter->dimension];
        for (auto ptr = grad4A_map.begin(); ptr != grad4A_map.end(); ptr++) {

            //Vec A_grad = ptr->second - parameter.lambdaA * row(rescalA, ptr->first);
            for (int i = 0; i < parameter->dimension; ++i) {
                A_grad[i] = ptr->second[i] - parameter->lambdaA * rescalA[ptr->first * parameter->dimension + i];
            }

            update_grad(rescalA + parameter->dimension * ptr->first, A_grad,
                        rescalA_G + parameter->dimension * ptr->first,
                        parameter->dimension, parameter);
        }
        delete[] A_grad;//cout<<"Free A-grd\n";
        delete[] grad4R;//cout<<"Free grad4R\n";

        for(auto pair:grad4A_map){
            delete[] pair.second;
        }
        //cout<<"Free grad4A_map\n";
        //cout<<"Exiting update\n";
    }


    void eval(const int epoch) {

        hit_rate testing_measure = eval_hit_rate(parameter, data, rescalA, rescalR);

        string prefix = "testing data >>> ";

        print_hit_rate(prefix, parameter->hit_rate_topk, testing_measure);
    }

    void output(const int epoch) {}

protected:

    void update_4_A(const Sample &sample, unordered_map<int, value_type*> &grad4A_map, const value_type p_pre) {
        //cout<<"Entering update4A\n";
        //DenseMatrix &R_k = rescalR[sample.relation_id];
        value_type * R_k = rescalR + sample.relation_id * parameter->dimension * parameter->dimension;

        //Vec p_tmp1 = prod(R_k, row(rescalA, sample.p_obj));
        value_type *p_tmp1 = new value_type[parameter->dimension];
        for (int i = 0; i < parameter->dimension; ++i) {
            p_tmp1[i] = 0;
            for (int j = 0; j < parameter->dimension; ++j) {
                p_tmp1[i] += R_k[i * parameter->dimension + j] * rescalA[sample.p_obj * parameter->dimension + j];
            }
        }

        //Vec p_tmp2 = prod(row(rescalA, sample.p_sub), R_k);
        value_type *p_tmp2 = new value_type[parameter->dimension];
        for (int i = 0; i < parameter->dimension; ++i) {
            p_tmp2[i] = 0;
            for (int j = 0; j < parameter->dimension; ++j) {
                p_tmp2[i] += rescalA[sample.p_sub * parameter->dimension + j] * R_k[j * parameter->dimension + i];
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


        delete [] p_tmp1;
        delete [] p_tmp2;
        //cout<<"Exiting update4A\n";
    }

    void update_4_R(const Sample &sample, value_type *grad4R, const value_type p_pre) {
        //cout<<"Entering update4R\n";
        value_type *p_sub = rescalA + sample.p_sub * parameter->dimension;
        value_type *p_obj = rescalA + sample.p_obj * parameter->dimension;


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
                grad4R[i * parameter->dimension + j] += p_pre * p_sub[i] * p_obj[j];
            }
        }

//        grad4R += - parameter->lambdaR * rescalR[sample.relation_id];
        value_type *R_k = rescalR + sample.relation_id * parameter->dimension * parameter->dimension;

        for (int i = 0; i < parameter->dimension; i++) {
            for (int j = 0; j < parameter->dimension; j++) {
                grad4R[i * parameter->dimension + j] +=
                        -parameter->lambdaR * R_k[i * parameter->dimension + j];
            }
        }
        //cout<<"Exiting update4R\n";
    }

public:
    explicit RESCAL_POS<OptimizerType>(Parameter &parameter, Data &data) : MarginBasedOptimizer<OptimizerType>(parameter, data) {}

    ~RESCAL_POS() {
        delete[] rescalA;
        delete[] rescalR;
        delete[] rescalA_G;
        delete[] rescalR_G;
    }
};

#endif //DISTRESCAL_RESCAL_POS_H