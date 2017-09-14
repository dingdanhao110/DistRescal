//
// Created by dhding on 9/13/17.
//

#ifndef DISTRESCAL_RESCAL_NAIVE_H
#define DISTRESCAL_RESCAL_NAIVE_H

#include "util/Base.h"
#include "util/RandomUtil.h"
#include "util/Monitor.h"
#include "util/FileUtil.h"
#include "util/CompareUtil.h"
#include "util/EvaluationUtil.h"
#include "util/Data.h"
#include "util/Calculator.h"
#include "util/Parameter.h"
#include "alg/MyOptimizer.h"

using namespace mf;
using namespace EvaluationUtil;
using namespace FileUtil;
using namespace Calculator;

template<typename OptimizerType>
class RESCAL_NAIVE : virtual public MyOptimizer<OptimizerType> {

protected:
    using MyOptimizer<OptimizerType>::data;
    using MyOptimizer<OptimizerType>::parameter;
    using MyOptimizer<OptimizerType>::current_epoch;
    using MyOptimizer<OptimizerType>::violations;
    using MyOptimizer<OptimizerType>::update_grad;

    value_type *rescalA;//DenseMatrix, UNSAFE!
    value_type **rescalR;//vector<DenseMatrix>, UNSAFE!
    value_type *rescalA_G;//DenseMatrix, UNSAFE!
    value_type **rescalR_G;//vector<DenseMatrix>, UNSAFE!

//    value_type cal_loss() {
//        return eval_rescal_train(parameter, data, rescalA, rescalR);
//    }

    void init_G(const int D) {
//        min_not_zeros.resize(D * D);
//        for (int i = 0; i < D * D; i++) {
//            min_not_zeros(i) = min_not_zero_value;
//        }

//        rescalA_G.resize(data.N, D);
//        rescalR_G.resize(data.K, DenseMatrix(D, D));
//        rescalA_G.clear();
        rescalA_G = new value_type[data.N * D];
        for (int i = 0; i < this->data.N * D; ++i) {
            rescalA_G[i] = 0;
        }
        rescalR_G = new value_type *[data.K];
        for (int i = 0; i < data.K; i++) {
            rescalR_G[i] = new value_type[D * D];
            for (int j = 0; j < D; ++j) {
                rescalR_G[i][j] = 0;
            }
        }
//        }
    }

    void initialize() {
        //cout<<"Entering init()\n";
        RandomUtil::init_seed();

        this->current_epoch = 1;

//        rescalA.resize(data->N, parameter->rescal_D);
//        rescalR.resize(data->K, DenseMatrix(parameter->rescal_D, parameter->rescal_D));
        rescalA = new value_type[data.N * parameter.rescal_D];

        rescalR = new value_type *[data.K];
        for (int i = 0; i < data.K; ++i) {
            rescalR[i] = new value_type[parameter.rescal_D * parameter.rescal_D];
        }

        init_G(parameter.rescal_D);

        value_type bnd = sqrt(6) / sqrt(data.N + parameter.rescal_D);

        for (int row = 0; row < data.N; row++) {
            for (int col = 0; col < parameter.rescal_D; col++) {
                rescalA[row * parameter.rescal_D + col] = RandomUtil::uniform_real(-bnd, bnd);
            }
        }

        bnd = sqrt(6) / sqrt(parameter.rescal_D + parameter.rescal_D);

        for (int R_i = 0; R_i < data.K; R_i++) {
            value_type *sub_R = rescalR[R_i];
            for (int row = 0; row < parameter.rescal_D; row++) {
                for (int col = 0; col < parameter.rescal_D; col++) {
                    sub_R[row * parameter.rescal_D + col] = RandomUtil::uniform_real(-bnd, bnd);
                }
            }
        }

        //cout<<"Exiting init()\n";
    }

    void update(const Sample &sample, const value_type weight = 1.0) {
        //cout<<"Entering update\n";
        value_type positive_score = cal_rescal_score(sample.relation_id, sample.p_sub, sample.p_obj, parameter.rescal_D,
                                                     data.N, rescalA, rescalR);
        value_type negative_score = cal_rescal_score(sample.relation_id, sample.n_sub, sample.n_obj, parameter.rescal_D,
                                                     data.N, rescalA, rescalR);

        value_type p_pre = 1;
        value_type n_pre = 1;

        // ToDo: ???
//        if (parameter->restore_from_hole) {
//            positive_score = sigmoid(positive_score);
//            negative_score = sigmoid(negative_score);
//            p_pre = g_sigmoid(positive_score);
//            n_pre = g_sigmoid(negative_score);
//        }

        if (positive_score - negative_score >= parameter.margin) {
            return;
        }

        violations++;

        //DenseMatrix grad4R(parameter.rescal_D, parameter.rescal_D);
        value_type *grad4R=new value_type[parameter.rescal_D* parameter.rescal_D];
        unordered_map<int, value_type *> grad4A_map;

        // Step 1: compute gradient descent

        update_4_R(sample, grad4R, p_pre, n_pre, weight);
        update_4_A(sample, grad4A_map, p_pre, n_pre, weight);

        // Step 2: do the update
        update_grad(rescalR[sample.relation_id], grad4R,
                    rescalR_G[sample.relation_id],
                    parameter.rescal_D * parameter.rescal_D,parameter, weight);

        value_type* A_grad=new value_type[parameter.rescal_D];
        for (auto ptr = grad4A_map.begin(); ptr != grad4A_map.end(); ptr++) {

            //Vec A_grad = ptr->second - parameter.lambdaA * row(rescalA, ptr->first);
            //TODO: DOUBLE CHECK
            for(int i=0;i<parameter.rescal_D;++i){
                A_grad[i]=ptr->second[i]-parameter.lambdaA*rescalA[ptr->first*parameter.rescal_D+i];
            }

            update_grad(rescalA + parameter.rescal_D * ptr->first, A_grad,
                        rescalA_G + parameter.rescal_D * ptr->first,
                        parameter.rescal_D,parameter, weight);
        }
        delete[] A_grad;//cout<<"Free A-grd\n";
        delete[] grad4R;//cout<<"Free grad4R\n";
        //TODO: DOUBLE FREE for grad4A_map???
        for(auto pair:grad4A_map){
            delete[] pair.second;
        }
        //cout<<"Free grad4A_map\n";
        //cout<<"Exiting update\n";
    }


    string eval(const int epoch) {

//        if (parameter->eval_train) {
//
//            hit_rate train_measure = eval_rescal_train(parameter, data, rescalA, rescalR, parameter->output_path + "/RESCAL_RANK_train_epoch_" + to_string(epoch));
//
//            string prefix = "sampled training data >>> ";
//
//            print_hit_rate_train(prefix, parameter->hit_rate_topk, train_measure);
//
//        }
//
//        hit_rate testing_measure = eval_hit_rate(Method::m_RESCAL_RANK, parameter, data, &rescalA, &rescalR,
//                                                 nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, parameter->output_path + "/RESCAL_RANK_test_epoch_" + to_string(epoch));
//
//        string prefix = "testing data >>> ";
//
//        print_hit_rate(prefix, parameter->hit_rate_topk, testing_measure);
//
//        if (parameter->eval_rel) {
//
//            hit_rate rel_measure = eval_relation_rescal(parameter, data, rescalA, rescalR, parameter->output_path + "/RESCAL_RANK_test_epoch_" + to_string(epoch));
//
//            string prefix = "testing data relation evalution >>> ";
//
//            print_hit_rate_rel(prefix, parameter->hit_rate_topk, rel_measure);
//        }
//
//        pair<value_type, value_type> map;
//        map.first = -1;
//        map.second = -1;
//
//        if (parameter->eval_map) {
//            map = eval_MAP(m_RESCAL_RANK, parameter, data, &rescalA, &rescalR, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
//            string prefix = "testing data MAP evalution >>> ";
//            print_map(prefix, parameter->num_of_replaced_entities, map);
//        }
//
        string log = "";
//        log.append("RESCAL_RANK,");
//        log.append(parameter->optimization + ",");
//        log.append(to_string(epoch) + ",");
//        log.append(to_string(parameter->rescal_D) + ",");
//        log.append(to_string(parameter->step_size) + ",");
//        log.append(to_string(parameter->margin) + ",");
//        log.append(to_string(parameter->lambdaA) + ",");
//        log.append(to_string(parameter->lambdaR) + ",");
//        if(parameter->optimization=="adadelta") {
//            log.append(to_string(parameter->Rho) + ",");
//        }
//
//        string count_s = (testing_measure.count_s == -1? "Not Computed" : to_string(testing_measure.count_s));
//        log.append(count_s + ",");
//
//        string count_o = (testing_measure.count_o == -1? "Not Computed" : to_string(testing_measure.count_o));
//        log.append(count_o + ",");
//
//        string count_s_ranking = (testing_measure.count_s_ranking == -1? "Not Computed" : to_string(testing_measure.count_s_ranking));
//        log.append(count_s_ranking + ",");
//
//        string count_o_ranking = (testing_measure.count_o_ranking == -1? "Not Computed" : to_string(testing_measure.count_o_ranking));
//        log.append(count_o_ranking + ",");
//
//        log.append(to_string(testing_measure.count_s_filtering) + ",");
//        log.append(to_string(testing_measure.count_o_filtering) + ",");
//        log.append(to_string(testing_measure.count_s_ranking_filtering) + ",");
//        log.append(to_string(testing_measure.count_o_ranking_filtering) + ",");
//
//        string map1 = (map.first == -1 ? "Not Computed" : to_string(map.first));
//        string map2 = (map.second == -1 ? "Not Computed" : to_string(map.second));
//
//        log.append(map1 + ",");
//        log.append(map2 + ",");
//
//        string inv_count_s_ranking = (testing_measure.inv_count_s_ranking == -1? "Not Computed" : to_string(testing_measure.inv_count_s_ranking));
//        log.append(count_s_ranking + ",");
//
//        string inv_count_o_ranking = (testing_measure.inv_count_o_ranking == -1? "Not Computed" : to_string(testing_measure.inv_count_o_ranking));
//        log.append(count_o_ranking + ",");
//
//        log.append(to_string(testing_measure.inv_count_s_ranking_filtering) + ",");
//        log.append(to_string(testing_measure.inv_count_o_ranking_filtering));

        return log;
    }

    void output(const int epoch) {

//        if(parameter->optimization=="sgd"){
//            output_matrices(rescalA, rescalR, epoch, parameter->output_path);
//        }else{
//            output_matrices(rescalA, rescalR, rescalA_G, rescalR_G, epoch, parameter->output_path);
//        }

    }

    string get_log_header() {

//        string header = "Method,Optimization,epoch,Dimension,step size,margin,lambdaA,lambdaR,";
//        header += ((parameter->optimization=="adadelta")?"Rho,":"");
//        header += "hit_rate_subject@" +
//                  to_string(parameter->hit_rate_topk) + ",hit_rate_object@" +
//                  to_string(parameter->hit_rate_topk) +
//                  ",subject_ranking,object_ranking,hit_rate_subject_filter@" +
//                  to_string(parameter->hit_rate_topk) + ",hit_rate_object_filter@" +
//                  to_string(parameter->hit_rate_topk) +
//                  ",subject_ranking_filter,object_ranking_filter,MAP_subject@" +
//                  to_string(parameter->num_of_replaced_entities) + ",MAP_object@" + to_string(parameter->num_of_replaced_entities) +
//                  ",MRR_subject,MRR_object,MRR_subject_filter,MRR_object_filter";
//        return header;
    }


protected:

    void update_4_A(const Sample &sample, unordered_map<int, value_type*> &grad4A_map, const value_type p_pre,
                    const value_type n_pre, const value_type weight) {
        //cout<<"Entering update4A\n";
        //DenseMatrix &R_k = rescalR[sample.relation_id];
        value_type * R_k = rescalR[sample.relation_id];

        //Vec p_tmp1 = prod(R_k, row(rescalA, sample.p_obj));
        value_type * p_tmp1 = new value_type[parameter.rescal_D];
        for(int i=0;i<parameter.rescal_D;++i){
            p_tmp1[i]=0;
            for(int j=0;j<parameter.rescal_D;++j){
                p_tmp1[i]+=R_k[i*parameter.rescal_D+j]*rescalA[sample.p_obj*parameter.rescal_D+j];
            }
        }

        //Vec p_tmp2 = prod(row(rescalA, sample.p_sub), R_k);
        value_type * p_tmp2 = new value_type[parameter.rescal_D];
        for(int i=0;i<parameter.rescal_D;++i) {
            p_tmp2[i] = 0;
            for (int j = 0; j < parameter.rescal_D; ++j) {
                p_tmp2[i]+=rescalA[sample.p_sub*parameter.rescal_D+j]*R_k[j*parameter.rescal_D+i];
            }
        }

        //Vec n_tmp1 = prod(R_k, row(rescalA, sample.n_obj));
        value_type * n_tmp1 = new value_type[parameter.rescal_D];
        for(int i=0;i<parameter.rescal_D;++i){
            n_tmp1[i]=0;
            for(int j=0;j<parameter.rescal_D;++j){
                n_tmp1[i]+=R_k[i*parameter.rescal_D+j]*rescalA[sample.n_obj*parameter.rescal_D+j];
            }
        }

        //Vec n_tmp2 = prod(row(rescalA, sample.n_sub), R_k);
        value_type * n_tmp2 = new value_type[parameter.rescal_D];
        for(int i=0;i<parameter.rescal_D;++i) {
            n_tmp2[i] = 0;
            for (int j = 0; j < parameter.rescal_D; ++j) {
                n_tmp2[i]+=rescalA[sample.n_sub*parameter.rescal_D+j]*R_k[j*parameter.rescal_D+i];
            }
        }


        //grad4A_map[sample.p_sub] = p_tmp1;
        grad4A_map[sample.p_sub]=new value_type[parameter.rescal_D];
        for(int i=0;i<parameter.rescal_D;++i){
            grad4A_map[sample.p_sub][i] = p_tmp1[i];
        }

        auto ptr = grad4A_map.find(sample.p_obj);
        if (ptr == grad4A_map.end()) {
            //grad4A_map[sample.p_obj] = p_pre * p_tmp2;
            grad4A_map[sample.p_obj] = new value_type[parameter.rescal_D];
            for(int i=0;i<parameter.rescal_D;++i){
                grad4A_map[sample.p_obj][i]=p_tmp2[i]*p_pre;
            }
        } else {
            //grad4A_map[sample.p_obj] += p_pre * p_tmp2;
            for(int i=0;i<parameter.rescal_D;++i) {
                grad4A_map[sample.p_obj][i] += p_pre * p_tmp2[i];
            }
        }

        ptr = grad4A_map.find(sample.n_sub);
        if (ptr == grad4A_map.end()) {
            //grad4A_map[sample.n_sub] = n_pre * (-n_tmp1);
            grad4A_map[sample.n_sub] = new value_type[parameter.rescal_D];
            for(int i=0;i<parameter.rescal_D;++i){
                grad4A_map[sample.n_sub][i]=n_pre * (-n_tmp1[i]);
            }
        } else {
            //grad4A_map[sample.n_sub] += n_pre * (-n_tmp1);
            for(int i=0;i<parameter.rescal_D;++i){
                grad4A_map[sample.n_sub][i]+=n_pre * (-n_tmp1[i]);
            }
        }

        ptr = grad4A_map.find(sample.n_obj);
        if (ptr == grad4A_map.end()) {
            //grad4A_map[sample.n_obj] = n_pre * (-n_tmp2);
            grad4A_map[sample.n_obj] = new value_type[parameter.rescal_D];
            for (int i = 0; i < parameter.rescal_D; ++i) {
                grad4A_map[sample.n_obj][i] = n_pre * (-n_tmp2[i]);
            }
        }
        else {
            //grad4A_map[sample.n_obj] += n_pre * (-n_tmp2);
            for (int i = 0; i < parameter.rescal_D; ++i) {
                grad4A_map[sample.n_obj][i] += n_pre * (-n_tmp2[i]);
            }
        }

        delete [] p_tmp1;
        delete [] p_tmp2;
        delete [] n_tmp1;
        delete [] n_tmp2;
        //cout<<"Exiting update4A\n";
    }

    void update_4_R(const Sample &sample, value_type* grad4R, const value_type p_pre, const value_type n_pre,
                    const value_type weight) {
        //cout<<"Entering update4R\n";
        value_type *p_sub = rescalA + sample.p_sub * parameter.rescal_D;
        value_type *p_obj = rescalA + sample.p_obj * parameter.rescal_D;

        value_type *n_sub = rescalA + sample.n_sub * parameter.rescal_D;
        value_type *n_obj = rescalA + sample.n_obj * parameter.rescal_D;


        //grad4R.clear();
        for (int i = 0; i < parameter.rescal_D; i++) {
            for (int j = 0; j < parameter.rescal_D; j++) {
                grad4R[i * parameter.rescal_D + j] = 0;
            }
        }


//        for (int i = 0; i < parameter->rescal_D; i++) {
//            for (int j = 0; j < parameter->rescal_D; j++) {
//                grad4R(i, j) += p_pre * p_sub[i] * p_obj[j] - n_pre * n_sub[i] * n_obj[j];
//            }
//        }
//
        for (int i = 0; i < parameter.rescal_D; i++) {
            for (int j = 0; j < parameter.rescal_D; j++) {
                grad4R[i*parameter.rescal_D+ j] += p_pre * p_sub[i] * p_obj[j] - n_pre * n_sub[i] * n_obj[j];
            }
        }

//        grad4R += - parameter->lambdaR * rescalR[sample.relation_id];
        for (int i = 0; i < parameter.rescal_D; i++) {
            for (int j = 0; j < parameter.rescal_D; j++) {
                grad4R[i*parameter.rescal_D+ j] += -parameter.lambdaR * rescalR[sample.relation_id][i*parameter.rescal_D+ j];
            }
        }
        //TODO: Double check.
        //cout<<"Exiting update4R\n";
    }

public:
    explicit RESCAL_NAIVE<OptimizerType>(Parameter &parameter, Data &data) : MyOptimizer<OptimizerType>(parameter, data) {}

    ~RESCAL_NAIVE() {}
};

#endif //DISTRESCAL_RESCAL_NAIVE_H
