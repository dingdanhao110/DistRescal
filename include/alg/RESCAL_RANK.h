#ifndef RESCAL_RANK_H
#define RESCAL_RANK_H

#include "alg/Optimizer.h"
#include "util/Base.h"
#include "util/RandomUtil.h"
#include "util/Monitor.h"
#include "util/FileUtil.h"
#include "util/CompareUtil.h"
#include "util/EvaluationUtil.h"
#include "util/Data.h"
#include "util/Calculator.h"
#include "util/Parameter.h"

using namespace mf;
using namespace EvaluationUtil;
using namespace FileUtil;
using namespace Calculator;

class RESCAL_RANK: virtual public Optimizer {

protected:
    DFTI_DESCRIPTOR_HANDLE descriptor;
    DenseMatrix rescalA;
    vector<DenseMatrix> rescalR;
    DenseMatrix rescalA_G;
    vector<DenseMatrix> rescalR_G;

//    value_type cal_loss() {
//        return eval_rescal_train(parameter, data, rescalA, rescalR);
//    }

    void init_G(const int D) {
        // ToDo: why SGD cannot run if min_not_zeros is not initialized
//        if(parameter->optimization=="adagrad" || parameter->optimization=="adadelta"){
            min_not_zeros.resize(D * D);
            for (int i = 0; i < D * D; i++) {
                min_not_zeros(i) = min_not_zero_value;
            }

            rescalA_G.resize(data->N, D);
            rescalR_G.resize(data->K, DenseMatrix(D, D));
            rescalA_G.clear();
            for (int i = 0; i < data->K; i++) {
                rescalR_G[i].clear();
            }
//        }
    }

    void initialize() {
        RandomUtil::init_seed();

        current_epoch = 1;

        rescalA.resize(data->N, parameter->rescal_D);
        rescalR.resize(data->K, DenseMatrix(parameter->rescal_D, parameter->rescal_D));
        init_G(parameter->rescal_D);

        value_type bnd = sqrt(6) / sqrt(data->N + parameter->rescal_D);

        for (int row = 0; row < data->N; row++) {
            for (int col = 0; col < parameter->rescal_D; col++) {
                rescalA(row, col) = RandomUtil::uniform_real(-bnd, bnd);
            }
        }

        bnd = sqrt(6) / sqrt(parameter->rescal_D + parameter->rescal_D);

        for (int R_i = 0; R_i < data->K; R_i++) {
            DenseMatrix &sub_R = rescalR[R_i];
            for (int row = 0; row < parameter->rescal_D; row++) {
                for (int col = 0; col < parameter->rescal_D; col++) {
                    sub_R(row, col) = RandomUtil::uniform_real(-bnd, bnd);
                }
            }
        }


    }

    void update(Sample &sample, const value_type weight = 1.0) {

        value_type positive_score = cal_rescal_score(sample.relation_id, sample.p_sub, sample.p_obj, parameter->rescal_D, rescalA, rescalR);
        value_type negative_score = cal_rescal_score(sample.relation_id, sample.n_sub, sample.n_obj, parameter->rescal_D, rescalA, rescalR);

        value_type p_pre = 1;
        value_type n_pre = 1;

        // ToDo: ???
//        if (parameter->restore_from_hole) {
//            positive_score = sigmoid(positive_score);
//            negative_score = sigmoid(negative_score);
//            p_pre = g_sigmoid(positive_score);
//            n_pre = g_sigmoid(negative_score);
//        }

        if(positive_score - negative_score >= parameter->margin) {
            return;
        }

        violations++;

        DenseMatrix grad4R(parameter->rescal_D, parameter->rescal_D);
        unordered_map<int, Vec> grad4A_map;

        // Step 1: compute gradient descent

        update_4_R(sample, grad4R, p_pre, n_pre, weight);
        update_4_A(sample, grad4A_map, p_pre, n_pre, weight);

        // Step 2: do the update
        (this->*update_grad)(rescalR[sample.relation_id].data().begin(), grad4R.data().begin(), rescalR_G[sample.relation_id].data().begin(),
           parameter->rescal_D * parameter->rescal_D , weight);

        for (auto ptr = grad4A_map.begin(); ptr != grad4A_map.end(); ptr++) {

            Vec A_grad = ptr->second - parameter->lambdaA * row(rescalA, ptr->first);

            (this->*update_grad)(rescalA.data().begin() + parameter->rescal_D * ptr->first, A_grad.data().begin(), rescalA_G.data().begin() + parameter->rescal_D * ptr->first,
                                 parameter->rescal_D, weight);
        }

    }


    string eval(const int epoch){

        if (parameter->eval_train) {

            hit_rate train_measure = eval_rescal_train(parameter, data, rescalA, rescalR, parameter->output_path + "/RESCAL_RANK_train_epoch_" + to_string(epoch));

            string prefix = "sampled training data >>> ";

            print_hit_rate_train(prefix, parameter->hit_rate_topk, train_measure);

        }

        hit_rate testing_measure = eval_hit_rate(Method::m_RESCAL_RANK, parameter, data, &rescalA, &rescalR,
                                                 nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, parameter->output_path + "/RESCAL_RANK_test_epoch_" + to_string(epoch));

        string prefix = "testing data >>> ";

        print_hit_rate(prefix, parameter->hit_rate_topk, testing_measure);

        if (parameter->eval_rel) {

            hit_rate rel_measure = eval_relation_rescal(parameter, data, rescalA, rescalR, parameter->output_path + "/RESCAL_RANK_test_epoch_" + to_string(epoch));

            string prefix = "testing data relation evalution >>> ";

            print_hit_rate_rel(prefix, parameter->hit_rate_topk, rel_measure);
        }

        pair<value_type, value_type> map;
        map.first = -1;
        map.second = -1;

        if (parameter->eval_map) {
            map = eval_MAP(m_RESCAL_RANK, parameter, data, &rescalA, &rescalR, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
            string prefix = "testing data MAP evalution >>> ";
            print_map(prefix, parameter->num_of_replaced_entities, map);
        }

        string log = "";
        log.append("RESCAL_RANK,");
        log.append(parameter->optimization + ",");
        log.append(to_string(epoch) + ",");
        log.append(to_string(parameter->rescal_D) + ",");
        log.append(to_string(parameter->step_size) + ",");
        log.append(to_string(parameter->margin) + ",");
        log.append(to_string(parameter->lambdaA) + ",");
        log.append(to_string(parameter->lambdaR) + ",");
        if(parameter->optimization=="adadelta") {
            log.append(to_string(parameter->Rho) + ",");
        }

        string count_s = (testing_measure.count_s == -1? "Not Computed" : to_string(testing_measure.count_s));
        log.append(count_s + ",");

        string count_o = (testing_measure.count_o == -1? "Not Computed" : to_string(testing_measure.count_o));
        log.append(count_o + ",");

        string count_s_ranking = (testing_measure.count_s_ranking == -1? "Not Computed" : to_string(testing_measure.count_s_ranking));
        log.append(count_s_ranking + ",");

        string count_o_ranking = (testing_measure.count_o_ranking == -1? "Not Computed" : to_string(testing_measure.count_o_ranking));
        log.append(count_o_ranking + ",");

        log.append(to_string(testing_measure.count_s_filtering) + ",");
        log.append(to_string(testing_measure.count_o_filtering) + ",");
        log.append(to_string(testing_measure.count_s_ranking_filtering) + ",");
        log.append(to_string(testing_measure.count_o_ranking_filtering) + ",");

        string map1 = (map.first == -1 ? "Not Computed" : to_string(map.first));
        string map2 = (map.second == -1 ? "Not Computed" : to_string(map.second));

        log.append(map1 + ",");
        log.append(map2 + ",");

        string inv_count_s_ranking = (testing_measure.inv_count_s_ranking == -1? "Not Computed" : to_string(testing_measure.inv_count_s_ranking));
        log.append(count_s_ranking + ",");

        string inv_count_o_ranking = (testing_measure.inv_count_o_ranking == -1? "Not Computed" : to_string(testing_measure.inv_count_o_ranking));
        log.append(count_o_ranking + ",");

        log.append(to_string(testing_measure.inv_count_s_ranking_filtering) + ",");
        log.append(to_string(testing_measure.inv_count_o_ranking_filtering));

        return log;
    }

    void output(const int epoch) {

        if(parameter->optimization=="sgd"){
            output_matrices(rescalA, rescalR, epoch, parameter->output_path);
        }else{
            output_matrices(rescalA, rescalR, rescalA_G, rescalR_G, epoch, parameter->output_path);
        }

    }

    string get_log_header() {

        string header = "Method,Optimization,epoch,Dimension,step size,margin,lambdaA,lambdaR,";
        header += ((parameter->optimization=="adadelta")?"Rho,":"");
        header += "hit_rate_subject@" +
                  to_string(parameter->hit_rate_topk) + ",hit_rate_object@" +
                  to_string(parameter->hit_rate_topk) +
                  ",subject_ranking,object_ranking,hit_rate_subject_filter@" +
                  to_string(parameter->hit_rate_topk) + ",hit_rate_object_filter@" +
                  to_string(parameter->hit_rate_topk) +
                  ",subject_ranking_filter,object_ranking_filter,MAP_subject@" +
                  to_string(parameter->num_of_replaced_entities) + ",MAP_object@" + to_string(parameter->num_of_replaced_entities) +
                  ",MRR_subject,MRR_object,MRR_subject_filter,MRR_object_filter";
        return header;
    }


protected:

    void update_4_A(Sample &sample, unordered_map<int, Vec> &grad4A_map, const value_type p_pre, const value_type n_pre, const value_type weight) {

        DenseMatrix &R_k = rescalR[sample.relation_id];


        Vec p_tmp1 = prod(R_k, row(rescalA, sample.p_obj));
        Vec p_tmp2 = prod(row(rescalA, sample.p_sub), R_k);
        Vec n_tmp1 = prod(R_k, row(rescalA, sample.n_obj));
        Vec n_tmp2 = prod(row(rescalA, sample.n_sub), R_k);


        grad4A_map[sample.p_sub] = p_tmp1;

        unordered_map<int, Vec>::iterator ptr = grad4A_map.find(sample.p_obj);
        if (ptr == grad4A_map.end()) {
            grad4A_map[sample.p_obj] = p_pre * p_tmp2;
        } else {

            grad4A_map[sample.p_obj] += p_pre * p_tmp2;

        }

        ptr = grad4A_map.find(sample.n_sub);
        if (ptr == grad4A_map.end()) {
            grad4A_map[sample.n_sub] = n_pre * (-n_tmp1);
        } else {


            grad4A_map[sample.n_sub] += n_pre * (-n_tmp1);

        }

        ptr = grad4A_map.find(sample.n_obj);
        if (ptr == grad4A_map.end()) {
            grad4A_map[sample.n_obj] = n_pre * (-n_tmp2);
        } else {
            grad4A_map[sample.n_obj] += n_pre * (-n_tmp2);

        }
    }

    void update_4_R(Sample &sample, DenseMatrix &grad4R, const value_type p_pre, const value_type n_pre, const value_type weight) {

        value_type *p_sub = rescalA.data().begin() + sample.p_sub * parameter->rescal_D;
        value_type *p_obj = rescalA.data().begin() + sample.p_obj * parameter->rescal_D;

        value_type *n_sub = rescalA.data().begin() + sample.n_sub * parameter->rescal_D;
        value_type *n_obj = rescalA.data().begin() + sample.n_obj * parameter->rescal_D;



        grad4R.clear();

        for (int i = 0; i < parameter->rescal_D; i++) {
            for (int j = 0; j < parameter->rescal_D; j++) {
                grad4R(i, j) += p_pre * p_sub[i] * p_obj[j] - n_pre * n_sub[i] * n_obj[j];
            }
        }

        grad4R += - parameter->lambdaR * rescalR[sample.relation_id];
    }

public:

    RESCAL_RANK(){};
    RESCAL_RANK(Parameter *parameter, Data *data) : Optimizer(parameter, data) {}
    ~RESCAL_RANK() {}
};

#endif //RESCAL_RANK_H
