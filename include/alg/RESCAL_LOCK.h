//
// Created by dhding on 9/9/17.
//

#ifndef DISTRESCAL_RESCAL_LOCK_H
#define DISTRESCAL_RESCAL_LOCK_H

#endif //DISTRESCAL_RESCAL_LOCK_H

#include "RESCAL_RANK.h"
#include <mutex>

class RESCAL_LOCK: public RESCAL_RANK{
protected:
    vector<std::mutex> A_locks;
    vector<std::mutex> R_locks;
    void update(Sample &sample, const value_type weight = 1.0)override {
        std::lock_guard<std::mutex> lock1(A_locks[sample.p_obj]);
        std::lock_guard<std::mutex> lock2(A_locks[sample.n_obj]);
        std::lock_guard<std::mutex> lock3(A_locks[sample.p_sub]);
        std::lock_guard<std::mutex> lock4(A_locks[sample.n_sub]);
        std::lock_guard<std::mutex> lock5(R_locks[sample.relation_id]);


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
public:
    RESCAL_LOCK(){}
    RESCAL_LOCK(Parameter *parameter, Data *data): RESCAL_RANK(parameter, data),A_locks(data->N),R_locks(data->K){}
    ~RESCAL_LOCK() {}
};