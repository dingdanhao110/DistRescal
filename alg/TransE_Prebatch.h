//
// Created by dhding on 1/25/18.
//

#ifndef DISTRESCAL_TRANSE_PREBATCH_H
#define DISTRESCAL_TRANSE_PREBATCH_H

#include "Prebatch_Optimizer.h"

template<typename OptimizerType>
virtual
class TRANSE_PREBATCH : virtual public PREBATCH_OPTIMIZER<OptimizerType> {
public:
    explicit TRANSE_PREBATCH(Parameter &parameter, Data &data) :
            PREBATCH_OPTIMIZER<OptimizerType>(parameter, data) {}

private:
    inline value_type cal_transe_score(const int rel_id, const int sub_id, const int obj_id) {

    }

    void update(Sample &sample) {

        bool subject_replace = (sample.n_sub != sample.p_sub); // true: subject is replaced, false: object is replaced.

        value_type positive_score = cal_transe_score(sample.p_sub, sample.p_obj, sample.relation_id);
        value_type negative_score = cal_transe_score(sample.n_sub, sample.n_obj, sample.relation_id);

        value_type margin = positive_score + parameter->margin - negative_score;

        if (parameter->margin_on)
            if (margin > 0) {
                violations++;
            } else {
                return;
            }
        value_type *p_sub_vec = transeA.data().begin() + sample.p_sub * parameter->dimension;
        value_type *p_obj_vec = transeA.data().begin() + sample.p_obj * parameter->dimension;

        value_type *n_sub_vec = transeA.data().begin() + sample.n_sub * parameter->dimension;
        value_type *n_obj_vec = transeA.data().begin() + sample.n_obj * parameter->dimension;

        value_type *rel_vec = transeR.data().begin() + sample.relation_id * parameter->dimension;

        Vec x = 2 * (row(transeA, sample.p_obj) - row(transeA, sample.p_sub) - row(transeR, sample.relation_id));

        if (parameter->L1_flag) {
            for (int i = 0; i < parameter->transe_D; i++) {
                if (x(i) > 0) {
                    x(i) = 1;
                } else {
                    x(i) = -1;
                }
            }
        }

        (this->update_grad)(rel_vec, x.data().begin(),
                             transeR_G.data().begin() + sample.relation_id * parameter->dimension, parameter->transe_D,
                             weight);
        (this->update_grad)(p_sub_vec, x.data().begin(), transeA_G.data().begin() + sample.p_sub * parameter->transe_D,
                             parameter->dimension, weight);
        (this->update_grad)(p_obj_vec, x.data().begin(), transeA_G.data().begin() + sample.p_obj * parameter->transe_D,
                             parameter->dimension, -weight);

        x = 2 * (row(transeA, sample.n_obj) - row(transeA, sample.n_sub) - row(transeR, sample.relation_id));

        if (parameter->L1_flag) {
            for (int i = 0; i < parameter->dimension; i++) {
                if (x(i) > 0) {
                    x(i) = 1;
                } else {
                    x(i) = -1;
                }
            }
        }

        (this->*update_grad)(rel_vec, x.data().begin(),
                             transeR_G.data().begin() + sample.relation_id * parameter->transe_D, parameter->transe_D,
                             -weight);
        if (subject_replace) {
            (this->*update_grad)(n_sub_vec, x.data().begin(),
                                 transeA_G.data().begin() + sample.n_sub * parameter->transe_D, parameter->transe_D,
                                 -weight);
            (this->*update_grad)(p_obj_vec, x.data().begin(),
                                 transeA_G.data().begin() + sample.p_obj * parameter->transe_D, parameter->transe_D,
                                 weight);
        } else {
            (this->*update_grad)(p_sub_vec, x.data().begin(),
                                 transeA_G.data().begin() + sample.p_sub * parameter->transe_D, parameter->transe_D,
                                 -weight);
            (this->*update_grad)(n_obj_vec, x.data().begin(),
                                 transeA_G.data().begin() + sample.n_obj * parameter->transe_D, parameter->transe_D,
                                 weight);
        }

        normalizeOne(rel_vec, parameter->dimension);
        normalizeOne(p_sub_vec, parameter->dimension);
        normalizeOne(p_obj_vec, parameter->dimension);

        if (subject_replace) {
            normalizeOne(n_sub_vec, parameter->dimension);
        } else {
            normalizeOne(n_obj_vec, parameter->dimension);
        }


    }

    inline void normalizeOne(value_type* vec, int dim){

    }
};


#endif //DISTRESCAL_TRANSE_PREBATCH_H
