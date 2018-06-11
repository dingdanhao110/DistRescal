#ifndef EVALUATIONUTIL_H
#define EVALUATIONUTIL_H

#include "Base.h"
#include "Data.h"
#include "CompareUtil.h"
#include "RandomUtil.h"
#include "Calculator.h"
#include "Parameter.h"
#include "../alg/Sampler.h"

using namespace Calculator;

#define mkl4IPRank

struct hit_rate {
    value_type count_s;
    value_type count_o;
    value_type count_s_ranking;
    value_type count_o_ranking;
    value_type count_s_filtering;
    value_type count_o_filtering;
    value_type count_s_ranking_filtering;
    value_type count_o_ranking_filtering;
    value_type inv_count_s_ranking;
    value_type inv_count_o_ranking;
    value_type inv_count_s_ranking_filtering;
    value_type inv_count_o_ranking_filtering;
};

namespace EvaluationUtil {

    inline void print_hit_rate(string prefix, const int hit_rate_topk, hit_rate result) {
        if (result.count_s != -1) {
            cout << prefix << "hit_rate_subject@" << hit_rate_topk << ": " << result.count_s
                 << ", hit_rate_object@" << hit_rate_topk << ": " << result.count_o << ", subject_ranking: "
                 << result.count_s_ranking << ", object_ranking: " << result.count_o_ranking << endl;
        }
        cout << prefix << "hit_rate_subject_filter@" << hit_rate_topk << ": " << result.count_s_filtering
             << ", hit_rate_object_filter@" << hit_rate_topk << ": " << result.count_o_filtering << ", subject_ranking_filter: "
             << result.count_s_ranking_filtering << ", object_ranking_filter: " << result.count_o_ranking_filtering << endl;

        if(result.inv_count_s_ranking!=-1){
            cout << prefix << "subject_MRR: " << result.inv_count_s_ranking << ", object_MRR: " << result.inv_count_o_ranking << endl;
        }

        cout << prefix << "subject_MRR_filter: " << result.inv_count_s_ranking_filtering << ", object_MRR_filter: " << result.inv_count_o_ranking_filtering << endl;
    }

    inline value_type cal_loss_single_thread(Parameter *parameter, Data *data, value_type *rescalA, value_type *rescalR) {

        value_type loss = 0;

        Sample sample;
        for (int i = 0; i < data->training_triples.size(); i++) {
            Sampler::random_sample(*data, sample, i);
            value_type positive_score = cal_rescal_score(sample.relation_id, sample.p_sub, sample.p_obj, rescalA, rescalR, parameter);
            value_type negative_score = cal_rescal_score(sample.relation_id, sample.n_sub, sample.n_obj, rescalA, rescalR, parameter);
            value_type tmp = positive_score - negative_score - parameter->margin;
            if(tmp < 0) {
                loss -= tmp;
            }
        }

        return loss;
    }

    inline hit_rate eval_hit_rate(Parameter *parameter, Data *data, value_type *rescalA, value_type *rescalR) {

        value_type *A = rescalA;
        value_type *Rs = rescalR;

        const int num_of_thread = parameter->num_of_eval_thread;
        const int hit_rate_topk = parameter->hit_rate_topk;

        Monitor timer;

        pool *eval_thread_pool = new pool(num_of_thread);

#ifdef detailed_eval
        vector<vector<ull> > rel_counts_s(num_of_thread, vector<ull>(data->num_of_relation, 0));
        vector<vector<ull> > rel_counts_o(num_of_thread, vector<ull>(data->num_of_relation, 0));

        vector<vector<ull> > rel_counts_s_ranking(num_of_thread, vector<ull>(data->num_of_relation, 0));
        vector<vector<ull> > rel_counts_o_ranking(num_of_thread, vector<ull>(data->num_of_relation, 0));

        vector<vector<value_type> > inv_rel_counts_s_ranking(num_of_thread, vector<value_type>(data->num_of_relation, 0));
        vector<vector<value_type> > inv_rel_counts_o_ranking(num_of_thread, vector<value_type>(data->num_of_relation, 0));

#endif
        vector<vector<ull> > rel_counts_s_filtering(num_of_thread, vector<ull>(data->num_of_relation, 0));
        vector<vector<ull> > rel_counts_o_filtering(num_of_thread, vector<ull>(data->num_of_relation, 0));

        vector<vector<ull> > rel_counts_s_ranking_filtering(num_of_thread, vector<ull>(data->num_of_relation, 0));
        vector<vector<ull> > rel_counts_o_ranking_filtering(num_of_thread, vector<ull>(data->num_of_relation, 0));

        vector<vector<value_type> > inv_rel_counts_s_ranking_filtering(num_of_thread, vector<value_type>(data->num_of_relation, 0));
        vector<vector<value_type> > inv_rel_counts_o_ranking_filtering(num_of_thread, vector<value_type>(data->num_of_relation, 0));

        int testing_size = data->num_of_testing_triples;

        value_type *AR = new value_type[data->num_of_entity * parameter->dimension];
        value_type *RAT = new value_type[parameter->dimension * data->num_of_entity];

        for (int relation_id = 0; relation_id < data->num_of_relation; relation_id++) {
            vector<Tuple<int> > &tuples = data->test_rel2tuples[relation_id];

            value_type *R = Rs + relation_id * parameter->dimension * parameter->dimension;

#ifdef mkl4IPRank
            mkl_set_num_threads(num_of_thread);
#endif

            timer.start();

            // calculate AR and RAT as cache
            matrix_product_mkl(A, R, AR, data->num_of_entity, parameter->dimension,
                               parameter->dimension, parameter->dimension);

            matrix_product_transpose_mkl(R, A, RAT, parameter->dimension, parameter->dimension,
                                         data->num_of_entity, parameter->dimension);

            timer.stop();

            int size4relation = tuples.size();

            int workload = size4relation / num_of_thread +
                           ((size4relation % num_of_thread == 0) ? 0 : 1);

#ifdef mkl4IPRank
            mkl_set_num_threads(1);
#endif

            std::function<void(int)> compute_func = [&](int thread_index) -> void {

                int start = thread_index * workload;
                int end = std::min(start + workload, size4relation);

                Monitor thread_timer;

#ifdef detailed_eval
                vector<ull> &counts_s = rel_counts_s[thread_index];
                vector<ull> &counts_o = rel_counts_o[thread_index];
                vector<ull> &counts_s_ranking = rel_counts_s_ranking[thread_index];
                vector<ull> &counts_o_ranking = rel_counts_o_ranking[thread_index];
                vector<value_type> &inv_counts_s_ranking = inv_rel_counts_s_ranking[thread_index];
                vector<value_type> &inv_counts_o_ranking = inv_rel_counts_o_ranking[thread_index];
#endif
                vector<ull> &counts_s_filtering = rel_counts_s_filtering[thread_index];
                vector<ull> &counts_o_filtering = rel_counts_o_filtering[thread_index];
                vector<ull> &counts_s_ranking_filtering = rel_counts_s_ranking_filtering[thread_index];
                vector<ull> &counts_o_ranking_filtering = rel_counts_o_ranking_filtering[thread_index];
                vector<value_type> &inv_counts_s_ranking_filtering = inv_rel_counts_s_ranking_filtering[thread_index];
                vector<value_type> &inv_counts_o_ranking_filtering = inv_rel_counts_o_ranking_filtering[thread_index];

                // first: entity id, second: score
#ifdef detailed_eval
                vector<pair<int, value_type> > result(data->num_of_entity);
#endif
                vector<pair<int, value_type> > result_filtering(data->num_of_entity);
                int result_filtering_index = 0;

                value_type *score_list = new value_type[data->num_of_entity];

                for (int n = start; n < end; n++) {

                    Tuple<int> &test_tuple = tuples[n];

                    // first replace subject with other entities
                    // AR*A_jt
                    value_type *A_row = A + test_tuple.object * parameter->dimension;

#ifdef mkl4IPRank
                    matrix_product_transpose_mkl(AR, A_row, score_list, data->num_of_entity, parameter->dimension, 1,
                                         parameter->dimension);
#else
                    matrix_product_transpose(AR, A_row, score_list, data->num_of_entity, parameter->dimension, 1,
                                             parameter->dimension);
#endif

                    result_filtering_index = 0;

                    for (int index = 0; index < data->num_of_entity; index++) {
                        value_type score = score_list[index];
#ifdef detailed_eval
                        result[index] = make_pair(index, score);
#endif
                        if ((index == test_tuple.subject) ||
                            (!data->faked_s_tuple_exist(index, relation_id, test_tuple.object))) {
                            result_filtering[result_filtering_index] = make_pair(index, score);
                            result_filtering_index++;
                        }
                    }

                    // sort
#ifdef detailed_eval
                    std::sort(result.begin(), result.end(), &CompareUtil::pairGreaterCompare<int>);
#endif
                    std::sort(result_filtering.begin(), result_filtering.begin() + result_filtering_index,
                              &CompareUtil::pairGreaterCompare<int>);

#ifdef detailed_eval
                    bool found = false;
#endif
                    bool found_filtering = false;

                    for (int i = 0; i < hit_rate_topk; i++) {
#ifdef detailed_eval
                        if (result[i].first == test_tuple.subject) {
                            counts_s[relation_id]++;
                            counts_s_ranking[relation_id] += i + 1;
                            inv_counts_s_ranking[relation_id] += 1.0 / (i + 1);
                            found = true;
                        }
#endif

                        if (result_filtering[i].first == test_tuple.subject) {
                            counts_s_filtering[relation_id]++;
                            counts_s_ranking_filtering[relation_id] += i + 1;
                            inv_counts_s_ranking_filtering[relation_id] += 1.0 / (i + 1);
                            found_filtering = true;
                        }
                    }

#ifdef detailed_eval
                    if (!found) {
                        for (int i = hit_rate_topk; i < data->num_of_entity; i++) {
                            if (result[i].first == test_tuple.subject) {
                                counts_s_ranking[relation_id] += i + 1;
                                inv_counts_s_ranking[relation_id] += 1.0 / (i + 1);
                                found = true;
                                break;
                            }
                        }
                    }
#endif

                    if (!found_filtering) {
                        for (int i = hit_rate_topk; i < result_filtering_index; i++) {
                            if (result_filtering[i].first == test_tuple.subject) {
                                counts_s_ranking_filtering[relation_id] += i + 1;
                                inv_counts_s_ranking_filtering[relation_id] += 1.0 / (i + 1);
                                found_filtering = true;
                                break;
                            }
                        }
                    }

#ifdef detailed_eval
                    if ((!found) || (!found_filtering)) {
                        cerr << "logical error in eval_hit_rate! (1)" << endl;
                        exit(1);
                    }
#else
                    if (!found_filtering) {
                        cerr << "logical error in eval_hit_rate! (1)" << endl;
                        exit(1);
                    }
#endif

                    // then replace object with other entities
                    // A_i * RAT
                    A_row = A + test_tuple.subject * parameter->dimension;

#ifdef mkl4IPRank
                    matrix_product_mkl(A_row, RAT, score_list, 1, parameter->dimension, parameter->dimension,
                               data->num_of_entity);
#else
                    matrix_product(A_row, RAT, score_list, 1, parameter->dimension, parameter->dimension,
                                   data->num_of_entity);
#endif

                    result_filtering_index = 0;

                    for (int index = 0; index < data->num_of_entity; index++) {

                        value_type score = score_list[index];

#ifdef detailed_eval
                        result[index] = make_pair(index, score);
#endif

                        if ((index == test_tuple.object) ||
                            (!data->faked_o_tuple_exist(test_tuple.subject, relation_id, index))) {
                            result_filtering[result_filtering_index] = make_pair(index, score);
                            result_filtering_index++;
                        }
                    }

                    // sort
#ifdef detailed_eval
                    std::sort(result.begin(), result.end(), &CompareUtil::pairGreaterCompare<int>);
#endif
                    std::sort(result_filtering.begin(), result_filtering.begin() + result_filtering_index,
                              &CompareUtil::pairGreaterCompare<int>);

                    found_filtering = false;
#ifdef detailed_eval
                    found = false;
#endif

                    for (int i = 0; i < hit_rate_topk; i++) {
#ifdef detailed_eval
                        if (result[i].first == test_tuple.object) {
                            counts_o[relation_id]++;
                            counts_o_ranking[relation_id] += i + 1;
                            inv_counts_o_ranking[relation_id] += 1.0 / (i + 1);
                            found = true;
                        }
#endif

                        if (result_filtering[i].first == test_tuple.object) {
                            counts_o_filtering[relation_id]++;
                            counts_o_ranking_filtering[relation_id] += i + 1;
                            inv_counts_o_ranking_filtering[relation_id] += 1.0 / (i + 1);
                            found_filtering = true;
                        }
                    }

#ifdef detailed_eval
                    if (!found) {
                        for (int i = hit_rate_topk; i < data->num_of_entity; i++) {
                            if (result[i].first == test_tuple.object) {
                                counts_o_ranking[relation_id] += i + 1;
                                inv_counts_o_ranking[relation_id] += 1.0 / (i + 1);
                                found = true;
                                break;
                            }
                        }
                    }
#endif

                    if (!found_filtering) {
                        for (int i = hit_rate_topk; i < result_filtering_index; i++) {
                            if (result_filtering[i].first == test_tuple.object) {
                                counts_o_ranking_filtering[relation_id] += i + 1;
                                inv_counts_o_ranking_filtering[relation_id] += 1.0 / (i + 1);
                                found_filtering = true;
                                break;
                            }
                        }
                    }

#ifdef detailed_eval
                    if ((!found) || (!found_filtering)) {
                        cerr << "logical error in eval_hit_rate! (2)" << endl;
                        exit(1);
                    }
#else
                    if (!found_filtering) {
                        cerr << "logical error in eval_hit_rate! (2)" << endl;
                        exit(1);
                    }
#endif
                }

                delete[] score_list;
            };

            for(int thread_index = 0; thread_index < num_of_thread; thread_index++) {
                eval_thread_pool->schedule(std::bind(compute_func, thread_index));
            }

            // wait until all threads finish
            eval_thread_pool->wait();
        }

        delete[] AR;
        delete[] RAT;

        hit_rate measure;

#ifdef detailed_eval
        measure.count_s = 0;
        measure.count_o = 0;
        measure.count_s_ranking = 0;
        measure.count_o_ranking = 0;
        measure.inv_count_s_ranking = 0;
        measure.inv_count_o_ranking = 0;
        vector<ull> rel_s(data->num_of_relation, 0);
        vector<ull> rel_o(data->num_of_relation, 0);
        vector<ull> rel_s_ranking(data->num_of_relation, 0);
        vector<ull> rel_o_ranking(data->num_of_relation, 0);
        vector<value_type > inv_rel_s_ranking(data->num_of_relation, 0);
        vector<value_type > inv_rel_o_ranking(data->num_of_relation, 0);
#else
        measure.count_s = -1;
        measure.count_o = -1;
        measure.count_s_ranking = -1;
        measure.count_o_ranking = -1;
        measure.inv_count_s_ranking = -1;
        measure.inv_count_o_ranking = -1;
#endif
        measure.count_s_filtering = 0;
        measure.count_o_filtering = 0;
        measure.count_s_ranking_filtering = 0;
        measure.count_o_ranking_filtering = 0;
        measure.inv_count_s_ranking_filtering = 0;
        measure.inv_count_o_ranking_filtering = 0;

        vector<ull> rel_s_filtering(data->num_of_relation, 0);
        vector<ull> rel_o_filtering(data->num_of_relation, 0);
        vector<ull> rel_s_ranking_filtering(data->num_of_relation, 0);
        vector<ull> rel_o_ranking_filtering(data->num_of_relation, 0);

        vector<value_type > inv_rel_s_ranking_filtering(data->num_of_relation, 0);
        vector<value_type > inv_rel_o_ranking_filtering(data->num_of_relation, 0);

        for (int thread_id = 0; thread_id < num_of_thread; thread_id++) {
#ifdef detailed_eval
            vector<ull> &counts_s = rel_counts_s[thread_id];
            vector<ull> &counts_o = rel_counts_o[thread_id];
            vector<ull> &counts_s_ranking = rel_counts_s_ranking[thread_id];
            vector<ull> &counts_o_ranking = rel_counts_o_ranking[thread_id];
            vector<value_type> &inv_counts_s_ranking = inv_rel_counts_s_ranking[thread_id];
            vector<value_type> &inv_counts_o_ranking = inv_rel_counts_o_ranking[thread_id];
#endif
            vector<ull> &counts_s_filtering = rel_counts_s_filtering[thread_id];
            vector<ull> &counts_o_filtering = rel_counts_o_filtering[thread_id];
            vector<ull> &counts_s_ranking_filtering = rel_counts_s_ranking_filtering[thread_id];
            vector<ull> &counts_o_ranking_filtering = rel_counts_o_ranking_filtering[thread_id];
            vector<value_type> &inv_counts_s_ranking_filtering = inv_rel_counts_s_ranking_filtering[thread_id];
            vector<value_type> &inv_counts_o_ranking_filtering = inv_rel_counts_o_ranking_filtering[thread_id];

            for (int relation_id = 0; relation_id < data->num_of_relation; relation_id++) {
#ifdef detailed_eval
                measure.count_s += counts_s[relation_id];
                measure.count_o += counts_o[relation_id];
                measure.count_s_ranking += counts_s_ranking[relation_id];
                measure.count_o_ranking += counts_o_ranking[relation_id];
                measure.inv_count_s_ranking += inv_counts_s_ranking[relation_id];
                measure.inv_count_o_ranking += inv_counts_o_ranking[relation_id];

                rel_s[relation_id] += counts_s[relation_id];
                rel_o[relation_id] += counts_o[relation_id];
                rel_s_ranking[relation_id] += counts_s_ranking[relation_id];
                rel_o_ranking[relation_id] += counts_o_ranking[relation_id];
                inv_rel_s_ranking[relation_id] += inv_counts_s_ranking[relation_id];
                inv_rel_o_ranking[relation_id] += inv_counts_o_ranking[relation_id];
#endif
                measure.count_s_filtering += counts_s_filtering[relation_id];
                measure.count_o_filtering += counts_o_filtering[relation_id];
                measure.count_s_ranking_filtering += counts_s_ranking_filtering[relation_id];
                measure.count_o_ranking_filtering += counts_o_ranking_filtering[relation_id];
                measure.inv_count_s_ranking_filtering += inv_counts_s_ranking_filtering[relation_id];
                measure.inv_count_o_ranking_filtering += inv_counts_o_ranking_filtering[relation_id];

                rel_s_filtering[relation_id] += counts_s_filtering[relation_id];
                rel_o_filtering[relation_id] += counts_o_filtering[relation_id];
                rel_s_ranking_filtering[relation_id] += counts_s_ranking_filtering[relation_id];
                rel_o_ranking_filtering[relation_id] += counts_o_ranking_filtering[relation_id];
                inv_rel_s_ranking_filtering[relation_id] += inv_counts_s_ranking_filtering[relation_id];
                inv_rel_o_ranking_filtering[relation_id] += inv_counts_o_ranking_filtering[relation_id];
            }
        }

#ifdef detailed_eval
        measure.count_s /= testing_size;
        measure.count_o /= testing_size;
        measure.count_s_ranking /= testing_size;
        measure.count_o_ranking /= testing_size;
        measure.inv_count_s_ranking /= testing_size;
        measure.inv_count_o_ranking /= testing_size;
#endif

        measure.count_s_filtering /= testing_size;
        measure.count_o_filtering /= testing_size;
        measure.count_s_ranking_filtering /= testing_size;
        measure.count_o_ranking_filtering /= testing_size;
        measure.inv_count_s_ranking_filtering /= testing_size;
        measure.inv_count_o_ranking_filtering /= testing_size;

        delete eval_thread_pool;

        return measure;
    }

    inline void transform_transe2rescal(const int rescal_d, Parameter *parameter, Data *data, value_type *transeA, value_type *transeR, value_type *rescalA, value_type *rescalR){

        for (int entity_index = 0; entity_index < data->num_of_entity; entity_index++) {
            value_type *rescalA_row = rescalA + entity_index * rescal_d;
            value_type *transeA_row = transeA + entity_index * parameter->dimension;
            for (int d = 0; d < parameter->dimension; d++) {
                rescalA_row[d] = 1;
            }

            for (int d = parameter->dimension; d < 2 * parameter->dimension; d++) {
                rescalA_row[d] = transeA_row[d - parameter->dimension];
            }

            rescalA_row[2 * parameter->dimension] = inner_product(transeA_row, transeA_row, parameter->dimension);
        }

        std::fill(rescalR, rescalR + data->num_of_relation * rescal_d * rescal_d, 0);

        for (int rel_id = 0; rel_id < data->num_of_relation; rel_id++) {
            value_type *RescalR_matrix = rescalR + rel_id * rescal_d * rescal_d;

            value_type *transeR_row = transeR + rel_id * parameter->dimension;

            RescalR_matrix[0] = - inner_product(transeR_row, transeR_row, parameter->dimension);

            for (int row = 0; row < parameter->dimension; row++) {
                *(RescalR_matrix + row * rescal_d + row + parameter->dimension) = 2 * transeR_row[row];
            }

            RescalR_matrix[2 * parameter->dimension] = -1;

            for (int row = parameter->dimension; row < 2 * parameter->dimension; row++) {
                *(RescalR_matrix + row * rescal_d + row - parameter->dimension) = -2 * transeR_row[row - parameter->dimension];
            }

            for (int row = parameter->dimension; row < 2 * parameter->dimension; row++) {
                *(RescalR_matrix + row * rescal_d + row) = 2;
            }

            *(RescalR_matrix + 2 * parameter->dimension * rescal_d) = -1;
        }

    }

    inline hit_rate eval_hit_rate_TransE(Parameter *parameter, Data *data, value_type *transeA, value_type *transeR) {
        int rescal_d =  2 * parameter->dimension + 1;
        value_type *rescalA = new value_type[data->num_of_entity * rescal_d];
        value_type *rescalR = new value_type[data->num_of_relation * rescal_d * rescal_d];

        transform_transe2rescal(rescal_d, parameter, data, transeA, transeR, rescalA, rescalR);
        int original_dim = parameter->dimension;
        parameter->dimension = rescal_d;

        cout << "transformation completed!\n";

        hit_rate result = eval_hit_rate(parameter, data, rescalA, rescalR);

        delete[] rescalA;
        delete[] rescalR;
        cout << "rescal_d: " << rescal_d << endl;
        parameter->dimension = original_dim;

        return result;
    }


}

#endif //EVALUATIONUTIL_H