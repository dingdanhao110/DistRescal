//
// Created by dhding on 9/30/17.
//

#ifndef DISTRESCAL_RESCAL_BATCH_H
#define DISTRESCAL_RESCAL_BATCH_H

#include "../util/Base.h"
#include "../util/RandomUtil.h"
#include "../util/Monitor.h"
#include "../util/FileUtil.h"
#include "../util/CompareUtil.h"
#include "../util/EvaluationUtil.h"
#include "../util/Data.h"
#include "../util/Calculator.h"
#include "../util/Parameter.h"
#include "../alg/Optimizer.h"
#include "../struct/Batch.h"

using namespace EvaluationUtil;
using namespace FileUtil;
using namespace Calculator;

/**
 * Margin based RESCAL_NOLOCK
 */
template<typename OptimizerType>
class RESCAL_BATCH {
public:
    /**
     * Start to train
     */
    void train() {

        RandomUtil::init_seed();
        initialize();

        Monitor timer;

        std::vector<int> indices(data->num_of_training_triples);
        std::iota(std::begin(indices), std::end(indices), 0);

//        if (parameter.print_log_header) {
//            print_log(get_log_header());
//        }

        int workload = data->num_of_training_triples / parameter->num_of_thread;

        value_type total_time = 0.0;

        for (int epoch = current_epoch; epoch <= parameter->epoch; epoch++) {
            //Sample all training data
            timer.start();
            //Bucket_assigner bucket_assigner(parameter->num_of_thread);
            Batch_assigner batch_assigner(parameter->num_of_thread);
            vector<Sample> training_samples(0);
            training_samples.reserve(data->num_of_training_triples);
            std::mutex mutex_scheduler;
            std::random_shuffle(indices.begin(), indices.end());
            for (int thread_index = 0; thread_index < parameter->num_of_thread; thread_index++) {

                computation_thread_pool->schedule(std::bind([&](const int thread_index) {
                    std::mt19937 *generator = new std::mt19937(clock() + std::hash<std::thread::id>()(std::this_thread::get_id()));
                    int start = thread_index * workload;
                    int end = std::min(start + workload, data->num_of_training_triples);
                    Sample sample;
                    for (int n = start; n < end; n++) {
                        if(parameter->num_of_thread == 1){
                            Sampler::random_sample(*data, sample, indices[n]);
                        } else {
                            Sampler::random_sample_multithreaded(*data, sample, indices[n],generator);
                        }
                        //cout<<"Thread "<<std::this_thread::get_id()<<": Work assigned"<<endl;
                        //update(sample);
                        {
                            //TODO: Collect samples...
                            std::lock_guard<std::mutex> lock(mutex_scheduler);
                            training_samples.push_back(sample);
                            batch_assigner.assign(sample);
                        }
                    }
                }, thread_index));
            }

            computation_thread_pool->wait();

            timer.stop();

            total_time += timer.getElapsedTime();
            cout << "------------------------" << endl;
            cout << "epoch " << epoch<<", sampling time "<<timer.getElapsedTime()<< " secs" << endl;

            int batch=0;
//            cout << "**************" << endl;
//            cout << "Batch: " << batch<<endl;
//            cout << "Bucket distribution: free count:"<<batch_assigner.get_free_count()<<endl;
//            cout<<"Samples in each bucket: ";
//            for(auto& b:batch_assigner.get_buckets()){
//                cout<<b.size()<<" ";
//            }
//            cout<<endl;
//            cout<<"Entities in each bucket: ";
//            for(auto& b:batch_assigner.get_buckets()){
//                cout<<b.entity_count()<<" ";
//            }
//            cout<<endl;
            //allocate samples and update in parallel
            violations = 0;
            loss = 0;
            timer.start();

            for (int thread_index = 0; thread_index < parameter->num_of_thread; thread_index++) {

                computation_thread_pool->schedule(std::bind([&](const int thread_index) {
                    //Allocate samples from each bucket
                    const auto& bucket=batch_assigner.get_buckets()[thread_index];
                    //Call update functions
                    for(auto& sample:bucket.get_samples()){
                        update(sample);
                    }
                }, thread_index));
            }
            computation_thread_pool->wait();

            while(!batch_assigner.is_finished()){
                ++batch;
                //assign next batch
                batch_assigner.next_batch();

//                cout << "**************" << endl;
//                cout << "Batch: " << batch<<endl;
//                cout << "Bucket distribution: free count:"<<batch_assigner.get_free_count()<<endl;
//                cout<<"Samples in each bucket: ";
//                for(auto& b:batch_assigner.get_buckets()){
//                    cout<<b.size()<<" ";
//                }
//                cout<<endl;
//                cout<<"Entities in each bucket: ";
//                for(auto& b:batch_assigner.get_buckets()){
//                    cout<<b.entity_count()<<" ";
//                }
//                cout<<endl;

                //initiate next round
                for (int thread_index = 0; thread_index < parameter->num_of_thread; thread_index++) {
                    computation_thread_pool->schedule(std::bind([&](const int thread_index) {
                        //Allocate samples from each bucket
                        const auto& bucket=batch_assigner.get_buckets()[thread_index];
                        //Call update functions
                        for(auto& sample:bucket.get_samples()){
                            update(sample);
                        }
                    }, thread_index));
                }
                computation_thread_pool->wait();
            }

            computation_thread_pool->wait();

            timer.stop();

            total_time += timer.getElapsedTime();

            cout  << "training time " << timer.getElapsedTime() << " secs" << endl;

            cout << "violations: " << violations << endl;

            if(parameter->show_loss) {

                timer.start();

                loss = cal_loss();

                cout << "loss: " << loss << endl;

                timer.stop();

                cout << "time for computing loss: " << timer.getElapsedTime() << " secs" << endl;
            }

            if (epoch % parameter->print_epoch == 0) {

                timer.start();

                eval(epoch);

                timer.stop();

                cout << "time for evaluation: " << timer.getElapsedTime() << " secs" << endl;

            }

            if (epoch % parameter->output_epoch == 0) {
                output(epoch);
            }

            cout << "------------------------" << endl;
        }

        cout << "Total Training Time: " << total_time << " secs" << endl;
    }
protected:
    // Functor object of update function
    OptimizerType update_grad;

    pool *computation_thread_pool = nullptr;
    Parameter *parameter= nullptr;
    Data *data= nullptr;
    int current_epoch=0;
    int violations=0;
    value_type loss=0;
    int block_size;

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
        //cout<<sample.relation_id<<" "<<sample.p_obj<<" "<<sample.p_sub<<" "<<sample.n_obj<<" "<<sample.n_sub<<endl;
        value_type positive_score = cal_rescal_score(sample.relation_id, sample.p_sub, sample.p_obj, rescalA, rescalR, parameter);
        value_type negative_score = cal_rescal_score(sample.relation_id, sample.n_sub, sample.n_obj, rescalA, rescalR, parameter);

        value_type p_pre = 1;
        value_type n_pre = 1;

//        if (positive_score - negative_score >= parameter->margin) {
//            return;
//        }
        if (positive_score - negative_score < parameter->margin) {
            violations++;
        }
        //DenseMatrix grad4R(parameter.rescal_D, parameter.rescal_D);
        value_type *grad4R = new value_type[parameter->dimension * parameter->dimension];
        unordered_map<int, value_type *> grad4A_map;

        // Step 1: compute gradient descent
        update_4_R(sample, grad4R, p_pre, n_pre);
        update_4_A(sample, grad4A_map, p_pre, n_pre);

        // Step 2: do the update
        update_grad(rescalR + sample.relation_id * parameter->dimension * parameter->dimension, grad4R,
                    rescalR_G + sample.relation_id * parameter->dimension * parameter->dimension,
                    parameter->dimension * parameter->dimension, parameter);

        value_type *A_grad = new value_type[parameter->dimension];
        for (auto ptr = grad4A_map.begin(); ptr != grad4A_map.end(); ptr++) {

            //Vec A_grad = ptr->second - parameter.lambdaA * row(rescalA, ptr->first);
            //TODO: DOUBLE CHECK
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

    void update_4_A(const Sample &sample, unordered_map<int, value_type*> &grad4A_map, const value_type p_pre,
                    const value_type n_pre) {
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

        //Vec n_tmp1 = prod(R_k, row(rescalA, sample.n_obj));
        value_type *n_tmp1 = new value_type[parameter->dimension];
        for (int i = 0; i < parameter->dimension; ++i) {
            n_tmp1[i] = 0;
            for (int j = 0; j < parameter->dimension; ++j) {
                n_tmp1[i] += R_k[i * parameter->dimension + j] * rescalA[sample.n_obj * parameter->dimension + j];
            }
        }

        //Vec n_tmp2 = prod(row(rescalA, sample.n_sub), R_k);
        value_type *n_tmp2 = new value_type[parameter->dimension];
        for (int i = 0; i < parameter->dimension; ++i) {
            n_tmp2[i] = 0;
            for (int j = 0; j < parameter->dimension; ++j) {
                n_tmp2[i] += rescalA[sample.n_sub * parameter->dimension + j] * R_k[j * parameter->dimension + i];
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
        value_type *p_sub = rescalA + sample.p_sub * parameter->dimension;
        value_type *p_obj = rescalA + sample.p_obj * parameter->dimension;

        value_type *n_sub = rescalA + sample.n_sub * parameter->dimension;
        value_type *n_obj = rescalA + sample.n_obj * parameter->dimension;


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
        value_type *R_k = rescalR + sample.relation_id * parameter->dimension * parameter->dimension;

        for (int i = 0; i < parameter->dimension; i++) {
            for (int j = 0; j < parameter->dimension; j++) {
                grad4R[i * parameter->dimension + j] +=
                        -parameter->lambdaR * R_k[i * parameter->dimension + j];
            }
        }
        //TODO: Double check.
        //cout<<"Exiting update4R\n";
    }

public:
    explicit RESCAL_BATCH<OptimizerType>(Parameter &parameter, Data &data) {
        this->parameter=&parameter;
        this->data=&data;
        computation_thread_pool = new pool(parameter.num_of_thread);
        block_size=this->data->num_of_entity/(parameter.num_of_thread*3+3)+1;
    }

    ~RESCAL_BATCH() {
        delete[] rescalA;
        delete[] rescalR;
        delete[] rescalA_G;
        delete[] rescalR_G;
        delete computation_thread_pool;
    }
};
#endif //DISTRESCAL_RESCAL_BATCH_H
