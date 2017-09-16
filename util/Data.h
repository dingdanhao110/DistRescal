#ifndef DATA_H
#define DATA_H

#include "Base.h"
#include "../struct/Tuple.h"
#include "../struct/Triple.h"
#include "FileUtil.h"

using namespace FileUtil;

class Data {
public:

    int num_of_entity; // num of entity
    int num_of_relation; // num of relation
    int num_of_training_triples;
    int num_of_testing_triples;
    int num_of_validation_triples;

    vector<Triple<string> > training_triple_strs;
    vector<Triple<int> > training_triples;
    vector<Triple<string> > testing_triple_strs;
    vector<Triple<string> > valiation_triple_strs;

    unordered_map<string, int> entity_encoder;
    unordered_map<string, int> relation_encoder;
    unordered_map<int, string> entity_decoder;
    unordered_map<int, string> relation_decoder;

    unordered_map<int, vector<Tuple<int> > > train_rel2tuples;
    unordered_map<int, vector<Tuple<int> > > test_rel2tuples;
    unordered_map<int, vector<Tuple<int> > > valid_rel2tuples;

    map<pair<int, int>, vector<int> > trainSubRel2Obj;
    map<pair<int, int>, vector<int> > trainObjRel2Sub;

    map<pair<int, int>, vector<int> > testSubRel2Obj;
    map<pair<int, int>, vector<int> > testObjRel2Sub;

    map<pair<int, int>, vector<int> > validSubRel2Obj;
    map<pair<int, int>, vector<int> > validObjRel2Sub;

    string get_info() {

        stringstream ss;
        ss << "num of training triples: " << num_of_training_triples << endl;
        ss << "num of testing triples: " << num_of_testing_triples << endl;
        ss << "num of validation triples: " << num_of_validation_triples << endl;
        ss << "Entity: " << num_of_entity << endl;
        ss << "Relation: " << num_of_relation << endl;

        return ss.str();
    }

    void prepare_data(const string train_data_path, const string valid_data_path, const string test_data_path){
        read_triple_data(train_data_path, training_triple_strs, num_of_training_triples);
        read_triple_data(test_data_path, testing_triple_strs, num_of_testing_triples);
        read_triple_data(valid_data_path, valiation_triple_strs, num_of_validation_triples);

        encode_triples();
    }

    // check whether faked triple already exists in training data
    bool faked_tuple_exist_train(const int relation_id, const int subject_id, const int object_id)const {
        auto train_ptr = train_rel2tuples.find(relation_id);
        if (train_ptr != train_rel2tuples.end()) {
            for (Tuple<int> tuple:train_ptr->second) {
                if ((tuple.subject == subject_id) && (tuple.object == object_id)) {
                    return true;
                }
            }
        }

        return false;
    }

    // check whether faked triple already exists in training data
    bool faked_tuple_exist_train(const int relation_id, const Tuple<int> &faked_tuple) {
        auto train_ptr = train_rel2tuples.find(relation_id);
        if (train_ptr != train_rel2tuples.end()) {
            for (Tuple<int> tuple:train_ptr->second) {
                if (tuple == faked_tuple) {
                    return true;
                }
            }
        }

        return false;
    }

    bool faked_tuple_exist_test(const int relation_id, const Tuple<int> &faked_tuple) {
        auto train_ptr = train_rel2tuples.find(relation_id);
        if (train_ptr != train_rel2tuples.end()) {
            for (Tuple<int> tuple:train_ptr->second) {
                if (tuple == faked_tuple) {
                    return true;
                }
            }
        }

        auto valid_ptr = valid_rel2tuples.find(relation_id);
        if (valid_ptr != valid_rel2tuples.end()) {
            for (Tuple<int> tuple:valid_ptr->second) {
                if (tuple == faked_tuple) {
                    return true;
                }
            }
        }

        auto test_ptr = test_rel2tuples.find(relation_id);
        if (test_ptr != test_rel2tuples.end()) {
            for (Tuple<int> tuple:test_ptr->second) {
                if (tuple == faked_tuple) {
                    return true;
                }
            }
        }

        return false;
    }

    bool faked_s_tuple_exist(const int subject_id, const int relation_id, const int object_id) {
        pair<int, int> key = make_pair(object_id, relation_id);

        auto train_ptr = trainObjRel2Sub.find(key);
        if (train_ptr != trainObjRel2Sub.end()) {
            for (int real_subject_id:train_ptr->second) {
                if (real_subject_id == subject_id) {
                    return true;
                }
            }
        }

        auto valid_ptr = validObjRel2Sub.find(key);
        if (valid_ptr != validObjRel2Sub.end()) {
            for (int real_subject_id:valid_ptr->second) {
                if (real_subject_id == subject_id) {
                    return true;
                }
            }
        }

        auto test_ptr = testObjRel2Sub.find(key);
        if (test_ptr != testObjRel2Sub.end()) {
            for (int real_subject_id:test_ptr->second) {
                if (real_subject_id == subject_id) {
                    return true;
                }
            }
        }

        return false;
    }

    bool faked_o_tuple_exist(const int subject_id, const int relation_id, const int object_id) {

        pair<int, int> key = make_pair(subject_id, relation_id);

        auto train_ptr = trainSubRel2Obj.find(key);
        if (train_ptr != trainSubRel2Obj.end()) {
            for (int real_object_id:train_ptr->second) {
                if (real_object_id == object_id) {
                    return true;
                }
            }
        }

        auto valid_ptr = validSubRel2Obj.find(key);
        if (valid_ptr != validSubRel2Obj.end()) {
            for (int real_object_id:valid_ptr->second) {
                if (real_object_id == object_id) {
                    return true;
                }
            }
        }

        auto test_ptr = testSubRel2Obj.find(key);
        if (test_ptr != testSubRel2Obj.end()) {
            for (int real_object_id:test_ptr->second) {
                if (real_object_id == object_id) {
                    return true;
                }
            }
        }

        return false;
    }

    void output_decoder(string output_folder) {

        boost::filesystem::create_directories(output_folder);

        std::ofstream output(output_folder + "/entity_decoder.dat");

        for (int i = 0; i < entity_decoder.size(); i++) {
            output << i << " " << entity_decoder.find(i)->second << endl;
        }

        output.close();

        output.open(output_folder + "/relation_decoder.dat");
        for (int i = 0; i < relation_decoder.size(); i++) {
            output << i << " " << relation_decoder.find(i)->second << endl;
        }
        output.close();
    }

private:

    void encode_triples() {

        int relation_id, subject_id, object_id;
        pair<int, int> sub_rel_pair;
        pair<int, int> obj_rel_pair;

        for (Triple<string> triple_str : training_triple_strs) {

            if (relation_encoder.find(triple_str.relation) == relation_encoder.end()) {
                relation_id = relation_encoder.size();
                relation_encoder[triple_str.relation] = relation_id;
                relation_decoder[relation_id] = triple_str.relation;
            } else {
                relation_id = relation_encoder[triple_str.relation];
            }

            if (entity_encoder.find(triple_str.subject) == entity_encoder.end()) {
                subject_id = entity_encoder.size();
                entity_encoder[triple_str.subject] = subject_id;
                entity_decoder[subject_id] = triple_str.subject;
            } else {
                subject_id = entity_encoder[triple_str.subject];
            }

            if (entity_encoder.find(triple_str.object) == entity_encoder.end()) {
                object_id = entity_encoder.size();
                entity_encoder[triple_str.object] = object_id;
                entity_decoder[object_id] = triple_str.object;
            } else {
                object_id = entity_encoder[triple_str.object];
            }

            if (train_rel2tuples.find(relation_id) == train_rel2tuples.end()) {
                train_rel2tuples[relation_id] = vector<Tuple<int> >();
            }

            train_rel2tuples[relation_id].push_back(Tuple<int>(subject_id, object_id));

            sub_rel_pair = make_pair(subject_id, relation_id);
            obj_rel_pair = make_pair(object_id, relation_id);

            if(trainSubRel2Obj.find(sub_rel_pair) == trainSubRel2Obj.end()){
                trainSubRel2Obj[sub_rel_pair] = vector<int>();
            }
            trainSubRel2Obj[sub_rel_pair].push_back(object_id);

            if(trainObjRel2Sub.find(obj_rel_pair) == trainObjRel2Sub.end()){
                trainObjRel2Sub[obj_rel_pair] = vector<int>();
            }
            trainObjRel2Sub[obj_rel_pair].push_back(subject_id);

            training_triples.push_back(Triple<int>(subject_id, relation_id, object_id));
        }

        for (Triple<string> triple_str: testing_triple_strs) {
            if (relation_encoder.find(triple_str.relation) == relation_encoder.end() ||
                entity_encoder.find(triple_str.subject) == entity_encoder.end() ||
                entity_encoder.find(triple_str.object) == entity_encoder.end()) {
                continue;
            }
            relation_id = relation_encoder[triple_str.relation];
            subject_id = entity_encoder[triple_str.subject];
            object_id = entity_encoder[triple_str.object];
            test_rel2tuples[relation_id].push_back(Tuple<int>(subject_id, object_id));

            sub_rel_pair = make_pair(subject_id, relation_id);
            obj_rel_pair = make_pair(object_id, relation_id);

            if (testSubRel2Obj.find(sub_rel_pair) == testSubRel2Obj.end()) {
                testSubRel2Obj[sub_rel_pair] = vector<int>();
            }
            testSubRel2Obj[sub_rel_pair].push_back(object_id);

            if (testObjRel2Sub.find(obj_rel_pair) == testObjRel2Sub.end()) {
                testObjRel2Sub[obj_rel_pair] = vector<int>();
            }
            testObjRel2Sub[obj_rel_pair].push_back(subject_id);
        }

        for (Triple<string> triple_str: valiation_triple_strs) {
            if (relation_encoder.find(triple_str.relation) == relation_encoder.end() ||
                entity_encoder.find(triple_str.subject) == entity_encoder.end() ||
                entity_encoder.find(triple_str.object) == entity_encoder.end()) {
                continue;
            }
            relation_id = relation_encoder[triple_str.relation];
            subject_id = entity_encoder[triple_str.subject];
            object_id = entity_encoder[triple_str.object];
            valid_rel2tuples[relation_id].push_back(Tuple<int>(subject_id, object_id));

            sub_rel_pair = make_pair(subject_id, relation_id);
            obj_rel_pair = make_pair(object_id, relation_id);

            if(validSubRel2Obj.find(sub_rel_pair) == validSubRel2Obj.end()){
                validSubRel2Obj[sub_rel_pair] = vector<int>();
            }
            validSubRel2Obj[sub_rel_pair].push_back(object_id);

            if(validObjRel2Sub.find(obj_rel_pair) == validObjRel2Sub.end()){
                validObjRel2Sub[obj_rel_pair] = vector<int>();
            }
            validObjRel2Sub[obj_rel_pair].push_back(subject_id);
        }

        num_of_entity = entity_decoder.size();
        num_of_relation = relation_decoder.size();
    }
};

#endif //DATA_H