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
        read_triple_data_double_escape(test_data_path, testing_triple_strs, num_of_testing_triples);
        read_triple_data_double_escape(valid_data_path, valiation_triple_strs, num_of_validation_triples);

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

        set<string> relation_counter;
        set<string> entity_counter;
        for (Triple<string> triple_str : training_triple_strs) {

            if (relation_counter.find(triple_str.relation) == relation_counter.end()) {
                relation_counter.insert(triple_str.relation);
            }
            if (entity_counter.find(triple_str.subject) == entity_counter.end()) {
                entity_counter.insert(triple_str.subject);
            }
            if (entity_counter.find(triple_str.object) == entity_counter.end()) {
                entity_counter.insert(triple_str.object);
            }
        }

        std::vector<int> rel2newid(relation_counter.size());
        std::iota(std::begin(rel2newid), std::end(rel2newid), 0);
//        std::random_shuffle(rel2newid.begin(), rel2newid.end());
//        std::vector<int> newid2rel(relation_counter.size());
//        for(int i=0;i<relation_counter.size();++i){
//            newid2rel[rel2newid[i]]=i;
//        }

        std::vector<int> entity2newid(entity_counter.size());
        std::iota(std::begin(entity2newid), std::end(entity2newid), 0);
//        std::random_shuffle(entity2newid.begin(), entity2newid.end());
//        std::vector<int> newid2entity(entity_counter.size());
//        for(int i=0;i<entity_counter.size();++i){
//            newid2entity[entity2newid[i]]=i;
//        }

        //Original code starts from here..
        for (Triple<string> triple_str : training_triple_strs) {

            if (relation_encoder.find(triple_str.relation) == relation_encoder.end()) {
                relation_id = rel2newid[relation_encoder.size()];
                relation_encoder[triple_str.relation] = relation_id;
                relation_decoder[relation_id] = triple_str.relation;

            } else {
                relation_id = relation_encoder[triple_str.relation];
            }

            if (entity_encoder.find(triple_str.subject) == entity_encoder.end()) {
                subject_id = entity2newid[entity_encoder.size()];
                entity_encoder[triple_str.subject] = subject_id;
                entity_decoder[subject_id] = triple_str.subject;

            } else {
                subject_id = entity_encoder[triple_str.subject];
            }

            if (entity_encoder.find(triple_str.object) == entity_encoder.end()) {
                object_id = entity2newid[entity_encoder.size()];
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

//        vector<string> s;
//        string training_triples_file = "training_triples.txt";
//
//        for (Triple<int> &t:training_triples) {
//            s.push_back(to_string(t.subject) + "," + to_string(t.relation) + "," + to_string(t.object));
//        }
//
//        ofstream output1(training_triples_file.c_str());
//        sort(s.begin(), s.end());
//        for(auto os:s){
//            output1 << os << endl;
//        }
//
//        output1.close();
//
//        string train_rel2tuples_file = "train_rel2tuples.txt";
//
//        for (auto ptr = train_rel2tuples.begin(); ptr != train_rel2tuples.end(); ptr++) {
//            for (Tuple<int> &t:ptr->second) {
//                s.push_back(to_string(ptr->first) + "," + to_string(t.subject) + "," + to_string(t.object));
//            }
//        }
//
//        ofstream output2(train_rel2tuples_file.c_str());
//        sort(s.begin(), s.end());
//
//        for(auto os:s){
//            output2 << os << endl;
//        }
//        output2.close();
//        s.clear();
//
//        string test_rel2tuples_file = "test_rel2tuples.txt";
//
//        for (auto ptr = test_rel2tuples.begin(); ptr != test_rel2tuples.end(); ptr++) {
//            for (Tuple<int> &t:ptr->second) {
//                s.push_back(to_string(ptr->first) + "," + to_string(t.subject) + "," + to_string(t.object));
//            }
//        }
//
//        ofstream output3(test_rel2tuples_file.c_str());
//        sort(s.begin(), s.end());
//
//        for(auto os:s){
//            output3 << os << endl;
//        }
//        output3.close();
//        s.clear();
//
//        string valid_rel2tuples_file = "valid_rel2tuples.txt";
//
//        for (auto ptr = valid_rel2tuples.begin(); ptr != valid_rel2tuples.end(); ptr++) {
//            for (Tuple<int> &t:ptr->second) {
//                s.push_back(to_string(ptr->first) + "," + to_string(t.subject) + "," + to_string(t.object));
//            }
//        }
//
//        ofstream output4(valid_rel2tuples_file.c_str());
//        sort(s.begin(), s.end());
//
//        for(auto os:s){
//            output4 << os << endl;
//        }
//        output4.close();
//        s.clear();
//
//        string trainSubRel2Obj_file = "trainSubRel2Obj.txt";
//
//        for (auto ptr = trainSubRel2Obj.begin(); ptr != trainSubRel2Obj.end(); ptr++) {
//            for (int &t:ptr->second) {
//                s.push_back(to_string(ptr->first.first) + "," + to_string(ptr->first.second) + "," + to_string(t));
//            }
//        }
//
//        ofstream output5(trainSubRel2Obj_file.c_str());
//        sort(s.begin(), s.end());
//        for(auto os:s){
//            output5 << os << endl;
//        }
//        output5.close();
//        s.clear();
//
//        string trainObjRel2Sub_file = "trainObjRel2Sub.txt";
//
//        for (auto ptr = trainObjRel2Sub.begin(); ptr != trainObjRel2Sub.end(); ptr++) {
//            for (int &t:ptr->second) {
//                s.push_back(to_string(ptr->first.first) + "," + to_string(ptr->first.second) + "," + to_string(t));
//
//            }
//        }
//
//        ofstream output6(trainObjRel2Sub_file.c_str());
//        sort(s.begin(), s.end());
//        for(auto os:s){
//            output6 << os << endl;
//        }
//        output6.close();
//        s.clear();
//
//        string testSubRel2Obj_file = "testSubRel2Obj.txt";
//        for (auto ptr = testSubRel2Obj.begin(); ptr != testSubRel2Obj.end(); ptr++) {
//            for (int &t:ptr->second) {
//                s.push_back(to_string(ptr->first.first) + "," + to_string(ptr->first.second) + "," + to_string(t));
//            }
//        }
//
//        ofstream output7(testSubRel2Obj_file.c_str());
//        sort(s.begin(), s.end());
//        for(auto os:s){
//            output7 << os << endl;
//        }
//        output7.close();
//        s.clear();
//
//        string testObjRel2Sub_file = "testObjRel2Sub.txt";
//        for (auto ptr = testObjRel2Sub.begin(); ptr != testObjRel2Sub.end(); ptr++) {
//            for (int &t:ptr->second) {
//                s.push_back(to_string(ptr->first.first) + "," + to_string(ptr->first.second) + "," + to_string(t));
//            }
//        }
//
//        ofstream output8(testObjRel2Sub_file.c_str());
//        sort(s.begin(), s.end());
//        for(auto os:s){
//            output8 << os << endl;
//        }
//        output8.close();
//        s.clear();
//
//        string validSubRel2Obj_file = "validSubRel2Obj.txt";
//
//        for (auto ptr = validSubRel2Obj.begin(); ptr != validSubRel2Obj.end(); ptr++) {
//            for (int &t:ptr->second) {
//                s.push_back(to_string(ptr->first.first) + "," + to_string(ptr->first.second) + "," + to_string(t));
//            }
//        }
//
//        ofstream output9(validSubRel2Obj_file.c_str());
//        sort(s.begin(), s.end());
//        for(auto os:s){
//            output9 << os << endl;
//        }
//        output9.close();
//        s.clear();
//
//        string validObjRel2Sub_file = "validObjRel2Sub.txt";
//
//        for (auto ptr = validObjRel2Sub.begin(); ptr != validObjRel2Sub.end(); ptr++) {
//            for (int &t:ptr->second) {
//                s.push_back(to_string(ptr->first.first) + "," + to_string(ptr->first.second) + "," + to_string(t));
//            }
//        }
//
//        ofstream output10(validObjRel2Sub_file.c_str());
//        sort(s.begin(), s.end());
//        for(auto os:s){
//            output10 << os << endl;
//        }
//        output10.close();
//        s.clear();
//
//        exit(1);
    }
};

#endif //DATA_H