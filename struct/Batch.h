//
// Created by dhding on 9/30/17.
//

#ifndef DISTRESCAL_BATCH_H
#define DISTRESCAL_BATCH_H

#include "../util/Base.h"
#include "../struct/Sample.h"

class Bucket {
private:
    set<int> entities;
    list<Sample> samples;

public:
    bool is_intersect(const Sample& sample)const {
        if (entities.find(sample.n_obj) != entities.end()) {
            //found
            return true;
        }
        if (entities.find(sample.n_sub) != entities.end()) {
            //found
            return true;
        }
        if (entities.find(sample.p_obj) != entities.end()) {
            //found
            return true;
        }
        if (entities.find(sample.p_sub) != entities.end()) {
            //found
            return true;
        }
        return false;
    }

    void insert(Sample sample) {
        entities.insert(sample.p_obj);
        entities.insert(sample.n_obj);
        entities.insert(sample.p_sub);
        entities.insert(sample.n_sub);
        samples.push_back(sample);
    }

    bool operator<(Bucket b2) {
        return this->samples.size() < b2.samples.size();
    }

    int size() const {
        return this->samples.size();
    }

    int entity_count() const { return entities.size(); }

    const set<int> &get_entities() const { return entities; }
    const list<Sample>& get_samples()const{return samples;}

    list<Sample> release_sample(){
        list<Sample> tmp=std::move(samples);
        entities.clear();
        samples=list<Sample>();
        return std::move(tmp);
    }

    void clear(){
        entities.clear();
        samples.clear();
    }

    Bucket() {}

    ~Bucket() {}

};

class Batch_assigner {
private:
    int num_of_bucket;
    int free_count = 0;
    vector<Bucket> buckets;// buckets[n] is for unassigned samples.
public:
    explicit Batch_assigner(int n) : buckets(n+1), num_of_bucket(n+1) {}

    void assign(Sample sample) {
        //TODO..Fix balance issue..
        int to_insert = -1;
        for (int i = 0; i < buckets.size()-1; ++i) {
            if (buckets[i].is_intersect(sample)) {
                if(to_insert<0) {
                    to_insert = i;
                }
                else{
                    //can be assign to two buckets at the same time;
                    buckets[buckets.size()-1].insert(sample);
                    return;
                }
            }
        }
        if (to_insert >= 0) {
            buckets[to_insert].insert(sample);
            return;
        }

        ++free_count;
        int min_size = std::numeric_limits<int>::max();
        to_insert = -1;
        for (
                int i = 0;
                i < buckets.

                        size();

                ++i) {
            //if (buckets[i].size() < min_size) {
            if (to_insert < 0 ||
                buckets[to_insert].entity_count() > buckets[i].entity_count()) {
                min_size = buckets[i].size();
                to_insert = i;
            }
            //}
        }
        buckets[to_insert].insert(sample);
    }

    vector<Bucket>&get_buckets() {
        return buckets;
    }

    int get_num_of_buckets() const {
        return num_of_bucket;
    }

    int get_free_count() const {
        return free_count;
    }

    vector<int> cal_conflicts() const {
        std::multiset<int> entries;
        set<int> unique_entries;
        for (int i=0;i<buckets.size()-1;++i) {
            auto &bucket=buckets[i];
            for (auto entry:bucket.get_entities()) {
                entries.insert(entry);
                unique_entries.insert(entry);
            }
        }
        vector<int> count(5, 0);
        for (auto &item:unique_entries) {
            ++count[entries.count(item)];
        }
        return count;
    }

    inline bool is_finished()const{
        if(buckets[buckets.size()-1].size())
            return false;
        else
            return true;
    }

    void next_batch(){
        list<Sample> to_asssign=buckets[buckets.size()-1].release_sample();
        for(int i=0;i<buckets.size()-1;++i){
            buckets[i].clear();
        }
        for(auto& s:to_asssign){
            assign(s);
        }
    }
};
#endif //DISTRESCAL_BATCH_H
