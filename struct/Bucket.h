//
// Created by dhding on 9/21/17.
//

#ifndef DISTRESCAL_BUCKET_H
#define DISTRESCAL_BUCKET_H

#include "../util/Base.h"
#include "../struct/Sample.h"

class Bucket {
private:
    set<int> entities;
    list<Sample> samples;

public:
    bool is_intersect(Sample sample) {
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

    Bucket() {}

    ~Bucket() {}

};

class Bucket_assigner {
private:
    int num_of_bucket;
    int free_count = 0;
    vector<Bucket> buckets;
public:
    explicit Bucket_assigner(int n) : buckets(n), num_of_bucket(n) {}

    void assign(Sample sample) {
        //TODO..Fix balance issue..
        int to_insert = -1;
        for (int i = 0; i < buckets.size(); ++i) {
            if (to_insert < 0 ||
                buckets[to_insert].entity_count() > buckets[i].entity_count()) {

                to_insert = i;
                continue;
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
        buckets
        [to_insert].
                insert(sample);

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
        for (auto &bucket:buckets) {
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

};

#endif //DISTRESCAL_BUCKET_H
