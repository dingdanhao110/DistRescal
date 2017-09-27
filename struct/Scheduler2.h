//
// Created by dhding on 9/27/17.
//

#ifndef DISTRESCAL_SCHEDULER2_H
#define DISTRESCAL_SCHEDULER2_H

#include "../util/Base.h"
#include "Sample.h"

struct Block {
    bool scheduled;
    set<int> samples;

    explicit Block() : scheduled(0) {}
};

class Scheduler2 {
private:
    int num_of_blocks;
    int block_size;
    vector<bool> lock_status;
    vector<Block> blocks;

    inline int block2id(int a, int b, int c) { return a * num_of_blocks * num_of_blocks + b * num_of_blocks + c; }

    inline void id2block(int id, int &a, int &b, int &c) {
        c = id % num_of_blocks;
        b = (id / num_of_blocks) % num_of_blocks;
        a = ((id / num_of_blocks) / num_of_blocks);
    }

public:
    unordered_set<int> pending_blocks;

    Scheduler2(int nb, int bs) : num_of_blocks(nb), lock_status(nb, 0), block_size(bs),
                                 blocks(num_of_blocks * num_of_blocks * num_of_blocks) {
    }

    void insert_sample(Sample &sample, int index) {
        //TODO:insert sample to corresponding blocks
        set<int> entities;
        entities.insert(entity2block(sample.n_obj));
        entities.insert(entity2block(sample.p_obj));
        entities.insert(entity2block(sample.p_sub));
        entities.insert(entity2block(sample.n_sub));
        vector<int> tmp(entities.begin(), entities.end());
        switch (entities.size()) {
            case 1:
                blocks[block2id(tmp[0], tmp[0], tmp[0])].samples.insert(index);
                pending_blocks.insert(block2id(tmp[0], tmp[0], tmp[0]));
                break;
            case 2:
                blocks[block2id(tmp[0], tmp[1], tmp[1])].samples.insert(index);
                pending_blocks.insert(block2id(tmp[0], tmp[1], tmp[1]));
                break;
            case 3:
                blocks[block2id(tmp[0], tmp[1], tmp[2])].samples.insert(index);
                pending_blocks.insert(block2id(tmp[0], tmp[1], tmp[2]));
                break;
            default:
                cerr << "Entities: " << sample.n_obj << " " << sample.p_obj << " " << sample.p_sub << " "
                     << sample.n_sub << endl;
                cerr << "Shall not reach here!\n";
                exit(-1);
        }
    }

    inline void report_finish(array<int, 3> &block) {
        //TODO:release lock for blocks
        for (int i:block) {
            if (i >= 0)
                lock_status[i] = 0;
        }
    }

    void schedule_next(array<int, 3> &block, set<int> &samples) {
        //TODO: Fix the scheduler!!!!!!!
        samples.clear();
        set<int> l;
        //lock entities
        //TODO: Draw block from set:pending
        for (int id:pending_blocks) {
            int a, b, c;
            id2block(id, a, b, c);
            if (lock_status[a] || lock_status[b] || lock_status[c])continue;
            //if (blocks[block2id(a, b, c)].scheduled) { continue; }
            set<int> tmp(l);
            tmp.insert(a);
            tmp.insert(b);
            tmp.insert(c);
            if (tmp.size() > 3) { continue; }
            l = std::move(tmp);

            if (pending_blocks.find(block2id(a, b, c)) != pending_blocks.end())
                pending_blocks.erase(block2id(a, b, c));

        }

        if (samples.size()) {
            vector<int> t(l.begin(), l.end());
            switch (t.size()) {
                case 3:
                    block[2] = t[2];
                    lock_status[t[2]] = 1;
                case 2:
                    block[1] = t[1];
                    lock_status[t[1]] = 1;
                case 1:
                    block[0] = t[0];
                    lock_status[t[0]] = 1;
                    break;
                default:
                    cerr << "Should not come here!\n";
                    exit(-1);
            }
            return;
        }
        //cerr<<"One empty draw\n";
    }


    bool finished() const {
        if (pending_blocks.size())
            return false;
        else
            return true;
    }

    inline void add_sample(set<int> &samples, int a, int b, int c) {

        for (auto &sample:blocks[block2id(a, b, c)].samples)
            samples.insert(sample);

    }

    int entity2block(int entity) {
        return entity / block_size;
    }

};

#endif //DISTRESCAL_SCHEDULER2_H
