//
// Created by dhding on 9/25/17.
//

#ifndef DISTRESCAL_BLOCK_H
#define DISTRESCAL_BLOCK_H

#include "../util/Base.h"
#include "Sample.h"

struct Block {
    bool scheduled;
    set<int> samples;

    explicit Block() : scheduled(0) {}
};

class Scheduler {
private:
    int num_of_blocks;
    int block_size;
    vector<bool> lock_status;
    vector<vector<vector<Block>>> blocks;

public:
    int count;

    Scheduler(int nb, int bs) : num_of_blocks(nb), lock_status(nb, 0), block_size(bs), count(0) {
        blocks.resize(nb);
        for (auto &i:blocks) {
            i.resize(nb);
            for (auto &j:i) {
                j.resize(nb);
            }
        }
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
                blocks[tmp[0]][tmp[0]][tmp[0]].samples.insert(index);
                break;
            case 2:
                blocks[tmp[0]][tmp[1]][tmp[1]].samples.insert(index);
                break;
            case 3:
                blocks[tmp[0]][tmp[1]][tmp[2]].samples.insert(index);
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
        vector<int> candidates;
        for (int i = 0; i < lock_status.size(); ++i) {
            if (!lock_status[i]) {
                candidates.push_back(i);
            }
        }

        if (candidates.size() >= 3) {
            //lock entities
            std::random_shuffle(candidates.begin(),candidates.end());
            do {
                int a = candidates[candidates.size()-1];
                int b = candidates[candidates.size()-2];
                int c = candidates[candidates.size()-3];
                block[0] = a;
                block[1] = b;
                block[2] = c;
                if (blocks[a][b][c].scheduled){continue;}
                lock_status[a] = 1;
                lock_status[b] = 1;
                lock_status[c] = 1;
                array<int,3> e={a,b,c};
                //mark blocks as scheduled, add to samples, and count
                for(int i:e) {
                    for (int j:e) {
                        for (int k:e) {
                            add_sample(samples, i, j, k);
                        }
                    }
                }
                if (!samples.size()) {
                    lock_status[a] = 0;
                    lock_status[b] = 0;
                    lock_status[c] = 0;
                } else {
                    return;
                }
                //cerr<<"One empty draw\n";
            } while (std::next_permutation(candidates.begin(), candidates.end()));

        } else {
            //should never go here
            block[0] = block[1] = block[2] = -1;
            cerr << "too few candidate piece of A\n";
            exit(-1);
        }
    }

    bool finished() const {
        if (count < num_of_blocks * num_of_blocks * num_of_blocks)
            return false;
        else
            return true;
    }

    inline void add_sample(set<int> &samples, int a, int b, int c) {
        if (!blocks[a][b][c].scheduled) {
            ++count;
            blocks[a][b][c].scheduled = 1;
            for (auto &sample:blocks[a][b][c].samples)
                samples.insert(sample);
        }
    }

    int entity2block(int entity) {
        return entity / block_size;
    }
};


#endif //DISTRESCAL_BLOCK_H
