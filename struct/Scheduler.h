//
// Created by dhding on 9/25/17.
//

#ifndef DISTRESCAL_BLOCK_H
#define DISTRESCAL_BLOCK_H

#include "../util/Base.h"

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
        //randomly select 3 segment of entities
        vector<int> candidates;
        for (int i = 0; i < lock_status.size(); ++i) {
            if (!lock_status[i]) {
                candidates.push_back(i);
            }
        }
        //TODO: check whether randomness is needed..
//        std::random_shuffle(candidates.begin(), candidates.end());
        if (candidates.size() >= 3) {

            //lock entities
            int a = candidates[0];
            int b = candidates[1];
            int c = candidates[2];
            if (!blocks[a][b][c].scheduled) {
                block[0] = a;
                block[1] = b;
                block[2] = c;
                lock_status[candidates[0]] = 1;
                lock_status[candidates[1]] = 1;
                lock_status[candidates[2]] = 1;
                //mark blocks as scheduled, add to samples, and count
                {
                    array<int, 3> e = {a, b, c};
                    std::sort(e.begin(), e.end());
                    do {
                        add_sample(samples, e[0], e[1], e[2]);
                    } while (std::next_permutation(e.begin(), e.begin() + 3));
                }
                {
                    array<int, 3> e = {a, b, a};
                    std::sort(e.begin(), e.end());
                    do {
                        add_sample(samples, e[0], e[1], e[2]);
                    } while (std::next_permutation(e.begin(), e.begin() + 3));
                }
                {
                    array<int, 3> e = {a, c, a};
                    std::sort(e.begin(), e.end());
                    do {
                        add_sample(samples, e[0], e[1], e[2]);
                    } while (std::next_permutation(e.begin(), e.begin() + 3));
                }
                {
                    array<int, 3> e = {a, b, b};
                    std::sort(e.begin(), e.end());
                    do {
                        add_sample(samples, e[0], e[1], e[2]);
                    } while (std::next_permutation(e.begin(), e.begin() + 3));
                }
                {
                    array<int, 3> e = {c, b, b};
                    std::sort(e.begin(), e.end());
                    do {
                        add_sample(samples, e[0], e[1], e[2]);
                    } while (std::next_permutation(e.begin(), e.begin() + 3));
                }
                {
                    array<int, 3> e = {a, c, c};
                    std::sort(e.begin(), e.end());
                    do {
                        add_sample(samples, e[0], e[1], e[2]);
                    } while (std::next_permutation(e.begin(), e.begin() + 3));
                }
                {
                    array<int, 3> e = {b, c, c};
                    std::sort(e.begin(), e.end());
                    do {
                        add_sample(samples, e[0], e[1], e[2]);
                    } while (std::next_permutation(e.begin(), e.begin() + 3));
                }

                add_sample(samples, a, a, a);
                add_sample(samples, b, b, b);
                add_sample(samples, c, c, c);

                if (!samples.size()) {
                    this->report_finish(block);
                } else {
                    return;
                }
            }


            while (std::next_permutation(candidates.begin(), candidates.end())) {
                //needs to fetch another block..By permutation..
                int a = candidates[0];
                int b = candidates[1];
                int c = candidates[2];
                if (blocks[a][b][c].scheduled) { continue; }
                block[0] = a;
                block[1] = b;
                block[2] = c;
                lock_status[candidates[0]] = 1;
                lock_status[candidates[1]] = 1;
                lock_status[candidates[2]] = 1;
                //mark blocks as scheduled, add to samples, and count
                {
                    array<int, 3> e = {a, b, c};
                    std::sort(e.begin(), e.end());
                    do {
                        add_sample(samples, e[0], e[1], e[2]);
                    } while (std::next_permutation(e.begin(), e.begin() + 3));
                }
                {
                    array<int, 3> e = {a, b, a};
                    std::sort(e.begin(), e.end());
                    do {
                        add_sample(samples, e[0], e[1], e[2]);
                    } while (std::next_permutation(e.begin(), e.begin() + 3));
                }
                {
                    array<int, 3> e = {a, c, a};
                    std::sort(e.begin(), e.end());
                    do {
                        add_sample(samples, e[0], e[1], e[2]);
                    } while (std::next_permutation(e.begin(), e.begin() + 3));
                }
                {
                    array<int, 3> e = {a, b, b};
                    std::sort(e.begin(), e.end());
                    do {
                        add_sample(samples, e[0], e[1], e[2]);
                    } while (std::next_permutation(e.begin(), e.begin() + 3));
                }
                {
                    array<int, 3> e = {c, b, b};
                    std::sort(e.begin(), e.end());
                    do {
                        add_sample(samples, e[0], e[1], e[2]);
                    } while (std::next_permutation(e.begin(), e.begin() + 3));
                }
                {
                    array<int, 3> e = {a, c, c};
                    std::sort(e.begin(), e.end());
                    do {
                        add_sample(samples, e[0], e[1], e[2]);
                    } while (std::next_permutation(e.begin(), e.begin() + 3));
                }
                {
                    array<int, 3> e = {b, c, c};
                    std::sort(e.begin(), e.end());
                    do {
                        add_sample(samples, e[0], e[1], e[2]);
                    } while (std::next_permutation(e.begin(), e.begin() + 3));
                }

                add_sample(samples, a, a, a);
                add_sample(samples, b, b, b);
                add_sample(samples, c, c, c);
                if (!samples.size()) {
                    this->report_finish(block);
                } else {
                    return;
                }
            }

        }
        if (candidates.size() == 2) {
            int a = candidates[0];
            int b = candidates[1];
            block[0] = a;
            block[1] = b;
            block[2] = -1;
            lock_status[candidates[0]] = 1;
            lock_status[candidates[1]] = 1;
            {
                array<int, 3> e = {a, b, b};
                std::sort(e.begin(), e.end());
                do {
                    add_sample(samples, e[0], e[1], e[2]);
                } while (std::next_permutation(e.begin(), e.begin() + 3));
            }
            {
                array<int, 3> e = {a, a, b};
                std::sort(e.begin(), e.end());
                do {
                    add_sample(samples, e[0], e[1], e[2]);
                } while (std::next_permutation(e.begin(), e.begin() + 3));
            }
            add_sample(samples, a, a, a);
            add_sample(samples, b, b, b);
            if (!samples.size()) {
                this->report_finish(block);
            } else {
                return;
            }
        }
        if (candidates.size() == 1) {
            int a = candidates[0];
            block[0] = a;
            block[1] = -1;
            block[2] = -1;
            lock_status[candidates[0]] = 1;
            add_sample(samples, a, a, a);
            if (!samples.size()) {
                this->report_finish(block);
            } else {
                return;
            }
        }
        if (candidates.size() == 0) {
            //should never go here
            block[0] = block[1] = block[2] = -1;
            return;
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
