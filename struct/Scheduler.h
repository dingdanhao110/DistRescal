//
// Created by dhding on 9/25/17.
//

#ifndef DISTRESCAL_BLOCK_H
#define DISTRESCAL_BLOCK_H

#include <../util/Base.h>
#include <mutex>
#include <vector>
#include <algorithm>


using std::vector;
using std::array;

struct Block{
    bool scheduled;
    set<int> samples;

    explicit Block():scheduled(0){}
};

class Scheduler{
private:
    int num_of_blocks;
    int block_size;
    vector<bool> lock_status;
    vector<vector<vector<Block>>> blocks;
    int count;

public:
    Scheduler(int nb,int bs):num_of_blocks(nb),lock_status(nb,0),block_size(bs)
    {
        blocks.resize(nb);
        for(auto& i:blocks){
            i.resize(nb);
            for(auto& j:i){
                j.resize(nb);
            }
        }
    }

    void insert_sample(Sample& sample,int index){
        //TODO:insert sample to corresponding blocks
        set<int> entities;
        entities.insert(entity2block(sample.n_obj));
        entities.insert(entity2block(sample.p_obj));
        entities.insert(entity2block(sample.p_sub));
        entities.insert(entity2block(sample.n_sub));
        vector<int> tmp(entities.begin(),entities.end());
        switch (entities.size()){
            case 2:
                blocks[tmp[0]][tmp[1]][tmp[1]].samples.insert(index);
                break;
            case 3:
                blocks[tmp[0]][tmp[1]][tmp[2]].samples.insert(index);
                break;
            default:
                cerr<<"Shall not reach here!\n";
                exit(-1);
        }
    }

    void report_finish(array<int,3>& block){
        //TODO:release lock for blocks
        for(int i:block){
            lock_status[i]=0;
        }
    }

    void schedule_next(array<int,3>& block,set<int>& samples){
        //TODO:randomly select 3 segment of entities

        //TODO:lock entities

        //TODO:mark blocks as scheduled, and count

        //TODO:if samples.empty(),reschedule blocks
    }

    bool finished(){
        if(count<num_of_blocks*num_of_blocks*num_of_blocks)
            return false;
        else
            return true;
    }

    int entity2block(int entity){
        return entity/block_size;
    }

    pair<int,int> block2entity(int block_id){
        return make_pair(block_id*block_size,(block_id+1)*block_size);
    };
};


#endif //DISTRESCAL_BLOCK_H
