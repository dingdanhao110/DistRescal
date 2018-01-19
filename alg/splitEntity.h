//
// Created by dhding on 1/18/18.
//

#ifndef DISTRESCAL_SPLITENTITY_H
#define DISTRESCAL_SPLITENTITY_H

#include "../util/Base.h"
#include "../struct/SHeap.h"


//

inline int split_entity(const vector<pair<int,int>>& heap_vec,const Parameter& parameter,unordered_set<int>& freq_entities){
    int split=heap_vec.size()-1;
    value_type thres=parameter.threshold_freq;

    int size=heap_vec.size();
    value_type sum = 0;
    value_type sum_square=0;

    for(const auto& pair:heap_vec){
        sum+=pair.second;
        sum_square+=pair.second*pair.second;
    }

    value_type base=(sum_square-sum*sum/size)/(size-1);
    //cout<<(sum_square-sum*sum/size)/(size-1)<<endl;
    for(int i=0;i<heap_vec.size()-1;++i){
        sum-=heap_vec[i].second;
        sum_square-=heap_vec[i].second*heap_vec[i].second;
        --size;
        if(size>1) {
            //cout << (sum_square - sum * sum / size) / (size - 1) << " ";
            //cout << (sum_square - sum * sum / size) / (size - 1) / base << " ";
            freq_entities.insert(heap_vec[i].first);
            if((sum_square - sum * sum / size) / (size - 1) / base<thres){
                split=i;
                break;
            }
        }
    }
    //cout<<endl;
    return split;
}







#endif //DISTRESCAL_SPLITENTITY_H
