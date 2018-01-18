//
// Created by dhding on 1/16/18.
//

#ifndef DISTRESCAL_SHEAP_H
#define DISTRESCAL_SHEAP_H

#include "../util/Base.h"

//<id,count>
struct comparator_lesser_than{
    bool operator() (pair<int,int> a, pair<int,int> b){
        if(a.second>b.second)return false;
        if(a.second<b.second)return true;
        if(a.first<b.first)return false;
        return true;
    }
};

struct comparator_bigger_than{
    bool operator() (pair<int,int> a, pair<int,int> b){
        if(a.second>b.second)return true;
        if(a.second<b.second)return false;
        if(a.first<b.first)return true;
        return false;
    }
};

typedef std::priority_queue<pair<int,int>, vector<pair<int,int>>,comparator_lesser_than> MyHeap;


#endif //DISTRESCAL_SHEAP_H
