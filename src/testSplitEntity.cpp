//
// Created by dhding on 1/18/18.
//

#include "../util/Calculator.h"
#include "../util/Base.h"
#include "../alg/splitEntity.h"

int main(int argc, char **argv) {

    std::vector<double> samples;
    samples.push_back(1.0);
    samples.push_back(2.0);
    samples.push_back(3.0);
    samples.push_back(4.0);
    samples.push_back(5.0);
    samples.push_back(6.0);
    samples.push_back(7.0);

    double variance = 0;
    double sum = 0;
    double sum_square=0;
    int size=samples.size();

    for(auto element:samples){
        sum+=element;
        sum_square+=element*element;
    }

    double t=samples[0];
    for (int i = 1; i < samples.size(); i++)
    {
        t += samples[i];
        double diff = ((i + 1) * samples[i]) - t;
        variance += (diff * diff) / ((i + 1.0) *i);
    }
    cout<< variance / (samples.size()-1)<<endl;

    double base=variance / (samples.size()-1);
    double last = variance / (samples.size()-1);
    cout<<(sum_square-sum*sum/size)/(size-1)<<endl;
    for(int i=0;i<samples.size()-1;++i){
        sum-=samples[i];
        sum_square-=samples[i]*samples[i];
        --size;
        if(size>1) {
            cout << (sum_square - sum * sum / size) / (size - 1) << " ";
            cout << (sum_square - sum * sum / size) / (size - 1) / base << " ";
            cout << (sum_square - sum * sum / size) / (size - 1) / last << endl;
            last = (sum_square - sum * sum / size) / (size - 1);
        }
    }



    return 0;
}