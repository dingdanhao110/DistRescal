#ifndef BASEUTIL_H
#define BASEUTIL_H

#include <iostream>
#include <fstream>
#include <numeric>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <list>
#include <functional>
#include <thread>
#include <string>
#include <mutex>
#include <chrono>
#include <mkl.h>
#include <mkl_vml.h>
#include <array>
#include <cmath>

using std::cout;
using std::cerr;
using std::endl;
using std::stringstream;
using std::mutex;
using std::string;
using std::vector;
using std::set;
using std::list;
using std::make_pair;
using std::pair;
using std::map;
using std::unordered_map;
using std::unordered_set;
using std::to_string;
using std::array;

////////////////// Boost //////////////////////

#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>
#include <boost/format.hpp>
#include <boost/filesystem.hpp>

namespace po = boost::program_options;

#include "../extern/boost/threadpool.hpp"
using namespace boost::threadpool;

////////////////// Boost //////////////////////

////////////////// Types //////////////////////
typedef unsigned long long ull;

#define use_double

#ifdef use_double
    typedef double value_type;
    #define cblas_xgemm cblas_dgemm
#else
    typedef float value_type;
    #define cblas_xgemm cblas_sgemm
#endif

////////////////// Types //////////////////////

const value_type min_not_zero_value = 1e-7;

//#define detailed_eval

#endif //BASEUTIL_H
