cmake_minimum_required(VERSION 2.8)
project(DistRESCAL)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11 -DNDEBUG -pthread -march=native -ftree-vectorize")

## look for boost
find_package(Boost COMPONENTS thread program_options system filesystem)
if(NOT ${Boost_FOUND})
    SET(BOOST_ROOT ~/local) # default
    SET(Boost_NO_SYSTEM_PATHS ON) # force to use own build
    find_package(Boost COMPONENTS thread program_options system filesystem)
endif(NOT ${Boost_FOUND})
if(Boost_FOUND)
    message(STATUS "Boost found")
    message(STATUS ${Boost_LIBRARIES})
    include_directories(${Boost_INCLUDE_DIR})
    link_libraries(${Boost_LIBRARIES})
endif(Boost_FOUND)

## set include directory
include_directories(include)
include_directories(extern)

## define lib library
set(lib_SRCS)
set(lib_HDRS
   include/util/Monitor.h
   include/util/EvaluationUtil.h
   include/util/Data.h
   include/util/Base.h
   include/util/FileUtil.h
   include/util/Monitor.h
   include/util/RandomUtil.h
   include/util/CompareUtil.h
   include/util/Calculator.h
   include/util/Parameter.h
   include/struct/Triple.h
   include/struct/Tuple.h
   include/alg/Sampler.h
   include/alg/RESCAL_NAIVE.h
   include/alg/MarginBasedOptimizer.h
        include/alg/RESCAL_LOCK.h include/alg/Optimizer.h)

set(threadpool
    extern/boost/threadpool.hpp)

add_library(lib SHARED ${lib_HDRS} ${lib_SRCS} ${threadpool})
set_target_properties(lib PROPERTIES LINKER_LANGUAGE CXX)

add_executable(testThreadPool src/testThreadPool.cpp)
target_link_libraries(testThreadPool lib)
target_link_libraries(testThreadPool ${Boost_LIBRARIES})

add_executable(runRESCAL_NAIVE src/runRESCAL_NAIVE.cpp)
target_link_libraries(runRESCAL_NAIVE lib)
target_link_libraries(runRESCAL_NAIVE ${Boost_LIBRARIES})

add_executable(runRESCAL_LOCK src/runRESCAL_LOCK.cpp)
target_link_libraries(runRESCAL_LOCK lib)
target_link_libraries(runRESCAL_LOCK ${Boost_LIBRARIES})

add_executable(testMatrixProduct src/testMatrixProduct.cpp)
target_link_libraries(testMatrixProduct lib)