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

## look for MKL
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/extern/cmake/")
find_package(MKL REQUIRED)
if(MKL_FOUND)
    message(STATUS "MKL found " ${MKL_INCLUDE_DIRS})
    include_directories(${MKL_INCLUDE_DIRS})
    link_directories(${MKL_LIBRARIES})
endif()

set(alg
    alg/Sampler.h
    alg/RESCAL_NAIVE.h
    alg/MarginBasedOptimizer.h
    alg/RESCAL_LOCK.h
    alg/RESCAL_NOLOCK.h
    alg/Optimizer.h alg/RESCAL_NOLOCK.h alg/RESCAL_BLOCK.h alg/RESCAL_BATCH.h alg/RESCAL_POS.h alg/RESCAL_NLL.h alg/RESCAL_ASYNC.h
        alg/RESCAL_PREBATCH.h
        alg/PreBatchAssigner.h alg/splitEntity.h alg/Prebatch_Optimizer.h alg/TransE_Prebatch.h alg/Naive_Optimizer.h alg/TransE_Naive.h alg/TransE_Lock.h alg/TransE_Prebatch_full.h
        alg/PreBatchAssigner_full.h alg/Prebatch_full_Optimizer.h
        alg/RESCAL_Prebatch_full.h alg/PreBatchAssigner_RelOnly.h
        alg/Prebatch_rel_Optimizer.h alg/RESCAL_PREBATCH_REL.h alg/H2_Assigner.h alg/H2_Optimizer.h alg/RESCAL_H2.h alg/Margin_multithread.h alg/RESCAL_NAIVE_mt.h)

set(util
    util/Monitor.h
    util/EvaluationUtil.h
    util/Data.h
    util/Base.h
    util/FileUtil.h
    util/Monitor.h
    util/RandomUtil.h
    util/CompareUtil.h
    util/Calculator.h
    util/Parameter.h)

set(struct
    struct/Triple.h
    struct/Tuple.h struct/Sample.h struct/Bucket.h struct/Scheduler.h struct/Scheduler2.h struct/Batch.h struct/Samples.h struct/SHeap.h alg/Naive_Optimizer.h)

set(threadpool
    extern/boost/threadpool.hpp)

add_library(lib SHARED ${lib_HDRS} ${lib_SRCS} ${threadpool})
set_target_properties(lib PROPERTIES LINKER_LANGUAGE CXX)

add_library(DistRESCAL SHARED ${alg} ${util} ${struct} ${threadpool})
set_target_properties(DistRESCAL PROPERTIES LINKER_LANGUAGE CXX)
link_libraries(DistRESCAL ${Boost_LIBRARIES} ${MKL_LIBRARIES})

add_executable(testThreadPool src/testThreadPool.cpp)
add_executable(testMatrixProduct src/testMatrixProduct.cpp)
add_executable(testScheduler src/testScheduler.cpp)

add_executable(runRESCAL_NAIVE src/runRESCAL_NAIVE.cpp)
add_executable(runRESCAL_LOCK src/runRESCAL_LOCK.cpp)
add_executable(runRESCAL_NOLOCK src/runRESCAL_NOLOCK.cpp)
add_executable(runRESCAL_BLOCK src/runRESCAL_BLOCK.cpp)
add_executable(runRESCAL_BATCH src/runRESCAL_BATCH.cpp)
add_executable(runRESCAL_POS src/runRESCAL_POS.cpp)
add_executable(runRESCAL_NLL src/runRESCAL_NLL.cpp)
add_executable(runRESCAL_ASYNC src/runRESCAL_ASYNC.cpp)
add_executable(runRESCAL_PREBATCH src/runRESCAL_PREBATCH.cpp)
add_executable(runRESCAL_PREBATCH_FULL src/runRESCAL_PREBATCH_FULL.cpp)
add_executable(runRESCAL_PREBATCH_REL src/runRESCAL_PREBATCH_REL.cpp)
add_executable(runRESCAL_H2 src/runRESCAL_H2.cpp)
add_executable(runRESCAL_NAIVE_mt src/runRESCAL_NAIVE_mt.cpp)

add_executable(runTRANSE_PREBATCH src/runTRANSE_PREBATCH.cpp)
add_executable(runTRANSE_PREBATCH_FULL src/runTRANSE_PREBATCH_full.cpp)
add_executable(runTRANSE_PREBATCH_REL src/runTRANSE_PREBATCH_rel.cpp)
add_executable(runTRANSE_NAIVE src/runTRANSE_NAIVE.cpp)
add_executable(runTRANSE_LOCK src/runTRANSE_LOCK.cpp)
add_executable(testSplitEntity src/testSplitEntity.cpp)

add_executable(runPRINT_STAT src/runPrintStatistics.cpp)