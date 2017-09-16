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
    alg/Optimizer.h)

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
    struct/Tuple.h)

set(threadpool
    extern/boost/threadpool.hpp)

add_library(lib SHARED ${lib_HDRS} ${lib_SRCS} ${threadpool})
set_target_properties(lib PROPERTIES LINKER_LANGUAGE CXX)

add_library(DistRESCAL SHARED ${alg} ${util} ${struct} ${threadpool})
set_target_properties(DistRESCAL PROPERTIES LINKER_LANGUAGE CXX)
link_libraries(DistRESCAL ${Boost_LIBRARIES} ${MKL_LIBRARIES})

add_executable(testThreadPool src/testThreadPool.cpp)
add_executable(testMatrixProduct src/testMatrixProduct.cpp)

add_executable(runRESCAL_NAIVE src/runRESCAL_NAIVE.cpp)
add_executable(runRESCAL_LOCK src/runRESCAL_LOCK.cpp)