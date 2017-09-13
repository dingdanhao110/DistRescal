cmake_minimum_required(VERSION 3.8)
project(DistRESCAL)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11 -DNDEBUG -pthread -march=native -ftree-vectorize")


## look for MKL
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/extern/cmake/")
find_package(MKL REQUIRED)
if(MKL_FOUND)
    message(STATUS "MKL found " ${MKL_INCLUDE_DIRS})
    include_directories(${MKL_INCLUDE_DIRS})
    link_directories(${MKL_LIBRARIES})
endif()

## look for TBB
find_package(TBB REQUIRED)
if(TBB_FOUND)
    message(STATUS "TBB found: " ${TBB_INCLUDE_DIRS})
    include_directories(${TBB_INCLUDE_DIRS})
    link_libraries(${TBB_LIBRARIES})
endif(TBB_FOUND)

## look for boost
find_package(Boost COMPONENTS thread program_options system filesystem)
if(NOT ${Boost_FOUND})
    SET(BOOST_ROOT ~/local) # default
    SET(Boost_NO_SYSTEM_PATHS ON) # force to use own build
    find_package(Boost COMPONENTS program_options system filesystem)
endif(NOT ${Boost_FOUND})
if(Boost_FOUND)
    message(STATUS "Boost found")
    message(STATUS ${Boost_LIBRARIES})
    include_directories(${Boost_INCLUDE_DIR})
    link_libraries(${Boost_LIBRARIES})
endif(Boost_FOUND)



## set include directory
include_directories(include)

## define lib library
set(lib_SRCS)
set(lib_HDRS
        include/boost/threadpool.hpp
        include/util/Monitor.h
        include/util/EvaluationUtil.h
        include/util/ThreadUtil.h
        include/util/Data.h
        include/util/Base.h
        include/util/FileUtil.h
        include/util/Monitor.h
        include/util/RandomUtil.h
        include/util/CompareUtil.h
        include/util/Calculator.h
        include/util/Parameter.h
        include/util/LRUtil.h
        include/struct/Triple.h
        include/struct/Tuple.h
        include/struct/Weight.h
        include/struct/Min_Max.h
        include/alg/Optimizer.h
        include/alg/Sampler.h
        include/alg/RESCAL_LOCK.h
        include/alg/RESCAL_NAIVE.h include/alg/RESCAL_RANK.h)


add_library(lib SHARED ${lib_HDRS} ${lib_SRCS})
set_target_properties(lib PROPERTIES LINKER_LANGUAGE CXX)

## executables
link_libraries(lib ${MKL_LIBRARIES} ${TBB_LIBRARIES} ${Boost_LIBRARIES} )

add_executable(testThreadPool src/testThreadPool.cpp)
add_executable(runRESCAL_LOCK src/runRESCAL_LOCK.cpp)
add_executable(runRESCAL_NAIVE src/runRESCAL_NAIVE.cpp)
