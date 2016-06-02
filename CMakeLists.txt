cmake_minimum_required(VERSION 3.5)
project(Sheet_4)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

find_package(OpenCV 3.0 CONFIG REQUIRED HINTS /work/opencv30/)
if (OpenCV_FOUND)
    message(STATUS "Found OpenCV v${OpenCV_VERSION}${OpenCV_VERSION_STATUS}")
    message("opencv include directories: ${OpenCV_INCLUDE_DIRS}")
    message("opencv libs: ${OpenCV_LIBS}")
endif ()

find_package(Eigen3 REQUIRED)
if(Eigen3_FOUND)
    message("Eigen3 found!")
endif()

set(SOURCE_FILES main.cpp MoGDistribution.hh MoGDistribution.h StringHelper.h)
add_executable(Sheet_4 ${SOURCE_FILES})
target_include_directories(Sheet_4 PUBLIC ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})
target_link_libraries(Sheet_4 ${OpenCV_LIBS})
