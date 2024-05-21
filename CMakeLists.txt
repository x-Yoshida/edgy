cmake_minimum_required(VERSION 3.10)
project(edgy_omp LANGUAGES CXX)


find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(edgy_omp edgy.cpp)

if(OpenMP_CXX_FOUND)
    target_link_libraries(edgy_omp
     PUBLIC OpenMP::OpenMP_CXX
     ${OpenCV_LIBS}
     )
endif()