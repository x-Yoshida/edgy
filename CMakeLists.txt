cmake_minimum_required(VERSION 3.10)
project(edgy LANGUAGES CXX CUDA)


find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(edgy_omp edgy_omp.cpp)

if(OpenMP_CXX_FOUND)
    target_link_libraries(edgy_omp
     PUBLIC OpenMP::OpenMP_CXX
     ${OpenCV_LIBS}
     )
endif()


if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

add_executable(edgy_cuda edgy_cuda.cu)

target_link_libraries(edgy_cuda ${OpenCV_LIBS})


add_executable(edgy_seq edgy_seq.cpp)

target_link_libraries(edgy_seq ${OpenCV_LIBS})