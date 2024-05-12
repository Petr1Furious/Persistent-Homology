cmake_minimum_required(VERSION 3.21)
set(CMAKE_CXX_STANDARD 17)
project(persistent_homology)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# add_compile_options(-fsanitize=thread)
# add_link_options(-fsanitize=thread)

add_custom_target(
    kernel_air
    COMMAND xcrun -sdk macosx metal -c ${CMAKE_CURRENT_SOURCE_DIR}/include/compute.metal -o ${CMAKE_CURRENT_BINARY_DIR}/compute.air
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/include/compute.metal
    BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/compute.air
)
add_custom_target(
    kernel_metallib
    COMMAND xcrun -sdk macosx metallib ${CMAKE_CURRENT_BINARY_DIR}/compute.air -o ${CMAKE_CURRENT_BINARY_DIR}/default.metallib
    DEPENDS kernel_air
    BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/default.metallib
)

include_directories(${CMAKE_CURRENT_LIST_DIR}/metal-cpp)
include_directories(${CMAKE_CURRENT_LIST_DIR}/include)

add_library(persistent_homology
    include/MetalSparseMatrix.cpp
    include/ParallelSparseMatrix.cpp
    include/SparseMatrix.cpp
    include/SparseMatrixBase.cpp
)

add_executable(persistent-homology
    cli/main.cpp
)
add_dependencies(persistent-homology kernel_metallib)
target_link_libraries(persistent-homology
    persistent_homology
    "-framework Metal"
    "-framework MetalKit"
    "-framework AppKit"
    "-framework Foundation"
    "-framework QuartzCore"
)
