cmake_minimum_required(VERSION 3.21)
set(CMAKE_CXX_STANDARD 17)
project(persistent_homology)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# add_compile_options(-fsanitize=address)
# add_link_options(-fsanitize=address)

include_directories(${CMAKE_CURRENT_LIST_DIR}/metal-cpp)

add_custom_target(
    kernel_air
    COMMAND xcrun -sdk macosx metal -c ${CMAKE_CURRENT_SOURCE_DIR}/src/compute.metal -o ${CMAKE_CURRENT_BINARY_DIR}/compute.air
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/compute.metal
    BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/compute.air
)
add_custom_target(
    kernel_metallib
    COMMAND xcrun -sdk macosx metallib ${CMAKE_CURRENT_BINARY_DIR}/compute.air -o ${CMAKE_CURRENT_BINARY_DIR}/default.metallib
    DEPENDS kernel_air
    BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/default.metallib
)

file(GLOB
    files_src
    CONFIGURE_DEPENDS
     "src/*.hpp"
     "src/*.cpp"
)

add_executable(persistent-homology ${files_src})
add_dependencies(persistent-homology kernel_metallib)
target_link_libraries(persistent-homology
    "-framework Metal"
    "-framework MetalKit"
    "-framework AppKit"
    "-framework Foundation"
    "-framework QuartzCore"
)
