cmake_minimum_required(VERSION 3.8)
project(my_project)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add precompiler definitions like that:
add_definitions(-DSOME_DEFINITION)

# Define the source file
set(SOURCES "${CMAKE_SOURCE_DIR}/main.cpp")

# Locate Eigen library
set(EIGEN3_INCLUDE_DIR /Users/artemnosenko/Projects/Dep/cpp/eigen-3.4.0)
# Add Eigen include directory
include_directories(${EIGEN3_INCLUDE_DIR})

add_executable(my_app ${SOURCES})

# Add more include directories if needed
#target_include_directories(my_app PUBLIC "${CMAKE_SOURCE_DIR}/include")

# If you have precompiled headers you can add them like this
#target_precompiled_headers(my_app PRIVATE "${CMAKE_SOURCE_DIR}/src/pch.h")

