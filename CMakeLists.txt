cmake_minimum_required(VERSION 3.19)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE OR NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Debug' as none was specified.")
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

if(CMAKE_BUILD_TYPE)
  set(TRIAL_OUTDIR_END_SUFFIX ${CMAKE_BUILD_TYPE})
  set(TRIAL_OUTDIR_SUFFIX ${TRIAL_OUTDIR_END_SUFFIX})
else()
  message(FATAL_ERROR "Build type unspecified!")
endif()

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/BuildUtilities.cmake")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(COMMON_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/common")

include_directories("$ENV{INTELOCLSDKROOT}/include")
include_directories(${COMMON_INCLUDE_DIR})

if(${CMAKE_GENERATOR_PLATFORM} STREQUAL "x86")
  link_directories("$ENV{INTELOCLSDKROOT}/lib/x86")
elseif(${CMAKE_GENERATOR_PLATFORM} STREQUAL "x64")
  link_directories("$ENV{INTELOCLSDKROOT}/lib/x64")
else()
  message(ERROR "Unknown OpenCL target generator platform!")
endif()

project(OpenGL-Trial VERSION 0.1.0)

# Set target OpenCL version to 2.0
add_compile_options("/DCL_TARGET_OPENCL_VERSION=200")

add_subdirectory(common)
add_subdirectory(hello_world)
add_subdirectory(gpu_profile)
add_subdirectory(matrix)
add_subdirectory(pixels_histogram)
add_subdirectory(vector_dot)
