cmake_minimum_required(VERSION 3.19)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE OR NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Debug' as none was specified.")
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

if(NOT CMAKE_BUILD_TYPE)
  message(FATAL_ERROR "Build type unspecified!")
endif()

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/BuildUtilities.cmake")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(COMMON_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/common")
SET(THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../ThirdParty")

include_directories(
  "$ENV{INTELOCLSDKROOT}/include"
  ${COMMON_INCLUDE_DIR}
)

# OpenGL staff
SET(GLAD_SOURCE_DIR ${THIRD_PARTY_DIR}/glad-460-core/src/)  # OpenGL staff
include_directories(${THIRD_PARTY_DIR}/glad-460-core/include)

# GLM staff
include_directories(${THIRD_PARTY_DIR}/glm)

# GLFW staff
add_subdirectory(${THIRD_PARTY_DIR}/glfw ${CMAKE_CURRENT_BINARY_DIR}/glfw)
include_directories(${THIRD_PARTY_DIR}/glfw/include)
link_directories(${CMAKE_CURRENT_BINARY_DIR}/glfw/src/${CMAKE_BUILD_TYPE})

enable_language(C CXX)
find_package(OpenCL REQUIRED)
if(NOT OpenCL_FOUND)
  message(ERROR "Can not found OpenCL SDK!")
endif()
include_directories("${OpenCL_INCLUDE_DIRS}")

get_filename_component(OPENCL_LIBRARY_DIR "${OpenCL_LIBRARY}" DIRECTORY)
message(STATUS "OpenCL link lib path: ${OPENCL_LIBRARY_DIR}")
link_directories("${OPENCL_LIBRARY_DIR}")

project(OpenGL-Trial VERSION 0.1.0)

add_subdirectory(common)
add_subdirectory(hello_world)
add_subdirectory(gpu_profile)
add_subdirectory(pixels_histogram)
add_subdirectory(vector_dot)
add_subdirectory(matrix)
add_subdirectory(sparse_matrix)
add_subdirectory(transpose_inplace)
add_subdirectory(heat_transfer)
