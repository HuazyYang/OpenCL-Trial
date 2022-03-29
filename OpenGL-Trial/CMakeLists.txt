cmake_minimum_required(VERSION 3.0.0)
project(HelloWorld VERSION 0.1.0)

if(WIN32 AND ${CMAKE_SIZEOF_VOID_P} EQUAL 8)
  list(APPEND CMAKE_PREFIX_PATH
    "$ENV{VCPKG_ROOT}/installed/x64-windows-static-md"
    "$ENV{VCPKG_ROOT}/installed/x64/windows-static-md/Debug"
    )
endif()

message("cmake prefix path: ${CMAKE_PREFIX_PATH}")

include(BuildUtilities.cmake)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE OR NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Debug' as none was specified.")
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../ThirdParty")

cmake_policy(SET CMP0012 NEW)
find_package(assimp REQUIRED)
if(NOT assimp_FOUND)
message(FATAL_ERROR "Could not find assimp")
else()
message(STATUS "assimp library: ${ASSIMP_INCLUDE_DIRS}")
endif()

# glfw
find_package(glfw3 CONFIG REQUIRED)

# stb
find_path(STB_INCLUDE_DIRS "stb.h")
include_directories(${STB_INCLUDE_DIRS})

#glm
include_directories(${THIRD_PARTY_DIR}/glm)

# glad
set(GLAD_SOURCE_DIR ${THIRD_PARTY_DIR}/glad-460-core)
add_library(glad STATIC
 ${GLAD_SOURCE_DIR}/src/glad.c
)
target_include_directories(glad AFTER PUBLIC ${GLAD_SOURCE_DIR}/include)
include_directories(${GLAD_SOURCE_DIR}/include)

# common
add_subdirectory(common)
include_directories(common)

if(MSVC)
  add_link_options("/NODEFAULTLIB:libcmt.lib")
endif(MSVC)

add_subdirectory(asset)
add_subdirectory(demos)

include(CTest)
enable_testing()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
