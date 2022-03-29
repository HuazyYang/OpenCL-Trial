cmake_minimum_required(VERSION 3.0)
project(common VERSION 0.1.0)

add_library(common
  GeometryGenerator.cpp
  ShaderProgram.cpp
  GameTimer.cpp
  GameTimer.hpp
  CommonUtils.cpp
  CommonUtils.hpp
  )
  target_link_libraries(
    common
    PRIVATE
    glad
  )
