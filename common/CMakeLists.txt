project(common VERSION 0.1.0)

add_library(
  ${PROJECT_NAME}
  cl_utils.cpp
  cl_utils.h
)
target_link_libraries(
  ${PROJECT_NAME}
  OpenCL
)