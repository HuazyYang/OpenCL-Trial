project(common VERSION 0.1.0)

add_library(
  ${PROJECT_NAME}
  cl_utils.cpp
  cl_utils.h
  common_miscs.cpp
  common_miscs.h
)
target_link_libraries(
  ${PROJECT_NAME}
  OpenCL
)