project(common VERSION 0.1.0)

add_library(
  ${PROJECT_NAME}
  cl_utils.cpp
  cl_utils.h
  common_miscs.cpp
  common_miscs.h

  # OpenGL staff
  ${GLAD_SOURCE_DIR}/glad.c
  glx_utils.cpp
  glx_utils.hpp
  glx_shader_program.cpp
  glx_shader_program.hpp
)

target_link_libraries(
  ${PROJECT_NAME}
  OpenCL
)