project(vector_dot VERSION 0.1.0)

set(ocl_src_files
  vector_dot.cl
)

add_executable(
  ${PROJECT_NAME}
  main.cpp
)
target_link_libraries(
  ${PROJECT_NAME}
  common
  "${OpenCL_LIBRARIES}"
)

copy_assets(ocl_src_files "" copied_${PROJECT_NAME}_ocl_files)

add_custom_target(
  ${PROJECT_NAME}CopyOCLFiles ALL
  DEPENDS ${copied_${PROJECT_NAME}_ocl_files}
)