project(matrix)

set(ocl_src_files
  matrix.cl
)

add_executable(
  ${PROJECT_NAME}
  main.cpp
)
target_link_libraries(
  ${PROJECT_NAME}
  common
  OpenCL
)

copy_assets(ocl_src_files "" copied_${PROJECT_NAME}_ocl_files)

add_custom_target(
  ${PROJECT_NAME}CopyOCLFiles ALL
  DEPENDS ${copied_${PROJECT_NAME}_ocl_files}
)