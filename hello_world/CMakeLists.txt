project(hello_word)

set(src_files
  # GPUOpenCLProjectforWindows.cpp
  # utils.cpp
  # utils.h
  main.cpp
  ${COMMON_INCLUDE_DIR}/cl_utils.h
  ${COMMON_INCLUDE_DIR}/cl_utils.cpp
)

set(ocl_src_files
  simple.cl
)

add_executable(
  ${PROJECT_NAME}
  ${src_files}
)

copy_assets(ocl_src_files "" copied_${PROJECT_NAME}_ocl_files)

add_custom_target(
  ${PROJECT_NAME}CopyOCLFiles ALL
  DEPENDS ${copied_${PROJECT_NAME}_ocl_files}
)
target_link_libraries(
  ${PROJECT_NAME}
  OpenCL
)