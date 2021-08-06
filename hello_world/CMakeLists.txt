project(hello_word)

set(src_files
  main.cpp
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
  common
)