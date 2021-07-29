project(heat_transfer VERSION 0.1.0)

set(src_files
  main.cpp
)

set(ocl_src_files
  heat_transfer.cl
)

set(glsl_src_files
  shaders/color.vs
  shaders/color.fs
)
source_group("shaders" FILES ${glsl_src_files})

add_executable(
  ${PROJECT_NAME}
  ${src_files}
)

copy_assets(ocl_src_files "" copied_${PROJECT_NAME}_ocl_files)

copy_assets(glsl_src_files "shaders" copied_${PROJECT_NAME}_glsl_files)

add_custom_target(
  ${PROJECT_NAME}CopyAssetFiles ALL
  DEPENDS ${copied_${PROJECT_NAME}_ocl_files} ${copied_${PROJECT_NAME}_glsl_files}
  SOURCES ${col_src_files} ${glsl_src_files}
)
target_link_libraries(
  ${PROJECT_NAME}
  common
  OpenCL
  glfw3
)