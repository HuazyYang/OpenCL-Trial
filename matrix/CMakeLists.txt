project(matrix)

set(ocl_src_files
  matrix.cl
  mat_mul_vec.cl
)

add_executable(
  ${PROJECT_NAME}
  main.cpp
)
target_compile_options(
  ${PROJECT_NAME}
  PRIVATE
  "/Qvec-report:1"
)
target_link_libraries(
  ${PROJECT_NAME}
  common
)

clang_oclxx_to_spirv(ocl_src_files "-D_USE_DOUBLE_FP=1 -I\"${COMMON_INCLUDE_DIR}\"" "OCL-SpirV" "matrix.spv" ${PROJECT_NAME}_spv_files)

add_custom_target(
  ${PROJECT_NAME}CompiledSpvFiles ALL
  DEPENDS ${${PROJECT_NAME}_spv_files}
  SOURCES ${ocl_src_files}
)