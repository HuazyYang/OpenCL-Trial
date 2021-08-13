project(tridiagonal)

set(ocl_src_files
  config.cl.h
  cr_kernels.cl
  pcr_kernels.cl
  cr_pcr_hybrid_kernels.cl
)

add_executable(
  ${PROJECT_NAME}
  main.cpp

  tridiagonal_mat.h
  cpu_serial.cpp
  cpu_serial.h
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

clang_oclxx_to_spirv(ocl_src_files "-D_USE_DOUBLE_FP=1 -I\"${COMMON_INCLUDE_DIR}\"" "OCL-SpirV" "tridiagonal.spv" ${PROJECT_NAME}_spv_files)

add_custom_target(
  ${PROJECT_NAME}CompiledSpvFiles ALL
  DEPENDS ${${PROJECT_NAME}_spv_files}
  SOURCES ${ocl_src_files}
)