project(transpose_inplace C CXX ASM_MASM)

set(ocl_src_files
)

add_executable(
  ${PROJECT_NAME}

  int128/int128.h
  int128/int128IO.cpp
  int128/int128x64_.asm
  int128/uint128x64_.asm

  main.cpp
  native_transpose.cpp
  native_transpose.h
  openmp_transpose.cpp
  openmp_transpose.h
  reduced_math.cpp
  reduced_math.h
  index.h

  page_allocator.h
  page_allocator.cpp
)
target_compile_options(
  ${PROJECT_NAME}
  PRIVATE
  "/Qvec-report:1"
  "/openmp"
)
target_link_libraries(
  ${PROJECT_NAME}
  common
)

copy_assets(ocl_src_files "" copied_${PROJECT_NAME}_ocl_files)

add_custom_target(
  ${PROJECT_NAME}CopyOCLFiles ALL
  DEPENDS ${copied_${PROJECT_NAME}_ocl_files}
)