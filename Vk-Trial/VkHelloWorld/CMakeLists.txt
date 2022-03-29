project(VkHelloWorld VERSION 0.1.0)

set(src_files
  Cube.cpp
  Sample.cpp
  main.cpp
)

set(spirv_gen_files
  ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/shaders/box.vs.spv
  ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/shaders/box.fs.spv
)

add_executable(${PROJECT_NAME}
 ${src_files}
 )

target_link_libraries(
  ${PROJECT_NAME}
  Common
  Microsoft::DirectXTex
  glfw
  ${Vulkan_LIBRARIES}
)

add_custom_command(
  OUTPUT
  ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/shaders/box.vs.spv
  ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/shaders/box.fs.spv
  ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/shaders/box.vert.spv
  ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/shaders/box.frag.spv
  COMMAND ${DXC} -T vs_6_0 -spirv -E VSMain -Fo ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/shaders/box.vs.spv  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/box.hlsl
  COMMAND ${DXC} -T ps_6_0 -spirv -E PSMain -Fo ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/shaders/box.fs.spv  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/box.hlsl
  COMMAND ${Vulkan_GLSLANG_VALIDATOR} -V ${CMAKE_CURRENT_SOURCE_DIR}/shaders/box.vert -o ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/shaders/box.vert.spv
  COMMAND ${Vulkan_GLSLANG_VALIDATOR} -V ${CMAKE_CURRENT_SOURCE_DIR}/shaders/box.frag -o ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/shaders/box.frag.spv
  DEPENDS shaders/box.hlsl shaders/box.vert shaders/box.frag
)

add_custom_target(
  GenSpirvFiles
  ALL
  DEPENDS
  ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/shaders/box.vs.spv
  ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/shaders/box.fs.spv
  ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/shaders/box.vert.spv
  ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/shaders/box.frag.spv
  SOURCES
  shaders/box.hlsl shaders/box.vert shaders/box.frag
)

# set(SPV_FILES shaders/box.vs.spv shaders/box.fs.spv)

# copy_assets(SPV_FILES "shaders" COMPILED_SPIRV_FILES)

# add_custom_target(
#   ${PROJECT_NAME}GenAssets ALL
#   DEPENDS ${COMPILED_SPIRV_FILES}
#   SOURCES ${SPV_FILES} ${SHADER_EXTRA_FILES}
# )



