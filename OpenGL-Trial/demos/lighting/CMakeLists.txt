cmake_minimum_required(VERSION 3.0)
project(lighting VERSION 0.1.0)

add_executable(lighting WIN32
    ${SOURCE_DIR}lighting.cpp
)
add_dependencies(${PROJECT_NAME} common)
target_link_libraries(lighting
  glfw
  common
)

file(GLOB SHADER_RESOURCES ${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.*)

copy_assets(SHADER_RESOURCES "shaders" copied_shader_files)

add_custom_target(
  ${PROJECT_NAME}CopyShaderFiles ALL
  DEPENDS ${copied_shader_files}
  SOURCES ${SHADER_RESOURCES}
)

source_group("Shaders" FILES ${SHADER_RESOURCES})