project(skeletal-animation VERSION 0.0.1)

add_executable(${PROJECT_NAME}
  skm_mesh.hpp
  skm_mesh.cpp
  skm_model.hpp
  skm_model.cpp
  material.hpp
  material.cpp
  animator.h
  animation.h
  bone.h
  assimp_glm_helpers.hpp
  skeletal-animation.cpp
)
add_dependencies(${PROJECT_NAME} common)
target_link_libraries(${PROJECT_NAME}
  glfw
  common
  assimp::assimp
)

file(GLOB SHADER_RESOURCES ${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.*)

copy_assets(SHADER_RESOURCES "shaders" copied_shader_files)

add_custom_target(
  ${PROJECT_NAME}CopyShaderFiles ALL
  DEPENDS ${copied_shader_files}
  SOURCES ${SHADER_RESOURCES}
)

source_group("Shaders" FILES ${SHADER_RESOURCES})