project(pbr-lighting)

set(PBR_SOURCES
  pbr/pbr_pre_compute.cpp
  pbr/pbr_pre_compute.hpp
)

set(SOURCES
  pch.cpp
  pbr-lighting.cpp
  equivrectangle_hdr_image_to_cube_images.cpp
)

add_executable(${PROJECT_NAME}
    ${SOURCES}
    ${PBR_SOURCES}
)
if(MSVC)
  target_compile_options(${PROJECT_NAME} BEFORE PRIVATE "/wd4996")
endif()
add_dependencies(${PROJECT_NAME} common)
target_link_libraries(${PROJECT_NAME}
    glfw
    common
)

file(GLOB SHADER_RESOURCES ${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.*)
file(GLOB PBR_SHADER_RESOURCES ${CMAKE_CURRENT_SOURCE_DIR}/pbr/shaders/*.*)

copy_assets(SHADER_RESOURCES "shaders" copied_shader_files)
copy_assets(PBR_SHADER_RESOURCES "pbr/shaders" pbr_copied_shader_files)

add_custom_target(
  ${PROJECT_NAME}CopyShaderFiles ALL
  DEPENDS ${copied_shader_files} ${pbr_copied_shader_files}
  SOURCES ${SHADER_RESOURCES} ${PBR_SHADER_RESOURCES}
)

source_group("shaders" FILES ${SHADER_RESOURCES})
source_group("pbr" FILES ${PBR_SOURCES})
source_group("pbr/shaders" FILES ${PBR_SHADER_RESOURCES})