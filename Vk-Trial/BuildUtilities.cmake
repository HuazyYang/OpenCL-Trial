
# Shader compilation
function(compile_glsl_to_spirv shader_files dir_name compiled_shaders)
foreach(shader ${${shader_files}})
	message("SHADER: ${shader}")
	get_filename_component(file_name ${shader} NAME)
	get_filename_component(full_path ${shader} ABSOLUTE)
	set(output_dir ${CMAKE_CURRENT_BINARY_DIR}/${TRIAL_OUTDIR_SUFFIX}/${dir_name})
	set(output_file ${output_dir}/${file_name}.spv)
	set(${compiled_shaders} ${${compiled_shaders}} ${output_file})
	set(${compiled_shaders} ${${compiled_shaders}} PARENT_SCOPE)
	set_source_files_properties(${shader} PROPERTIES HEADER_FILE_ONLY TRUE)
	if (WIN32)
        add_custom_command(
            OUTPUT ${output_file}
            COMMAND ${Vulkan_GLSLANG_VALIDATOR} -V ${full_path} -o ${output_file}
            DEPENDS ${full_path}
        )
    else()
        add_custom_command(
            OUTPUT ${output_file}
            COMMAND mkdir --parents ${output_dir} && ${Vulkan_GLSLANG_VALIDATOR} -V ${full_path} -o ${output_file}
            DEPENDS ${full_path}
        )
    endif()
endforeach()
endfunction(compile_glsl_to_spirv)

function(copy_assets asset_files dir_name copied_files)
foreach(asset ${${asset_files}})
  #message("asset: ${asset}")
  get_filename_component(file_name ${asset} NAME)
  get_filename_component(full_path ${asset} ABSOLUTE)
  set(output_dir ${CMAKE_CURRENT_BINARY_DIR}/${TRIAL_OUTDIR_SUFFIX}/${dir_name})
  set(output_file ${output_dir}/${file_name})
  set(${copied_files} ${${copied_files}} ${output_file})
  set(${copied_files} ${${copied_files}} PARENT_SCOPE)
  set_source_files_properties(${asset} PROPERTIES HEADER_FILE_ONLY TRUE)
  if (WIN32)
    add_custom_command(
      OUTPUT ${output_file}
      #COMMAND mklink \"${output_file}\" \"${full_path}\"
      COMMAND xcopy \"${full_path}\" \"${output_file}*\" /Y /Q /F
      DEPENDS ${full_path}
    )
  else()
    add_custom_command(
      OUTPUT ${output_file}
      COMMAND mkdir --parents ${output_dir} && cp --force --link \"${full_path}\" \"${output_file}\"
      DEPENDS ${full_path}
    )
  endif()
endforeach()
endfunction(copy_assets)