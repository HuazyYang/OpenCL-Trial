# Shader compilation
find_program(CLANG_CXX_EXECUTABLE
  NAMES clang++ clang)
if(NOT CLANG_CXX_EXECUTABLE)
  message(FATAL_ERROR "clang++ executable can not found in host environment!")
endif()

find_program(LLVM_LINK NAMES llvm-lib)
if(NOT LLVM_LINK)
  message(FATAL_ERROR "llvm-link executable can not found in host environment!")
endif()

find_program(LLVM_SPIRV NAMES llvm-spirv)
if(NOT LLVM_SPIRV)
  message(FATAL_ERROR "llvm-spirv executable can not found in host environment!")
endif()

function(clang_oclxx_to_spirv oclxx_source_files aux_options dir_name linked_file compiled_spirvs)
set(list_compiled_files "")
set(output_dir ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/${dir_name})
foreach(shader ${${oclxx_source_files}})
  get_filename_component(file_ext ${shader} LAST_EXT)

  if((NOT ${file_ext} STREQUAL ".hxx") AND (NOT ${file_ext} STREQUAL ".h") AND
    (NOT ${file_ext} STREQUAL ".hpp"))
    message("Compile OpenCL source to SPIRV: ${shader}")
    get_filename_component(file_name ${shader} NAME)
    get_filename_component(full_path ${shader} ABSOLUTE)
    set(output_file ${output_dir}/${file_name}.bc)
    set(${compiled_spirvs} ${${compiled_spirvs}} ${output_file})
    set(${compiled_spirvs} ${${compiled_spirvs}} PARENT_SCOPE)
    set_source_files_properties(${shader} PROPERTIES HEADER_FILE_ONLY TRUE)
    if (WIN32)
      add_custom_command(
          OUTPUT ${output_file}
          COMMAND ${CLANG_CXX_EXECUTABLE} -cc1 -emit-llvm -finclude-default-header -triple=spir64-unknown-unknown
                    -cl-std=CL2.0 ${aux_options} -x cl -o ${output_file} ${full_path}
          DEPENDS ${full_path}
          WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      )
    else()
      add_custom_command(
        OUTPUT ${output_file}
        COMMAND mkdir --parents ${output_dir} &&
          ${CLANG_CXX_EXECUTABLE} -cc1 -emit-spirv -triple=spir64-unknown-unknown
            -cl-std=CL2.0 ${aux_options} -x cl -o ${output_file} ${full_path}
        DEPENDS ${full_path}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      )
    endif()
    list(APPEND list_compiled_files ${output_file})
  endif()
endforeach()

list(LENGTH list_compiled_files list_compiled_files_length)
if(${list_compiled_files_length} GREATER 0)
  # link all the compiled spv files into one.
  get_filename_component(file_name ${linked_file} NAME)
  set(output_file ${output_dir}/${file_name}.bc)
  set(output_spv_file ${output_dir}/${linked_file})
  set(${compiled_spirvs} ${${compiled_spirvs}} ${output_spv_file})
  set(${compiled_spirvs} ${${compiled_spirvs}} PARENT_SCOPE)
  
  if(${list_compiled_files_length} GREATER 1)
    message("Link LLVM IR into one: ${list_compiled_files} --> ${output_file}")
    add_custom_command(
      OUTPUT ${output_file}
      COMMAND ${LLVM_LINK} ${list_compiled_files} -o ${output_file}
      DEPENDS ${list_compiled_files}
      WORKING_DIRECTORY ${output_dir}
    )
    message("Convert LLVM IR to SPIR-V: ${output_file} --> ${output_spv_file}")
    add_custom_command(
      OUTPUT ${output_spv_file}
      COMMAND ${LLVM_SPIRV} ${output_file} -o ${output_spv_file}
      DEPENDS ${output_file}
      WORKING_DIRECTORY ${output_dir}
    )
  else()
    message("Convert LLVM IR to SPIR-V: ${list_compiled_files} --> ${output_spv_file}")
    add_custom_command(
      OUTPUT ${output_spv_file}
      COMMAND ${LLVM_SPIRV} ${list_compiled_files} -o ${output_spv_file} 
      DEPENDS ${list_compiled_files}
      WORKING_DIRECTORY ${output_dir}
    )
  endif()
endif()
endfunction(clang_oclxx_to_spirv)

function(copy_assets asset_files dir_name copied_files)
foreach(asset ${${asset_files}})
  #message("asset: ${asset}")
  get_filename_component(file_name ${asset} NAME)
  get_filename_component(full_path ${asset} ABSOLUTE)
  set(output_dir ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/${dir_name})
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