# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

#  include_directories("${CMAKE_CURRENT_BINARY_DIR}/onnx")
  file(GLOB_RECURSE onnxruntime_providers_openvino_cc_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/openvino/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/openvino/*.cc"
    "${ONNXRUNTIME_ROOT}/core/providers/openvino/*.hpp"
    "${ONNXRUNTIME_ROOT}/core/providers/openvino/*.cpp"
    "${ONNXRUNTIME_ROOT}/core/providers/shared_library/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/shared_library/*.cc"
  )

  # Header paths
  find_package(OpenVINO REQUIRED COMPONENTS Runtime ONNX)
  if(OpenVINO_VERSION VERSION_LESS 2026.0)
    message(FATAL_ERROR "OpenVINO 2026.0 and newer are supported. Please, use latest OpenVINO release")
  endif()

  if(OpenVINO_VERSION VERSION_GREATER_EQUAL 2024.4)
    add_definitions(-DUSE_OVEP_NPU_MEMORY=1)
  endif()

  # If building RelWithDebInfo and OV package does not have that configuration map to Release
  get_target_property(ov_rt_implib_rwdi openvino::runtime IMPORTED_IMPLIB_RELWITHDEBINFO)
  if ((CMAKE_BUILD_TYPE STREQUAL RelWithDebInfo) AND NOT ov_rt_implib_rwdi)
    set_target_properties(openvino::runtime PROPERTIES
      MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release
    )
  endif()

  list(APPEND OPENVINO_LIB_LIST openvino::frontend::onnx openvino::runtime ${PYTHON_LIBRARIES})
  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_openvino_cc_srcs})
  onnxruntime_add_shared_library_module(onnxruntime_providers_openvino ${onnxruntime_providers_openvino_cc_srcs} "${ONNXRUNTIME_ROOT}/core/dll/onnxruntime.rc")

  # Propagate leak check define if enabled at top level
  if(onnxruntime_ENABLE_MEMLEAK_CHECKER)
    target_compile_definitions(onnxruntime_providers_openvino PRIVATE ONNXRUNTIME_ENABLE_MEMLEAK_CHECK)
  endif()

  onnxruntime_add_include_to_target(onnxruntime_providers_openvino onnxruntime_common onnx nlohmann_json::nlohmann_json)
  install(FILES ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/providers/openvino/openvino_provider_factory.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/)
  set_target_properties(onnxruntime_providers_openvino PROPERTIES CXX_STANDARD 20)
  set_target_properties(onnxruntime_providers_openvino PROPERTIES LINKER_LANGUAGE CXX)
  set_target_properties(onnxruntime_providers_openvino PROPERTIES FOLDER "ONNXRuntime")

  target_compile_options(onnxruntime_providers_openvino PRIVATE
  $<$<CONFIG:Release>:-DRELEASE>
  )

  if(NOT MSVC)
    target_compile_options(onnxruntime_providers_openvino PRIVATE "-Wno-parentheses")
  endif()
  add_dependencies(onnxruntime_providers_openvino onnxruntime_providers_shared ${onnxruntime_EXTERNAL_DEPENDENCIES})
  target_include_directories(onnxruntime_providers_openvino SYSTEM PUBLIC ${ONNXRUNTIME_ROOT} ${CMAKE_CURRENT_BINARY_DIR} ${OpenVINO_INCLUDE_DIR} ${OPENVINO_INCLUDE_DIR_LIST} ${PYTHON_INCLUDE_DIRS} $ENV{OPENCL_INCS} $ENV{OPENCL_INCS}/../../cl_headers/)
  target_link_libraries(onnxruntime_providers_openvino ${ONNXRUNTIME_PROVIDERS_SHARED} Boost::mp11 ${OPENVINO_LIB_LIST} ${ABSEIL_LIBS} Eigen3::Eigen onnx_proto)

  # ETW TraceLogging depends on Advapi32 on Windows
  if(WIN32)
    target_link_libraries(onnxruntime_providers_openvino advapi32)
  endif()

  target_compile_definitions(onnxruntime_providers_openvino PRIVATE FILE_NAME=\"onnxruntime_providers_openvino.dll\")

  if(MSVC)
    target_compile_options(onnxruntime_providers_openvino PRIVATE /wd4099 /wd4275 /wd4100 /wd4005)
  endif()

  # Needed for the provider interface, as it includes training headers when training is enabled
  if (onnxruntime_ENABLE_TRAINING_OPS)
    target_include_directories(onnxruntime_providers_openvino PRIVATE ${ORTTRAINING_ROOT})
  endif()

  if(APPLE)
    set_property(TARGET onnxruntime_providers_openvino APPEND_STRING PROPERTY LINK_FLAGS "-Xlinker -exported_symbols_list ${ONNXRUNTIME_ROOT}/core/providers/openvino/exported_symbols.lst")
  elseif(UNIX)
    set_property(TARGET onnxruntime_providers_openvino APPEND_STRING PROPERTY LINK_FLAGS "-Xlinker --version-script=${ONNXRUNTIME_ROOT}/core/providers/openvino/version_script.lds -Xlinker --gc-sections")
  elseif(WIN32)
    set_property(TARGET onnxruntime_providers_openvino APPEND_STRING PROPERTY LINK_FLAGS "-DEF:${ONNXRUNTIME_ROOT}/core/providers/openvino/symbols.def")
  else()
    message(FATAL_ERROR "onnxruntime_providers_openvino unknown platform, need to specify shared library exports for it")
  endif()

  if (CMAKE_OPENVINO_LIBRARY_INSTALL_DIR)
    install(TARGETS onnxruntime_providers_openvino
            ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY  DESTINATION ${CMAKE_OPENVINO_LIBRARY_INSTALL_DIR}
            RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})
  else()
    install(TARGETS onnxruntime_providers_openvino
            ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})
  endif()

set_target_properties(onnxruntime_providers_openvino PROPERTIES
  MAP_IMPORTED_CONFIG_RELEASE RelWithDebInfo
  MAP_IMPORTED_CONFIG_DEBUG RelWithDebInfo
  )

# ---------------------------------------------------------------------------
# Windows SxS (Side-by-Side) loading support
#
# Embeds private-assembly manifests into OV/TBB DLLs and the provider DLL so
# that Windows loads them from the EP's own directory, avoiding DLL collisions
# when another application in the same process also uses OpenVINO.
# ---------------------------------------------------------------------------
if(WIN32)
  # Require mt.exe for manifest embedding
  if(NOT CMAKE_MT)
    message(FATAL_ERROR
      "mt.exe not found — SxS manifest embedding is required on Windows.\n"
      "Set -DCMAKE_MT=<path/to/mt.exe> or ensure the Windows SDK is on PATH.")
  endif()

  # --- Enumerate OV and TBB DLL filenames for the SxS assembly manifest ---
  # Glob patterns selecting the OV binaries needed by the EP.
  set(_ORT_OV_PATTERNS
    "cache.json"
    "*openvino.*"
    "*openvinod.*"
    "*openvino*plugin*"
    "*openvino*compiler*"
    "*openvino_ir_frontend*"
    "*openvino_onnx_frontend*")

  function(_ort_glob_ov _out_var _dir)
    set(_result "")
    foreach(_pat IN LISTS _ORT_OV_PATTERNS)
      file(GLOB _tmp "${_dir}/${_pat}")
      list(APPEND _result ${_tmp})
    endforeach()
    set(${_out_var} "${_result}" PARENT_SCOPE)
  endfunction()

  if(DEFINED ENV{INTEL_OPENVINO_DIR})
    file(TO_CMAKE_PATH "$ENV{INTEL_OPENVINO_DIR}" _ORT_OV_ROOT)
    set(_ov_bin "${_ORT_OV_ROOT}/runtime/bin/intel64")
    set(_tbb_bin "${_ORT_OV_ROOT}/runtime/3rdparty/tbb/bin")

    # TBB: split into non-debug (Release/RelWithDebInfo) and debug subsets
    file(GLOB _all_tbb "${_tbb_bin}/tbb*.dll")
    set(_tbb_release "")
    set(_tbb_debug   "")
    foreach(_f IN LISTS _all_tbb)
      get_filename_component(_lname "${_f}" NAME)
      string(TOLOWER "${_lname}" _lname_lower)
      if(_lname_lower MATCHES "debug")
        list(APPEND _tbb_debug   "${_f}")
      else()
        list(APPEND _tbb_release "${_f}")
      endif()
    endforeach()

    if(EXISTS "${_ov_bin}/Release")
      _ort_glob_ov(_ov_release "${_ov_bin}/Release")
      set(ORT_OV_INSTALL_FILES_Release ${_ov_release} ${_tbb_release})
    endif()
    if(EXISTS "${_ov_bin}/Debug")
      _ort_glob_ov(_ov_debug "${_ov_bin}/Debug")
      set(ORT_OV_INSTALL_FILES_Debug ${_ov_debug} ${_tbb_debug})
    endif()
    if(EXISTS "${_ov_bin}/RelWithDebInfo")
      _ort_glob_ov(_ov_rwdi "${_ov_bin}/RelWithDebInfo")
      set(ORT_OV_INSTALL_FILES_RelWithDebInfo ${_ov_rwdi} ${_tbb_release})
    elseif(DEFINED ORT_OV_INSTALL_FILES_Release)
      set(ORT_OV_INSTALL_FILES_RelWithDebInfo ${ORT_OV_INSTALL_FILES_Release})
    endif()
  else()
    # Python wheel layout — all binaries flat in <site-packages>/openvino/libs/
    set(Python3_FIND_VIRTUALENV FIRST)
    find_package(Python3 QUIET COMPONENTS Interpreter)
    if(Python3_FOUND)
      execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c
          "import openvino, pathlib; print(pathlib.Path(openvino.__file__).parent)"
        OUTPUT_VARIABLE _ORT_OV_WHEEL_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _ORT_OV_WHEEL_RESULT)
    endif()
    if(Python3_FOUND AND _ORT_OV_WHEEL_RESULT EQUAL 0 AND _ORT_OV_WHEEL_DIR)
      file(TO_CMAKE_PATH "${_ORT_OV_WHEEL_DIR}" _ORT_OV_ROOT)
      set(_libs "${_ORT_OV_ROOT}/libs")
      if(EXISTS "${_libs}")
        _ort_glob_ov(_ov_wheel "${_libs}")
        file(GLOB _tbb_wheel "${_libs}/tbb*.dll")
        set(ORT_OV_INSTALL_FILES_wheel ${_ov_wheel} ${_tbb_wheel})
      endif()
    endif()
  endif()

  # Collect unique DLL filenames for SxS dep manifest embedding
  set(_all_files
    ${ORT_OV_INSTALL_FILES_Release}
    ${ORT_OV_INSTALL_FILES_Debug}
    ${ORT_OV_INSTALL_FILES_RelWithDebInfo}
    ${ORT_OV_INSTALL_FILES_wheel})
  set(ORT_OV_TBB_DLL_NAMES "")
  foreach(_f IN LISTS _all_files)
    get_filename_component(_ext  "${_f}" EXT)
    get_filename_component(_name "${_f}" NAME)
    if(_ext STREQUAL ".dll")
      list(APPEND ORT_OV_TBB_DLL_NAMES "${_name}")
    endif()
  endforeach()
  list(REMOVE_DUPLICATES ORT_OV_TBB_DLL_NAMES)

  if(ORT_OV_TBB_DLL_NAMES)
    message(STATUS "OpenVINO SxS: DLLs for manifest: ${ORT_OV_TBB_DLL_NAMES}")

    # --- Generate assembly manifest ---
    set(ORT_SXS_VERSION "${ORT_VERSION}")
    set(_file_entries "")
    foreach(_dll IN LISTS ORT_OV_TBB_DLL_NAMES)
      string(APPEND _file_entries "  <file name=\"${_dll}\" />\n")
    endforeach()
    set(ORT_SXS_ASSEMBLY_FILE_ENTRIES "${_file_entries}")

    configure_file(
      "${CMAKE_CURRENT_LIST_DIR}/sxs/assembly.manifest.in"
      "${CMAKE_BINARY_DIR}/cmake/sxs/openvino_runtime.manifest"
      @ONLY)
    install(FILES "${CMAKE_BINARY_DIR}/cmake/sxs/openvino_runtime.manifest"
      DESTINATION ${CMAKE_INSTALL_BINDIR})

    # --- Install OV+TBB binaries alongside the provider DLL ---
    if(DEFINED ORT_OV_INSTALL_FILES_wheel)
      install(FILES ${ORT_OV_INSTALL_FILES_wheel} DESTINATION ${CMAKE_INSTALL_BINDIR})
    else()
      foreach(_config IN ITEMS Release Debug RelWithDebInfo)
        if(ORT_OV_INSTALL_FILES_${_config})
          install(FILES ${ORT_OV_INSTALL_FILES_${_config}}
            DESTINATION ${CMAKE_INSTALL_BINDIR}
            CONFIGURATIONS ${_config})
        endif()
      endforeach()
    endif()

    # --- Embed SxS manifests at install time ---
    install(CODE "set(CMAKE_MT         \"${CMAKE_MT}\")")
    install(CODE "set(SXS_SOURCE_DIR   \"${CMAKE_CURRENT_LIST_DIR}/sxs\")")
    install(CODE "set(EP_FILE_VERSION  \"${ORT_VERSION}.0\")")
    install(CODE "set(OV_TBB_DLL_NAMES \"${ORT_OV_TBB_DLL_NAMES}\")")
    install(CODE "set(ORT_SXS_BIN_DIR  \"\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}\")")
    install(SCRIPT "${CMAKE_CURRENT_LIST_DIR}/sxs/embed_manifests.cmake")
  else()
    message(WARNING "OpenVINO SxS: No OV/TBB DLLs found — SxS manifest embedding skipped.\n"
      "Ensure INTEL_OPENVINO_DIR is set (setupvars.bat) or the openvino wheel is installed.")
  endif()
endif()
