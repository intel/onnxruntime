# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# include_directories("${CMAKE_CURRENT_BINARY_DIR}/onnx")
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

if(OpenVINO_VERSION VERSION_LESS 2024.5)
  message(FATAL_ERROR "OpenVINO 2024.5 and newer are supported. Please, use latest OpenVINO release")
endif()

if(OpenVINO_VERSION VERSION_GREATER_EQUAL 2024.4)
  add_definitions(-DUSE_OVEP_NPU_MEMORY=1)
endif()

# If building RelWithDebInfo and OV package does not have that configuration map to Release
get_target_property(ov_rt_implib_rwdi openvino::runtime IMPORTED_IMPLIB_RELWITHDEBINFO)

if((CMAKE_BUILD_TYPE STREQUAL RelWithDebInfo) AND NOT ov_rt_implib_rwdi)
  set_target_properties(openvino::runtime PROPERTIES
    MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release
  )
endif()

list(APPEND OPENVINO_LIB_LIST openvino::frontend::onnx openvino::runtime ${PYTHON_LIBRARIES})

source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_openvino_cc_srcs})

if(WIN32)
  # Create a static library target with the common source code. Currently not unified due to unresolved linux linking
  # issues regarding dropped export symbols
  set(onnxruntime_providers_openvino_target onnxruntime_providers_openvino_lib)
  onnxruntime_add_static_library(onnxruntime_providers_openvino_lib ${onnxruntime_providers_openvino_cc_srcs})
else()
  set(onnxruntime_providers_openvino_target onnxruntime_providers_openvino)
  onnxruntime_add_shared_library_module(${onnxruntime_providers_openvino_target} ${onnxruntime_providers_openvino_cc_srcs} "${ONNXRUNTIME_ROOT}/core/dll/onnxruntime.rc")

  if(APPLE)
    set_property(TARGET ${onnxruntime_providers_openvino_target} APPEND_STRING PROPERTY LINK_FLAGS "-Xlinker -exported_symbols_list ${ONNXRUNTIME_ROOT}/core/providers/openvino/exported_symbols.lst")
    target_compile_definitions(${onnxruntime_providers_openvino_target} PRIVATE FILE_NAME=\"${onnxruntime_providers_openvino_target}.so\")
  elseif(UNIX)
    set_property(TARGET ${onnxruntime_providers_openvino_target} APPEND_STRING PROPERTY LINK_FLAGS
      "-Xlinker --version-script=${ONNXRUNTIME_ROOT}/core/providers/openvino/version_script.lds -Xlinker --gc-sections"
    )
    target_compile_definitions(${onnxruntime_providers_openvino_target} PRIVATE FILE_NAME=\"${onnxruntime_providers_openvino_target}.so\")
  endif()
endif()

# Common configuration for the library
onnxruntime_add_include_to_target(${onnxruntime_providers_openvino_target} onnxruntime_common onnx nlohmann_json::nlohmann_json)
set_target_properties(${onnxruntime_providers_openvino_target} PROPERTIES CXX_STANDARD 20)
set_target_properties(${onnxruntime_providers_openvino_target} PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(${onnxruntime_providers_openvino_target} PROPERTIES FOLDER "ONNXRuntime")

target_compile_options(${onnxruntime_providers_openvino_target} PRIVATE
  $<$<CONFIG:Release>:-DRELEASE>
)

if(NOT MSVC)
  target_compile_options(${onnxruntime_providers_openvino_target} PRIVATE "-Wno-parentheses")
endif()

add_dependencies(${onnxruntime_providers_openvino_target} onnxruntime_providers_shared ${onnxruntime_EXTERNAL_DEPENDENCIES})
target_include_directories(${onnxruntime_providers_openvino_target} SYSTEM PUBLIC ${ONNXRUNTIME_ROOT} ${CMAKE_CURRENT_BINARY_DIR} ${OpenVINO_INCLUDE_DIR} ${OPENVINO_INCLUDE_DIR_LIST} ${PYTHON_INCLUDE_DIRS})
target_link_libraries(${onnxruntime_providers_openvino_target} ${ONNXRUNTIME_PROVIDERS_SHARED} Boost::mp11 ${OPENVINO_LIB_LIST} ${ABSEIL_LIBS} Eigen3::Eigen)

if(MSVC)
  target_compile_options(${onnxruntime_providers_openvino_target} PRIVATE /wd4099 /wd4275 /wd4100 /wd4005)
endif()

# Needed for the provider interface, as it includes training headers when training is enabled
if(onnxruntime_ENABLE_TRAINING_OPS)
  target_include_directories(${onnxruntime_providers_openvino_target} PRIVATE ${ORTTRAINING_ROOT})
endif()

# Function to create a shared library from the static library
function(create_openvino_shared_library TARGET_NAME SYMBOL_FILE_ROOT)
  onnxruntime_add_shared_library(${TARGET_NAME} "${ONNXRUNTIME_ROOT}/core/dll/onnxruntime.rc")

  # Link against the static library
  target_link_libraries(${TARGET_NAME} ${onnxruntime_providers_openvino_target})

  set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "ONNXRuntime")
  set_target_properties(${TARGET_NAME} PROPERTIES LINKER_LANGUAGE CXX)

  # Platform-specific export settings
  if(WIN32)
    set_property(TARGET ${TARGET_NAME} APPEND_STRING PROPERTY LINK_FLAGS "-DEF:${SYMBOL_FILE_ROOT}/symbols.def")
    target_compile_definitions(${TARGET_NAME} PRIVATE FILE_NAME=\"${TARGET_NAME}.dll\")
  else()
    message(FATAL_ERROR "${TARGET_NAME} unknown platform, need to specify shared library exports for it")
  endif()

  set_target_properties(${TARGET_NAME} PROPERTIES
    MAP_IMPORTED_CONFIG_RELEASE RelWithDebInfo
    MAP_IMPORTED_CONFIG_DEBUG RelWithDebInfo
  )
endfunction()

if(WIN32)
  # Create the standard OpenVINO shared library
  create_openvino_shared_library(onnxruntime_providers_openvino
    "${ONNXRUNTIME_ROOT}/core/providers/openvino")

  # Create the Windows-specific OpenVINO shared library (if on Windows)
  create_openvino_shared_library(onnxruntime_providers_openvino_plugin
    "${ONNXRUNTIME_ROOT}/core/providers/openvino/plugin")
endif()

# Install header file (only once)
install(FILES ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/providers/openvino/openvino_provider_factory.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/)

# Install targets
if(WIN32)
  set(OPENVINO_TARGETS onnxruntime_providers_openvino)
  list(APPEND OPENVINO_TARGETS onnxruntime_providers_openvino_plugin)
else()
  set(OPENVINO_TARGETS ${onnxruntime_providers_openvino_target})
endif()

if(CMAKE_OPENVINO_LIBRARY_INSTALL_DIR)
  install(TARGETS ${OPENVINO_TARGETS}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_OPENVINO_LIBRARY_INSTALL_DIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
else()
  install(TARGETS ${OPENVINO_TARGETS}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
