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

  if (WIN32)
      set(CMAKE_MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release)
  endif()

  # Header paths
  find_package(OpenVINO REQUIRED COMPONENTS Runtime ONNX)
  if(OpenVINO_VERSION VERSION_LESS 2023.0)
    message(FATAL_ERROR "OpenVINO 2023.0 and newer are supported. Please, latest OpenVINO release")
  endif()

  if (WIN32)
    unset(CMAKE_MAP_IMPORTED_CONFIG_RELWITHDEBINFO)
  endif()

  # Get the INTEL_OPENVINO_DIR environment variable
  file(TO_CMAKE_PATH "$ENV{INTEL_OPENVINO_DIR}" OpenVINO_BASE_DIR)

  # Combine the base directory with the suffix path
  file(TO_CMAKE_PATH "${OpenVINO_BASE_DIR}/${OPENVINO_SUFFIX_PATH}" OPENVINO_STATIC_LIB_DIR)

  # Check if the combined directory exists, If the directory exists, proceed with setting up the static libraries
  if(IS_DIRECTORY "${OPENVINO_STATIC_LIB_DIR}")
    # Initialize an empty list to hold the found static libraries
    set(OPENVINO_FOUND_STATIC_LIBS)

    # Use the appropriate file extension for static libraries based on the host operating system
    if(CMAKE_HOST_WIN32)
      set(OPENVINO_STATIC_LIB_EXT "*.lib")
    elseif(CMAKE_HOST_UNIX)
      set(OPENVINO_STATIC_LIB_EXT "*.a")
    endif()

    # Use GLOB_RECURSE to find all static library files in the specified directory based on the OS
    file(GLOB_RECURSE OPENVINO_POSSIBLE_LIBS "${OPENVINO_STATIC_LIB_DIR}/${OPENVINO_STATIC_LIB_EXT}")

    # Iterate over each possible library and check if it exists before appending
    foreach(lib ${OPENVINO_POSSIBLE_LIBS})
      if(EXISTS "${lib}")
        list(APPEND OPENVINO_FOUND_STATIC_LIBS "${lib}")
      endif()
    endforeach()

    # Append the found static library files to the OPENVINO_LIB_LIST
    list(APPEND OPENVINO_LIB_LIST ${OPENVINO_FOUND_STATIC_LIBS})
  else()
    message(FATAL_ERROR "The specified OpenVINO static library directory does not exist: ${OPENVINO_STATIC_LIB_DIR}")
  endif()

  list(APPEND OPENVINO_LIB_LIST openvino::frontend::onnx openvino::runtime ${PYTHON_LIBRARIES})
  if ((DEFINED ENV{OPENCL_LIBS}) AND (DEFINED ENV{OPENCL_INCS}))
    add_definitions(-DIO_BUFFER_ENABLED=1)
    list(APPEND OPENVINO_LIB_LIST $ENV{OPENCL_LIBS})
  endif()

  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_openvino_cc_srcs})
  onnxruntime_add_shared_library_module(onnxruntime_providers_openvino ${onnxruntime_providers_openvino_cc_srcs} "${ONNXRUNTIME_ROOT}/core/dll/onnxruntime.rc")
  onnxruntime_add_include_to_target(onnxruntime_providers_openvino onnxruntime_common onnx)
  install(FILES ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/providers/openvino/openvino_provider_factory.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/)
  set_target_properties(onnxruntime_providers_openvino PROPERTIES LINKER_LANGUAGE CXX)
  set_target_properties(onnxruntime_providers_openvino PROPERTIES FOLDER "ONNXRuntime")
  if(NOT MSVC)
    target_compile_options(onnxruntime_providers_openvino PRIVATE "-Wno-parentheses")
  endif()
  add_dependencies(onnxruntime_providers_openvino onnxruntime_providers_shared ${onnxruntime_EXTERNAL_DEPENDENCIES})
  target_include_directories(onnxruntime_providers_openvino SYSTEM PUBLIC ${ONNXRUNTIME_ROOT} ${CMAKE_CURRENT_BINARY_DIR} ${eigen_INCLUDE_DIRS} ${OpenVINO_INCLUDE_DIR} ${OPENVINO_INCLUDE_DIR_LIST} ${PYTHON_INCLUDE_DIRS} $ENV{OPENCL_INCS} $ENV{OPENCL_INCS}/../../cl_headers/)
  target_link_libraries(onnxruntime_providers_openvino ${ONNXRUNTIME_PROVIDERS_SHARED} Boost::mp11 ${OPENVINO_LIB_LIST} ${ABSEIL_LIBS})

  target_compile_definitions(onnxruntime_providers_openvino PRIVATE VER_MAJOR=${VERSION_MAJOR_PART})
  target_compile_definitions(onnxruntime_providers_openvino PRIVATE VER_MINOR=${VERSION_MINOR_PART})
  target_compile_definitions(onnxruntime_providers_openvino PRIVATE VER_BUILD=${VERSION_BUILD_PART})
  target_compile_definitions(onnxruntime_providers_openvino PRIVATE VER_PRIVATE=${VERSION_PRIVATE_PART})
  target_compile_definitions(onnxruntime_providers_openvino PRIVATE VER_STRING=\"${VERSION_STRING}\")
  target_compile_definitions(onnxruntime_providers_openvino PRIVATE FILE_NAME=\"onnxruntime_providers_openvino.dll\")

  if(MSVC)
    target_compile_options(onnxruntime_providers_openvino PUBLIC /wd4099 /wd4275 /wd4100 /wd4005 /wd4244 /wd4267)
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
