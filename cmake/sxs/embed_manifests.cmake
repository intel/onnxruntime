# Copyright (C) Intel Corporation
# Licensed under the MIT License
#
# Install-time CMake script: embeds SxS manifests into OV+TBB DLLs and
# onnxruntime_providers_openvino.dll.
# Run via install(SCRIPT ...) after OV/TBB binaries are installed.
#
# Variables injected by install(CODE) in onnxruntime_providers_openvino.cmake:
#   CMAKE_MT        — absolute path to mt.exe
#   SXS_SOURCE_DIR  — cmake/sxs/ source directory
#   EP_FILE_VERSION — EP version in A.B.C.D form (e.g. 1.28.0.0)
#   OV_TBB_DLL_NAMES — semicolon-separated list of OV+TBB DLL filenames

cmake_minimum_required(VERSION 3.28)

set(PROVIDER_DLL_NAME "onnxruntime_providers_openvino")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Embed a manifest into a DLL as RT_MANIFEST resource ID 2.
# If the DLL already has a resource ID 2 manifest it is extracted and merged first.
# Temporary files are written next to the DLL and deleted on completion.
function(ort_sxs_embed_manifest DLL_PATH MANIFEST_PATH)
  get_filename_component(_dll_dir    "${DLL_PATH}" DIRECTORY)
  get_filename_component(_dll_name_we "${DLL_PATH}" NAME_WE)
  set(_existing "${_dll_dir}/existing_${_dll_name_we}.manifest")

  # Try to extract any existing RT_MANIFEST resource ID 2.
  execute_process(
    COMMAND "${CMAKE_MT}" -nologo
      "-inputresource:${DLL_PATH};2"
      "-out:${_existing}"
    RESULT_VARIABLE _rc
    OUTPUT_QUIET ERROR_QUIET)

  if(_rc EQUAL 0 AND EXISTS "${_existing}")
    # Merge existing + new dep manifest, then re-embed.
    execute_process(
      COMMAND "${CMAKE_MT}" -nologo
        "-manifest" "${_existing}" "${MANIFEST_PATH}"
        "-outputresource:${DLL_PATH};2"
      RESULT_VARIABLE _rc2)
    file(REMOVE "${_existing}")
    if(NOT _rc2 EQUAL 0)
      message(FATAL_ERROR
        "SxS manifest embedding: mt.exe merge failed for '${DLL_PATH}' (exit ${_rc2}).")
    endif()
  else()
    # No existing manifest — embed the new one directly.
    execute_process(
      COMMAND "${CMAKE_MT}" -nologo
        "-manifest" "${MANIFEST_PATH}"
        "-outputresource:${DLL_PATH};2"
      RESULT_VARIABLE _rc2)
    if(NOT _rc2 EQUAL 0)
      message(FATAL_ERROR
        "SxS manifest embedding: mt.exe embed failed for '${DLL_PATH}' (exit ${_rc2}).")
    endif()
  endif()
endfunction()

# ---------------------------------------------------------------------------
# Resolve install directory
# ---------------------------------------------------------------------------

if(DEFINED ORT_SXS_BIN_DIR AND NOT ORT_SXS_BIN_DIR STREQUAL "")
  set(BIN_DIR "${ORT_SXS_BIN_DIR}")
else()
  set(BIN_DIR "${CMAKE_INSTALL_PREFIX}/bin")
endif()
message(STATUS "SxS manifest embedding: processing '${BIN_DIR}'")

# ---------------------------------------------------------------------------
# Step 1: Embed dep manifests into each OV+TBB DLL
# ---------------------------------------------------------------------------

foreach(_name IN LISTS OV_TBB_DLL_NAMES)
  set(_dll "${BIN_DIR}/${_name}")
  if(NOT EXISTS "${_dll}")
    continue()  # DLL absent for this config (e.g. debug-only DLL in a Release build)
  endif()
  get_filename_component(_name_we "${_name}" NAME_WE)
  set(DLL_BASE_NAME "${_name_we}")
  set(_dep_manifest "${BIN_DIR}/${_name_we}.dep.manifest")
  configure_file("${SXS_SOURCE_DIR}/dep.manifest.in" "${_dep_manifest}")
  ort_sxs_embed_manifest("${_dll}" "${_dep_manifest}")
  file(REMOVE "${_dep_manifest}")
  message(STATUS "SxS manifest embedding: embedded dep manifest in ${_name_we}.dll")
endforeach()

# ---------------------------------------------------------------------------
# Step 2: Embed dep manifest into onnxruntime_providers_openvino.dll
# ---------------------------------------------------------------------------

# The provider DLL may be installed to bin/ or lib/ depending on whether it's
# a SHARED or MODULE library.  Check both locations.
set(_provider_dll "${BIN_DIR}/${PROVIDER_DLL_NAME}.dll")
if(NOT EXISTS "${_provider_dll}")
  # Try the lib/ sibling directory (MODULE libraries install to LIBRARY dest)
  get_filename_component(_prefix "${BIN_DIR}" DIRECTORY)
  set(_provider_dll "${_prefix}/lib/${PROVIDER_DLL_NAME}.dll")
endif()
if(NOT EXISTS "${_provider_dll}")
  message(FATAL_ERROR
    "SxS manifest embedding: '${PROVIDER_DLL_NAME}.dll' not found in '${BIN_DIR}' or '${_prefix}/lib'.")
endif()

set(DLL_BASE_NAME "${PROVIDER_DLL_NAME}")
set(_dep_manifest "${BIN_DIR}/${PROVIDER_DLL_NAME}.dep.manifest")
configure_file("${SXS_SOURCE_DIR}/dep.manifest.in" "${_dep_manifest}")
ort_sxs_embed_manifest("${_provider_dll}" "${_dep_manifest}")
file(REMOVE "${_dep_manifest}")
message(STATUS "SxS manifest embedding: embedded dep manifest in ${PROVIDER_DLL_NAME}.dll")

message(STATUS "SxS manifest embedding: done")
