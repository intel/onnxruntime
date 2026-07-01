# Copyright (C) Intel Corporation
# Licensed under the MIT License
#
# cmake -P script: remove the Authenticode digital signature from a PE file.
#
# Variables (passed via -D):
#   SIGNTOOL  — absolute path to signtool.exe
#   DLL_PATH  — absolute path to the PE file
#
# Uses signtool verify to detect whether the file is signed before attempting
# removal, so unsigned files are skipped cleanly without parsing error messages.
cmake_minimum_required(VERSION 3.28)

execute_process(
  COMMAND "${SIGNTOOL}" verify /pa "${DLL_PATH}"
  RESULT_VARIABLE _signed
  OUTPUT_QUIET ERROR_QUIET
)

if(NOT _signed EQUAL 0)
  message(STATUS "remove_signature: '${DLL_PATH}' is not signed — skipped")
  return()
endif()

execute_process(
  COMMAND "${SIGNTOOL}" remove /s "${DLL_PATH}"
  RESULT_VARIABLE _rc
  OUTPUT_VARIABLE _out
  ERROR_VARIABLE  _err
)

if(NOT _rc EQUAL 0)
  message(FATAL_ERROR
    "remove_signature: signtool remove failed for '${DLL_PATH}' (exit ${_rc}):\n${_out}${_err}")
endif()
