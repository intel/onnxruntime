# Building from source is currently unsupported.
# Please follow instructions in the documentation to install OpenVINO binaries.

include (ExternalProject)

string(TOLOWER ${CMAKE_SYSTEM_PROCESSOR} OPENVINO_ARCH)
if(OPENVINO_ARCH STREQUAL "x86_64" OR OPENVINO_ARCH STREQUAL "amd64") # Windows detects Intel's 64-bit CPU as AMD64
    set(OPENVINO_ARCH intel64)
elseif(OPENVINO_ARCH STREQUAL "i386")
    set(OPENVINO_ARCH ia32)
else()
    set(OPENVINO_ARCH armv7l)
endif()

 # Libraries for python package.
if (WIN32)
    set(OPENVINO_CPU_EXTENSION_LIB cpu_extension.dll)
else()
    set(OPENVINO_CPU_EXTENSION_LIB libcpu_extension.so)
endif()

set(OPENVINO_EXTENSIONS_DIR $ENV{INTEL_OPENVINO_DIR}/deployment_tools/inference_engine/samples)
set(OPENVINO_INSTALL ${CMAKE_CURRENT_BINARY_DIR}/openvino/install)
set(OPENVINO_BUILD ${CMAKE_CURRENT_BINARY_DIR}/openvino)
set(OPENVINO_CPU_EXTENSION_DIR ${OPENVINO_BUILD}/${OPENVINO_ARCH}/Release/lib)

ExternalProject_Add(project_openvino_cpu_extensions
    SOURCE_DIR ${OPENVINO_EXTENSIONS_DIR}
    BINARY_DIR ${OPENVINO_BUILD}
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${OPENVINO_INSTALL}
)
add_library(ie_cpu_extension SHARED IMPORTED)