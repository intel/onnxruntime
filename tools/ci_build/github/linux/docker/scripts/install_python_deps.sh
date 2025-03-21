#!/bin/bash
set -e -x

INSTALL_DEPS_TRAINING=false
INSTALL_DEPS_DISTRIBUTED_SETUP=false
TARGET_ROCM=false
CU_VER="11.8"
TORCH_VERSION='2.0.0'
USE_CONDA=false

while getopts p:h:d:v:tmurc parameter_Option
do case "${parameter_Option}"
in
p) PYTHON_VER=${OPTARG};;
h) TORCH_VERSION=${OPTARG};;
d) DEVICE_TYPE=${OPTARG};;
v) CU_VER=${OPTARG};;
t) INSTALL_DEPS_TRAINING=true;;
m) INSTALL_DEPS_DISTRIBUTED_SETUP=true;;
r) TARGET_ROCM=true;;
c) USE_CONDA=true;;
esac
done

echo "Python version=$PYTHON_VER"

DEVICE_TYPE=${DEVICE_TYPE:=Normal}

if [[ $USE_CONDA = true ]]; then
  # conda python version has already been installed by
  # tools/ci_build/github/linux/docker/Dockerfile.ubuntu_gpu_training.
  # so, /home/onnxruntimedev/miniconda3/bin/python should point
  # to the correct version of the python version
   PYTHON_EXE="/home/onnxruntimedev/miniconda3/bin/python"
elif [[ "$PYTHON_VER" = "3.10" && -d "/opt/python/cp310-cp310"  ]]; then
   PYTHON_EXE="/opt/python/cp310-cp310/bin/python3.10"
elif [[ "$PYTHON_VER" = "3.11" && -d "/opt/python/cp311-cp311"  ]]; then
   PYTHON_EXE="/opt/python/cp311-cp311/bin/python3.11"
elif [[ "$PYTHON_VER" = "3.12" && -d "/opt/python/cp312-cp312"  ]]; then
   PYTHON_EXE="/opt/python/cp312-cp312/bin/python3.12"
else
   PYTHON_EXE="/usr/bin/python${PYTHON_VER}"
fi

export ONNX_ML=1
export CMAKE_ARGS="-DONNX_GEN_PB_TYPE_STUBS=OFF -DONNX_WERROR=OFF"
${PYTHON_EXE} -m pip install -r ${0/%install_python_deps\.sh/requirements\.txt}
