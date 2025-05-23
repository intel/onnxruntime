#!/bin/bash

# This script will run a full ORT build and use the python package built to generate ort format test files,
# which will be used in build_minimal_ort_and_run_tests.sh and nnapi_minimal_build_minimal_ort_and_run_tests.sh

set -e
set -x

BUILD_DIR=${1:?"usage: $0 <build directory>"}

python3 -m pip install --user -r tools/ci_build/github/linux/python/requirements.txt
# Validate the operator kernel registrations, as the ORT model uses hashes of the kernel registration details
# to find kernels. If the hashes from the registration details are incorrect we will produce a model that will break
# when the registration is fixed in the future.
python3 tools/ci_build/op_registration_validator.py

# Run a full build of ORT.
# We need the ORT python package to generate the ORT format files and the required ops config files.
# We do not run tests in this command since those are covered by other CIs.
# Both the NNAPI and CoreML EPs are enabled.
python3 tools/ci_build${BUILD_DIR}.py \
    --build_dir ${BUILD_DIR} --cmake_generator Ninja \
    --config Debug \
    --skip_submodule_sync \
    --parallel --use_vcpkg --use_vcpkg_ms_internal_asset_cache --use_binskim_compliant_compile_flags \
    --build_wheel \
    --skip_tests \
    --enable_training_ops \
    --use_nnapi \
    --use_coreml

# Install the ORT python wheel
python3 -m pip install --user ${BUILD_DIR}/Debug/dist/*

# Convert all the E2E ONNX models to ORT format
python3 tools/python/convert_onnx_models_to_ort.py \
    onnxruntime/test/testdata/ort_minimal_e2e_test_data

# Do it again using the conversion script from the python package to validate that also works
python3 -m onnxruntime.tools.convert_onnx_models_to_ort \
    onnxruntime/test/testdata/ort_minimal_e2e_test_data

# Create configs with just the required ops for ORT format models in testdata
# These are used by build_minimal_ort_and_run_tests.sh later in the linux-cpu-minimal-build-ci-pipeline CI
# and will include ops for the E2E models we just converted

# Config without type reduction
python3 tools/python/create_reduced_build_config.py --format ORT \
    onnxruntime/test/testdata \
    ${BUILD_DIR}/.test_data/required_ops.ort_models.config

# Config with type reduction
python3 tools/python/create_reduced_build_config.py --format ORT --enable_type_reduction \
    onnxruntime/test/testdata \
    ${BUILD_DIR}/.test_data/required_ops_and_types.ort_models.config

# Append the info for ops involved from inside custom ops. These can't be read from the models as they're
# dynamically created at runtime when the kernel is created.
cat onnxruntime/test/testdata/ort_minimal_e2e_test_data/required_ops.standalone_invoker.config >> \
    ${BUILD_DIR}/.test_data/required_ops.ort_models.config
cat onnxruntime/test/testdata/ort_minimal_e2e_test_data/required_ops.standalone_invoker.config >> \
    ${BUILD_DIR}/.test_data/required_ops_and_types.ort_models.config

# Test that we can convert an ONNX model with custom ops to ORT format
mkdir ${BUILD_DIR}/.test_data/custom_ops_model
cp onnxruntime/test/testdata/custom_op_library/*.onnx ${BUILD_DIR}/.test_data/custom_ops_model/
python3 tools/python/convert_onnx_models_to_ort.py \
    --custom_op_library ${BUILD_DIR}/Debug/libcustom_op_library.so \
    ${BUILD_DIR}/.test_data/custom_ops_model
rm -rf ${BUILD_DIR}/.test_data/custom_ops_model
