// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_cxx_api.h"
#include "core/graph/constants.h"
#include "providers.h"
#include <memory>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <atomic>
#include <thread>
#include <gtest/gtest.h>
#include "test_allocator.h"
#include "../shared_lib/test_fixture.h"
#include <stdlib.h>

struct Input {
  const char* name = nullptr;
  std::vector<int64_t> dims;
  std::vector<float> values;
};

extern std::unique_ptr<Ort::Env> ort_env;
static constexpr PATH_TYPE MODEL_URI = TSTR("testdata/squeezenet/model.onnx");
class CApiTestGlobalThreadPoolsWithProvider : public testing::Test, public ::testing::WithParamInterface<int> {
};

template <typename OutT>
static void RunSession(OrtAllocator& allocator, Ort::Session& session_object,
                       std::vector<Input>& inputs,
                       const char* output_name,
                       const std::vector<int64_t>& dims_y,
                       const std::vector<OutT>& values_y,
                       Ort::Value* output_tensor) {
  std::vector<Ort::Value> ort_inputs;
  std::vector<const char*> input_names;
  for (size_t i = 0; i < inputs.size(); i++) {
    input_names.emplace_back(inputs[i].name);
    ort_inputs.emplace_back(Ort::Value::CreateTensor<float>(allocator.Info(&allocator), inputs[i].values.data(), inputs[i].values.size(), inputs[i].dims.data(), inputs[i].dims.size()));
  }

  std::vector<Ort::Value> ort_outputs;
  if (output_tensor)
    session_object.Run(Ort::RunOptions{nullptr}, input_names.data(), ort_inputs.data(), ort_inputs.size(), &output_name, output_tensor, 1);
  else {
    ort_outputs = session_object.Run(Ort::RunOptions{nullptr}, input_names.data(), ort_inputs.data(), ort_inputs.size(), &output_name, 1);
    ASSERT_EQ(ort_outputs.size(), 1u);
    output_tensor = &ort_outputs[0];
  }

  auto type_info = output_tensor->GetTensorTypeAndShapeInfo();
  ASSERT_EQ(type_info.GetShape(), dims_y);
  // size_t total_len = type_info.GetElementCount();
  ASSERT_EQ(values_y.size(), static_cast<size_t>(5));

// test inference is using onnxruntime_shared_lib_test_LIBS, so HasCudaEnvironment(800) isn't available
#ifdef USE_CUDA
  const float tolerance = 1e-5f;
#else
  const float tolerance = 1e-6f;
#endif
  OutT* f = output_tensor->GetTensorMutableData<OutT>();
  for (size_t i = 0; i != static_cast<size_t>(5); ++i) {
    ASSERT_NEAR(values_y[i], f[i], tolerance);
  }
}

template <typename T, typename OutT>
static Ort::Session GetSessionObj(Ort::Env& env, T model_uri, int provider_type) {
  Ort::SessionOptions session_options;
  session_options.DisablePerSessionThreads();

  if (provider_type == 1) {
#ifdef USE_CUDA
    OrtCUDAProviderOptionsV2* options;
    Ort::ThrowOnError(Ort::GetApi().CreateCUDAProviderOptions(&options));
    session_options.AppendExecutionProvider_CUDA_V2(*options);
    std::cout << "Running simple inference with cuda provider" << std::endl;
#else
    return Ort::Session(nullptr);
#endif
  } else if (provider_type == 2) {
#ifdef USE_DNNL
    OrtDnnlProviderOptions dnnl_options;
    dnnl_options.use_arena = 1;
    dnnl_options.threadpool_args = nullptr;
    session_options.AppendExecutionProvider_Dnnl(dnnl_options);
    // Ort::ThrowOnError(OrtApis::SessionOptionsAppendExecutionProvider_Dnnl(session_options, &dnnl_options));
    std::cout << "Running simple inference with dnnl provider" << std::endl;
#else
    return Ort::Session(nullptr);
#endif
  } else {
    std::cout << "Running simple inference with default provider" << std::endl;
  }

  // if session creation passes, model loads fine
  return Ort::Session(env, model_uri, session_options);
}

template <typename T, typename OutT>
static void TestInference(Ort::Session& session,
                          std::vector<Input>& inputs,
                          const char* output_name,
                          const std::vector<int64_t>& expected_dims_y,
                          const std::vector<OutT>& expected_values_y) {
  auto default_allocator = std::make_unique<MockedOrtAllocator>();
  Ort::Value value_y = Ort::Value::CreateTensor<float>(default_allocator.get(), expected_dims_y.data(), expected_dims_y.size());

  RunSession<OutT>(*default_allocator,
                   session,
                   inputs,
                   output_name,
                   expected_dims_y,
                   expected_values_y,
                   &value_y);
}

static void GetInputsAndExpectedOutputs(std::vector<Input>& inputs,
                                        std::vector<int64_t>& expected_dims_y,
                                        std::vector<float>& expected_values_y,
                                        std::string& output_name) {
  inputs.resize(1);
  Input& input = inputs.back();
  input.name = "data_0";
  input.dims = {1, 3, 224, 224};
  size_t input_tensor_size = 224 * 224 * 3;
  input.values.resize(input_tensor_size);
  auto& input_tensor_values = input.values;
  for (unsigned int i = 0; i < input_tensor_size; i++)
    input_tensor_values[i] = (float)i / (input_tensor_size + 1);

  // prepare expected inputs and outputs
  expected_dims_y = {1, 1000, 1, 1};
  // For this test I'm checking for the first 5 values only since the global thread pool change
  // doesn't affect the core op functionality
  expected_values_y = {0.000045f, 0.003846f, 0.000125f, 0.001180f, 0.001317f};

  output_name = "softmaxout_1";
}

// All tests below use global threadpools

// Test 1
// run inference on a model using just 1 session
TEST_P(CApiTestGlobalThreadPoolsWithProvider, simple) {
  // prepare inputs/outputs
  std::vector<Input> inputs;
  std::vector<int64_t> expected_dims_y;
  std::vector<float> expected_values_y;
  std::string output_name;
  GetInputsAndExpectedOutputs(inputs, expected_dims_y, expected_values_y, output_name);

  // create session
  Ort::Session session = GetSessionObj<PATH_TYPE, float>(*ort_env, MODEL_URI, GetParam());

  // run session
  if (session) {
    TestInference<PATH_TYPE, float>(session, inputs, output_name.c_str(), expected_dims_y, expected_values_y);
  }
}

static std::thread::id caller_tid = std::this_thread::get_id();
static std::atomic_bool atomic_wait{false};

void AsyncCallback(void* user_data, OrtValue** outputs, size_t num_outputs, OrtStatusPtr status_ptr) {
  const float* expected_result = reinterpret_cast<const float*>(user_data);
  auto callee_tid = std::this_thread::get_id();
  EXPECT_NE(caller_tid, callee_tid);
  Ort::Status status(status_ptr);
  EXPECT_TRUE(status.IsOK());
  EXPECT_EQ(num_outputs, 1UL);
  Ort::Value output_value(outputs[0]);
  EXPECT_NEAR(output_value.GetTensorData<float>()[1], expected_result[1], 0.001);
  output_value.release();
  atomic_wait.store(true);
}

TEST_P(CApiTestGlobalThreadPoolsWithProvider, simpleAsync) {
  Ort::Session session = GetSessionObj<PATH_TYPE, float>(*ort_env, MODEL_URI, GetParam());
  if (!session) {
    return;
  }

  std::vector<Input> inputs;
  std::vector<int64_t> expected_dims_y;
  std::vector<float> expected_values_y;
  std::string output_name;
  GetInputsAndExpectedOutputs(inputs, expected_dims_y, expected_values_y, output_name);

  auto allocator = std::make_unique<MockedOrtAllocator>();
  std::vector<Ort::Value> ort_inputs;
  std::vector<const char*> input_names;
  for (size_t i = 0; i < inputs.size(); i++) {
    input_names.emplace_back(inputs[i].name);
    ort_inputs.emplace_back(Ort::Value::CreateTensor<float>(allocator->Info(),
                                                            inputs[i].values.data(),
                                                            inputs[i].values.size(),
                                                            inputs[i].dims.data(),
                                                            inputs[i].dims.size()));
  }
  std::vector<const char*> output_names = {output_name.c_str()};
  std::vector<Ort::Value> ort_outputs;
  ort_outputs.emplace_back(Ort::Value{nullptr});

  atomic_wait.store(false);
  session.RunAsync(Ort::RunOptions{nullptr},
                   input_names.data(),
                   ort_inputs.data(),
                   ort_inputs.size(),
                   output_names.data(),
                   ort_outputs.data(),
                   1,
                   AsyncCallback,
                   expected_values_y.data());

  std::chrono::duration<double, std::milli> dur{100};
  // timeout in about 10 secs
  for (int i = 0; i < 100 && !atomic_wait.load(); ++i) {
    std::this_thread::sleep_for(dur);
  }
  EXPECT_EQ(atomic_wait.load(), true);
}

// Test 2
// run inference on the same model using 2 sessions
// destruct the 2 sessions only at the end
TEST_P(CApiTestGlobalThreadPoolsWithProvider, simple2) {
  // prepare inputs/outputs
  std::vector<Input> inputs;
  std::vector<int64_t> expected_dims_y;
  std::vector<float> expected_values_y;
  std::string output_name;
  GetInputsAndExpectedOutputs(inputs, expected_dims_y, expected_values_y, output_name);

  // create sessions
  Ort::Session session1 = GetSessionObj<PATH_TYPE, float>(*ort_env, MODEL_URI, GetParam());
  Ort::Session session2 = GetSessionObj<PATH_TYPE, float>(*ort_env, MODEL_URI, GetParam());

  // run session
  if (session1 && session2) {
    TestInference<PATH_TYPE, float>(session1, inputs, output_name.c_str(), expected_dims_y, expected_values_y);
    TestInference<PATH_TYPE, float>(session2, inputs, output_name.c_str(), expected_dims_y, expected_values_y);
  }
}

// Test 3
// run inference on the same model using 2 sessions
// one after another destructing first session first
// followed by second session
TEST_P(CApiTestGlobalThreadPoolsWithProvider, simple3) {
  // prepare inputs/outputs
  std::vector<Input> inputs;
  std::vector<int64_t> expected_dims_y;
  std::vector<float> expected_values_y;
  std::string output_name;
  GetInputsAndExpectedOutputs(inputs, expected_dims_y, expected_values_y, output_name);

  // first session
  {
    // create session
    Ort::Session session = GetSessionObj<PATH_TYPE, float>(*ort_env, MODEL_URI, GetParam());

    // run session
    if (session) {
      TestInference<PATH_TYPE, float>(session, inputs, output_name.c_str(), expected_dims_y, expected_values_y);
    }
  }

  // second session
  {
    // create session
    Ort::Session session = GetSessionObj<PATH_TYPE, float>(*ort_env, MODEL_URI, GetParam());

    // run session
    if (session) {
      TestInference<PATH_TYPE, float>(session, inputs, output_name.c_str(), expected_dims_y, expected_values_y);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(CApiTestGlobalThreadPoolsWithProviders,
                         CApiTestGlobalThreadPoolsWithProvider,
                         ::testing::Values(0, 1, 2, 3, 4));
