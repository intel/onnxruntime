// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.



#include <filesystem>
#include <string>
#include "core/framework/provider_options.h"
#include "core/framework/tensor_shape.h"
#include "core/graph/model.h"
#include "core/common/logging/logging.h"
#include "test/util/include/test_utils.h"
#include "test/util/include/asserts.h"
#include "test/util/include/test/test_environment.h"
#include "test/util/include/default_providers.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/session/inference_session.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "test/unittest_util/qdq_test_utils.h"


using namespace ONNX_NAMESPACE;
using namespace onnxruntime::logging;

extern std::unique_ptr<Ort::Env> ort_env;

class OVEPWorkloadTypeTests : public ::testing::Test {
protected:
  // Helper function to check if NPU is available
  static bool IsNPUAvailable() {
    try {
      Ort::SessionOptions test_options;
      std::unordered_map<std::string, std::string> ov_options;
      ov_options["device_type"] = "NPU";
      test_options.AppendExecutionProvider_OpenVINO_V2(ov_options);
      return true;
    } catch (...) {
      return false;
    }
  }
};

namespace onnxruntime {
namespace test {

// Test: SetEpDynamicOptions with workload_type transitions should not error
// baseline -> Efficient -> Default.
TEST_F(OVEPWorkloadTypeTests, OVEPWorkloadTypeDynamicSwitch) {
  // Skip test if NPU is not available
  if (!IsNPUAvailable()) {
    GTEST_SKIP() << "NPU device not available, skipping workload type test";
  }

  Ort::SessionOptions session_options;
  std::unordered_map<std::string, std::string> ov_options;
  ov_options["device_type"] = "NPU";

  const std::unordered_map<std::string, int> domain_to_version = {{"", 13}};
  auto& logging_manager = DefaultLoggingManager();
  logging_manager.SetDefaultLoggerSeverity(logging::Severity::kERROR);

  onnxruntime::Model model("WorkloadType_Test_Model", false, ModelMetaData(),
                           PathString(), IOnnxRuntimeOpSchemaRegistryList(),
                           domain_to_version, {},
                           logging_manager.DefaultLogger());

  auto& graph = model.MainGraph();

  // Input: X [1, 3, 2, 2] float
  ONNX_NAMESPACE::TypeProto float_tensor;
  float_tensor.mutable_tensor_type()->set_elem_type(
      ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  auto* shape = float_tensor.mutable_tensor_type()->mutable_shape();
  shape->add_dim()->set_dim_value(1);
  shape->add_dim()->set_dim_value(3);
  shape->add_dim()->set_dim_value(2);
  shape->add_dim()->set_dim_value(2);

  auto& input_arg = graph.GetOrCreateNodeArg("X", &float_tensor);
  auto& output_arg = graph.GetOrCreateNodeArg("Y", &float_tensor);

  // Constant initializer: scalar 2.0
  ONNX_NAMESPACE::TensorProto multiplier;
  multiplier.set_name("Multiplier");
  multiplier.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  multiplier.add_dims(1);
  multiplier.add_float_data(2.0f);
  graph.AddInitializedTensor(multiplier);

  auto& multiplier_arg = graph.GetOrCreateNodeArg("Multiplier", nullptr);

  graph.AddNode("mul_node", "Mul", "Multiply by 2",
                {&input_arg, &multiplier_arg}, {&output_arg});
  graph.SetInputs({&input_arg});
  graph.SetOutputs({&output_arg});

  ASSERT_STATUS_OK(graph.Resolve());

  // Serialize the model to a string.
  std::string model_data;
  model.ToProto().SerializeToString(&model_data);
  const auto model_data_span = AsByteSpan(model_data.data(), model_data.size());

  session_options.SetIntraOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);
  session_options.AppendExecutionProvider_OpenVINO_V2(ov_options);

  Ort::Session session(*ort_env, model_data_span.data(),
                       model_data_span.size(), session_options);

  // Prepare input: 12 floats (shape 1x3x2x2) all set to 1.0
  Ort::AllocatorWithDefaultOptions allocator;
  std::string input_name = session.GetInputNameAllocated(0, allocator).get();
  std::string output_name = session.GetOutputNameAllocated(0, allocator).get();
  const char* input_names[] = {input_name.c_str()};
  const char* output_names[] = {output_name.c_str()};

  std::vector<int64_t> input_shape = {1, 3, 2, 2};
  std::vector<float> input_values(12, 1.0f);
  Ort::MemoryInfo mem_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  auto run_and_verify = [&](const std::string& phase_label) {
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info, input_values.data(), input_values.size(),
        input_shape.data(), input_shape.size());
    auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names,
                               &input_tensor, 1, output_names, 1);
    ASSERT_EQ(outputs.size(), 1u) << phase_label;
    const float* output_data = outputs[0].GetTensorData<float>();
    size_t num_elements =
        outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    ASSERT_EQ(num_elements, 12u) << phase_label;
    for (size_t i = 0; i < num_elements; ++i) {
      EXPECT_NEAR(output_data[i], 2.0f, 1e-5f) << phase_label << " index " << i;
    }
  };

  const char* const keys[] = {"ep.dynamic.workload_type"};

  // Phase 1: Baseline (no workload type set)
  run_and_verify("Baseline");

  // Phase 2: Efficient
  const char* const eff_val[] = {"Efficient"};
  session.SetEpDynamicOptions(keys, eff_val, 1);
  run_and_verify("Efficient");

  // Phase 3: Default
  const char* const def_val[] = {"Default"};
  session.SetEpDynamicOptions(keys, def_val, 1);
  run_and_verify("Default");
}

// Test: Multiple inferences per workload mode
// This validates sustained correctness under each workload type and ensures
// no degradation or resource leaks across multiple inferences.
TEST_F(OVEPWorkloadTypeTests, OVEPWorkloadTypeMultipleInferencesPerMode) {
  // Skip test if NPU is not available
  if (!IsNPUAvailable()) {
    GTEST_SKIP() << "NPU device not available, skipping workload type test";
  }

  Ort::SessionOptions session_options;
  std::unordered_map<std::string, std::string> ov_options;
  ov_options["device_type"] = "NPU";

  const std::unordered_map<std::string, int> domain_to_version = {{"", 13}};
  auto& logging_manager = DefaultLoggingManager();
  logging_manager.SetDefaultLoggerSeverity(logging::Severity::kERROR);

  onnxruntime::Model model("WorkloadType_MultiRun_Model", false, ModelMetaData(),
                           PathString(), IOnnxRuntimeOpSchemaRegistryList(),
                           domain_to_version, {},
                           logging_manager.DefaultLogger());

  auto& graph = model.MainGraph();

  ONNX_NAMESPACE::TypeProto float_tensor;
  float_tensor.mutable_tensor_type()->set_elem_type(
      ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  auto* shape = float_tensor.mutable_tensor_type()->mutable_shape();
  shape->add_dim()->set_dim_value(1);
  shape->add_dim()->set_dim_value(3);
  shape->add_dim()->set_dim_value(2);
  shape->add_dim()->set_dim_value(2);

  auto& input_arg = graph.GetOrCreateNodeArg("X", &float_tensor);
  auto& output_arg = graph.GetOrCreateNodeArg("Y", &float_tensor);

  ONNX_NAMESPACE::TensorProto multiplier;
  multiplier.set_name("Multiplier");
  multiplier.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  multiplier.add_dims(1);
  multiplier.add_float_data(2.0f);
  graph.AddInitializedTensor(multiplier);

  auto& multiplier_arg = graph.GetOrCreateNodeArg("Multiplier", nullptr);

  graph.AddNode("mul_node", "Mul", "Multiply by 2",
                {&input_arg, &multiplier_arg}, {&output_arg});
  graph.SetInputs({&input_arg});
  graph.SetOutputs({&output_arg});

  ASSERT_STATUS_OK(graph.Resolve());

  std::string model_data;
  model.ToProto().SerializeToString(&model_data);
  const auto model_data_span = AsByteSpan(model_data.data(), model_data.size());

  session_options.SetIntraOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);
  session_options.AppendExecutionProvider_OpenVINO_V2(ov_options);

  Ort::Session session(*ort_env, model_data_span.data(),
                       model_data_span.size(), session_options);

  Ort::AllocatorWithDefaultOptions allocator;
  std::string input_name = session.GetInputNameAllocated(0, allocator).get();
  std::string output_name = session.GetOutputNameAllocated(0, allocator).get();
  const char* input_names[] = {input_name.c_str()};
  const char* output_names[] = {output_name.c_str()};

  std::vector<int64_t> input_shape = {1, 3, 2, 2};
  std::vector<float> input_values(12, 1.0f);
  Ort::MemoryInfo mem_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  auto run_and_verify = [&](const std::string& phase_label) {
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info, input_values.data(), input_values.size(),
        input_shape.data(), input_shape.size());
    auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names,
                               &input_tensor, 1, output_names, 1);
    ASSERT_EQ(outputs.size(), 1u) << phase_label;
    const float* output_data = outputs[0].GetTensorData<float>();
    size_t num_elements =
        outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    ASSERT_EQ(num_elements, 12u) << phase_label;
    for (size_t i = 0; i < num_elements; ++i) {
      EXPECT_NEAR(output_data[i], 2.0f, 1e-5f) << phase_label << " index " << i;
    }
  };

  const char* const keys[] = {"ep.dynamic.workload_type"};
  const char* const eff_val[] = {"Efficient"};
  const char* const def_val[] = {"Default"};

  constexpr int kIterationsPerMode = 10;

  // Phase 1: Baseline - 10 runs without workload type
  for (int i = 0; i < kIterationsPerMode; ++i) {
    run_and_verify("Baseline iter " + std::to_string(i));
  }

  // Phase 2: Efficient - 10 runs with Efficient workload type
  session.SetEpDynamicOptions(keys, eff_val, 1);
  for (int i = 0; i < kIterationsPerMode; ++i) {
    run_and_verify("Efficient iter " + std::to_string(i));
  }

  // Phase 3: Default - 10 runs with Default workload type
  session.SetEpDynamicOptions(keys, def_val, 1);
  for (int i = 0; i < kIterationsPerMode; ++i) {
    run_and_verify("Default iter " + std::to_string(i));
  }
}

}  // namespace test
}  // namespace onnxruntime
