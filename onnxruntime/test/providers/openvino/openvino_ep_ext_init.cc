// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <filesystem>
#include <map>
#include <string>
#include <optional>

#include "core/session/onnxruntime_cxx_api.h"

#include "test/util/include/test/test_environment.h"
#include "test/unittest_util/qdq_test_utils.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "onnxruntime_session_options_config_keys.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::logging;

extern std::unique_ptr<Ort::Env> ort_env;

namespace {

std::optional<std::vector<uint8_t>> LoadFileToMemory(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    return std::nullopt;
  }
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<uint8_t> buffer(static_cast<size_t>(size));
  if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
    return std::nullopt;
  }
  return buffer;
}

auto ProbeDevice(const std::string& device) {
  static std::map<std::string, bool> is_present;
  if (is_present.find(device) == is_present.end()) {
    Ort::SessionOptions sessionOptions;
    std::unordered_map<std::string, std::string> ov_options;
    ov_options["device_type"] = device;
    try {
      sessionOptions.AppendExecutionProvider_OpenVINO_V2(ov_options);
      is_present[device] = true;
    } catch (...) {
      is_present[device] = false;
    }
  }
  return is_present[device];
}

}  // namespace

namespace onnxruntime {
namespace test {

class OVEP_ExtInit_Tests : public ::testing::TestWithParam<std::string> {
 public:
  static void SetUpTestSuite() {
    // 1. Create initializers
    initializer_data_.reserve(num_initializers_);
    for (size_t i = 0; i < num_initializers_; ++i)
      initializer_data_.emplace_back(floats_per_initializer_, static_cast<float>(i + 1));  // W0:1, W1:2...

    // 2. Build ONNX model with external initializers, and ADD nodes
    {
      ModelProto model_proto;
      model_proto.set_ir_version(7);
      model_proto.set_producer_name("openvino_extinit_test");
      model_proto.set_producer_version("1.0");
      model_proto.set_domain("");
      model_proto.set_model_version(1);

      auto* graph = model_proto.mutable_graph();
      graph->set_name("TestGraph");

      // Input: shape [floats_per_initializer]
      auto* input = graph->add_input();
      input->set_name("X");
      auto* input_type = input->mutable_type()->mutable_tensor_type();
      input_type->set_elem_type(TensorProto_DataType_FLOAT);
      input_type->mutable_shape()->add_dim()->set_dim_value(floats_per_initializer_);

      // Output: shape [floats_per_initializer]
      auto* output = graph->add_output();
      output->set_name("Y");
      auto* output_type = output->mutable_type()->mutable_tensor_type();
      output_type->set_elem_type(TensorProto_DataType_FLOAT);
      output_type->mutable_shape()->add_dim()->set_dim_value(floats_per_initializer_);

      auto* opset_import = model_proto.add_opset_import();
      opset_import->set_domain("");
      opset_import->set_version(19);

      // Add initializers as external data
      size_t offset = 0;
      std::vector<std::string> initializer_names;
      initializer_names.reserve(num_initializers_);
      for (size_t i = 0; i < num_initializers_; ++i) {
        std::string name = "W" + std::to_string(i);
        initializer_names.push_back(name);
        TensorProto* initializer = graph->add_initializer();
        initializer->set_name(name);
        initializer->set_data_type(TensorProto_DataType_FLOAT);
        initializer->add_dims(floats_per_initializer_);
        initializer->set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL);
        auto* ext = initializer->add_external_data();
        ext->set_key("location");
        ext->set_value(weights_path_);
        ext = initializer->add_external_data();
        ext->set_key("offset");
        ext->set_value(std::to_string(offset));
        ext = initializer->add_external_data();
        ext->set_key("length");
        ext->set_value(std::to_string(floats_per_initializer_ * sizeof(float)));
        offset += floats_per_initializer_ * sizeof(float);
      }

      // nodes: X -> Add with Init[0] -> ... -> output Y
      std::string prev_output = "X";
      std::string node_output;
      for (size_t i = 0; i < num_initializers_; ++i) {
        node_output = (i == num_initializers_ - 1) ? "Y" : "A" + std::to_string(i);
        auto* add_node = graph->add_node();
        add_node->set_op_type("Add");
        add_node->add_input(prev_output);
        add_node->add_input(initializer_names[i]);
        add_node->add_output(node_output);
        prev_output = node_output;
      }

      // Save model
      std::ofstream model_file(model_path_, std::ios::binary);
      ASSERT_TRUE(model_file.is_open()) << "Failed to open model file";
      ASSERT_TRUE(model_proto.SerializeToOstream(&model_file)) << "Failed to serialize model";
    }

    // 3. Save weights file (concatenate all initializers)
    {
      std::ofstream weights_file(weights_path_, std::ios::binary);
      ASSERT_TRUE(weights_file.is_open()) << "Failed to open weights file";
      for (const auto& w : initializer_data_) {
        weights_file.write(reinterpret_cast<const char*>(w.data()), w.size() * sizeof(float));
      }
      ASSERT_TRUE(weights_file.good()) << "Failed to write all weights to file";
    }

    // 4. Load model and weights into memory (once for all tests)
    model_data_ = LoadFileToMemory(model_path_);
    weights_data_ = LoadFileToMemory(weights_path_);
    ASSERT_TRUE(model_data_.has_value()) << "Failed to load model into memory";
    ASSERT_TRUE(weights_data_.has_value()) << "Failed to load weights into memory";
  }

  static void TearDownTestSuite() {
    // Cleanup files and release memory
    std::filesystem::remove(model_path_);
    std::filesystem::remove(weights_path_);

    // Release memory
    model_data_.reset();
    weights_data_.reset();
    initializer_data_.clear();
  }

 protected:
  inline static constexpr const char* model_path_ = "ovep_ext_init_test.onnx";
  inline static constexpr const char* weights_path_ = "ovep_ext_init_test.onnx.data";
  inline static constexpr size_t num_initializers_ = 8;
  inline static constexpr size_t floats_per_initializer_ = 64 * 1024 * 1024;  // 64 million floats per initializer, 256MB
  inline static std::vector<std::vector<float>> initializer_data_;
  inline static std::optional<std::vector<uint8_t>> model_data_;
  inline static std::optional<std::vector<uint8_t>> weights_data_;
};

TEST_P(OVEP_ExtInit_Tests, ModelFromExtInit) {
  const auto& device = GetParam();
  if (!ProbeDevice(device))
    GTEST_SKIP() << device + " is not available on this machine";

  // 5. Prepare external initializer info
  std::string temp_name(weights_path_);
  PathString weights_name_path(temp_name.begin(), temp_name.end());
  std::vector<PathString> names_path = {weights_name_path};
  std::vector<char*> buffers = {reinterpret_cast<char*>(weights_data_.value().data())};
  std::vector<size_t> buffer_sizes = {weights_data_.value().size()};

  // 6. Set up session options with OpenVINO
  Ort::SessionOptions session_options;
  session_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1");
  session_options.SetIntraOpNumThreads(1);
  std::unordered_map<std::string, std::string> ov_options = {{"device_type", device}};
  session_options.AppendExecutionProvider_OpenVINO_V2(ov_options);
  session_options.AddExternalInitializersFromFilesInMemory(names_path, buffers, buffer_sizes);

  // 7. Create session from memory
  Ort::Session session(*ort_env, model_data_.value().data(), model_data_.value().size(), session_options);

  // 8. Run inference to verify weights are loaded
  std::vector<float> input_data(floats_per_initializer_, 2.0f);
  std::vector<int64_t> input_shape = {static_cast<int64_t>(floats_per_initializer_)};
  Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(mem_info, input_data.data(), input_data.size(), input_shape.data(), input_shape.size());

  std::vector<const char*> input_names = {"X"};
  std::vector<const char*> output_names = {"Y"};
  std::vector<Ort::Value> output_tensors(1);

  session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(), output_tensors.data(), 1);

  // Check output: should be input + W0 + W1 + W2...
  auto* out_data = output_tensors[0].GetTensorMutableData<float>();
  float expected = input_data[0];
  for (size_t i = 0; i < num_initializers_; ++i) {
    expected += initializer_data_[i][0];
  }

  for (size_t i = 0; i < floats_per_initializer_; ++i)
    ASSERT_FLOAT_EQ(out_data[i], expected);
}

INSTANTIATE_TEST_SUITE_P(OVEP_Tests,
                         OVEP_ExtInit_Tests,
                         ::testing::Values("CPU", "GPU", "NPU"));

}  // namespace test
}  // namespace onnxruntime
