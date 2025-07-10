#include <filesystem>
#include <gsl/span>

#include "gtest/gtest.h"
#include "core/common/common.h"
#include "onnxruntime_cxx_api.h"
#include "api_asserts.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

extern std::unique_ptr<Ort::Env> ort_env;

struct OrtEpLibraryOv : public ::testing::Test {
  static const inline std::filesystem::path library_path =
#if _WIN32
      "onnxruntime_providers_openvino_plugin.dll";
#else
      "libonnxruntime_providers_openvino.so";
#endif
  static const inline std::string registration_name = "openvino_ep";

  void SetUp() override {
#ifndef _WIN32
    GTEST_SKIP() << "Skipping OpenVINO EP tests as the OpenVINO plugin is not built.";
#endif
    ort_env->RegisterExecutionProviderLibrary(registration_name.c_str(), library_path.c_str());
  }

  void TearDown() override {
#ifndef _WIN32
    GTEST_SKIP() << "Skipping OpenVINO EP tests as the OpenVINO plugin is not built.";
#endif
    ort_env->UnregisterExecutionProviderLibrary(registration_name.c_str());
  }

  void RunModelWithPluginEp(Ort::SessionOptions& session_options) {
    Ort::Session session(*ort_env, ORT_TSTR("testdata/mul_1.onnx"), session_options);
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    std::vector<int64_t> shape = {3, 2};
    std::vector<float> input0_data(6, 2.0f);
    std::vector<Ort::Value> ort_inputs;
    std::vector<const char*> ort_input_names;
    ort_inputs.emplace_back(Ort::Value::CreateTensor<float>(
        memory_info, input0_data.data(), input0_data.size(), shape.data(), shape.size()));
    ort_input_names.push_back("X");
    std::array<const char*, 1> output_names{"Y"};
    std::vector<Ort::Value> ort_outputs = session.Run(Ort::RunOptions{nullptr}, ort_input_names.data(), ort_inputs.data(),
                                                      ort_inputs.size(), output_names.data(), output_names.size());
    Ort::Value& ort_output = ort_outputs[0];
    const float* output_data = ort_output.GetTensorData<float>();
    gsl::span<const float> output_span(output_data, 6);
    EXPECT_THAT(output_span, ::testing::ElementsAre(2, 4, 6, 8, 10, 12));
  }
};

TEST_F(OrtEpLibraryOv, LoadUnloadPluginLibrary) {
  auto ep_devices = ort_env->GetEpDevices();
  auto test_ep_device = std::find_if(ep_devices.begin(), ep_devices.end(),
                                     [this](Ort::ConstEpDevice& device) {
                                       return std::string_view(device.EpName()).find(registration_name) != std::string::npos;
                                     });
  ASSERT_NE(test_ep_device, ep_devices.end()) << "Expected an OrtEpDevice to have been created by the test library.";
  ASSERT_STREQ(test_ep_device->EpVendor(), "Intel");
  Ort::ConstHardwareDevice device = test_ep_device->Device();
  ASSERT_EQ(device.Type(), OrtHardwareDeviceType_CPU);
  ASSERT_GE(device.VendorId(), 0);
  ASSERT_GE(device.DeviceId(), 0);
  ASSERT_NE(device.Vendor(), nullptr);
  Ort::ConstKeyValuePairs device_metadata = device.Metadata();
  std::unordered_map<std::string, std::string> metadata_entries = device_metadata.GetKeyValuePairs();
  ASSERT_GT(metadata_entries.size(), 0);
}

TEST_F(OrtEpLibraryOv, MetaDevicesAvailable) {
  auto ep_devices = ort_env->GetEpDevices();
  auto expected_meta_devices = {"AUTO"};

  for (auto& expected_meta_device : expected_meta_devices) {
    std::string expected_ep_name = registration_name + "." + expected_meta_device;
    auto it = std::find_if(ep_devices.begin(), ep_devices.end(),
                           [&](Ort::ConstEpDevice& device) {
                             return std::string_view(device.EpName()).find(expected_ep_name) != std::string::npos;
                           });
    bool meta_device_found = it != ep_devices.end();
    ASSERT_TRUE(meta_device_found) << "Expected to find " << expected_ep_name;
  }
}

TEST_F(OrtEpLibraryOv, PluginEp_AppendV2_MulInference) {
  auto ep_devices = ort_env->GetEpDevices();
  Ort::ConstEpDevice plugin_ep_device;
  for (Ort::ConstEpDevice& device : ep_devices) {
    if (std::string_view(device.EpName()).find(registration_name) != std::string::npos) {
      plugin_ep_device = device;
      break;
    }
  }

  ASSERT_NE(plugin_ep_device, nullptr);

  Ort::SessionOptions session_options;
  std::unordered_map<std::string, std::string> ep_options;
  session_options.AppendExecutionProvider_V2(*ort_env, std::vector<Ort::ConstEpDevice>{plugin_ep_device}, ep_options);
  RunModelWithPluginEp(session_options);
}

TEST_F(OrtEpLibraryOv, PluginEp_PreferCpu_MulInference) {
  Ort::SessionOptions session_options;
  session_options.SetEpSelectionPolicy(OrtExecutionProviderDevicePolicy_PREFER_CPU);
  RunModelWithPluginEp(session_options);
}
