// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include "core/providers/openvino/ov_interface.h"

#define ORT_API_MANUAL_INIT
#include "core/session/onnxruntime_cxx_api.h"
#include "core/providers/shared_library/provider_api.h"
#include "core/providers/openvino/backend_utils.h"
#include <format>

namespace onnxruntime {
namespace openvino_ep {

template <typename Func, typename... Args>
inline auto OvExeceptionBoundary(Func func, std::format_string<Args...>&& fmt, Args&&... args) {
  try {
    return func();
  } catch (const ov::Exception& e) {
    ORT_THROW(log_tag + std::vformat(fmt.get(), std::make_format_args(args...)) + ": " + std::string(e.what()));
  } catch (...) {
    ORT_THROW(log_tag + std::vformat(fmt.get(), std::make_format_args(args...)));
  }
}

#ifndef NDEBUG
void printDebugInfo(const ov::CompiledModel& obj) {
  if (onnxruntime::openvino_ep::backend_utils::IsDebugEnabled()) {
    // output of the actual settings that the device selected
    auto supported_properties = obj.get_property(ov::supported_properties);
    std::cout << "Model:" << std::endl;
    for (const auto& cfg : supported_properties) {
      if (cfg == ov::supported_properties)
        continue;
      auto prop = obj.get_property(cfg);
      if (cfg == ov::device::properties) {
        auto devices_properties = prop.as<ov::AnyMap>();
        for (auto& item : devices_properties) {
          std::cout << "  " << item.first << ": " << std::endl;
          for (auto& item2 : item.second.as<ov::AnyMap>()) {
            OPENVINO_SUPPRESS_DEPRECATED_START
            if (item2.first == ov::supported_properties || item2.first == "SUPPORTED_CONFIG_KEYS)" ||
                item2.first == "SUPPORTED_METRICS")
              continue;
            OPENVINO_SUPPRESS_DEPRECATED_END
            std::cout << "    " << item2.first << ": " << item2.second.as<std::string>() << std::endl;
          }
        }
      } else {
        std::cout << "  " << cfg << ": " << prop.as<std::string>() << std::endl;
      }
    }
  }
}
#endif

// Function to check if a given OV property is enabled
std::optional<bool> queryOVProperty(const std::string& property, const std::string& device_type) {
  try {
    // Get the property value
    auto supported_properties = OVCore::Get()->core.get_property(device_type, ov::supported_properties);
    return std::find(supported_properties.begin(), supported_properties.end(), property) != supported_properties.end();
  } catch (const std::exception&) {
    return std::nullopt;  // Property not found or invalid
  }
}

std::shared_ptr<OVNetwork> OVCore::ReadModel(std::string&& model, const std::string& model_path) {
  return OvExeceptionBoundary([&]() {
    std::istringstream modelStringStream(std::move(model));
    std::istream& modelStream = modelStringStream;
    // Try to load with FrontEndManager
    ov::frontend::FrontEndManager manager;
    ov::frontend::FrontEnd::Ptr FE;
    ov::frontend::InputModel::Ptr inputModel;

    ov::AnyVector params{&modelStream, model_path};

    FE = manager.load_by_model(params);
    if (FE) {
      inputModel = FE->load(params);
      return FE->convert(inputModel);
    } else {
      ORT_THROW(log_tag + "Unknown exception while Reading network");
    }
  },
                              "Exception while Reading network");
}

OVExeNetwork OVCore::StatefulCompileModel(std::shared_ptr<OVNetwork>& model,
                                          std::string& hw_target,
                                          const ov::AnyMap& device_config) {
  ov::CompiledModel compiled_model;
  ov::AnyMap config = device_config;

  if (onnxruntime::openvino_ep::backend_utils::IsDebugEnabled()) {
    std::cout << "Stateless OV Model Statistic:" << std::endl;
    LogBasicModelInfo(model);
  }

  LOGS_DEFAULT(INFO) << log_tag << "Converting from Stateless OV Model to Stateful OV Model" << std::endl;
  bool model_status = IsStateful(model);
  LOGS_DEFAULT(INFO) << log_tag << "Model IsStateful() Status:\t" << (model_status ? "True" : "False");
  if (!model_status) {
    PatchStatefulDecoder(model);
  }

  if (onnxruntime::openvino_ep::backend_utils::IsDebugEnabled()) {
    std::cout << "Stateful OV Model Statistic:" << std::endl;
    LogBasicModelInfo(model);
  }

  auto kv_pos = GetKVAxesPos(model);

  if (hw_target.find("NPU") != std::string::npos) {
    KVDesc kv_desc;
    auto parse_genai_config = [&](const std::string& key, unsigned int default_value) {
      return (config.count(key) && !config.at(key).empty() && config.at(key).as<std::string>() != "0") ? config.at(key).as<unsigned int>() : default_value;
    };

    kv_desc.max_prompt_len = parse_genai_config("MAX_PROMPT_LEN", CausalLMConfig().max_prompt_len);
    kv_desc.min_response_len = parse_genai_config("MIN_RESPONSE_LEN", CausalLMConfig().min_response_len);

    // For compilation, MAX_PROMPT_LEN & MIN_RESPONSE_LEN should not be 0
    if (kv_desc.max_prompt_len == 0 || kv_desc.min_response_len == 0) {
      ORT_THROW(log_tag + "MAX_PROMPT_LEN and MIN_RESPONSE_LEN cannot be 0 or empty");
    }

    if (onnxruntime::openvino_ep::backend_utils::IsDebugEnabled()) {
      std::cout << "kv_pos.batch = " << kv_pos.batch << std::endl;
      std::cout << "kv_pos.seq_len = " << kv_pos.seq_len << std::endl;
      std::cout << "kv_desc.max_prompt_len:\t" << kv_desc.max_prompt_len << std::endl;
      std::cout << "kv_desc.min_response_len:\t" << kv_desc.min_response_len << std::endl;
    }

    UpdateNPUConfig(config, kv_pos, kv_desc);
  } else {
    // This patches the OV IR model so that it only produces the logits required for sampling.
    // Actually either way that happens within NPUW::LLMCompiledModel creation for NPU device,
    // while this is here mostly to align this behavior for other devices viz. (CPU, GPU).
    ApplySliceBeforeMatmulTransformation(model);
  }

  LOGS_DEFAULT(INFO) << log_tag << "Compiling OV Model using Stateful Transformation flow";
  compiled_model = OVCore::Get()->core.compile_model(model, hw_target, config);
  OVExeNetwork exe(compiled_model, hw_target, true);
  return exe;
}

OVExeNetwork OVCore::CompileModel(std::shared_ptr<const OVNetwork>& ie_cnn_network,
                                  std::string& hw_target,
                                  ov::AnyMap& device_config,
                                  bool enable_causallm,
                                  const std::string& name) {
  return OvExeceptionBoundary([&]() {
    ov::CompiledModel obj;
    obj = core.compile_model(ie_cnn_network, hw_target, device_config);
#ifndef NDEBUG
    printDebugInfo(exe.Get());
#endif

    return exe;
  },
                              "Exception while Loading Network for graph");
}

OVExeNetwork OVCore::CompileModel(const std::string& onnx_model,
                                  std::string& hw_target,
                                  ov::AnyMap& device_config,
                                  const std::string& name) {
  return OvExeceptionBoundary([&]() {
    ov::CompiledModel obj;

    obj = core.compile_model(onnx_model, ov::Tensor(), hw_target, device_config);
#ifndef NDEBUG
    printDebugInfo(obj);
#endif
    OVExeNetwork exe(obj, hw_target);
    return exe;
  },
                              "Exception while Loading Network for graph");
}

OVExeNetwork OVCore::ImportModel(std::istream& model_stream,
                                 std::string hw_target,
                                 const ov::AnyMap& device_config,
                                 std::string name) {
  return OvExeceptionBoundary([&]() {
    ov::CompiledModel obj;
    obj = core.import_model(model_stream, hw_target, device_config);
#ifndef NDEBUG
    printDebugInfo(exe.Get());
#endif
    OVExeNetwork exe(obj, hw_target);
    return exe;
  },
                              "Exception while Loading Network for graph");
}

void OVCore::SetCache(const std::string& cache_dir_path) {
  core.set_property(ov::cache_dir(cache_dir_path));
}

#ifdef IO_BUFFER_ENABLED
OVExeNetwork OVCore::CompileModel(std::shared_ptr<const OVNetwork>& model,
                                  OVRemoteContextPtr context, std::string name) {
  OvExeceptionBoundary([&]() {
    auto obj = core.compile_model(model, *context);
#ifndef NDEBUG
    printDebugInfo(obj);
#endif
    return OVExeNetwork(obj);
  },
                       "Exception while Loading Network for graph: {}", name);
}
OVExeNetwork OVCore::ImportModel(std::shared_ptr<std::istringstream> model_stream,
                                 OVRemoteContextPtr context, std::string name) {
  return OvExeceptionBoundary([&]() {
    auto obj = core.import_model(*model_stream, *context);
#ifndef NDEBUG
    printDebugInfo(obj);
#endif
    OVExeNetwork exe(obj);
    return exe;
  },
                              "Exception while Loading Network for graph: {}", name);
}
#endif

std::vector<std::string> OVCore::GetAvailableDevices() const {
  std::vector<std::string> available_devices = core.get_available_devices();
  return available_devices;
}

std::vector<std::string> OVCore::GetAvailableDevices(const std::string& device_type) const {
  std::vector<std::string> available_devices;
  std::vector<std::string> devicesIDs;
  // Uses logic from OpenVINO to only return available devices of the specified type (e.g. CPU, NPU or GPU)
  try {
    devicesIDs = core.get_property(device_type, ov::available_devices);
  } catch (const ov::Exception&) {
    // plugin is not created by e.g. invalid env
    // Empty device list will be returned
  } catch (const std::exception& ex) {
    ORT_THROW(log_tag + "An exception occurred while trying to create the ",
              device_type,
              " device: ",
              ex.what());
  } catch (...) {
    ORT_THROW(log_tag + "Unknown exception occurred while trying to create the ",
              device_type,
              " device");
  }

  if (devicesIDs.size() > 1 ||
      (devicesIDs.size() == 1 && devicesIDs[0] == "0")) {
    for (const auto& deviceID : devicesIDs) {
      available_devices.push_back(device_type + '.' + deviceID);
    }
  }
  if (!devicesIDs.empty()) {
    available_devices.push_back(device_type);
  }

  return available_devices;
}

void OVCore::SetStreams(const std::string& device_type, int num_streams) {
  core.set_property(device_type, {ov::num_streams(num_streams)});
}

OVInferRequest OVExeNetwork::CreateInferRequest() {
  return OvExeceptionBoundary([&]() {
    auto infReq = obj.create_infer_request();
    OVInferRequest inf_obj(std::move(infReq));
    return inf_obj;
  },
                              "Exception while creating InferRequest object");
}

OVTensorPtr OVInferRequest::GetTensor(const std::string& input_name) {
  return OvExeceptionBoundary([&]() {
    auto tobj = ovInfReq.get_tensor(input_name);
    OVTensorPtr blob = std::make_shared<OVTensor>(tobj);
    return blob;
  },
                              " Cannot access IE Blob for input: {}", input_name);
}

std::string OVInferRequest::GetInputTensorName(uint32_t index) {
  return OvExeceptionBoundary([&]() {
    const auto& model = ovInfReq.get_compiled_model();
    return *model.input(index).get_names().begin();
  },
                              " Cannot access IE Blob for input number: {}", index);
}

void OVInferRequest::SetTensor(const std::string& name, OVTensorPtr& blob) {
  OvExeceptionBoundary([&]() {
    ovInfReq.set_tensor(name, *(blob.get()));
  },
                       " Cannot set Remote Blob for output: {}", name);
}

uint32_t OVInferRequest::GetNumInputs() {
  return static_cast<uint32_t>(ovInfReq.get_compiled_model().inputs().size());
}

void OVInferRequest::Infer() {
  OvExeceptionBoundary([&]() {
    ovInfReq.infer();
  },
                       "In Error Couldn't start Inference");
}

}  // namespace openvino_ep
}  // namespace onnxruntime
