// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include "core/providers/openvino/ov_interface.h"

#define ORT_API_MANUAL_INIT
#include "core/session/onnxruntime_cxx_api.h"
#include "core/providers/shared_library/provider_api.h"
#include "core/providers/openvino/backend_utils.h"

// for make stateful utility function(s)
#include "core/providers/openvino/ov_stateful_patch_utils.h"

using Exception = ov::Exception;

namespace onnxruntime {
namespace openvino_ep {

static const std::string log_tag = "[OpenVINO-EP] ";

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
  try {
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
      ORT_THROW(log_tag + "[OpenVINO-EP] Unknown exception while Reading network");
    }
  } catch (const Exception& e) {
    ORT_THROW(log_tag + "[OpenVINO-EP] Exception while Reading network: " + std::string(e.what()));
  } catch (...) {
    ORT_THROW(log_tag + "[OpenVINO-EP] Unknown exception while Reading network");
  }
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
  bool status = IsStateful(model);
  std::cout << "IsStateful Status:\t" << status << std::endl;
  if (!status) {
    PatchStatefulDecoder(model);
  }

  if (onnxruntime::openvino_ep::backend_utils::IsDebugEnabled()) {
    std::cout << "Stateful OV Model Statistic:" << std::endl;
    LogBasicModelInfo(model);
  }

  auto kv_pos = GetKVAxesPos(model);
  if (onnxruntime::openvino_ep::backend_utils::IsDebugEnabled()) {
    std::cout << "kv_pos.batch = " << kv_pos.batch << std::endl;
    std::cout << "kv_pos.seq_len = " << kv_pos.seq_len << std::endl;
  }

  if (hw_target.find("NPU") != std::string::npos) {
    KVDesc kv_desc;
    kv_desc.max_prompt_len = PopIntAndCast(config, "MAX_PROMPT_LEN").value_or(1024u);
    kv_desc.min_response_len = PopIntAndCast(config, "MIN_RESPONSE_LEN").value_or(128u);

    if (onnxruntime::openvino_ep::backend_utils::IsDebugEnabled()) {
      std::cout << "kv_desc.max_prompt_len:\t" << kv_desc.max_prompt_len << std::endl;
      std::cout << "kv_desc.min_response_len:\t" << kv_desc.min_response_len << std::endl;
    }

    UpdateNPUConfig(config, kv_pos, kv_desc);
  } else {
    // This patches the model so that it only produces the logits required for sampling.
    // Actually either way that happens within NPUW::LLMCompiledModel creation, but this is
    // here mostly to align this behavior for other devices (CPU, GPU).
    ApplySliceBeforeMatmulTransformation(model);
  }

  std::cout << "Compiling Stateful OV Model ..." << std::endl;
  compiled_model = OVCore::Get()->core.compile_model(model, hw_target, config);
  std::cout << "Stateful OV Model Compilation Complete" << std::endl;

  OVExeNetwork exe(compiled_model);
  return exe;
}

OVExeNetwork OVCore::CompileModel(std::shared_ptr<const OVNetwork>& ie_cnn_network,
                                  std::string& hw_target,
                                  ov::AnyMap& device_config,
                                  bool enable_causallm,
                                  const std::string& name) {
  ov::CompiledModel obj;
  try {
    if (enable_causallm) {
      auto mutable_model = ie_cnn_network->clone();
      auto compiled_model = OVCore::Get()->StatefulCompileModel(mutable_model, hw_target, device_config);
      obj = compiled_model.Get();
    } else {
      obj = core.compile_model(ie_cnn_network, hw_target, device_config);
    }
#ifndef NDEBUG
    printDebugInfo(obj);
#endif
    OVExeNetwork exe(obj);
    return exe;
  } catch (const Exception& e) {
    ORT_THROW(log_tag + " Exception while Loading Network for graph: " + name + e.what());
  } catch (...) {
    ORT_THROW(log_tag + " Exception while Loading Network for graph " + name);
  }
}

OVExeNetwork OVCore::CompileModel(const std::string& onnx_model,
                                  std::string& hw_target,
                                  ov::AnyMap& device_config,
                                  const std::string& name) {
  ov::CompiledModel obj;
  try {
    obj = core.compile_model(onnx_model, ov::Tensor(), hw_target, device_config);
#ifndef NDEBUG
    printDebugInfo(obj);
#endif
    OVExeNetwork exe(obj);
    return exe;
  } catch (const Exception& e) {
    ORT_THROW(log_tag + " Exception while Loading Network for graph: " + name + e.what());
  } catch (...) {
    ORT_THROW(log_tag + " Exception while Loading Network for graph " + name);
  }
}

OVExeNetwork OVCore::ImportModel(std::istream& model_stream,
                                 std::string hw_target,
                                 const ov::AnyMap& device_config,
                                 bool enable_causallm,
                                 std::string name) {
  try {
    ov::CompiledModel obj;

    // Check if it's XML
    std::streampos originalPos = model_stream.tellg();
    // Allocate space for "<?xml"
    std::string header(5, '\0');
    model_stream.read(&header[0], 5);

    // Clear any read errors
    model_stream.clear();
    // Restore the stream position (important for reusing the stream)
    model_stream.seekg(originalPos);

    if (header != "<?xml") {
      obj = core.import_model(model_stream, hw_target, device_config);
    } else {
      // Get path to bin file
      std::string bin_file;
      if (name.size() >= 5 && name.substr(name.size() - 5) == ".onnx") {
        bin_file = name;
        bin_file.replace(name.size() - 5, 5, ".bin");
      } else {
        throw std::runtime_error("Invalid model name. Make sure *.onnx, *.xml, and *.bin carry the same name.");
      }

      // Read the model XML into a string
      std::stringstream xml_stream;
      xml_stream << model_stream.rdbuf();
      std::string xml_content = xml_stream.str();

      // Read model.bin into a vector
      std::ifstream bin_stream;
      bin_stream.open(bin_file, std::ios::binary);
      if (!bin_stream.is_open()) {
        throw std::runtime_error("Failed to open " + bin_file);
      }

      bin_stream.seekg(0, std::ios::end);
      std::streamsize size = bin_stream.tellg();
      bin_stream.seekg(0, std::ios::beg);
      std::vector<uint8_t> bin_data(size);
      if (!bin_stream.read(reinterpret_cast<char*>(bin_data.data()), size)) {
        throw std::runtime_error("Failed to read binary data from " + bin_file);
      }

      // Create an ov::Tensor for weights
      ov::Tensor weights_tensor(ov::element::u8, {bin_data.size()}, bin_data.data());

      // Load the model explicitly with XML content and weights
      std::shared_ptr<ov::Model> model = core.read_model(xml_content, weights_tensor);

      if (enable_causallm) {
        auto compiled_model = OVCore::Get()->StatefulCompileModel(model, hw_target, device_config);
        obj = compiled_model.Get();
      } else {
        obj = core.compile_model(model, hw_target, device_config);
      }
    }

#ifndef NDEBUG
    printDebugInfo(obj);
#endif
    OVExeNetwork exe(obj);
    return exe;
  } catch (const Exception& e) {
    ORT_THROW(log_tag + " Exception while Loading Network for graph: " + name + e.what());
  } catch (...) {
    ORT_THROW(log_tag + " Exception while Loading Network for graph " + name);
  }
}

void OVCore::SetCache(const std::string& cache_dir_path) {
  core.set_property(ov::cache_dir(cache_dir_path));
}

#ifdef IO_BUFFER_ENABLED
OVExeNetwork OVCore::CompileModel(std::shared_ptr<const OVNetwork>& model,
                                  OVRemoteContextPtr context, std::string name) {
  try {
    auto obj = core.compile_model(model, *context);
#ifndef NDEBUG
    printDebugInfo(obj);
#endif
    return OVExeNetwork(obj);
  } catch (const Exception& e) {
    ORT_THROW(log_tag + " Exception while Loading Network for graph: " + name + e.what());
  } catch (...) {
    ORT_THROW(log_tag + " Exception while Loading Network for graph " + name);
  }
}
OVExeNetwork OVCore::ImportModel(std::shared_ptr<std::istringstream> model_stream,
                                 OVRemoteContextPtr context, std::string name) {
  try {
    auto obj = core.import_model(*model_stream, *context);
#ifndef NDEBUG
    printDebugInfo(obj);
#endif
    OVExeNetwork exe(obj);
    return exe;
  } catch (const Exception& e) {
    ORT_THROW(log_tag + " Exception while Loading Network for graph: " + name + e.what());
  } catch (...) {
    ORT_THROW(log_tag + " Exception while Loading Network for graph " + name);
  }
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
  } catch (const std::runtime_error& ex) {
    // plugin is not created by e.g. invalid env
    // Empty device list will be returned
    ORT_THROW("[ERROR] [OpenVINO] An exception occurred while trying to create the ",
              device_type,
              " device: ",
              ex.what());
  } catch (const std::exception& ex) {
    ORT_THROW("[ERROR] [OpenVINO] An exception occurred while trying to create the ",
              device_type,
              " device: ",
              ex.what());
  } catch (...) {
    ORT_THROW("[ERROR] [OpenVINO] Unknown exception occurred while trying to create the ",
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
  try {
    auto infReq = obj.create_infer_request();
    OVInferRequest inf_obj(std::move(infReq));
    return inf_obj;
  } catch (const Exception& e) {
    ORT_THROW(log_tag + "Exception while creating InferRequest object: " + e.what());
  } catch (...) {
    ORT_THROW(log_tag + "Exception while creating InferRequest object.");
  }
}

OVTensorPtr OVInferRequest::GetTensor(const std::string& input_name) {
  try {
    auto tobj = ovInfReq.get_tensor(input_name);
    OVTensorPtr blob = std::make_shared<OVTensor>(tobj);
    return blob;
  } catch (const Exception& e) {
    ORT_THROW(log_tag + " Cannot access IE Blob for input: " + input_name + e.what());
  } catch (...) {
    ORT_THROW(log_tag + " Cannot access IE Blob for input: " + input_name);
  }
}

std::string OVInferRequest::GetInputTensorName(uint32_t index) {
  try {
    const auto& model = ovInfReq.get_compiled_model();
    return *model.input(index).get_names().begin();
  } catch (const Exception& e) {
    ORT_THROW(log_tag + " Cannot access IE Blob for input number: ", index, e.what());
  } catch (...) {
    ORT_THROW(log_tag + " Cannot access IE Blob for input number: ", index);
  }
}

void OVInferRequest::SetTensor(const std::string& name, OVTensorPtr& blob) {
  try {
    ovInfReq.set_tensor(name, *(blob.get()));

    if (name == "input_ids") {
      // Since we can't seem to set at ORT GenAI layer right now, we just set it here
      // as a workaround.
      // TODO: Fix this.
      ov::Tensor beam_idx = ov::Tensor(ov::element::i32, {1});
      std::fill_n(beam_idx.data<int32_t>(), 1, 0);
      ovInfReq.set_tensor("beam_idx", beam_idx);
    }

  } catch (const Exception& e) {
    ORT_THROW(log_tag + " Cannot set Remote Blob for output: " + name + e.what());
  } catch (...) {
    ORT_THROW(log_tag + " Cannot set Remote Blob for output: " + name);
  }
}

uint32_t OVInferRequest::GetNumInputs() {
  return static_cast<uint32_t>(ovInfReq.get_compiled_model().inputs().size());
}

void OVInferRequest::StartAsync() {
  try {
    ovInfReq.start_async();
  } catch (const Exception& e) {
    ORT_THROW(log_tag + " Couldn't start Inference: " + e.what());
  } catch (...) {
    ORT_THROW(log_tag + " In Error Couldn't start Inference");
  }
}

void OVInferRequest::Infer() {
  try {
    ovInfReq.infer();
  } catch (const Exception& e) {
    ORT_THROW(log_tag + " Couldn't start Inference: " + e.what());
  } catch (...) {
    ORT_THROW(log_tag + " In Error Couldn't start Inference");
  }
}

void OVInferRequest::WaitRequest() {
  try {
    ovInfReq.wait();
  } catch (const Exception& e) {
    ORT_THROW(log_tag + " Wait Model Failed: " + e.what());
  } catch (...) {
    ORT_THROW(log_tag + " Wait Mode Failed");
  }
}

void OVInferRequest::QueryStatus() {
  std::cout << "ovInfReq.query_state()"
            << " ";
}
}  // namespace openvino_ep
}  // namespace onnxruntime
