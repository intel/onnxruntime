// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include "core/providers/openvino/ov_interface.h"

#define ORT_API_MANUAL_INIT
#include "core/session/onnxruntime_cxx_api.h"
#include "core/providers/shared_library/provider_api.h"
#include "core/providers/openvino/backend_utils.h"
#include "core/providers/openvino/backends/basic_backend.h"
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
      return (config.count(key) && !config.at(key).empty() && config.at(key).as<std::string>() != "0") ?
         config.at(key).as<unsigned int>() : default_value;
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
    // This patches the model so that it only produces the logits required for sampling.
    // Actually either way that happens within NPUW::LLMCompiledModel creation, but this is
    // here mostly to align this behavior for other devices (CPU, GPU).
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
  OVExeNetwork exe;
  try {
    if (enable_causallm) {
      auto mutable_model = ie_cnn_network->clone();
      exe = OVCore::Get()->StatefulCompileModel(mutable_model, hw_target, device_config);
    } else {
      auto obj = core.compile_model(ie_cnn_network, hw_target, device_config);
      exe = OVExeNetwork(obj, hw_target);
    }

#ifndef NDEBUG
    printDebugInfo(exe.Get());
#endif

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
    OVExeNetwork exe(obj, hw_target);
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
    OVExeNetwork exe;

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
      auto obj = core.import_model(model_stream, hw_target, device_config);
      exe = OVExeNetwork(obj, hw_target);
    } else {
      // If the model is XML, we need to load it with the XML content in read_model()
      // where weights from bin file is directly consumed
      std::string xml_file_name = name;
      if (name.size() >= 5 && name.substr(name.size() - 5) == ".onnx") {
        xml_file_name = name;
        xml_file_name.replace(name.size() - 5, 5, ".xml");
      } else {
        throw std::runtime_error("Invalid model name. Make sure *.onnx, *.xml, and *.bin carry the same name.");
      }

      // Load the model explicitly with XML contents
      std::shared_ptr<ov::Model> model = core.read_model(xml_file_name);

      if (enable_causallm) {
        exe = OVCore::Get()->StatefulCompileModel(model, hw_target, device_config);
      } else {
        auto obj = core.compile_model(model, hw_target, device_config);
        exe = OVExeNetwork(obj, hw_target);
      }
    }

#ifndef NDEBUG
    printDebugInfo(exe.Get());
#endif
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
  } catch (const std::runtime_error&) {
    // plugin is not created by e.g. invalid env
    // Empty device list will be returned
  } catch (const std::exception& ex) {
    ORT_THROW("[ERROR] [OpenVINO] An exception is thrown while trying to create the ",
              device_type,
              " device: ",
              ex.what());
  } catch (...) {
    ORT_THROW("[ERROR] [OpenVINO] Unknown exception is thrown while trying to create the ",
              device_type,
              " device");
  }

  if (devicesIDs.size() > 1) {
    for (const auto& deviceID : devicesIDs) {
      available_devices.push_back(device_type + '.' + deviceID);
    }
  } else if (!devicesIDs.empty()) {
    available_devices.push_back(device_type);
  }

  return available_devices;
}

void OVCore::SetStreams(const std::string& device_type, int num_streams) {
  core.set_property(device_type, {ov::num_streams(num_streams)});
}

std::shared_ptr<OVInferRequest> OVExeNetwork::CreateInferRequest() {
  try {
    auto infReq = obj.create_infer_request();
    std::shared_ptr<OVInferRequest> ovInfReq;
    if (_stateful_llm) {
      ovInfReq = std::make_shared<StatefulOVInferRequest>(std::move(infReq), _device);
    } else {
      ovInfReq = std::make_shared<OVInferRequest>(std::move(infReq));
    }
    return ovInfReq;
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

StatefulOVInferRequest::StatefulOVInferRequest(ov::InferRequest infer_request, std::string d)
    : OVInferRequest(std::move(infer_request)), device(d) {
  if ((device.find("NPU") != std::string::npos) || (device.find("GPU") != std::string::npos)) {
    prefill_use_full_chat_history = true;
  }
}

void StatefulOVInferRequest::PreProcessInferRequest() {
  // Since we can't seem to set at ORT GenAI layer right now, we just set it here
  // as a workaround.
  // TODO: Fix this.
  ov::Tensor beam_idx = ov::Tensor(ov::element::i32, {1});
  std::fill_n(beam_idx.data<int32_t>(), 1, 0);
  ovInfReq.set_tensor("beam_idx", beam_idx);

  // If 'prefill full chat history' mode is enabled, we need to cache input_ids and position_ids.
  if (prefill_use_full_chat_history) {
    auto input_ids_tensor = ovInfReq.get_tensor("input_ids");

    // add input_ids to our cache
    {
      auto* pData = input_ids_tensor.data<int64_t>();
      for (size_t i = 0; i < input_ids_tensor.get_size(); i++) {
        cached_input_ids.push_back(pData[i]);
      }
    }

    // add position_ids to our cache
    {
      auto position_ids = ovInfReq.get_tensor("position_ids");
      auto* pData = position_ids.data<int64_t>();
      for (size_t i = 0; i < position_ids.get_size(); i++) {
        cached_position_ids.push_back(pData[i]);
      }
    }

    // if we're about to run prefill model
    if (input_ids_tensor.get_size() > 1) {
      // if the input_ids size doesn't equal cached size of the input_ids
      //  then it means that we're running 2nd (or later) prompt.
      if (input_ids_tensor.get_shape()[1] != cached_input_ids.size()) {
        // Clear the internal KVCache state (note: this is a no-op for NPU)
        ovInfReq.reset_state();

        // set a new input_ids tensor with the content of our cached input_ids
        {
          auto new_shape = input_ids_tensor.get_shape();
          new_shape[1] = cached_input_ids.size();
          auto new_input_ids = ov::Tensor(input_ids_tensor.get_element_type(), new_shape);
          auto* pNewInputIds = new_input_ids.data<int64_t>();
          std::memcpy(pNewInputIds, cached_input_ids.data(), cached_input_ids.size() * sizeof(int64_t));
          ovInfReq.set_tensor("input_ids", new_input_ids);
        }

        // set a new position_ids tensor with the content of our cached position_ids
        {
          auto position_ids_tensor = ovInfReq.get_tensor("position_ids");
          auto new_shape = position_ids_tensor.get_shape();
          new_shape[1] = cached_position_ids.size();
          auto new_position_ids = ov::Tensor(position_ids_tensor.get_element_type(), new_shape);
          auto* pNewPositionIds = new_position_ids.data<int64_t>();
          std::memcpy(pNewPositionIds, cached_position_ids.data(), cached_position_ids.size() * sizeof(int64_t));
          ovInfReq.set_tensor("position_ids", new_position_ids);
        }
      }
    }
  }
}

void StatefulOVInferRequest::StartAsync() {
  PreProcessInferRequest();
  OVInferRequest::StartAsync();
}

void StatefulOVInferRequest::Infer() {
  PreProcessInferRequest();
  OVInferRequest::Infer();
}

void StatefulOVInferRequest::RewindKVCache(size_t index) {
  LOGS_DEFAULT(INFO) << log_tag << "RewindKVCache: Rewinding OpenVINO-internal KVCache state to index=" << index << std::endl;

  if (prefill_use_full_chat_history) {
    // Clear the internal KVCache state (note: this is a no-op for NPU)
    ovInfReq.reset_state();

    if (cached_input_ids.size() > index) {
      cached_input_ids.resize(index);
    }

    if (cached_position_ids.size() > index) {
      cached_position_ids.resize(index);
    }
  } else {
    if (index == 0) {
      // in this case, since we're trimming *all* of the KVCache, just reset the state.
      ovInfReq.reset_state();
    } else {
      // retrieve kvcache states, and trim...
      // Most of this code was grabbed from here:
      // https://github.com/openvinotoolkit/openvino.genai/blob/releases/2025/1/src/cpp/src/utils.cpp#L329
      auto states = ovInfReq.query_state();
      for (auto& state : states) {
        ov::Tensor old_tensor = state.get_state();
        // [BATCH_SIZE, num_kv_heads, seq_len, head_size]
        auto shape = old_tensor.get_shape();

        if (shape[2] > index) {
          shape[2] = index;

          ov::Coordinate new_shape_begin{0, 0, 0, 0};
          ov::Coordinate new_shape_end{shape};

          auto trimmed_tensor = ov::Tensor(old_tensor, new_shape_begin, new_shape_end);

          ov::Tensor new_tensor(old_tensor.get_element_type(), shape);
          trimmed_tensor.copy_to(new_tensor);

          state.set_state(new_tensor);
        }
      }
    }
  }
}

}  // namespace openvino_ep
}  // namespace onnxruntime
