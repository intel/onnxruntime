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

OVExeNetwork OVCore::CompileModel(std::shared_ptr<const OVNetwork>& ie_cnn_network,
                                  std::string& hw_target,
                                  ov::AnyMap& device_config,
                                  const std::string& name) {
  ov::CompiledModel obj;
  try {
    if (true) {
      ov::AnyMap config;

      // Create a clone of ie_cnn_network, since it's a const ov::Model, and we need to patch it..
      //  Note! With this default path, the model runs but produces garbage (for NPUW). For CPU it's fine.
      auto mutable_model = ie_cnn_network->clone();

      // uncomment to override ov::Model with one produced by OV's ONNX front-end.
      // For some reason, this makes it work -- even though model.onnx is the same model read by ORT GenAI.
      // auto mutable_model = core.read_model("C:\\Users\\LNL\\Workspace\\ORT\\deepseek_r1_distill_qwen_1.5B_int4_ort_qdq\\model.onnx");

      std::cout << "stateless model" << std::endl;
      logBasicModelInfo(mutable_model);

      std::cout << "making stateful..." << std::endl;
      patch_stateful_decoder(mutable_model);

      std::cout << "after stateful transition:" << std::endl;
      logBasicModelInfo(mutable_model);

      // This patches the model so that it only produces the logits required for sampling.
      // Actually either way that happens within NPUW::LLMCompiledModel creation, but this is
      // here mostly to align this behavior for other devices (CPU, GPU).
      apply_slice_before_matmul_transformation(mutable_model);

      auto kv_pos = get_kv_axes_pos(mutable_model);
      std::cout << "kv_pos.batch = " << kv_pos.batch << std::endl;
      std::cout << "kv_pos.seq_len = " << kv_pos.seq_len << std::endl;

      if (hw_target.find("NPU") != std::string::npos) {
        KVDesc kv_desc;
        kv_desc.max_prompt_len = pop_int_and_cast(device_config, "MAX_PROMPT_LEN").value_or(1024u);
        kv_desc.min_response_len = pop_int_and_cast(device_config, "MIN_RESPONSE_LEN").value_or(128u);

        std::cout << "kv_desc.max_prompt_len = " << kv_desc.max_prompt_len << std::endl;
        std::cout << "kv_desc.min_response_len = " << kv_desc.min_response_len << std::endl;

        update_npu_config(config, mutable_model, kv_pos, kv_desc);
      }

      std::cout << "calling compile on stateful model..." << std::endl;
      obj = core.compile_model(mutable_model, hw_target, config);
      std::cout << "done calling compile on stateful model..." << std::endl;
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
                                 std::string name) {
  try {
    ov::CompiledModel obj;
    obj = core.import_model(model_stream, hw_target, device_config);
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
