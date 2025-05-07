// Copyright (C) Intel Corporation
// Licensed under the MIT License

#pragma once

#include <memory>

#define ORT_API_MANUAL_INIT
#include <vector>
#include <iostream>
#include <string>
#include <condition_variable>
#include <mutex>
#include <map>
#include <functional>
#include <algorithm>
#include <utility>

#include "core/session/onnxruntime_cxx_api.h"
#include "core/providers/openvino/contexts.h"
#include "core/providers/openvino/ibackend.h"
#include "core/providers/openvino/ov_interface.h"
#include "core/providers/openvino/backend_utils.h"

namespace onnxruntime {
namespace openvino_ep {

struct OnnxToOvNetworkBindings {
  std::vector<ParameterInfo> network_outputs_;
  std::vector<ParameterInfo> network_inputs_;

  OnnxToOvNetworkBindings(OVExeNetwork& exec_network, SubGraphContext& subgraph_context, SessionContext& session_context) {
    auto populate = [&](auto& input_output_map, const SubGraphContext::string_index_map_t& onnx_input_map, const auto& ov_parameters) {
      for (const auto& [onnx_name, onnx_param_index] : onnx_input_map) {
        auto it = std::find_if(ov_parameters.begin(), ov_parameters.end(),
                               [&onnx_name](const auto& ov_parameter_info) { return ov_parameter_info.get_names().contains(onnx_name); });

        // For Stateful Model Compilation, the ONNX model includes KV cache (past/present) tensors.
        // However, these tensors are internally converted to a stateful representation, which removes them.
        // To prevent runtime exceptions, we simply continue processing here.
        if ((onnx_name.empty() || onnx_name == "beam_idx" ||
            onnx_name.find("past_key_values") != std::string::npos ||
            onnx_name.find("present") != std::string::npos) &&
            session_context.enable_causallm) {
          continue;
        }

        ORT_ENFORCE(it != ov_parameters.end(), backend_utils::log_tag,
                    "Input names mismatch between OpenVINO and ONNX. ", onnx_name,
                    " doesn't exist in the list of OpenVINO input tensor names");

        auto ov_param_index = std::distance(ov_parameters.begin(), it);

        auto shape = ov_parameters[ov_param_index].get_partial_shape();
        auto type = ov_parameters[ov_param_index].get_element_type();
        ParameterInfo info{onnx_name, ov_param_index, onnx_param_index, type, shape};
        input_output_map.push_back(std::move(info));
      }
    };

    populate(network_inputs_, subgraph_context.input_names, exec_network.Get().inputs());
    populate(network_outputs_, subgraph_context.output_names, exec_network.Get().outputs());
  }
};

class InferRequestPool;
class BasicBackend : public IBackend {
 public:
  BasicBackend(std::unique_ptr<ONNX_NAMESPACE::ModelProto>& model_proto,
               SessionContext& session_context,
               const SubGraphContext& subgraph_context,
               SharedContext& shared_context,
               ptr_stream_t& model_stream);

  void Infer(OrtKernelContext* context) override;
  ~BasicBackend() override = default;
  ov::CompiledModel GetOVCompiledModel() override {
    return exe_network_.Get();
  }
  void RewindKVCache(size_t index) override;

 private:
  bool ValidateSubgraph(std::map<std::string, std::shared_ptr<ov::Node>>& const_outputs_map);
  void PopulateConfigValue(ov::AnyMap& device_config);
  void EnableCaching();
  void EnableGPUThrottling(ov::AnyMap& device_config);
  void EnableStreams();
  void SetNumThreads(ov::AnyMap& device_config);

#ifdef IO_BUFFER_ENABLED
  void RemoteInfer(Ort::KernelContext& context, std::shared_ptr<OVInferRequest> infer_request);
#endif

  SessionContext& session_context_;
  SubGraphContext subgraph_context_;
  SharedContext& shared_context_;
  mutable std::mutex compute_lock_;
  OVExeNetwork exe_network_;
  std::map<std::string, std::shared_ptr<ov::Node>> const_outputs_map_;
  std::unique_ptr<InferRequestPool> infer_req_pool_;
#if defined IO_BUFFER_ENABLED
  OVRemoteContextPtr remote_context_;
#endif

  using ort_tensor_key_t = const std::string;
  std::unique_ptr<OnnxToOvNetworkBindings> bindings_;
};

class InferRequestPool {
 public:
  struct GuardedInferReq {
    OVInferRequestPtr infer_request_;

    GuardedInferReq(InferRequestPool& queue, OVInferRequestPtr& infer_req) : queue_(queue), infer_request_(infer_req) {}
    ~GuardedInferReq() { queue_.putIdleRequest(std::move(infer_request_)); }

    // Movable but not copyable
    GuardedInferReq(const GuardedInferReq&) = delete;
    GuardedInferReq& operator=(const GuardedInferReq&) = delete;
    GuardedInferReq(GuardedInferReq&&) = default;
    GuardedInferReq& operator=(GuardedInferReq&&) = default;

   private:
    InferRequestPool& queue_;
    friend class InferRequestPool;
  };

  InferRequestPool(OVExeNetwork& net, size_t nireq, std::function<void(OVInferRequestPtr)> initializer) {
    OVInferRequestPtr infer_request;
    live_threads=nireq;
    for (size_t id = 0; id < nireq; id++) {
      infer_request = net.CreateInferRequest();
      initializer(infer_request);
      infer_requests_.push_back(infer_request);
    }
  }
  ~InferRequestPool() = default;

  GuardedInferReq getIdleRequest() {
    std::unique_lock<std::mutex> lock(_mutex);
    if(live_threads==0) {
      return nullptr;
    }

    _cv.wait(lock, [this] { return infer_requests_.size() > 0; });
    auto request = infer_requests_.back();
    infer_requests_.pop_back();
    return GuardedInferReq(*this, request);
  }

  void deleteRequest() {
    std::unique_lock<std::mutex> lock(_mutex);
    live_threads=live_threads-1;
    std::cout << "delete Request" << live_threads << "\n";
  }

 private:
  void putIdleRequest(OVInferRequestPtr&& infer_request) {
    if (infer_request) {
      std::unique_lock<std::mutex> lock(_mutex);
      infer_requests_.push_back(infer_request);
      _cv.notify_one();
    }
  }

  std::mutex _mutex;
  std::condition_variable _cv;
  std::vector<OVInferRequestPtr> infer_requests_;
  int live_threads;
};

}  // namespace openvino_ep
}  // namespace onnxruntime
