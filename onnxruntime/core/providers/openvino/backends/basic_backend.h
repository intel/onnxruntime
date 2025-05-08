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

  OnnxToOvNetworkBindings(OVExeNetwork& exec_network, SubGraphContext& subgraph_context) {
    auto populate = [&](auto& input_output_map, const SubGraphContext::string_index_map_t& onnx_input_map, const auto& ov_parameters) {
      for (const auto& [onnx_name, onnx_param_index] : onnx_input_map) {
        auto it = std::find_if(ov_parameters.begin(), ov_parameters.end(),
                               [&onnx_name](const auto& ov_parameter_info) { return ov_parameter_info.get_names().contains(onnx_name); });

        ORT_ENFORCE(it != ov_parameters.end(), log_tag,
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

  InferRequestPool(OVExeNetwork& net, size_t initial_size, std::function<void(OVInferRequestPtr)> initializer) : exe_network_(net), initializer_(std::move(initializer)) {
    OVInferRequestPtr infer_request;
    for (size_t id = 0; id < initial_size; id++) {
      putIdleRequest(createInferRequest());
    }
  }
  ~InferRequestPool() = default;

  GuardedInferReq getRequest() {
    std::unique_lock<std::mutex> lock(_mutex);
    if (infer_requests_.empty()) {
      infer_requests_.emplace_back(createInferRequest());
    }
    auto request = infer_requests_.back();
    infer_requests_.pop_back();
    return GuardedInferReq(*this, request);
  }

 private:
  void putIdleRequest(OVInferRequestPtr&& infer_request) {
    if (infer_request) {
      std::unique_lock<std::mutex> lock(_mutex);
      infer_requests_.push_back(infer_request);
    }
  }

  OVInferRequestPtr createInferRequest() {
    auto infer_request = std::make_shared<OVInferRequest>(exe_network_.CreateInferRequest());
    initializer_(infer_request);
    return infer_request;
  }

  std::mutex _mutex;
  std::vector<OVInferRequestPtr> infer_requests_;
  OVExeNetwork& exe_network_;
  std::function<void(OVInferRequestPtr)> initializer_;
};

}  // namespace openvino_ep
}  // namespace onnxruntime
