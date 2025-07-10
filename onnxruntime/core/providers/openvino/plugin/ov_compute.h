// Copyright (C) Intel Corporation
// Licensed under the MIT License

#pragma once

#include <string>
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <array>
#include <memory>
#include <functional>
#include <mutex>
#include <gsl/narrow>

#include "ov_provider.h"

#include "openvino/openvino.hpp"
#include "core/providers/utils/ort_graph_to_proto.h"

namespace onnxruntime {
namespace openvino_ep {

using string_index_map_t = std::unordered_map<std::string, uint32_t>;

class WrappedInferRequest {
  struct ov_tensor_data_t {
    std::unique_ptr<ov::Tensor> tensor_ptr;
    const void* ort_ptr;
  };

 protected:
  ov::InferRequest ov_inf_req_;
  std::unordered_map<std::string, ov_tensor_data_t> bindings_cache_;

 public:
  std::string GetInputTensorName(uint32_t index);

  // Set tensor call infer req tensor if ort_ptr differs from last set ptr.
  void SetTensor(const std::string& name, const ov::element::Type& type, const ov::Shape& shape, void* ort_ptr) {
    auto& cached_binding = bindings_cache_[name];
    if (cached_binding.ort_ptr != ort_ptr) {
      cached_binding.tensor_ptr.reset();
      auto ov_tensor = std::make_unique<ov::Tensor>(type, shape, ort_ptr);
      ov_inf_req_.set_tensor(name, *ov_tensor);
      cached_binding = {std::move(ov_tensor), ort_ptr};
    }
  }
  void GetTensor(const std::string& name, ov::Tensor& out_tensor) {
    auto it = bindings_cache_.find(name);
    if (it != bindings_cache_.end()) {
      out_tensor = *(it->second.tensor_ptr);
    } else {
      throw std::runtime_error("Tensor not found: " + name);
    }
  }

  void Infer() { ov_inf_req_.infer(); }
  ov::InferRequest& ov() { return ov_inf_req_; }
  explicit WrappedInferRequest(ov::InferRequest&& obj) : ov_inf_req_(std::move(obj)) {}
  // virtual void RewindKVCache(size_t index) {}
};

class InferRequestPool {
 public:
  using OVInferRequestPtr = std::unique_ptr<WrappedInferRequest>;
  struct GuardedInferReq {
    OVInferRequestPtr infer_request_;
    GuardedInferReq(InferRequestPool& queue, OVInferRequestPtr&& infer_req) : infer_request_(std::move(infer_req)), queue_(queue) {}
    ~GuardedInferReq() { queue_.putIdleRequest(std::move(infer_request_)); }

    // Movable but not copyable
    OVEP_DISABLE_COPY(GuardedInferReq);
    GuardedInferReq(GuardedInferReq&&) = default;
    GuardedInferReq& operator=(GuardedInferReq&&) = default;

   private:
    InferRequestPool& queue_;
    friend class InferRequestPool;
  };

  InferRequestPool(ov::CompiledModel& model, size_t initial_size, std::function<void(OVInferRequestPtr&)> initializer) : compile_model_(model), initializer_(std::move(initializer)) {
    for (size_t id = 0; id < initial_size; id++) {
      infer_requests_.emplace_back(createInferRequest());
    }
  }
  ~InferRequestPool() = default;

  GuardedInferReq getRequest() {
    std::unique_lock<std::mutex> lock(_mutex);
    if (infer_requests_.empty()) {
      infer_requests_.emplace_back(createInferRequest());
    }
    auto request = std::move(infer_requests_.back());
    infer_requests_.pop_back();
    return GuardedInferReq(*this, std::move(request));
  }

  template <typename Func>
  void forEachIdleRequest(Func&& func) {
    std::unique_lock<std::mutex> lock(_mutex);
    for (auto& infer_request : infer_requests_) {
      func(infer_request);
    }
  }

 private:
  void putIdleRequest(OVInferRequestPtr&& infer_request) {
    if (infer_request) {
      std::unique_lock<std::mutex> lock(_mutex);
      infer_requests_.emplace_back(std::move(infer_request));
    }
  }

  OVInferRequestPtr createInferRequest() {
    auto infer_request = std::make_unique<WrappedInferRequest>(std::move(compile_model_.create_infer_request()));
    initializer_(infer_request);
    return infer_request;
  }

 private:
  std::mutex _mutex;
  std::vector<OVInferRequestPtr> infer_requests_;
  ov::CompiledModel& compile_model_;
  std::function<void(OVInferRequestPtr&)> initializer_;
};

struct ParameterShape {
  using ort_shape_t = std::vector<int64_t>;

  static ov::PartialShape ToOvPartialShape(const ort_shape_t& ort_shape) {
    std::vector<ov::Dimension> ov_shape(ort_shape.size());
    std::transform(ort_shape.begin(), ort_shape.end(), ov_shape.begin(), [](int64_t dim) {
      return dim == -1 ? ov::Dimension::dynamic() : ov::Dimension(dim);
    });
    return ov::PartialShape(ov_shape);
  }

  static ort_shape_t
  ToOrtShape(const ov::PartialShape& ov_shape) {
    ort_shape_t ort_shape(ov_shape.size());
    std::transform(ov_shape.begin(), ov_shape.end(), ort_shape.begin(), [](const auto& dim) {
      return dim.is_dynamic() ? -1 : dim.get_length();
    });
    return ort_shape;
  }

  static ort_shape_t ToOrtShape(const ov::Shape& ov_shape) {
    ort_shape_t ort_shape(ov_shape.size());
    std::transform(ov_shape.begin(), ov_shape.end(), ort_shape.begin(), [](const auto& dim) {
      return gsl::narrow<int64_t>(dim);
    });
    return ort_shape;
  }

  operator ov::Shape() const { return ov_.get_shape(); }
  operator const ov::PartialShape&() const { return ov_; }
  operator const ort_shape_t&() const { return ort_; }
  ort_shape_t& ort() { return ort_; }

  explicit ParameterShape(const ort_shape_t& ort_shape) : ort_(ort_shape), ov_(ToOvPartialShape(ort_shape)) {}
  explicit ParameterShape(const ov::PartialShape& ov_partial_shape) : ort_(ToOrtShape(ov_partial_shape)), ov_(ov_partial_shape) {}

 private:
  ort_shape_t ort_;
  ov::PartialShape ov_;
};

struct OnnxIOMapping {
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;

  OrtStatus* Init(const OrtApi& ort_api, const OrtNode& node) {
    std::vector<const OrtValueInfo*> io;
    size_t num_elements = 0;

    RETURN_IF_ERROR(ort_api.Node_GetNumInputs(&node, &num_elements));
    io.resize(num_elements);
    RETURN_IF_ERROR(ort_api.Node_GetInputs(&node, io.data(), io.size()));
    RETURN_IF_ERROR(PopulateNames(ort_api, io, input_names));

    RETURN_IF_ERROR(ort_api.Node_GetNumOutputs(&node, &num_elements));
    io.resize(num_elements);
    RETURN_IF_ERROR(ort_api.Node_GetOutputs(&node, io.data(), io.size()));
    RETURN_IF_ERROR(PopulateNames(ort_api, io, output_names));
    return nullptr;
  }

  OrtStatus* Init(const OrtApi& ort_api, const OrtGraph& graph) {
    std::vector<const OrtValueInfo*> io;
    size_t num_elements = 0;

    RETURN_IF_ERROR(ort_api.Graph_GetNumInputs(&graph, &num_elements));
    io.resize(num_elements);
    RETURN_IF_ERROR(ort_api.Graph_GetInputs(&graph, io.data(), io.size()));
    RETURN_IF_ERROR(PopulateNames(ort_api, io, input_names));

    RETURN_IF_ERROR(ort_api.Graph_GetNumOutputs(&graph, &num_elements));
    io.resize(num_elements);
    RETURN_IF_ERROR(ort_api.Graph_GetOutputs(&graph, io.data(), io.size()));
    RETURN_IF_ERROR(PopulateNames(ort_api, io, output_names));
    return nullptr;
  }

 private:
  static OrtStatus* PopulateNames(const OrtApi& ort_api, std::vector<const OrtValueInfo*>& io_nodes, std::vector<std::string>& names_vec) {
    for (const auto& value_info : io_nodes) {
      const char* name = nullptr;
      RETURN_IF_ERROR(ort_api.GetValueInfoName(value_info, &name));
      if (name) {
        names_vec.emplace_back(name);
      }
    }
    return nullptr;
  }
};

struct ParameterInfo {
  std::string name;
  uint32_t ov_index;
  uint32_t onnx_index;
  ov::element::Type type;
  ParameterShape shape;
};

struct SessionContext {
  bool enable_causallm = false;
};

struct OnnxToOvNetworkBindings {
  std::vector<ParameterInfo> network_outputs_;
  std::vector<ParameterInfo> network_inputs_;
  bool has_dynamic_io_ = false;

  inline static const std::array special_io_names_{
      "beam_idx",
      "past_key_values",
      "present",
  };

  OnnxToOvNetworkBindings(const ov::CompiledModel& exec_network, const OnnxIOMapping& io_mapping, const SessionContext& session_context) {
    auto populate = [&](auto& input_output_map, const std::vector<std::string>& onnx_io, const auto& ov_parameters) {
      for (uint32_t i = 0; i < onnx_io.size(); i++) {
        const auto& onnx_name = onnx_io[i];
        const uint32_t onnx_param_index = i;

        auto it = std::find_if(ov_parameters.begin(), ov_parameters.end(),
                               [&onnx_name](const auto& ov_parameter_info) { return ov_parameter_info.get_names().contains(onnx_name); });
        bool matched_names = it != ov_parameters.end();

        // For Stateful Model Compilation, the ONNX model includes KV cache (past/present) tensors.
        // However, these tensors are internally converted to a stateful representation, which removes them.
        // It's also possible that the onnx model does not contain tensors such as beam_idx, whereas our converted
        // stateful representation has introduced these new tensors, creating a name mismatch (matched_names=false).
        // So, if there is a name mismatch, or the name matches our special io list, we simply continue processing
        // here to prevent runtime exceptions.
        if (session_context.enable_causallm) {
          if (!matched_names ||
              std::any_of(special_io_names_.begin(), special_io_names_.end(),
                          [&onnx_name](const std::string& name) { return onnx_name.find(name) != std::string::npos; })) {
            // This case also requires dynamic shape inference, so we'll mark the bindings as dynamic.
            has_dynamic_io_ = true;
            continue;
          }
        }

        // ORT_ENFORCE(matched_names, log_tag,
        //             "Input names mismatch between OpenVINO and ONNX. ", onnx_name,
        //             " doesn't exist in the list of OpenVINO input tensor names");

        uint32_t ov_param_index = gsl::narrow<uint32_t>(std::distance(ov_parameters.begin(), it));

        auto shape = ov_parameters[ov_param_index].get_partial_shape();
        auto type = ov_parameters[ov_param_index].get_element_type();
        ParameterInfo info{onnx_name, ov_param_index, onnx_param_index, type, ParameterShape{shape}};

        input_output_map.push_back(std::move(info));
      }
    };

    // Init inputs and outputs
    populate(network_inputs_, io_mapping.input_names, exec_network.inputs());
    populate(network_outputs_, io_mapping.output_names, exec_network.outputs());
  }
};

struct EpContextNode;
struct OvComputeInfo : OrtNodeComputeInfo, ApiPtrs {
  OvComputeInfo(ApiPtrs apis, ov::Core& ov_core);

  OrtStatus* CreateState(OrtNodeComputeContext* compute_context,
                         void** compute_state);
  OrtStatus* Compute(void* compute_state,
                     OrtKernelContext* kernel_context);
  void ReleaseState(void* compute_state);

  OrtStatus* Init(const std::string& ov_device, OnnxIOMapping, EpContextNode ep_context_node);
  OrtStatus* Init(const std::string& ov_device, OnnxIOMapping, std::unique_ptr<onnx::GraphProto> ep_context_node);

 private:
  ov::CompiledModel compiled_model_;
  ov::Core& ov_core_;
  std::unique_ptr<InferRequestPool> infer_request_pool_;
  std::unique_ptr<OnnxToOvNetworkBindings> onnx_to_ov_bindings_;

 public:
  // Static wrapper functions for C API compatibility
  static OrtStatus* ORT_API_CALL CreateStateImpl(OrtNodeComputeInfo* this_ptr,
                                                 OrtNodeComputeContext* compute_context,
                                                 void** compute_state) {
    auto* compute_info = static_cast<OvComputeInfo*>(this_ptr);
    return compute_info->CreateState(compute_context, compute_state);
  }

  static OrtStatus* ORT_API_CALL ComputeImpl(OrtNodeComputeInfo* this_ptr, void* compute_state,
                                             OrtKernelContext* kernel_context) {
    auto* compute_info = static_cast<OvComputeInfo*>(this_ptr);
    return compute_info->Compute(compute_state, kernel_context);
  }

  static void ORT_API_CALL ReleaseStateImpl(OrtNodeComputeInfo* this_ptr, void* compute_state) {
    auto* compute_info = static_cast<OvComputeInfo*>(this_ptr);
    compute_info->ReleaseState(compute_state);
  }
};
}  // namespace openvino_ep
}  // namespace onnxruntime
