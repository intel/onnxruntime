// Copyright (C) Intel Corporation
// Licensed under the MIT License

#pragma once

#include <vector>
#include <string>

#include "onnxruntime_c_api.h"

#define OVEP_DISABLE_MOVE(class_name) \
  class_name(class_name&&) = delete;  \
  class_name& operator=(class_name&&) = delete;

#define OVEP_DISABLE_COPY(class_name)     \
  class_name(const class_name&) = delete; \
  class_name& operator=(const class_name&) = delete;

#define OVEP_DISABLE_COPY_AND_MOVE(class_name) \
  OVEP_DISABLE_COPY(class_name)                \
  OVEP_DISABLE_MOVE(class_name)

struct ApiPtrs {
  const OrtApi& ort_api;
  const OrtEpApi& ep_api;
};

class OpenVINOEpPlugin : public OrtEp,
                         public ApiPtrs {
 public:
  OpenVINOEpPlugin(ApiPtrs apis, const std::string& name, const OrtSessionOptions& session_options, const OrtLogger& logger, const std::string ov_device_type);
  ~OpenVINOEpPlugin();

  OVEP_DISABLE_COPY_AND_MOVE(OpenVINOEpPlugin)

  // Member functions that implement the OpenVINO EP functionality
  const char* GetName() const {
    return name_.c_str();
  }
  OrtStatus* GetCapability(const OrtGraph* graph, OrtEpGraphSupportInfo* graph_support_info);
  OrtStatus* Compile(const OrtGraph** graphs, const OrtNode** fused_nodes, size_t count, OrtNodeComputeInfo** node_compute_infos, OrtNode** ep_context_nodes);
  void ReleaseNodeComputeInfos(OrtNodeComputeInfo** node_compute_infos, size_t num_node_compute_infos);

  // Static wrapper functions for C API compatibility
  static const char* ORT_API_CALL GetNameImpl(const OrtEp* this_ptr) {
    const auto* ep = static_cast<const OpenVINOEpPlugin*>(this_ptr);
    return ep->GetName();
  }

  static OrtStatus* ORT_API_CALL GetCapabilityImpl(OrtEp* this_ptr, const OrtGraph* graph,
                                                   OrtEpGraphSupportInfo* graph_support_info) {
    auto* ep = static_cast<OpenVINOEpPlugin*>(this_ptr);
    return ep->GetCapability(graph, graph_support_info);
  }

  static OrtStatus* ORT_API_CALL CompileImpl(OrtEp* this_ptr, const OrtGraph** graphs, const OrtNode** fused_nodes,
                                             size_t count, OrtNodeComputeInfo** node_compute_infos,
                                             OrtNode** ep_context_nodes) {
    auto* ep = static_cast<OpenVINOEpPlugin*>(this_ptr);
    return ep->Compile(graphs, fused_nodes, count, node_compute_infos, ep_context_nodes);
  }

  static void ORT_API_CALL ReleaseNodeComputeInfosImpl(OrtEp* this_ptr,
                                                       OrtNodeComputeInfo** node_compute_infos,
                                                       size_t num_node_compute_infos) {
    auto* ep = static_cast<OpenVINOEpPlugin*>(this_ptr);
    ep->ReleaseNodeComputeInfos(node_compute_infos, num_node_compute_infos);
  }

 private:
  std::string name_;
  std::vector<const OrtHardwareDevice*> hardware_devices_;
  const OrtLogger& logger_;
  std::string ov_device_type_;  // OpenVINO device type (CPU, GPU, NPU, AUTO, etc.)
};
