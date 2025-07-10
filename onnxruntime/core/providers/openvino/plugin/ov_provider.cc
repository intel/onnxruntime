// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include <memory>
#include <string>
#include <vector>
#include "ov_provider.h"

// Implementation class definition (PIMPL idiom)
OpenVINOEpPlugin::OpenVINOEpPlugin(ApiPtrs apis, const std::string& name,
                                   const OrtSessionOptions& /*session_options*/,
                                   const OrtLogger& logger,
                                   const std::string ov_device_type)
    : ApiPtrs(apis),
      name_(name),
      logger_(logger),
      ov_device_type_(ov_device_type) {
  ort_version_supported = ORT_API_VERSION;  // set to the ORT version we were compiled with.

  OrtEp::GetName = GetNameImpl;
  OrtEp::GetCapability = GetCapabilityImpl;
  OrtEp::Compile = CompileImpl;
  OrtEp::ReleaseNodeComputeInfos = ReleaseNodeComputeInfosImpl;
}

OpenVINOEpPlugin::~OpenVINOEpPlugin() = default;

OrtStatus* OpenVINOEpPlugin::GetCapability(const OrtGraph* graph, OrtEpGraphSupportInfo* graph_support_info) {
  (void)graph;
  (void)graph_support_info;
  return ort_api.CreateStatus(ORT_NOT_IMPLEMENTED, "OpenVINO EP GetCapability not implemented yet.");
}

OrtStatus* OpenVINOEpPlugin::Compile(const OrtGraph** graphs, const OrtNode** fused_nodes,
                                     size_t count, OrtNodeComputeInfo** node_compute_infos, OrtNode** ep_context_nodes) {
  // Dummy usage of all parameters to avoid compiler warnings
  (void)graphs;
  (void)fused_nodes;
  (void)count;
  (void)node_compute_infos;
  (void)ep_context_nodes;

  return ort_api.CreateStatus(ORT_NOT_IMPLEMENTED, "OpenVINO EP Compile not implemented yet.");
}

void OpenVINOEpPlugin::ReleaseNodeComputeInfos(OrtNodeComputeInfo** node_compute_infos,
                                               size_t num_node_compute_infos) {
  // Clean up any compute info objects
  for (size_t i = 0; i < num_node_compute_infos; i++) {
    delete node_compute_infos[i];
  }
}
