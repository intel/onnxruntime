// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <functional>

#include "ov_provider.h"
#include "ov_compute.h"
#include "ov_ep_context.h"
#include "../common/ov_supported_ops.h"

#define ORT_EP_UTILS_ORT_GRAPH_TO_PROTO_IMPL
#include "core/providers/utils/ort_graph_to_proto.h"

using namespace onnxruntime::openvino_ep;

// Implementation class definition
OpenVINOEpPlugin::OpenVINOEpPlugin(ApiPtrs apis, const std::string& name,
                                   const OrtSessionOptions& /*session_options*/,
                                   const OrtLogger& logger,
                                   const std::string ov_device_type,
                                   std::shared_ptr<ov::Core> ov_core)
    : ApiPtrs(apis),
      name_(name),
      logger_(logger),
      ov_device_type_(ov_device_type),
      ov_core_(ov_core) {
  ort_version_supported = ORT_API_VERSION;  // set to the ORT version we were compiled with.

  OrtEp::GetName = GetNameImpl;
  OrtEp::GetCapability = GetCapabilityImpl;
  OrtEp::Compile = CompileImpl;
  OrtEp::ReleaseNodeComputeInfos = ReleaseNodeComputeInfosImpl;
}

OpenVINOEpPlugin::~OpenVINOEpPlugin() = default;

struct NodeView {
  std::string_view domain;
  std::string_view op_type;
  bool is_ep_context = false;

  OrtStatus* Init(const OrtApi& ort_api, const OrtNode* node_ptr) {
    const char* op_type_char = nullptr;
    RETURN_IF_ERROR(ort_api.Node_GetOperatorType(node_ptr, &op_type_char));
    op_type = op_type_char;

    const char* domain_char = nullptr;
    RETURN_IF_ERROR(ort_api.Node_GetDomain(node_ptr, &domain_char));

    // Use empty string if domain is null (default domain)
    domain = domain_char ? domain_char : "";

    is_ep_context = SupportedOps::Get().IsEpContextNode(op_type, domain);

    return nullptr;
  }
};

OrtStatus* OpenVINOEpPlugin::GetCapability(const OrtGraph* graph, OrtEpGraphSupportInfo* graph_support_info) {
  size_t num_nodes = 0;
  RETURN_IF_ERROR(ort_api.Graph_GetNumNodes(graph, &num_nodes));

  std::vector<const OrtNode*> nodes(num_nodes);
  RETURN_IF_ERROR(ort_api.Graph_GetNodes(graph, nodes.data(), nodes.size()));

  auto add_nodes_to_fuse = [this, graph_support_info](std::vector<const OrtNode*> nodes) -> OrtStatus* {
    OrtNodeFusionOptions options{};
    options.ort_version_supported = ORT_API_VERSION;
    options.drop_constant_initializers = true;
    return ep_api.EpGraphSupportInfo_AddNodesToFuse(graph_support_info, nodes.data(), nodes.size(), &options);
  };

  // Get supported operations instance
  const auto& supported_ops = SupportedOps::Get();

  // Check each node for support
  std::vector<const OrtNode*> supported_nodes;
  for (const auto& node_ptr : nodes) {
    NodeView node_info;
    RETURN_IF_ERROR(node_info.Init(ort_api, node_ptr));

    // Check if this operation is supported
    bool is_ep_context_node = supported_ops.IsEpContextNode(node_info.op_type, node_info.domain);
    if (supported_ops.IsOpSupported(node_info.op_type, node_info.domain) && !is_ep_context_node) {
      supported_nodes.push_back(node_ptr);
    } else {
      // Fuse all the supported nodes collected so far
      if (!supported_nodes.empty()) {
        RETURN_IF_ERROR(add_nodes_to_fuse(supported_nodes));
      }
      supported_nodes.clear();

      if (is_ep_context_node) {
        RETURN_IF_ERROR(add_nodes_to_fuse(std::vector<const OrtNode*>{node_ptr}));
      }
    }
  }

  if (!supported_nodes.empty()) {
    RETURN_IF_ERROR(add_nodes_to_fuse(supported_nodes));
  }

  return nullptr;
}

OrtStatus* OpenVINOEpPlugin::Compile(const OrtGraph** graphs, const OrtNode** fused_nodes,
                                     size_t count, OrtNodeComputeInfo** node_compute_infos, OrtNode** ep_context_nodes) {
  // Dummy usage of all parameters to avoid compiler warnings
  (void)ep_context_nodes;
  (void)fused_nodes;

  // Process all graphs
  for (size_t i = 0; i < count; ++i) {
    const OrtGraph* graph = graphs[i];
    if (graph == nullptr) {
      return ort_api.CreateStatus(ORT_INVALID_ARGUMENT, "Graph is null");
    }

    size_t num_nodes = 0;
    RETURN_IF_ERROR(ort_api.Graph_GetNumNodes(graph, &num_nodes));

    std::vector<const OrtNode*> nodes(num_nodes);
    RETURN_IF_ERROR(ort_api.Graph_GetNodes(graph, nodes.data(), nodes.size()));

    auto ov_compute = std::make_unique<OvComputeInfo>(*this, *ov_core_);

    OnnxIOMapping io_mapping;
    RETURN_IF_ERROR(io_mapping.Init(ort_api, *graph));

    if (num_nodes == 1) {
      // Only supporting single ep context node per graph for now
      NodeView node_info;
      const OrtNode* node = nodes[0];
      RETURN_IF_ERROR(node_info.Init(ort_api, node));
      if (node_info.is_ep_context) {
        EpContextNode ep_context_node(*this, node);
        RETURN_IF_ERROR(ov_compute->Init(ov_device_type_, std::move(io_mapping), std::move(ep_context_node)));
        node_compute_infos[i] = ov_compute.release();
        continue;
      }
    }

    std::unique_ptr<onnx::GraphProto> graph_proto = std::make_unique<onnx::GraphProto>();
    RETURN_IF_ERROR(OrtEpUtils::OrtGraphToProto(*graph, *graph_proto));
    RETURN_IF_ERROR(ov_compute->Init(ov_device_type_, std::move(io_mapping), std::move(graph_proto)));

    node_compute_infos[i] = ov_compute.release();
  }

  return nullptr;
}

void OpenVINOEpPlugin::ReleaseNodeComputeInfos(OrtNodeComputeInfo** node_compute_infos,
                                               size_t num_node_compute_infos) {
  // Clean up any compute info objects
  for (size_t i = 0; i < num_node_compute_infos; i++) {
    delete node_compute_infos[i];
  }
}
