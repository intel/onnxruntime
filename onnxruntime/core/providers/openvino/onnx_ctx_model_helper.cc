// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include <string>
#include <fstream>
#include <vector>
#include <utility>

#include "core/providers/openvino/onnx_ctx_model_helper.h"

namespace onnxruntime {
namespace openvino_ep {

/* Export the serialized blob string embedded onto an EPContext Node
 * along with other metadata necessary to validate the graph on import
 */

Status EPCtxHandler::ExportEPCtxModel(const GraphViewer& graph_viewer,
                                      const std::string& graph_name,
                                      const logging::Logger& logger,
                                      const bool& ep_context_embed_mode,
                                      const std::string& model_blob_str,
                                      const std::string& openvino_sdk_version) const {
  std::unique_ptr<onnxruntime::Model> model_build = graph_viewer.CreateModel(logger);
  onnxruntime::Graph& graph_build = model_build->MainGraph();

  // Reserve space for inputs and outputs to avoid reallocations
  std::vector<onnxruntime::NodeArg*> inputs, outputs;
  inputs.reserve(graph_viewer.GetInputs().size());
  outputs.reserve(graph_viewer.GetOutputs().size());

  for (const onnxruntime::NodeArg* input : graph_viewer.GetInputs()) {
      inputs.emplace_back(&graph_build.GetOrCreateNodeArg(input->Name(), input->TypeAsProto()));
  }
  for (const onnxruntime::NodeArg* output : graph_viewer.GetOutputs()) {
      outputs.emplace_back(&graph_build.GetOrCreateNodeArg(output->Name(), output->TypeAsProto()));
  }

  std::unique_ptr<ONNX_NAMESPACE::NodeAttributes> node_attributes = ONNX_NAMESPACE::NodeAttributes::Create();
  node_attributes->reserve(4);

  {
    // Create EP context node attributes
    std::unique_ptr<ONNX_NAMESPACE::AttributeProto> embed_mode_attr = ONNX_NAMESPACE::AttributeProto::Create();
    embed_mode_attr->set_name(EMBED_MODE);
    embed_mode_attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
    embed_mode_attr->set_i(ep_context_embed_mode);
    node_attributes->emplace(EMBED_MODE, std::move(*embed_mode_attr));

    std::unique_ptr<ONNX_NAMESPACE::AttributeProto> ep_cache_context_attr = ONNX_NAMESPACE::AttributeProto::Create();
    ep_cache_context_attr->set_name(EP_CACHE_CONTEXT);
    ep_cache_context_attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_STRING);
    ep_cache_context_attr->set_s(model_blob_str);
    node_attributes->emplace(EP_CACHE_CONTEXT, std::move(*ep_cache_context_attr));

    std::unique_ptr<ONNX_NAMESPACE::AttributeProto> sdk_version_attr = ONNX_NAMESPACE::AttributeProto::Create();
    sdk_version_attr->set_name(EP_SDK_VER);
    sdk_version_attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_STRING);
    sdk_version_attr->set_s(openvino_sdk_version);
    node_attributes->emplace(EP_SDK_VER, std::move(*sdk_version_attr));

    std::unique_ptr<ONNX_NAMESPACE::AttributeProto> source_attr = ONNX_NAMESPACE::AttributeProto::Create();
    source_attr->set_name(SOURCE);
    source_attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_STRING);
    source_attr->set_s(kOpenVINOExecutionProvider);
    node_attributes->emplace(SOURCE, std::move(*source_attr));
  }

  // Create EP context node
  graph_build.AddNode(graph_name, EPCONTEXT_OP, "", inputs, outputs, std::move(node_attributes.get()), kMSDomain);
  ORT_ENFORCE(graph_build.Resolve().IsOK());

  {
    // Serialize modelproto to file
    std::unique_ptr<ONNX_NAMESPACE::ModelProto> model_proto = model_build->ToProto();
    model_proto->set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

    std::ofstream epctx_onnx_model(graph_name, std::ios::out | std::ios::trunc | std::ios::binary);
    if (!epctx_onnx_model) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unable to create epctx onnx model file");
    }

    if (!model_proto->SerializeToOstream(epctx_onnx_model)) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to serialize model to file");
    }

    // Explicitly close the file stream
    epctx_onnx_model.close();
  }

  LOGS_DEFAULT(VERBOSE) << "[OpenVINO EP] Export blob as EPContext Node";

  return Status::OK();
}

Status EPCtxHandler::ImportBlobFromEPCtxModel(const GraphViewer& graph_viewer) {
  auto node = graph_viewer.GetNode(0);
  auto& attrs = node->GetAttributes();
  ORT_ENFORCE(attrs.count(EP_CACHE_CONTEXT) > 0);
  model_stream_ = std::make_shared<std::istringstream>(attrs.at(EP_CACHE_CONTEXT).s());
  LOGS_DEFAULT(VERBOSE) << "[OpenVINO EP] Read blob from EPContext Node";

  is_valid_ep_ctx_graph_ = true;
  return Status::OK();
}

bool EPCtxHandler::CheckForOVEPCtxNode(const GraphViewer& graph_viewer, std::string openvino_sdk_version) const {
  for (int i = 0; i < graph_viewer.MaxNodeIndex(); ++i) {
    auto node = graph_viewer.GetNode(i);
    auto& attrs = node->GetAttributes();

    // Check for correct Op Type, EP SOURCE, and SDK version
    if (node != nullptr && node->OpType() == EPCONTEXT_OP) {
      if (attrs.at(SOURCE).s() == kOpenVINOExecutionProvider) {
        if (attrs.at(EP_SDK_VER).s() == openvino_sdk_version) {
          return true;
        } else {
          ORT_THROW("[Invalid Graph] Versions of OpenVINO used to export blob (" + attrs.at(EP_SDK_VER).s() +
                    ") and current runtime (" + openvino_sdk_version + ") don't match.");
        }
      }
    }
  }
  return false;
}

}  // namespace openvino_ep
}  // namespace onnxruntime
