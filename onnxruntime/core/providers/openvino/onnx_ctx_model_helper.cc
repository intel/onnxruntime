// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include <string>
#include <fstream>
#include <vector>
#include <algorithm>

#include "core/providers/openvino/constants.h"
#include "core/providers/openvino/onnx_ctx_model_helper.h"

namespace onnxruntime {
namespace openvino_ep {

EPCtxHandler::EPCtxHandler(const logging::Logger& logger,
                           fs::path path) : logger_{logger}, ep_context_model_path{path} {
  epctx_model_ = Model::Create("ovep_context_model", false, logger_);
}

Status EPCtxHandler::AddOVEPCtxNodeToGraph(const GraphViewer& graph_viewer,
                                           const std::string& graph_name,
                                           const bool embed_mode,
                                           std::string&& model_blob_str) const {
  auto& graph = epctx_model_->MainGraph();

  // Get graph inputs and outputs
  const auto& viewer_inputs = graph_viewer.GetInputs();
  const auto& viewer_outputs = graph_viewer.GetOutputs();
  std::vector<onnxruntime::NodeArg*> inputs(viewer_inputs.size()), outputs(viewer_outputs.size());
  auto transform_f = [&](const onnxruntime::NodeArg* iter) { return &graph.GetOrCreateNodeArg(iter->Name(), iter->TypeAsProto()); };
  auto fill_vectors = [transform_f](auto& src, auto& dst) {
    std::transform(src.begin(), src.end(), dst.begin(), transform_f);
  };
  fill_vectors(viewer_inputs, inputs);
  fill_vectors(viewer_outputs, outputs);

  // Create EP context node attributes
  auto node_attributes = ONNX_NAMESPACE::NodeAttributes::Create();
  node_attributes->reserve(4);
  {
    // Create EP context node attributes

    // embed mode
    auto embed_mode_attr = ONNX_NAMESPACE::AttributeProto::Create();
    embed_mode_attr->set_name(EMBED_MODE);
    embed_mode_attr->set_type(onnx::AttributeProto_AttributeType_INT);
    embed_mode_attr->set_i(embed_mode);
    node_attributes->emplace(EMBED_MODE, std::move(*embed_mode_attr));

    // ep context
    auto ep_cache_context_attr = ONNX_NAMESPACE::AttributeProto::Create();
    ep_cache_context_attr->set_name(EP_CACHE_CONTEXT);
    ep_cache_context_attr->set_type(onnx::AttributeProto_AttributeType_STRING);
    ep_cache_context_attr->set_s(std::move(model_blob_str));
    node_attributes->emplace(EP_CACHE_CONTEXT, std::move(*ep_cache_context_attr));

    // sdk version
    auto sdk_version_attr = ONNX_NAMESPACE::AttributeProto::Create();
    sdk_version_attr->set_name(EP_SDK_VER);
    sdk_version_attr->set_type(onnx::AttributeProto_AttributeType_STRING);
    sdk_version_attr->set_s(constants::ov_version::name.data());
    node_attributes->emplace(EP_SDK_VER, std::move(*sdk_version_attr));

    // source
    auto source_attr = ONNX_NAMESPACE::AttributeProto::Create();
    source_attr->set_name(SOURCE);
    source_attr->set_type(onnx::AttributeProto_AttributeType_STRING);
    source_attr->set_s(kOpenVINOExecutionProvider);
    node_attributes->emplace(SOURCE, std::move(*source_attr));
  }

  // Create EP context node
  graph.AddNode(graph_name, EPCONTEXT_OP, "", inputs, outputs, std::move(*node_attributes), kMSDomain);

  ORT_ENFORCE(graph.Resolve().IsOK());

  return Status::OK();
}

std::unique_ptr<std::istream> EPCtxHandler::GetModelBlobStream(const std::filesystem::path& so_context_file_path, const GraphViewer& graph_viewer) const {
  auto first_index = *graph_viewer.GetNodesInTopologicalOrder().begin();
  auto node = graph_viewer.GetNode(first_index);
  ORT_ENFORCE(node != nullptr);
  auto& attrs = node->GetAttributes();

  ORT_ENFORCE(attrs.count(EP_CACHE_CONTEXT) == 1);
  const auto& ep_cache_context = attrs.at(EP_CACHE_CONTEXT).s();

  ORT_ENFORCE(attrs.count(EMBED_MODE) == 1);
  bool embed_mode = static_cast<bool>(attrs.at(EMBED_MODE).i());

  std::unique_ptr<std::istream> result;
  if (embed_mode) {
    result.reset((std::istream*)new std::istringstream(ep_cache_context));
  } else {
    auto blob_filepath = so_context_file_path;
    if (blob_filepath.empty() && !graph_viewer.ModelPath().empty()) {
      blob_filepath = graph_viewer.ModelPath();
    }
    blob_filepath = blob_filepath.parent_path() / ep_cache_context;
    ORT_ENFORCE(std::filesystem::exists(blob_filepath), "Blob file not found: ", blob_filepath.string());
    result.reset((std::istream*)new std::ifstream(blob_filepath, std::ios_base::binary | std::ios_base::in));
  }
  LOGS_DEFAULT(VERBOSE) << "[OpenVINO EP] Read blob from EPContext Node";
  return result;
}

bool EPCtxHandler::CheckForOVEPCtxNodeInGraph(const GraphViewer& graph_viewer) const {
  if (graph_viewer.NumberOfNodes() == 1) {
    auto first_index = *graph_viewer.GetNodesInTopologicalOrder().begin();
    if (auto node = graph_viewer.GetNode(first_index); (node != nullptr) && CheckForOVEPCtxNode(*node)) {
      return true;
    }
  }
  return false;
}

bool EPCtxHandler::CheckForOVEPCtxNode(const Node& node) const {
  // Check for correct Op Type, EP SOURCE, and SDK version
  if (node.OpType() == EPCONTEXT_OP) {
    auto& attrs = node.GetAttributes();
    bool result = (attrs.count(SOURCE) == 1) && (attrs.at(SOURCE).s() == kOpenVINOExecutionProvider);
    result &= (attrs.count(EP_SDK_VER) == 1) && (attrs.at(EP_SDK_VER).s() == constants::ov_version::name);
    result &= attrs.count(EMBED_MODE) == 1;
    result &= attrs.count(EP_CACHE_CONTEXT) == 1;
    return result;
  }
  return false;
}

InlinedVector<const Node*> EPCtxHandler::GetEPCtxNodes() const {
  const auto& epctx_nodes{epctx_model_->MainGraph().Nodes()};
  return InlinedVector<const Node*>(epctx_nodes.begin(), epctx_nodes.end());
}

// Putting these structures here for now
struct Header {
  uint32_t bin_version{current_header_version};
  std::streampos weight_pos{0};
  std::streampos blobs_pos{0};
  std::streampos weight_map_pos{0};
  std::streampos blob_map_pos{0};
  constexpr static uint32_t current_header_version = 1;
};
static_assert(std::is_trivially_copyable_v<Header>, "Header is not trivial");

bool EPCtxHandler::StartReadingContextBin(const std::filesystem::path& bin_file_path, openvino_ep::weight_info_map& shared_weight_info_) {
  ORT_ENFORCE(!context_binary_.is_open(), "Unexpected open context binary file");

  context_binary_.open(bin_file_path, std::ios::in | std::ios::binary);

  // Get header
  Header header;
  context_binary_ >> header;

  ORT_ENFORCE(header.bin_version == header.current_header_version, "Binary file version mismatch");

  // Get blob information
  context_binary_.seekg(header.blob_map_pos);
  context_binary_ >> compiled_models_info_;

  // Get weight map
  context_binary_.seekg(header.weight_map_pos);
  context_binary_ >> shared_weight_info_;

  // Get blobs
  // for (const auto &blob_info : blob_info_map) {
  //  context_binary_.seekg(blob_info.pos);
  //  context_binary_.read(p_somewhere, blob_info.size);
  //}

  return true;
}

bool EPCtxHandler::FinishReadingContextBin() {
  ORT_ENFORCE(context_binary_.is_open(), "Expected open context binary file");

  context_binary_.close();

  return true;
}

std::ostream& EPCtxHandler::PreInsertBlob() {
  // Save stream position
  pre_blob_insert_ = context_binary_.tellg();
  return *(std::ostream*)&context_binary_;
}

void EPCtxHandler::PostInsertBlob(const std::string& blob_name) {
  // Save stream position
  std::streampos post_blob_insert = context_binary_.tellg();

  // Compute difference
  //  Enter data in blob map
}

bool EPCtxHandler::StartWritingContextBin(const fs::path& context_binary_name) {
  ORT_ENFORCE(!context_binary_.is_open(), "Unexpected open context binary file");

  // Mock header
  Header header{3, 4};

  auto context_binary_path_name = ep_context_model_path / context_binary_name;
  context_binary_.open(context_binary_path_name, std::ios::out | std::ios::binary);
  if (context_binary_.is_open()) {
    context_binary_ << header;
  }

  return true;
}

bool EPCtxHandler::FinishWritingContextBin(const openvino_ep::weight_info_map& shared_weight_info) {
  ORT_ENFORCE(context_binary_.is_open(), "Expected open context binary file");

  // Write maps
  context_binary_ << compiled_models_info_;
  context_binary_ << shared_weight_info;

  context_binary_.close();

  /////////////////////////////////////////////////////////////////////////////////////////////
  //
  // TEMP CODE
  // Metadata is always read from epctx model location
  //
  fs::path metadata_filename = ep_context_model_path / constants::metadata_bin_name;
  if (std::basic_fstream<std::byte> file{metadata_filename, std::ios::out + std::ios::binary}) {
    file << shared_weight_info;
  }

  // Validate serialization round trip
  if (auto filein = std::basic_fstream<std::byte>(metadata_filename, std::ios::in + std::ios::binary)) {
    openvino_ep::weight_info_map read_weight_info;
    filein >> read_weight_info;

    ORT_ENFORCE(read_weight_info == shared_weight_info);
  }
  /////////////////////////////////////////////////////////////////////////////////////////////

  return true;
}

byte_iostream& operator<<(byte_iostream& stream, const EPCtxHandler::compiled_model_info_value& value) {
  stream << value.start;
  stream << value.end;
  return stream;
}

byte_iostream& operator>>(byte_iostream& stream, EPCtxHandler::compiled_model_info_value& value) {
  stream >> value.start;
  stream >> value.end;
  return stream;
}

bool EPCtxHandler::compiled_model_info_value::operator==(const EPCtxHandler::compiled_model_info_value& other) const {
  return (start == other.start) && (end == other.end);
}

}  // namespace openvino_ep
}  // namespace onnxruntime
