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
struct context_bin_header : streamable<context_bin_header> {
  context_bin_header() = default;
  context_bin_header(uint32_t bv,
                     std::streampos wp,
                     std::streampos cmp,
                     std::streampos wmp,
                     std::streampos cmmp) : bin_version{bv},
                                            weight_pos{wp},
                                            compiled_models_pos{cmp},
                                            weight_map_pos{wmp},
                                            compiled_model_map_pos{cmmp} {
  }
  uint32_t bin_version{constants::expected_bin_version};
  std::streampos weight_pos{0};
  std::streampos compiled_models_pos{0};
  std::streampos weight_map_pos{0};
  std::streampos compiled_model_map_pos{0};

  template <typename S>
  friend void write_bytes(S& stream, const context_bin_header& value) {
    write_bytes(stream, value.bin_version);
    write_bytes(stream, value.weight_pos);
    write_bytes(stream, value.compiled_models_pos);
    write_bytes(stream, value.weight_map_pos);
    write_bytes(stream, value.compiled_model_map_pos);
  }

  template <typename S>
  friend void read_bytes(S& stream, context_bin_header& value) {
    read_bytes(stream, value.bin_version);
    read_bytes(stream, value.weight_pos);
    read_bytes(stream, value.compiled_models_pos);
    read_bytes(stream, value.weight_map_pos);
    read_bytes(stream, value.compiled_model_map_pos);
  }
};
static_assert(std::is_trivially_copyable_v<context_bin_header>, "Header is not trivial");

bool EPCtxHandler::StartReadingContextBin(const std::filesystem::path& bin_file_path, openvino_ep::weight_info_map& shared_weight_info_) {
  ORT_ENFORCE(!context_bin_stream_.is_open(), "Unexpected open context binary file");

  context_bin_stream_.open(bin_file_path, std::ios::in | std::ios::binary);

  // Get header
  context_bin_header header;
  read_bytes(context_bin_stream_, header);

  ORT_ENFORCE(header.bin_version == constants::expected_bin_version, "Binary file version mismatch");

  // Get compiled model information
  read_bytes(context_bin_stream_, compiled_models_info_, header.compiled_model_map_pos);

  // Get weight map
  read_bytes(context_bin_stream_, shared_weight_info_, header.weight_map_pos);

  // Get blobs
  // for (const auto &blob_info : blob_info_map) {
  //  context_bin_stream_.seekg(blob_info.pos);
  //  context_bin_stream_.read(p_somewhere, blob_info.size);
  //}

  return true;
}

bool EPCtxHandler::FinishReadingContextBin() {
  ORT_ENFORCE(context_bin_stream_.is_open(), "Expected open context binary file");

  context_bin_stream_.close();

  return true;
}

std::ostream& EPCtxHandler::PreInsertBlob() {
  // Save stream position
  pre_blob_insert_ = context_bin_stream_.tellg();
  return *(std::ostream*)&context_bin_stream_;
}

void EPCtxHandler::PostInsertBlob(const std::string& blob_name) {
  // Save stream position
  std::streampos post_blob_insert = context_bin_stream_.tellg();

  // Compute difference
  //  Enter data in blob map
}

bool EPCtxHandler::StartWritingContextBin(const fs::path& context_bin_name) {
  ORT_ENFORCE(!context_bin_stream_.is_open(), "Unexpected open context binary file");

  // Mock header
  context_bin_header header{1, 2, 3, 4, 5};

  auto context_bin_path_name = ep_context_model_path / context_bin_name;
  context_bin_stream_.open(context_bin_path_name, std::ios::out | std::ios::binary);
  if (context_bin_stream_.is_open()) {
    write_bytes(context_bin_stream_, header);
  }

  return true;
}

bool EPCtxHandler::FinishWritingContextBin(const openvino_ep::weight_info_map& shared_weight_info) {
  ORT_ENFORCE(context_bin_stream_.is_open(), "Expected open context binary file");

  // Write maps
  write_bytes(context_bin_stream_, compiled_models_info_);
  write_bytes(context_bin_stream_, shared_weight_info);

  context_bin_stream_.close();

  /////////////////////////////////////////////////////////////////////////////////////////////
  //
  // TEMP CODE
  // Metadata is always read from epctx model location
  //
  fs::path metadata_filename = ep_context_model_path / constants::metadata_bin_name;
  if (std::ofstream file{metadata_filename, std::ios::binary}) {
    write_bytes(file, shared_weight_info);
  }

  // Validate serialization round trip
  if (auto filein = std::ifstream(metadata_filename, std::ios::binary)) {
    openvino_ep::weight_info_map read_weight_info;
    read_bytes(filein, read_weight_info);

    ORT_ENFORCE(read_weight_info == shared_weight_info);
  }
  /////////////////////////////////////////////////////////////////////////////////////////////

  return true;
}

template <typename S>
void write_bytes(S& stream, const EPCtxHandler::compiled_model_info_value& value) {
  write_bytes(stream, value.start);
  write_bytes(stream, value.end);
}

template <typename S>
void read_bytes(S& stream, EPCtxHandler::compiled_model_info_value& value) {
  read_bytes(stream, value.start);
  read_bytes(stream, value.end);
}

bool EPCtxHandler::compiled_model_info_value::operator==(const EPCtxHandler::compiled_model_info_value& other) const {
  return (start == other.start) && (end == other.end);
}

}  // namespace openvino_ep
}  // namespace onnxruntime
