// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include <string>
#include <fstream>
#include <vector>
#include <algorithm>

#include "core/providers/openvino/constants.h"
#include "core/providers/openvino/contexts.h"
#include "core/providers/openvino/onnx_ctx_model_helper.h"

namespace onnxruntime {
namespace openvino_ep {

using namespace constants;

std::optional<fs::path> GetExternalWeightsRelativePath(const openvino_ep::weight_info_map& shared_weights_info) {
  // Assume all entries contain the same file name
  ORT_ENFORCE(shared_weights_info.size() > 0, "Expected a populated map");
  return shared_weights_info.begin()->second.location;
}

EPCtxHandler::EPCtxHandler(const SessionContext& session_context) : logger_{session_context.logger} {
  epctx_model_ = Model::Create("ovep_context_model", false, logger_);

  if (session_context.so_context_file_path.empty()) {
    ep_context_model_path_ = session_context.onnx_model_path_name.parent_path();
  } else {
    ep_context_model_path_ = session_context.so_context_file_path.parent_path();
  }
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
    namespace ep_ctx_constants = ep_context::attributes;

    // embed mode
    auto embed_mode_attr = ONNX_NAMESPACE::AttributeProto::Create();
    embed_mode_attr->set_name(ep_ctx_constants::embed_mode.data());
    embed_mode_attr->set_type(onnx::AttributeProto_AttributeType_INT);
    embed_mode_attr->set_i(embed_mode);
    node_attributes->emplace(ep_ctx_constants::embed_mode.data(), std::move(*embed_mode_attr));

    // ep context
    auto ep_cache_context_attr = ONNX_NAMESPACE::AttributeProto::Create();
    ep_cache_context_attr->set_name(ep_ctx_constants::ep_cache_context.data());
    ep_cache_context_attr->set_type(onnx::AttributeProto_AttributeType_STRING);
    ep_cache_context_attr->set_s(std::move(model_blob_str));
    node_attributes->emplace(ep_ctx_constants::ep_cache_context.data(), std::move(*ep_cache_context_attr));

    // sdk version
    auto sdk_version_attr = ONNX_NAMESPACE::AttributeProto::Create();
    sdk_version_attr->set_name(ep_ctx_constants::ep_sdk_version.data());
    sdk_version_attr->set_type(onnx::AttributeProto_AttributeType_STRING);
    sdk_version_attr->set_s(ov_version::name.data());
    node_attributes->emplace(ep_ctx_constants::ep_sdk_version.data(), std::move(*sdk_version_attr));

    // source
    auto source_attr = ONNX_NAMESPACE::AttributeProto::Create();
    source_attr->set_name(ep_ctx_constants::source.data());
    source_attr->set_type(onnx::AttributeProto_AttributeType_STRING);
    source_attr->set_s(kOpenVINOExecutionProvider);
    node_attributes->emplace(ep_ctx_constants::source.data(), std::move(*source_attr));
  }

  // Create EP context node
  graph.AddNode(graph_name, ep_context::op_name.data(), "", inputs, outputs, std::move(*node_attributes), kMSDomain);

  ORT_ENFORCE(graph.Resolve().IsOK());

  return Status::OK();
}

std::unique_ptr<std::istream> EPCtxHandler::GetModelBlobStream(const std::filesystem::path& so_context_file_path, const GraphViewer& graph_viewer) {
  auto first_index = *graph_viewer.GetNodesInTopologicalOrder().begin();
  auto node = graph_viewer.GetNode(first_index);
  ORT_ENFORCE(node != nullptr);
  auto& attrs = node->GetAttributes();

  ORT_ENFORCE(attrs.count(ep_context::attributes::ep_cache_context.data()) == 1);
  const auto& ep_cache_context = attrs.at(ep_context::attributes::ep_cache_context.data()).s();

  ORT_ENFORCE(attrs.count(ep_context::attributes::embed_mode.data()) == 1);
  bool embed_mode = static_cast<bool>(attrs.at(ep_context::attributes::embed_mode.data()).i());

  std::unique_ptr<std::istream> result;
  if (embed_mode) {
    result.reset((std::istream*)new std::istringstream(ep_cache_context));
  } else {
    ORT_ENFORCE(so_context_file_path.has_parent_path(), "Expected ep.context_file_path to contain a parent path");
    auto blob_filepath = so_context_file_path.parent_path() / ep_cache_context;
    ORT_ENFORCE(std::filesystem::exists(blob_filepath), "Blob file not found: ", blob_filepath.string());
    result.reset((std::istream*)new std::ifstream(blob_filepath, std::ios_base::binary | std::ios_base::in));
  }
  LOGS_DEFAULT(VERBOSE) << "[OpenVINO EP] Read blob from EPContext Node";
  return result;
}

bool EPCtxHandler::CheckForOVEPCtxNodeInGraph(const GraphViewer& graph_viewer) {
  if (graph_viewer.NumberOfNodes() == 1) {
    auto first_index = *graph_viewer.GetNodesInTopologicalOrder().begin();
    if (auto node = graph_viewer.GetNode(first_index); (node != nullptr) && CheckForOVEPCtxNode(*node)) {
      return true;
    }
  }
  return false;
}

bool EPCtxHandler::CheckForOVEPCtxNode(const Node& node) {
  namespace ep_ctx_constants = ep_context::attributes;

  // Check for correct Op Type, EP SOURCE, and SDK version
  if (node.OpType() == ep_context::op_name.data()) {
    auto& attrs = node.GetAttributes();
    bool result = (attrs.count(ep_ctx_constants::source.data()) == 1) && (attrs.at(ep_ctx_constants::source.data()).s() == kOpenVINOExecutionProvider);
    result &= (attrs.count(ep_ctx_constants::ep_sdk_version.data()) == 1) && (attrs.at(ep_ctx_constants::ep_sdk_version.data()).s() == ov_version::name);
    result &= attrs.count(ep_ctx_constants::embed_mode.data()) == 1;
    result &= attrs.count(ep_ctx_constants::ep_cache_context.data()) == 1;
    return result;
  }
  return false;
}

InlinedVector<const Node*> EPCtxHandler::GetEPCtxNodes() const {
  const auto& epctx_nodes{epctx_model_->MainGraph().Nodes()};
  return InlinedVector<const Node*>(epctx_nodes.begin(), epctx_nodes.end());
}

EPCtxBinReader::EPCtxBinReader(std::unique_ptr<std::istream>& context_binary_stream,
                               openvino_ep::weight_info_map& shared_weights_info,
                               fs::path context_model_parent_path) : context_bin_stream_{std::move(context_binary_stream)} {
  ORT_ENFORCE(context_bin_stream_, "Invalid context binary stream");

  auto& stream = *context_bin_stream_;

  // Read header
  read_bytes(stream, header_);

  ORT_ENFORCE(header_.bin_version == ep_context::expected_bin_version, "Binary file version mismatch");

  // Read compiled model information
  stream.seekg(header_.sections.compiled_models_map);
  read_bytes(stream, compiled_models_info_);
  // read_bytes(stream, compiled_models_info_, header_.sections.compiled_models_map);

  // Read weight map
  stream.seekg(header_.sections.weights_map);
  read_bytes(stream, shared_weights_info);
  // read_bytes(stream, shared_weights_info, header_.sections.weights_map);

  if (header_.sections.weights == 0) {
    // Create input stream for external weights
    auto relative_filepath = GetExternalWeightsRelativePath(shared_weights_info);
    ORT_ENFORCE(relative_filepath.has_value(), "Expected a valid external weight relative path");
    auto external_weights_filepath = context_model_parent_path / relative_filepath.value();
    ORT_ENFORCE(fs::exists(external_weights_filepath));
    external_weights_stream_.open(external_weights_filepath, std::ios::binary);
    ORT_ENFORCE(external_weights_stream_, "Error opening external weight file at ", external_weights_filepath.string());
  }
}

std::unique_ptr<std::istringstream> EPCtxBinReader::GetCompiledModelStream(const std::string& subgraph_name) const {
  if (compiled_models_info_.contains(subgraph_name)) {
    const auto& info = compiled_models_info_.at(subgraph_name);
    std::istream& stream = *context_bin_stream_;
    stream.seekg(info.start);
    std::string temp;
    temp.resize(info.size);
    stream.read(temp.data(), temp.size());
    return std::make_unique<std::istringstream>(std::move(temp));
  }
  return {};
}

EPCtxBinReader::~EPCtxBinReader() {
  // if (!context_bin_stream_.is_open()) {
  //   LOGS_DEFAULT(WARNING) << "Expected open context binary file\n";
  // }
  // context_bin_stream_.close();
}

// context_bin_name: full path to the context binary name
// external_weights: path to external weights if available
EPCtxBinWriter::EPCtxBinWriter(EPCtxHandler& ep_ctx_handler,
                               const fs::path& context_bin_name,
                               const fs::path& external_weights_full_path,
                               const openvino_ep::weight_info_map& shared_weights_info) : ep_ctx_handler_{ep_ctx_handler},
                                                                                          shared_weights_info_{shared_weights_info} {
  ORT_ENFORCE(!context_bin_stream_.is_open(), "Unexpected open context binary file");

  context_bin_path_name_ = ep_ctx_handler_.ep_context_model_path_ / context_bin_name;
  context_bin_stream_.open(context_bin_path_name_, std::ios::out | std::ios::binary);
  ORT_ENFORCE(context_bin_stream_.is_open(), "Context binary file failed to open");

  // Write the uninitialized header to advance the stream. The actual header
  // data will be written after all the context binary sections are written
  auto weights_section_pos = write_bytes(context_bin_stream_, header_);

  // External weights can either be copied as a hard link next to the context
  // model or stored inside the context binary. For now that external weights
  // are always stored in the context binary
  if (!external_weights_full_path.empty()) {
    if constexpr (save_weights_in_context_bin) {
      // Store weights in context binary
      header_.sections.weights = weights_section_pos;
      if (auto weights_stream = std::ifstream(external_weights_full_path, std::ios::binary)) {
        context_bin_stream_ << weights_stream.rdbuf();
        header_.sections.compiled_models = context_bin_stream_.tellp();
      }
    } else {
      // Skip weights section, compiled model
      header_.sections.compiled_models = weights_section_pos;

      // Create a hard link or copy
      auto new_weights_file_path = ep_ctx_handler_.ep_context_model_path_ / external_weights_full_path.filename();
      auto& original_weights_path = external_weights_full_path;
      try {
        std::filesystem::create_hard_link(original_weights_path, new_weights_file_path);
      } catch (const std::filesystem::filesystem_error& e) {
        LOGS_DEFAULT(WARNING) << "Failed to create hard link for weights: " << e.what() << " Falling back to copy.";
        std::filesystem::copy_file(original_weights_path, new_weights_file_path);
      }
    }
  }
}

EPCtxBinWriter::~EPCtxBinWriter() {
  if (!context_bin_stream_.is_open()) {
    LOGS_DEFAULT(WARNING) << "Expected open context binary file\n";
  }

  // Write maps
  header_.sections.compiled_models_map = context_bin_stream_.tellp();
  header_.sections.weights_map = write_bytes(context_bin_stream_, compiled_models_info_);
  write_bytes(context_bin_stream_, shared_weights_info_);

  // Update header
  context_bin_stream_.seekp(0);
  write_bytes(context_bin_stream_, header_);
  // write_bytes(context_bin_stream_, header_, 0);

  context_bin_stream_.close();

  /////////////////////////////////////////////////////////////////////////////////////////////
  //
  // TEMP CODE
  // Metadata is always read from epctx model location
  //
  if (auto filein = std::ifstream(context_bin_path_name_, std::ios::binary)) {
    context_bin_header read_header;
    read_bytes(filein, read_header);
    if (header_ != read_header) {
      LOGS_DEFAULT(ERROR) << "Shared weight info written/read incorrectly\n";
    }

    compiled_model_info_map read_compiled_models_info;
    filein.seekg(read_header.sections.compiled_models_map);
    read_bytes(filein, read_compiled_models_info);
    // read_bytes(filein, read_compiled_models_info, read_header.sections.compiled_models_map);
    if (compiled_models_info_ != read_compiled_models_info) {
      LOGS_DEFAULT(ERROR) << "Shared weight info written/read incorrectly\n";
    }

    weight_info_map read_shared_weights_info;
    filein.seekg(read_header.sections.weights_map);
    read_bytes(filein, read_shared_weights_info);
    // read_bytes(filein, read_shared_weights_info, read_header.sections.weights_map);
    if (shared_weights_info_ != read_shared_weights_info) {
      LOGS_DEFAULT(ERROR) << "Shared weight info written/read incorrectly\n";
    }
  }
  /////////////////////////////////////////////////////////////////////////////////////////////
}

std::ostream& EPCtxBinWriter::GetContextBinStream() {
  // Save stream position
  pre_blob_insert_ = context_bin_stream_.tellp();
  return context_bin_stream_;
}

void EPCtxBinWriter::PostInsertBlob(const std::string& blob_name) {
  std::streampos post_blob_insert = context_bin_stream_.tellp();
  std::streamoff size = post_blob_insert - pre_blob_insert_;
  compiled_models_info_[blob_name] = {pre_blob_insert_, size};
}

///////////////////////////////////////////////////////////////////////////////
// weight_map_value
// std::streampos write_bytes(KeyT &,struct onnxruntime::openvino_ep::weight_map_value const &)
std::streampos write_bytes(std::ostream& stream, const weight_map_value& value) {
  write_bytes(stream, value.location);
  write_bytes(stream, value.data_offset);
  write_bytes(stream, value.size);
  write_bytes(stream, value.dimensions);
  return write_bytes(stream, value.element_type);
}

void read_bytes(std::istream& stream, weight_map_value& value) {
  read_bytes(stream, value.location);
  read_bytes(stream, value.data_offset);
  read_bytes(stream, value.size);
  read_bytes(stream, value.dimensions);
  read_bytes(stream, value.element_type);
}

///////////////////////////////////////////////////////////////////////////////
// compiled_model_info_value
//
std::streampos write_bytes(std::ostream& stream, const compiled_model_info_value& value) {
  write_bytes(stream, value.start);
  return write_bytes(stream, value.size);
}

void read_bytes(std::istream& stream, compiled_model_info_value& value) {
  read_bytes(stream, value.start);
  read_bytes(stream, value.size);
}

bool compiled_model_info_value::operator==(const compiled_model_info_value& other) const {
  return (start == other.start) && (size == other.size);
}

///////////////////////////////////////////////////////////////////////////////
// context_bin_header
//
std::streampos write_bytes(std::ostream& stream, const context_bin_header& value) {
  write_bytes(stream, value.bin_version);
  write_bytes(stream, (std::streamoff)value.sections.weights);
  write_bytes(stream, (std::streamoff)value.sections.compiled_models);
  write_bytes(stream, (std::streamoff)value.sections.weights_map);
  return write_bytes(stream, (std::streamoff)value.sections.compiled_models_map);
}

void read_bytes(std::istream& stream, context_bin_header& value) {
  read_bytes(stream, value.bin_version);
  std::streamoff size;
  read_bytes(stream, size);
  value.sections.weights = std::streampos(size);
  read_bytes(stream, size);
  value.sections.compiled_models = std::streampos(size);
  read_bytes(stream, size);
  value.sections.weights_map = std::streampos(size);
  read_bytes(stream, size);
  value.sections.compiled_models_map = std::streampos(size);
}

bool context_bin_header::operator==(const context_bin_header& value) const {
  return (bin_version == value.bin_version) &&
         (sections.weights == value.sections.weights) &&
         (sections.compiled_models == value.sections.compiled_models) &&
         (sections.weights_map == value.sections.weights_map) &&
         (sections.compiled_models_map == value.sections.compiled_models_map);
}

}  // namespace openvino_ep
}  // namespace onnxruntime
