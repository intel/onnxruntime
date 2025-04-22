// Copyright (C) Intel Corporation
// Licensed under the MIT License

#pragma once

#include <sstream>
#include <string>
#include <memory>

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/openvino/ov_interface.h"
#include "core/providers/openvino/serialization_helper.h"

namespace fs = std::filesystem;

namespace onnxruntime {
namespace openvino_ep {

// Utilities to handle EPContext node export and parsing of an EPContext node
// to create the compiled_model object to infer on

struct SessionContext;
struct EPCtxBinReader;
struct EPCtxBinWriter;

class EPCtxHandler {
 public:
  EPCtxHandler(const SessionContext& session_context);
  EPCtxHandler(const EPCtxHandler&) = delete;  // No copy constructor
  bool static CheckForOVEPCtxNodeInGraph(const GraphViewer& graph_viewer);
  bool static CheckForOVEPCtxNode(const Node& node);
  Status AddOVEPCtxNodeToGraph(const GraphViewer& graph_viewer,
                               const std::string& graph_name,
                               const bool embed_mode,
                               std::string&& model_blob_str) const;
  std::unique_ptr<std::istream> GetModelBlobStream(const std::filesystem::path& so_context_file_path, const GraphViewer& graph_viewer) const;
  InlinedVector<const Node*> GetEPCtxNodes() const;

 private:
  friend EPCtxBinReader;
  friend EPCtxBinWriter;

 private:
  const logging::Logger& logger_;
  fs::path ep_context_model_path_;
  std::unique_ptr<Model> epctx_model_;
  std::ifstream context_bin_stream_;
  std::optional<std::iostream> external_weights_stream_;
};

struct weight_map_value : streamable<weight_map_value> {
  weight_map_value() = default;
  weight_map_value(auto l, auto d_o, auto s, auto d, auto et) : location{l}, data_offset{d_o}, size{s}, dimensions{d}, element_type{et} {}
  bool operator==(const weight_map_value& other) const;

  template <typename S>
  friend std::streampos write_bytes(S& stream, const weight_map_value& value) {
    write_bytes(stream, value.location);
    write_bytes(stream, value.data_offset);
    write_bytes(stream, value.size);
    write_bytes(stream, value.dimensions);
    return write_bytes(stream, value.element_type);
  }

  template <typename S>
  friend void read_bytes(S& stream, weight_map_value& value) {
    read_bytes(stream, value.location);
    read_bytes(stream, value.data_offset);
    read_bytes(stream, value.size);
    read_bytes(stream, value.dimensions);
    read_bytes(stream, value.element_type);
  }

  std::string location;
  unsigned int data_offset{0};
  unsigned int size{0};
  std::vector<size_t> dimensions;
  std::int32_t element_type{0};
  std::shared_ptr<ov::Tensor> tensor;
};

using weight_info_map = io_unordered_map<std::string, weight_map_value>;

struct compiled_model_info_value : streamable<compiled_model_info_value> {
  compiled_model_info_value() = default;
  compiled_model_info_value(std::streampos s, std::streamoff o) : start{s}, offset{o} {}
  bool operator==(const compiled_model_info_value& other) const;

  std::streampos start;
  std::streamoff offset;
};

using compiled_model_info_map = io_unordered_map<std::string, compiled_model_info_value>;

struct context_bin_header : streamable<context_bin_header> {
  context_bin_header() = default;

  uint32_t bin_version{constants::ep_context::expected_bin_version};
  struct {
    std::streampos weights{0};
    std::streampos compiled_models{0};
    std::streampos weights_map{0};
    std::streampos compiled_models_map{0};
  } sections;

  template <typename S>
  friend std::streampos write_bytes(S& stream, const context_bin_header& value) {
    write_bytes(stream, value.bin_version);
    write_bytes(stream, (std::streamoff)value.sections.weights);
    write_bytes(stream, (std::streamoff)value.sections.compiled_models);
    write_bytes(stream, (std::streamoff)value.sections.weights_map);
    return write_bytes(stream, (std::streamoff)value.sections.compiled_models_map);
    // write_bytes(stream, value.sections.weights.state());
    // write_bytes(stream, value.sections.compiled_models.state());
    // write_bytes(stream, value.sections.weights_map.state());
    // return write_bytes(stream, value.sections.compiled_models_map.state());
    // stream << value.sections.weights;
    // stream << value.sections.compiled_models;
    // stream << value.sections.weights_map;
    // stream << value.sections.compiled_models_map;
    return stream.tellp();
  }

  template <typename S>
  friend void read_bytes(S& stream, context_bin_header& value) {
    read_bytes(stream, value.bin_version);
    // read_bytes(stream, value.sections.weights);
    // read_bytes(stream, value.sections.compiled_models);
    // read_bytes(stream, value.sections.weights_map);
    // read_bytes(stream, value.sections.compiled_models_map);
    // stream >> value.sections.weights;
    // stream >> value.sections.compiled_models;
    // stream >> value.sections.weights_map;
    // stream >> value.sections.compiled_models_map;
    std::streamoff offset;
    read_bytes(stream, offset);
    value.sections.weights = std::streampos(offset);
    read_bytes(stream, offset);
    value.sections.compiled_models = std::streampos(offset);
    read_bytes(stream, offset);
    value.sections.weights_map = std::streampos(offset);
    read_bytes(stream, offset);
    value.sections.compiled_models_map = std::streampos(offset);
  }

  bool operator==(const context_bin_header& value) const {
    return (bin_version == value.bin_version) &&
           (sections.weights == value.sections.weights) &&
           (sections.compiled_models == value.sections.compiled_models) &&
           (sections.weights_map == value.sections.weights_map) &&
           (sections.compiled_models_map == value.sections.compiled_models_map);
  }
};

struct EPCtxBinReader {
  EPCtxBinReader(EPCtxHandler& ep_ctx_handler,
                 const fs::path& context_binary_name,
                 weight_info_map& shared_weight_info);
  ~EPCtxBinReader();
  friend EPCtxHandler;

 private:
  EPCtxHandler& ep_ctx_handler_;
  std::ifstream context_bin_stream_;
  compiled_model_info_map compiled_models_info_;
  context_bin_header header_;
  std::optional<std::ifstream> external_weights_stream_;
};

struct EPCtxBinWriter {
  EPCtxBinWriter(EPCtxHandler& ep_ctx_handler,
                 const fs::path& context_bin_name,
                 std::optional<fs::path> external_weights,
                 const weight_info_map& shared_weights_info);
  ~EPCtxBinWriter();

  std::ostream& GetContextBinStream();
  void PostInsertBlob(const std::string& blob_name);
  std::string GetContextBinPath() { return context_bin_path_name_.string(); }

 private:
  EPCtxHandler& ep_ctx_handler_;
  const weight_info_map& shared_weights_info_;
  std::ofstream context_bin_stream_;
  std::streampos pre_blob_insert_;
  compiled_model_info_map compiled_models_info_;
  context_bin_header header_;
  fs::path context_bin_path_name_;
  std::streampos context_bin_stream_start_pos;
};

}  // namespace openvino_ep
}  // namespace onnxruntime
