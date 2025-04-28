// Copyright (C) Intel Corporation
// Licensed under the MIT License

#pragma once

#include <sstream>
#include <string>
#include <memory>
#include <functional>

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
  std::unique_ptr<std::istream> static GetModelBlobStream(const std::filesystem::path& so_context_file_path, const GraphViewer& graph_viewer);
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

  friend std::streampos write_bytes(std::ostream&, const weight_map_value&);
  friend void read_bytes(std::istream&, weight_map_value&);

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
  compiled_model_info_value(std::streampos s, std::streamoff o) : start{s}, size{o} {}
  bool operator==(const compiled_model_info_value& other) const;

  friend std::streampos write_bytes(std::ostream&, const compiled_model_info_value&);
  friend void read_bytes(std::istream&, compiled_model_info_value&);

  std::streampos start;
  std::streamoff size;
};

using compiled_model_info_map = io_unordered_map<std::string, compiled_model_info_value>;

struct context_bin_header : streamable<context_bin_header> {
  context_bin_header() = default;
  bool operator==(const context_bin_header& value) const;

  friend std::streampos write_bytes(std::ostream&, const context_bin_header&);
  friend void read_bytes(std::istream&, context_bin_header&);

  uint32_t bin_version{constants::ep_context::expected_bin_version};
  struct {
    std::streampos weights{0};
    std::streampos compiled_models{0};
    std::streampos weights_map{0};
    std::streampos compiled_models_map{0};
  } sections;
};

struct EPCtxBinReader {
  EPCtxBinReader(std::unique_ptr<std::istream>& context_binary_stream,
                 weight_info_map& shared_weight_info,
                 fs::path context_model_parent_path);
  ~EPCtxBinReader();
  std::unique_ptr<std::istringstream> GetCompiledModelStream(const std::string& subgraph_name) const;
  friend EPCtxHandler;

 private:
  std::unique_ptr<std::istream> context_bin_stream_;
  compiled_model_info_map compiled_models_info_;
  context_bin_header header_;
  std::ifstream external_weights_stream_;
};

struct EPCtxBinWriter {
  EPCtxBinWriter(EPCtxHandler& ep_ctx_handler,
                 const fs::path& context_bin_name,
                 const fs::path& external_weights_full_path,
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
