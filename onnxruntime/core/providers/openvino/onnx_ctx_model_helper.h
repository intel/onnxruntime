// Copyright (C) Intel Corporation
// Licensed under the MIT License

#pragma once

#include <sstream>
#include <string>
#include <memory>

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/openvino/contexts.h"

namespace onnxruntime {
namespace openvino_ep {

// Utilities to handle EPContext node export and parsing of an EPContext node
// to create the compiled_model object to infer on
static const char EPCONTEXT_OP[] = "EPContext";
static const char EMBED_MODE[] = "embed_mode";
static const char EP_CACHE_CONTEXT[] = "ep_cache_context";
static const char EP_SDK_VER[] = "ep_sdk_version";
static const char SOURCE[] = "source";

class EPCtxHandler {
 public:
  EPCtxHandler(const logging::Logger& logger, fs::path path);
  EPCtxHandler(const EPCtxHandler&) = delete;  // No copy constructor
  bool CheckForOVEPCtxNodeInGraph(const GraphViewer& graph_viewer) const;
  bool CheckForOVEPCtxNode(const Node& node) const;
  Status AddOVEPCtxNodeToGraph(const GraphViewer& graph_viewer,
                               const std::string& graph_name,
                               const bool embed_mode,
                               std::string&& model_blob_str) const;
  std::unique_ptr<std::istream> GetModelBlobStream(const std::filesystem::path& so_context_file_path, const GraphViewer& graph_viewer) const;
  InlinedVector<const Node*> GetEPCtxNodes() const;
  bool StartReadingContextBin(const fs::path& context_binary_name, openvino_ep::weight_info_map& shared_weight_info_);
  bool FinishReadingContextBin();
  std::ostream& PreInsertBlob();
  void PostInsertBlob(const std::string& blob_name);
  bool StartWritingContextBin(const fs::path& context_binary_name);
  bool FinishWritingContextBin(const openvino_ep::weight_info_map& shared_weight_info_);

 private:
  struct compiled_model_info_value : streamable<compiled_model_info_value> {
    compiled_model_info_value() = default;
    compiled_model_info_value(std::streampos s, std::streampos e) : start{s}, end{e} {}
    bool operator==(const compiled_model_info_value& other) const;

    std::streampos start;
    std::streampos end;
  };

  using compiled_model_info_map = io_unordered_map<std::string, compiled_model_info_value>;

  const logging::Logger& logger_;
  const fs::path ep_context_model_path;
  std::unique_ptr<Model> epctx_model_;
  std::fstream context_bin_stream_;
  std::streampos pre_blob_insert_;
  compiled_model_info_map compiled_models_info_;
};

}  // namespace openvino_ep
}  // namespace onnxruntime
