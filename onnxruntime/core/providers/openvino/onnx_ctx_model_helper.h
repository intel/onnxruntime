// Copyright (C) Intel Corporation
// Licensed under the MIT License

#pragma once

#include <sstream>
#include <string>
#include <memory>
#include <streambuf>

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
  EPCtxHandler(std::string ov_sdk_version, const logging::Logger& logger);
  EPCtxHandler(const EPCtxHandler&) = delete;  // No copy constructor
  Status ExportEPCtxModel(const std::string& model_name);
  bool CheckForOVEPCtxNodeInGraph(const GraphViewer& graph_viewer) const;
  bool CheckForOVEPCtxNode(const Node& node) const;
  Status AddOVEPCtxNodeToGraph(const GraphViewer& graph_viewer,
                               const std::string& graph_name,
                               const bool embed_mode,
                               std::string&& model_blob_str) const;
  std::unique_ptr<std::istream> GetModelBlobStream(SharedContext& shared_context_,
                                                   const std::string &subgraph_name,
                                                   const GraphViewer& graph_viewer) const;
  InlinedVector<const Node*> GetEPCtxNodes() const;

 private:
  const std::string openvino_sdk_version_;
  std::unique_ptr<Model> epctx_model_;
  const logging::Logger& logger_;
};

// class LimitedFileStreambuf : public std::streambuf {
// private:
//     std::fstream& file; // Reference to the existing file stream
//     long start, end; // Start and end positions

// protected:
//     int_type underflow() override {
//         if (file.tellg() >= end || file.eof())
//             return traits_type::eof(); // Stop reading if we reach the limit

//         return file.get(); // Read next character directly from the file
//     }

// public:
//     LimitedFileStreambuf(std::fstream& bin_file_, long start, long end)
//         : file(bin_file_), start(start), end(end) {
//         file.clear(); // Clear error flags in case of previous reads
//         file.seekg(start); // Move file pointer to the start position
//     }
// };

}  // namespace openvino_ep
}  // namespace onnxruntime
