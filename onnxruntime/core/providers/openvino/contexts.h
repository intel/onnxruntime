// Copyright (C) Intel Corporation
// Licensed under the MIT License

#pragma once

#include <vector>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <filesystem>
#include <memory>
#include "core/common/common.h"
#include "core/providers/shared_library/provider_api.h"
#include "core/providers/openvino/ov_interface.h"
#include "core/providers/shared_library/provider_api.h"

namespace onnxruntime {
namespace openvino_ep {

namespace fs = std::filesystem;

class SharedContext : public WeakSingleton<SharedContext> {
  // Keep the core alive as long as the shared SharedContext are alive.
  std::shared_ptr<OVCore> OVCore_;

 public:
  SharedContext() : OVCore_(OVCore::Get()) {}
  struct SharedWeights {
    struct Header {
      uint32_t bin_version = 1;
      uint64_t footer_offset = 0;
    } header_;
    struct Footer {
      uint64_t subgraph_offset;
      size_t subgraph_length;
      uint64_t metadata_offset;
      size_t metadata_length;
    } footer_;

    struct Metadata {
      struct Key {
        std::string name;
        bool operator==(const Key&) const = default;
      };
      struct Hash {
        std::size_t operator()(const Key& key) const noexcept {
          return std::hash<std::string>()(key.name);
        }
      };
      struct Value {
        std::string location;
        uint32_t data_offset;
        uint32_t size;
        std::vector<size_t> dimensions;
        std::int32_t element_type;
        std::shared_ptr<ov::Tensor> tensor;
      };
      using Map = std::unordered_map<Key, Value, Hash>;
      void writeMetadataToBinaryFile(SharedContext& shared_context, const Metadata::Map& metadata);
      void readMetadataFromBinaryFile(SharedContext& shared_context, Metadata::Map& metadata);
    } metadata_;

    struct SubgraphMetadata {
      struct Key {
        std::string name;
        bool operator==(const Key&) const = default;
      };
      struct Hash {
        std::size_t operator()(const Key& key) const noexcept {
          return std::hash<std::string>()(key.name);
        }
      };
      struct Value {
        uint64_t epctx_offset;
        size_t epctx_length;
      };
      using Map = std::unordered_map<Key, Value, Hash>;
      void writeSubgraphDataToBinaryFile(SharedContext& shared_context,
                                         const SubgraphMetadata::Map& subgraph_metadata);
      void readSubgraphDataFromBinaryFile(SharedContext& shared_context,
                                          SubgraphMetadata::Map& subgraph_metadata);
    } subgraph_metadata_;

    struct WeightsFile {
      ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(WeightsFile);
      WeightsFile() = delete;
      explicit WeightsFile(std::filesystem::path filename);

      void load_weights(size_t file_offset, void* data, size_t size);

     private:
      std::ifstream file_;
      size_t weights_size_;
    };

    struct SharedBinFile {
      fs::path shared_bin_filename;
      std::fstream bin_file_;
      size_t bin_size_;

      SharedBinFile() = default;  // Default constructor
      ~SharedBinFile() {
        if (bin_file_.is_open()) {
          bin_file_.close();  // Close file when object is destroyed
        }
      }

      void openBinFile(const fs::path shared_bin_filename) {
        // Check if the file exists before trying to open
        if (!fs::exists(shared_bin_filename)) {
          std::ofstream createFile(shared_bin_filename, std::ios::binary);  // Create an empty binary file
          if (!createFile) {
            ORT_THROW("Failed to create the shared bin file!");
          }
          createFile.close();
        }

        // Check if the file is accessible for reading and writing
        fs::perms file_perms = fs::status(shared_bin_filename).permissions();

        if ((file_perms & fs::perms::owner_read) == fs::perms::none ||
            (file_perms & fs::perms::owner_write) == fs::perms::none) {
          ORT_THROW("Failed to open shared bin file! Insufficient permissions for file " + shared_bin_filename + ".");
        }

        if (!bin_file_.is_open()) {  // Prevent reopening
          bin_file_.open(shared_bin_filename, std::ios::in | std::ios::out | std::ios::binary);
          bin_size_ = bin_file_.seekg(0, std::ios::end).tellg();
          bin_file_.seekg(0, std::ios::beg);  // Reset to the beginning of the file

          if (!bin_file_) {
            ORT_THROW("Failed to open shared bin file!");
          }
        }
      }
      void readBinFile(SharedContext& shared_context_);
      void dumpBinFile(SharedContext& shared_context_);
    } shared_bin_file;

    fs::path external_weight_filename;
    std::unique_ptr<WeightsFile> mapped_weights;
    Metadata::Map metadata;
    SubgraphMetadata::Map subgraph_metadata;
  } shared_weights;
};

using config_t = std::map<std::string, ov::AnyMap>;
using reshape_t = std::map<std::string, ov::PartialShape>;

struct ProviderInfo {
  std::string device_type{""};             // [device_type]: Overrides the accelerator hardware type and
                                           // precision with these values at runtime.
  std::string precision{""};               // [precision]: Sets the inference precision for execution.
                                           // Supported precision for devices are
                                           // CPU=FP32, GPU=FP32,FP16, NPU=FP16.
                                           // Not setting precision will execute with optimized precision for
                                           // best inference latency. set Precision=ACCURACY for executing
                                           // models with input precision for best accuracy.
  uint32_t num_of_threads{0};              // [num_of_threads]: Overrides the accelerator default value of
                                           // number of threads with this value at runtime.
  config_t load_config{};                  // JSON config map to load custom OV parameters.
  fs::path cache_dir{""};                  // [cache_dir]: specify the path to
                                           // dump and load the blobs for the model caching/kernel caching
                                           // (GPU) feature. If blob files are already present,
                                           // it will be directly loaded.
  reshape_t reshape{};                     // Used for reshaping the ov input tensor shape at runtime.
  std::string model_priority{"DEFAULT"};   // High-level OpenVINO model priority hint
                                           // Defines what model should be provided with more performant
                                           // bounded resource first
  uint32_t num_streams{1};                 // [num_streams]: Option that specifies the number of parallel
                                           // inference requests to be processed on a given `device_type`.
                                           // Overrides the accelerator default value of number of streams
                                           // with this value at runtime.
  void* context{nullptr};                  // OpenCL context
  bool enable_opencl_throttling{false};    // [enable_opencl_throttling]: Enables OpenCL queue throttling for
                                           // GPU device (Reduces CPU Utilization when using GPU)
  bool disable_dynamic_shapes{false};      // [disable_dynamic_shapes]:  Rewrite dynamic shaped models to
                                           // static shape at runtime and execute.
  bool enable_qdq_optimizer{false};        // Enables QDQ pruning for efficient inference latency with NPU
  bool enable_causallm{false};             // Enables Causal LM Compilation for ORT GenAI OVEP Pass
  bool so_context_enable{false};           // ORT session option
  bool so_disable_cpu_ep_fallback{false};  // ORT session option
  bool so_context_embed_mode{false};       // ORT session option
  bool so_share_ep_contexts{false};        // ORT session option
  fs::path so_context_file_path{};         // ORT session option
  const ConfigOptions* config_options{NULL};
  const std::unordered_set<std::string> valid_provider_keys = {"device_type", "device_id", "device_luid", "cache_dir", "precision",
                                                               "load_config", "context", "num_of_threads", "model_priority", "num_streams", "enable_opencl_throttling", "enable_qdq_optimizer",
                                                               "enable_causallm", "disable_dynamic_shapes", "reshape_input"};
};

// Holds context applicable to the entire EP instance.
struct SessionContext : ProviderInfo {
  SessionContext(const ProviderInfo& info) : ProviderInfo{info} {}
  std::vector<bool> deviceAvailableList = {true, true, true, true, true, true, true, true};
  std::filesystem::path onnx_model_path_name;
  uint32_t onnx_opset_version{0};
  mutable bool is_wholly_supported_graph = false;  // Value is set to mutable to modify from capability
  mutable bool has_external_weights = false;       // Value is set to mutable to modify from capability
  const std::vector<uint32_t> OpenVINO_Version = {OPENVINO_VERSION_MAJOR, OPENVINO_VERSION_MINOR};
  const std::string openvino_sdk_version = std::to_string(OPENVINO_VERSION_MAJOR) + "." + std::to_string(OPENVINO_VERSION_MINOR);
};

// Holds context specific to subgraph.
struct SubGraphContext {
  using string_index_map_t = std::unordered_map<std::string, uint32_t>;
  bool has_dynamic_input_shape = false;
  bool enable_batching = false;
  bool set_npu_config = false;
  bool is_constant = false;
  void* context = 0;
  std::string subgraph_name;
  string_index_map_t input_names;
  string_index_map_t output_names;
  std::string model_precision;
  bool is_ep_ctx_graph = false;
};

}  // namespace openvino_ep
}  // namespace onnxruntime
