// Copyright (C) Intel Corporation
// Licensed under the MIT License

#pragma once

#include <vector>
#include <map>
#include <unordered_map>
#include <string>
#include <filesystem>
#include <memory>
#include "core/common/common.h"
#include "core/providers/openvino/ov_interface.h"

namespace onnxruntime {
namespace openvino_ep {

namespace fs = std::filesystem;

struct SharedContext {
  struct SharedWeights {
    struct Header {
      uint32_t bin_version=1;
      uint32_t footer_offset;
      Header(uint32_t bin_in, uint32_t footer_in) :
        bin_version(bin_in), footer_offset(footer_in){}
    };
    struct Footer {
      uint32_t subgraph_offset;
      uint32_t subgraph_length;
      uint32_t metadata_offset;
      uint32_t metadata_length;
      Footer(uint32_t subgraph_offset_in, uint32_t subgraph_length_in,
             uint32_t metadata_offset_in, uint32_t metadata_length_in) :
        subgraph_offset(subgraph_offset_in), subgraph_length(subgraph_length_in),
        metadata_offset(metadata_offset_in), metadata_length(metadata_length_in) {}
    };
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
      friend std::ostream& operator<<(std::ostream& right, const Metadata::Map& metadata);
      friend std::istream& operator>>(std::istream& right, Metadata::Map& metadata);
    };

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
        uint32_t epctx_offset;
        uint32_t epctx_length;
      };
      using Map = std::unordered_map<Key, Value, Hash>;
      friend std::ostream& operator<<(std::ostream& right, const SubgraphMetadata::Map& subgraph_metadata);
      friend std::istream& operator>>(std::istream& right, SubgraphMetadata::Map& subgraph_metadata);
    };

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
      // ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SharedBinFile);
      // SharedBinFile() = delete;
      // SharedBinFile(fs::path shared_bin_filename) :
      // bin_file_(shared_bin_filename, std::ios::out | std::ios::app| std::ios::binary) {
      //   if(bin_file_.is_open())
      //     std::cout << " Bin file opened " << std::endl;
      // }
      fs::path shared_bin_filename;
      std::ofstream bin_file_;

      SharedBinFile() = default;  // Default constructor
      ~SharedBinFile() = default; // Prevent closing the file automatically

      void openBinFile(fs::path shared_bin_filename) {
        if (!bin_file_.is_open()) {  // Prevent reopening
          bin_file_.open(shared_bin_filename, std::ios::out | std::ios::app | std::ios::binary);
          if (!bin_file_) {
              throw std::runtime_error("Failed to open log file!");
          }
        }
      }
    }shared_bin_file;

    fs::path external_weight_filename;
    std::unique_ptr<WeightsFile> mapped_weights;
    std::unique_ptr<Header> header_;
    std::unique_ptr<Footer> footer_;
    // std::unique_ptr<SharedBinFile> shared_bin_file;
    Metadata::Map metadata;
    SubgraphMetadata::Map subgraph_metadata;
  }shared_weights;
};

using config_t = std::map<std::string, ov::AnyMap>;

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
  bool so_context_enable{false};           // ORT session option
  bool so_disable_cpu_ep_fallback{false};  // ORT session option
  bool so_context_embed_mode{false};       // ORT session option
  bool so_share_ep_contexts{false};        // ORT session option
  fs::path so_context_file_path{};         // ORT session option
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
