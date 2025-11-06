// Copyright (C) Intel Corporation
// Licensed under the MIT License

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <filesystem>
#include <fstream>
#include <shared_mutex>
#include <mutex>

#include "openvino/runtime/core.hpp"
#include "ov_shared_resource_manager.h"
#include "weak_singleton.h"

namespace onnxruntime {
namespace openvino_ep {

class SharedContext {
 public:
  SharedContext() = default;

  struct Metadata {
    struct Value {
      struct {
        std::filesystem::path location{};
        size_t data_offset{0};
        size_t size{0};
      } serialized;

      std::shared_ptr<const ov::Tensor> tensor;
    };
    using Map = std::unordered_map<std::string, Value>;
  };

  bool IsSharedWeight(const std::string& name) const {
    std::shared_lock lock(mutex_);
    return metadata_.contains(name);
  }

  void AddExternalWeight(const std::string& name, size_t offset, size_t size, const std::filesystem::path& location) {
    Metadata::Value value;
    value.serialized.data_offset = offset;
    value.serialized.size = size;
    value.serialized.location = location;
    std::unique_lock lock(mutex_);
    metadata_[name] = std::move(value);
  }

  Metadata::Map GetMetadataCopy() const {
    std::shared_lock lock(mutex_);
    return metadata_;
  }

  void SetSharedWeightsOnInferRequest(ov::InferRequest& ir, const std::filesystem::path& model_dir);

 private:
  struct WeightsFile {
    ORT_DISALLOW_COPY_AND_ASSIGNMENT(WeightsFile);
    WeightsFile() = delete;
    virtual ~WeightsFile() = default;
    explicit WeightsFile(const std::filesystem::path& filename);
    void LoadWeights(size_t file_offset, void* data, size_t size);
    void* TryGetOrCreateDeviceMapping(std::optional<ov::RemoteContext>& remote_context);
    size_t Size() const { return weights_size_; }

   private:
    std::ifstream file_;
    std::filesystem::path file_path_;
    size_t weights_size_;
    struct MappingContainer {
      void* ptr_{nullptr};
      ov::Tensor tensor_;
    };
    std::map<std::string, MappingContainer> imported_device_tensors_;
  };

  void LoadTensorFromFile(
      Metadata::Value& value,
      const std::filesystem::path& model_dir,
      std::optional<ov::RemoteContext>& remote_context,
      const ov::element::Type& element_type,
      const ov::Shape& dimensions);

  mutable std::shared_mutex mutex_;
  std::unordered_map<std::filesystem::path, std::unique_ptr<WeightsFile>> weight_files_;
  Metadata::Map metadata_;
};

class SharedContextManager : public WeakSingleton<SharedContextManager> {
 public:
  std::shared_ptr<SharedContext> GetOrCreateActiveSharedContext(const std::filesystem::path& model_path) {
    return manager_.GetActiveResourceOrCreate(model_path);
  }

  std::shared_ptr<SharedContext> GetOrCreateSharedContext(const std::filesystem::path& model_path) {
    return manager_.GetOrCreateResource(model_path);
  }

  void ClearActiveSharedContext() {
    manager_.ClearActiveResource();
  }

 private:
  SharedResourceManager<std::filesystem::path, SharedContext> manager_;
};

}  // namespace openvino_ep
}  // namespace onnxruntime
