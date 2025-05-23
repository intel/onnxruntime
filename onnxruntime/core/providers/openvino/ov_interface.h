// Copyright (C) Intel Corporation
// Licensed under the MIT License

#pragma once

#include <vector>
#include <memory>
#include <fstream>
#include <sstream>
#include <utility>
#include <optional>

#include "openvino/openvino.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"
#include "openvino/pass/convert_fp32_to_fp16.hpp"
#include "openvino/frontend/manager.hpp"

#ifdef IO_BUFFER_ENABLED
#include <openvino/runtime/intel_gpu/ocl/ocl.hpp>
#endif

#include <string>

namespace onnxruntime {
namespace openvino_ep {
class OVCore;
class OVInferRequest;
class OVExeNetwork;

typedef ov::Tensor OVTensor;
typedef ov::ProfilingInfo OVProfilingInfo;
typedef ov::Model OVNetwork;
typedef std::shared_ptr<OVInferRequest> OVInferRequestPtr;
typedef std::shared_ptr<OVTensor> OVTensorPtr;

#ifdef IO_BUFFER_ENABLED
typedef ov::intel_gpu::ocl::ClContext* OVRemoteContextPtr;
typedef ov::RemoteContext OVRemoteContext;
#endif

struct ParameterShape {
  using ort_shape_t = std::vector<int64_t>;

  static ov::PartialShape ToOvPartialShape(const ort_shape_t& ort_shape) {
    std::vector<ov::Dimension> ov_shape(ort_shape.size());
    std::transform(ort_shape.begin(), ort_shape.end(), ov_shape.begin(), [](int64_t dim) {
      return dim == -1 ? ov::Dimension::dynamic() : ov::Dimension(dim);
    });
    return ov::PartialShape(ov_shape);
  }

  static ort_shape_t ToOrtShape(const ov::PartialShape& ov_shape) {
    ort_shape_t ort_shape(ov_shape.size());
    std::transform(ov_shape.begin(), ov_shape.end(), ort_shape.begin(), [](const auto& dim) {
      return dim.is_dynamic() ? -1 : dim.get_length();
    });
    return ort_shape;
  }

  ov::Shape ov_shape() const { return ov_.get_shape(); }
  const ov::PartialShape& ov() const { return ov_; }
  const ort_shape_t& ort() const { return ort_; }

  ParameterShape(const ort_shape_t& ort_shape) : ort_(ort_shape), ov_(ToOvPartialShape(ort_shape)) {
  }
  ParameterShape(const ov::PartialShape& ov_partial_shape) : ov_(ov_partial_shape), ort_(ToOrtShape(ov_partial_shape)) {
  }

 private:
  ort_shape_t ort_;
  ov::PartialShape ov_;
};

struct ParameterInfo {
  std::string name;
  uint32_t ov_index;
  uint32_t onnx_index;
  ov::element::Type type;
  ParameterShape shape;
};

std::optional<bool> queryOVProperty(const std::string& property, const std::string& device_type);

template <typename T>
class WeakSingleton {
 public:
  static std::shared_ptr<T> Get() {
    static std::weak_ptr<T> instance;
    static std::mutex mutex;

    auto ptr = instance.lock();
    if (!ptr) {
      std::lock_guard<std::mutex> lock(mutex);
      // ensure another thread didn't create an instance while this thread was waiting
      ptr = instance.lock();
      if (!ptr) {
        ptr = std::make_shared<T>();
        instance = ptr;
      }
    }
    return ptr;
  }

 protected:
  WeakSingleton() = default;
  virtual ~WeakSingleton() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(WeakSingleton);
};

struct OVCore : WeakSingleton<OVCore> {
  ov::Core core;

  // OV Interface For Reading Model
  std::shared_ptr<OVNetwork> ReadModel(std::string&& model_stream, const std::string& model_path);

  // OV Interface for Compiling OV Model Type
  OVExeNetwork CompileModel(std::shared_ptr<const OVNetwork>& ie_cnn_network,
                            std::string& hw_target,
                            ov::AnyMap& device_config,
                            const std::string& name);
  // OV Interface for Fast Compile
  OVExeNetwork CompileModel(const std::string& onnx_model,
                            std::string& hw_target,
                            ov::AnyMap& device_config,
                            const std::string& name);
  // OV Interface for Import model Stream
  OVExeNetwork ImportModel(std::istream& model_stream,
                           std::string hw_target,
                           const ov::AnyMap& device_config,
                           std::string name);
#ifdef IO_BUFFER_ENABLED
  OVExeNetwork CompileModel(std::shared_ptr<const OVNetwork>& model,
                            OVRemoteContextPtr context,
                            std::string name);
  OVExeNetwork ImportModel(std::shared_ptr<std::istringstream> model_stream,
                           OVRemoteContextPtr context,
                           std::string name);
#endif
  std::vector<std::string> GetAvailableDevices() const;
  std::vector<std::string> GetAvailableDevices(const std::string& device_type) const;
  void SetCache(const std::string& cache_dir_path);
  void SetStreams(const std::string& device_type, int num_streams);
};

class OVExeNetwork {
  ov::CompiledModel obj;

 public:
  explicit OVExeNetwork(ov::CompiledModel md) : obj(md) {}
  OVExeNetwork() : obj(ov::CompiledModel()) {}
  ov::CompiledModel& Get() { return obj; }
  OVInferRequest CreateInferRequest();
};

class OVInferRequest {
  struct ov_tensor_data_t {
    OVTensorPtr tensor_ptr;
    const void* ort_ptr;
  };

  ov::InferRequest ovInfReq;
  std::unordered_map<std::string, ov_tensor_data_t> bindings_cache_;

 public:
  uint32_t GetNumInputs();
  OVTensorPtr GetTensor(const std::string& name);
  std::string GetInputTensorName(uint32_t index);

  // Set tensor described param_info and ort_ptr. Call infer req tensor if ort_ptr is last set.
  void SetTensor(const ParameterInfo& param_info, void* ort_ptr) {
    SetTensorShapeOverride(param_info, param_info.shape, ort_ptr);
  }

  // Set tensor described param_info and ort_ptr. Overrides shape in param_info with shape_override. Call infer req tensor if ort_ptr is last set.
  void SetTensorShapeOverride(const ParameterInfo& param_info, const ParameterShape& shape_override, void* ort_ptr) {
    auto& cached_binding = bindings_cache_[param_info.name];
    if (cached_binding.ort_ptr != ort_ptr) {
      auto tensor_ptr = std::make_shared<ov::Tensor>(param_info.type, shape_override.ov_shape(), const_cast<void*>(ort_ptr));
      SetTensor(param_info.name, tensor_ptr);
      cached_binding = {tensor_ptr, ort_ptr};
    }
  }

  void SetTensor(const std::string& name, OVTensorPtr& blob);
  void Infer();
  explicit OVInferRequest(ov::InferRequest obj) : ovInfReq(std::move(obj)) {}
  OVInferRequest() : ovInfReq(ov::InferRequest()) {}
  ov::InferRequest& GetNewObj() {
    return ovInfReq;
  }
};
}  // namespace openvino_ep
}  // namespace onnxruntime
