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
  using onnx_shape_t = std::vector<int64_t>;

 private:
  onnx_shape_t onnx_;
  ov::PartialShape ov_;

 public:
  static ov::PartialShape ToOvPartialShape(const onnx_shape_t& onnx_shape) {
    std::vector<ov::Dimension> ov_shape(onnx_shape.size());
    std::transform(onnx_shape.begin(), onnx_shape.end(), ov_shape.begin(), [](int64_t dim) {
      return dim == -1 ? ov::Dimension::dynamic() : ov::Dimension(dim);
    });
    return ov::PartialShape(ov_shape);
  }

  static ov::Shape ToOvShape(const onnx_shape_t& onnx_shape) {
    return ToOvPartialShape(onnx_shape).get_shape();
  }

  static onnx_shape_t ToOnnxShape(const ov::PartialShape& ov_shape) {
      onnx_shape_t onnx_shape(ov_shape.size());
    std::transform(ov_shape.begin(), ov_shape.end(), onnx_shape.begin(), [](const auto& dim) {
      return dim.is_dynamic() ? -1 : dim.get_length();
    });
      return onnx_shape;
  }

  static bool IsDynamic(const ov::PartialShape& ov_shape) {
    return ov_shape.is_dynamic();
  }
  static bool IsDynamic(const onnx_shape_t& onnx_shape) {
    return std::any_of(onnx_shape.begin(), onnx_shape.end(), [](const auto& dim) { return dim == -1; });
  }

  ov::Shape ov_shape() const { return ov_.get_shape(); }

  const ov::PartialShape& ov() const { return ov_; }
  const onnx_shape_t& onnx() const { return onnx_; }

  ParameterShape reshape(const onnx_shape_t& new_onnx_shape) const {
    return ParameterShape(new_onnx_shape);
  };
  ParameterShape reshape(const ov::Shape& new_ov_shape) const {
    return ParameterShape(new_ov_shape);
  };

  ParameterShape(const onnx_shape_t& onnx_shape) : onnx_(onnx_shape), ov_(ToOvPartialShape(onnx_shape)) {
  }
  ParameterShape(const ov::PartialShape& ov_partial_shape) : ov_(ov_partial_shape), onnx_(ToOnnxShape(ov_partial_shape)) {
  }
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

  OVExeNetwork StatefulCompileModel(std::shared_ptr<OVNetwork>& model,
                                    std::string& hw_target,
                                    const ov::AnyMap& device_config);
  // OV Interface for Compiling OV Model Type
  OVExeNetwork CompileModel(std::shared_ptr<const OVNetwork>& ie_cnn_network,
                            std::string& hw_target,
                            ov::AnyMap& device_config,
                            bool enable_causallm,
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
  std::vector<std::string> GetAvailableDevices() const;
  std::vector<std::string> GetAvailableDevices(const std::string& device_type) const;
  void SetCache(const std::string& cache_dir_path);
  void SetStreams(const std::string& device_type, int num_streams);
};

class OVExeNetwork {
  ov::CompiledModel compiled_model_obj;
  std::string target_device;
  bool is_stateful_causallm;

 public:
  explicit OVExeNetwork(ov::CompiledModel compiled_model, std::string device, bool stateful_causallm = false)
      : compiled_model_obj(compiled_model), target_device(device), is_stateful_causallm(stateful_causallm) {}
  OVExeNetwork() : compiled_model_obj(ov::CompiledModel()) {}
  ov::CompiledModel& Get() { return compiled_model_obj; }
  std::shared_ptr<OVInferRequest> CreateInferRequest();
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
    auto& cached_binding = bindings_cache_[param_info.name];
    if (cached_binding.ort_ptr != ort_ptr) {
      auto tensor_ptr = std::make_shared<ov::Tensor>(param_info.type, param_info.shape.ov_shape(), const_cast<void*>(ort_ptr));
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
  virtual void RewindKVCache(size_t index) {}
};

class StatefulOVInferRequest : public OVInferRequest {
 public:
  explicit StatefulOVInferRequest(ov::InferRequest infer_request, std::string device);

  void StartAsync() override;
  void Infer() override;
  void RewindKVCache(size_t index) override;
  void FillTensor(const std::string& tensor_name, const ov::element::Type& type,
                   const std::vector<size_t>& shape, int32_t fill_value);
  void CacheTensor(const std::string& tensor_name, std::vector<int64_t>& cache);
  void SetTensorFromCache(const std::string& tensor_name, const std::vector<int64_t>& cache_data);
  std::optional<ov::Tensor> FindTensor(const std::string& tensor_name);

 private:
  void PreProcessInferRequest();
  std::string target_device;

  // If prefill_use_full_chat_history is true, cache the "input_ids" & "position_ids" tensors,
  // and ensure that full chat history is passed for each prefill call.
  bool prefill_use_full_chat_history = false;
  std::vector<int64_t> cached_input_ids;
  std::vector<int64_t> cached_position_ids;
};

}  // namespace openvino_ep
}  // namespace onnxruntime
