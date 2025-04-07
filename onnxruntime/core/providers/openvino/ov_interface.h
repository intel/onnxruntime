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
                           bool enable_causallm,
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
  std::string _device;
  bool _stateful_llm;
 public:
  explicit OVExeNetwork(ov::CompiledModel md, std::string device, bool stateful_llm = false) 
      : obj(md), _device(device), _stateful_llm(stateful_llm) {}
  OVExeNetwork() : obj(ov::CompiledModel()) {}
  ov::CompiledModel& Get() { return obj; }
  std::shared_ptr<OVInferRequest> CreateInferRequest();
};

class OVInferRequest {
 protected:
  ov::InferRequest ovInfReq;

 public:
  uint32_t GetNumInputs();
  OVTensorPtr GetTensor(const std::string& name);
  std::string GetInputTensorName(uint32_t index);
  void SetTensor(const std::string& name, OVTensorPtr& blob);
  virtual void StartAsync();
  virtual void Infer();
  void WaitRequest();
  void QueryStatus();
  explicit OVInferRequest(ov::InferRequest obj) : ovInfReq(std::move(obj)) {}
  OVInferRequest() : ovInfReq(ov::InferRequest()) {}
  ov::InferRequest& GetNewObj() {
    return ovInfReq;
  }
  virtual void RewindKVCache(size_t index) {};
};

class StatefulOVInferRequest : public OVInferRequest {
 public:
  explicit StatefulOVInferRequest(ov::InferRequest obj, std::string d) : OVInferRequest(std::move(obj)), device(d) {}

  void StartAsync() override;
  void Infer() override;
  void RewindKVCache(size_t index) override;

 private:
  void _pre_infer();
  std::string device;

  // For NPU, we need to cache input_ids & position_ids to support chat-mode.
  std::vector<int64_t> cached_input_ids;
  std::vector<int64_t> cached_position_ids;
};

}  // namespace openvino_ep
}  // namespace onnxruntime
