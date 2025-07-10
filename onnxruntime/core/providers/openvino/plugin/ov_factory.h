// Copyright (C) Intel Corporation
// Licensed under the MIT License

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <set>

#include "ov_provider.h"
#include "openvino/openvino.hpp"

namespace onnxruntime {
namespace openvino_ep {

class OpenVINOEpPluginFactory : public OrtEpFactory, public ApiPtrs {
 public:
  OpenVINOEpPluginFactory(const std::string& ep_name, ApiPtrs apis, const std::string& ov_device, std::shared_ptr<ov::Core> ov_core);
  ~OpenVINOEpPluginFactory() = default;

  OVEP_DISABLE_COPY_AND_MOVE(OpenVINOEpPluginFactory)

  static const std::vector<std::string>& GetOvDevices();

  std::vector<std::string> GetOvDevices(const std::string& device_type) {
    std::vector<std::string> filtered_devices;
    const auto& devices = GetOvDevices();
    std::copy_if(devices.begin(), devices.end(), std::back_inserter(filtered_devices),
                 [&device_type](const std::string& device) {
                   return device.find(device_type) != std::string::npos;
                 });
    return filtered_devices;
  }

  static const std::vector<std::string>& GetOvMetaDevices();

  // Member functions
  const char* GetName() const {
    return ep_name_.c_str();
  }

  const char* GetVendor() const {
    return vendor_;
  }
  OrtStatus* GetSupportedDevices(const OrtHardwareDevice* const* devices,
                                 size_t num_devices,
                                 OrtEpDevice** ep_devices,
                                 size_t max_ep_devices,
                                 size_t* p_num_ep_devices);
  OrtStatus* CreateEp(const OrtHardwareDevice* const* devices,
                      const OrtKeyValuePairs* const* ep_metadata,
                      size_t num_devices,
                      const OrtSessionOptions* session_options,
                      const OrtLogger* logger,
                      OrtEp** ep);

  void ReleaseEp(OrtEp* ep) {
    delete static_cast<OpenVINOEpPlugin*>(ep);
  }
  OrtStatus* CreateAllocator(const OrtMemoryInfo* memory_info,
                             const OrtKeyValuePairs* allocator_options,
                             OrtAllocator** allocator) noexcept;
  void ReleaseAllocator(OrtAllocator* allocator) noexcept;
  OrtStatus* CreateDataTransfer(OrtDataTransferImpl** data_transfer) noexcept;

  bool IsMetaDeviceFactory() const {
    return known_meta_devices_.find(device_type_) != known_meta_devices_.end();
  }

  // Constants
  static constexpr const char* vendor_ = "Intel";
  static constexpr uint32_t vendor_id_{0x8086};  // Intel's PCI vendor ID
  static constexpr const char* ov_device_key_ = "ov_device";

 private:
  std::string ep_name_;
  std::string device_type_;
  std::vector<std::string> ov_devices_;
  std::shared_ptr<ov::Core> ov_core_;
  inline static const std::set<std::string> known_meta_devices_ = {
      "AUTO"};

 public:
  // Static callback methods for the OrtEpFactory interface
  static const char* ORT_API_CALL GetNameImpl(const OrtEpFactory* this_ptr) {
    const auto* factory = static_cast<const OpenVINOEpPluginFactory*>(this_ptr);
    return factory->GetName();
  }

  static const char* ORT_API_CALL GetVendorImpl(const OrtEpFactory* this_ptr) {
    const auto* factory = static_cast<const OpenVINOEpPluginFactory*>(this_ptr);
    return factory->GetVendor();
  }

  static OrtStatus* ORT_API_CALL GetSupportedDevicesImpl(OrtEpFactory* this_ptr,
                                                         const OrtHardwareDevice* const* devices,
                                                         size_t num_devices,
                                                         OrtEpDevice** ep_devices,
                                                         size_t max_ep_devices,
                                                         size_t* p_num_ep_devices) {
    auto* factory = static_cast<OpenVINOEpPluginFactory*>(this_ptr);
    return factory->GetSupportedDevices(devices, num_devices, ep_devices, max_ep_devices, p_num_ep_devices);
  }
  static OrtStatus* ORT_API_CALL CreateEpImpl(OrtEpFactory* this_ptr,
                                              _In_reads_(num_devices) const OrtHardwareDevice* const* devices,
                                              _In_reads_(num_devices) const OrtKeyValuePairs* const* ep_metadata,
                                              _In_ size_t num_devices,
                                              _In_ const OrtSessionOptions* session_options,
                                              _In_ const OrtLogger* logger,
                                              _Out_ OrtEp** ep) {
    auto* factory = static_cast<OpenVINOEpPluginFactory*>(this_ptr);
    return factory->CreateEp(devices, ep_metadata, num_devices, session_options, logger, ep);
  }
  static void ORT_API_CALL ReleaseEpImpl(OrtEpFactory* this_ptr, OrtEp* ep) {
    auto* factory = static_cast<OpenVINOEpPluginFactory*>(this_ptr);
    factory->ReleaseEp(ep);
  }
  static OrtStatusPtr ORT_API_CALL CreateAllocatorImpl(OrtEpFactory* this_ptr,
                                                       const OrtMemoryInfo* memory_info,
                                                       const OrtKeyValuePairs* allocator_options,
                                                       OrtAllocator** allocator) noexcept {
    auto* factory = static_cast<OpenVINOEpPluginFactory*>(this_ptr);
    return factory->CreateAllocator(memory_info, allocator_options, allocator);
  }

  static void ORT_API_CALL ReleaseAllocatorImpl(OrtEpFactory* this_ptr, OrtAllocator* allocator) noexcept {
    auto* factory = static_cast<OpenVINOEpPluginFactory*>(this_ptr);
    factory->ReleaseAllocator(allocator);
  }

  static OrtStatusPtr ORT_API_CALL CreateDataTransferImpl(OrtEpFactory* this_ptr,
                                                          OrtDataTransferImpl** data_transfer) noexcept {
    auto* factory = static_cast<OpenVINOEpPluginFactory*>(this_ptr);
    return factory->CreateDataTransfer(data_transfer);
  }

  static const char* ORT_API_CALL GetVersionImpl(const OrtEpFactory*) noexcept {
    return OVEP_PLUGIN_VERSION;
  }
};

}  // namespace openvino_ep
}  // namespace onnxruntime
