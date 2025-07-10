// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include <memory>
#include <map>
#include <string>
#include <algorithm>
#include <vector>
#include <ranges>
#include <format>

#include "onnxruntime_c_api.h"
#include "ov_factory.h"
#include "openvino/openvino.hpp"
#include "../common/weak_singleton.h"

using namespace onnxruntime::openvino_ep;
using ov_core_singleton = onnxruntime::openvino_ep::WeakSingleton<ov::Core>;

static void InitCxxApi(const OrtApiBase& ort_api_base) {
  static std::once_flag init_api;
  std::call_once(init_api, [&]() {
    const OrtApi* ort_api = ort_api_base.GetApi(ORT_API_VERSION);
    Ort::InitApi(ort_api);
  });
}

OpenVINOEpPluginFactory::OpenVINOEpPluginFactory(const std::string& ep_name, ApiPtrs apis, const std::string& ov_metadevice_name, std::shared_ptr<ov::Core> core)
    : ApiPtrs{apis},
      ep_name_(ov_metadevice_name.empty() ? ep_name : ep_name + "." + ov_metadevice_name),
      device_type_(ov_metadevice_name),
      ov_core_(std::move(core)) {
  OrtEpFactory::GetName = GetNameImpl;
  OrtEpFactory::GetVendor = GetVendorImpl;
  OrtEpFactory::GetSupportedDevices = GetSupportedDevicesImpl;
  OrtEpFactory::CreateEp = CreateEpImpl;
  OrtEpFactory::ReleaseEp = ReleaseEpImpl;
  OrtEpFactory::CreateAllocator = CreateAllocatorImpl;
  OrtEpFactory::ReleaseAllocator = ReleaseAllocatorImpl;
  OrtEpFactory::CreateDataTransfer = CreateDataTransferImpl;
  OrtEpFactory::GetVersion = GetVersionImpl;
  ort_version_supported = ORT_API_VERSION;  // Set to the ORT version we were compiled with.
}

const std::vector<std::string>& OpenVINOEpPluginFactory::GetOvDevices() {
  static std::vector<std::string> devices = ov_core_singleton::Get()->get_available_devices();
  return devices;
}

const std::vector<std::string>& OpenVINOEpPluginFactory::GetOvMetaDevices() {
  static std::vector<std::string> virtual_devices = [ov_core = ov_core_singleton::Get()] {
    std::vector<std::string> supported_virtual_devices{};
    for (const auto& meta_device : known_meta_devices_) {
      try {
        ov_core->get_property(meta_device, ov::supported_properties);
        supported_virtual_devices.push_back(meta_device);
      } catch (ov::Exception&) {
        // meta device isn't supported.
      }
    }
    return supported_virtual_devices;
  }();

  return virtual_devices;
}

OrtStatus* OpenVINOEpPluginFactory::GetSupportedDevices(const OrtHardwareDevice* const* devices,
                                                        size_t num_devices,
                                                        OrtEpDevice** ep_devices,
                                                        size_t max_ep_devices,
                                                        size_t* p_num_ep_devices) {
  size_t& num_ep_devices = *p_num_ep_devices;

  // Create a map for device type mapping
  static const std::map<OrtHardwareDeviceType, std::string> ort_to_ov_device_name = {
      {OrtHardwareDeviceType::OrtHardwareDeviceType_CPU, "CPU"},
      {OrtHardwareDeviceType::OrtHardwareDeviceType_GPU, "GPU"},
      {OrtHardwareDeviceType::OrtHardwareDeviceType_NPU, "NPU"},
  };

  for (size_t i = 0; i < num_devices && num_ep_devices < max_ep_devices; ++i) {
    const OrtHardwareDevice& device = *devices[i];
    if (ort_api.HardwareDevice_VendorId(&device) != vendor_id_) {
      // Not an Intel Device.
      continue;
    }

    auto device_type = ort_api.HardwareDevice_Type(&device);
    auto device_it = ort_to_ov_device_name.find(device_type);
    if (device_it == ort_to_ov_device_name.end()) {
      // We don't know about this device type
      continue;
    }

    const auto& ov_device_type = device_it->second;
    std::string ov_device_name;
    auto get_pci_device_id = [&](const std::string& ov_device) {
      try {
        ov::device::PCIInfo pci_info = ov_core_->get_property(ov_device, ov::device::pci_info);
        return pci_info.device;
      } catch (ov::Exception&) {
        return 0u;  // If we can't get the PCI info, we won't have a device ID.
      }
    };

    auto filtered_devices = GetOvDevices(ov_device_type);
    auto matched_device = filtered_devices.begin();
    if (filtered_devices.size() > 1 && device_type == OrtHardwareDeviceType::OrtHardwareDeviceType_GPU) {
      // If there are multiple devices of the same type, we need to match by device ID.
      matched_device = std::find_if(filtered_devices.begin(), filtered_devices.end(), [&](const std::string& ov_device) {
        uint32_t ort_device_id = ort_api.HardwareDevice_DeviceId(&device);
        return ort_device_id == get_pci_device_id(ov_device);
      });
    }

    if (matched_device == filtered_devices.end()) {
      // We didn't find a matching OpenVINO device for the OrtHardwareDevice.
      continue;
    }

    // these can be returned as nullptr if you have nothing to add.
    OrtKeyValuePairs* ep_metadata = nullptr;
    OrtKeyValuePairs* ep_options = nullptr;
    ort_api.CreateKeyValuePairs(&ep_metadata);
    ort_api.AddKeyValuePair(ep_metadata, ov_device_key_, matched_device->c_str());

    // Create EP device
    auto* status = ort_api.GetEpApi()->CreateEpDevice(this, &device, ep_metadata, ep_options,
                                                      &ep_devices[num_ep_devices++]);

    ort_api.ReleaseKeyValuePairs(ep_metadata);
    ort_api.ReleaseKeyValuePairs(ep_options);

    if (status != nullptr) {
      return status;
    }
  }

  return nullptr;
}

OrtStatus* OpenVINOEpPluginFactory::CreateEp(const OrtHardwareDevice* const* devices,
                                             const OrtKeyValuePairs* const* ep_metadata,
                                             size_t num_devices,
                                             const OrtSessionOptions* session_options,
                                             const OrtLogger* logger,
                                             OrtEp** ep) {
  (void)devices;
  *ep = nullptr;

  // Create the execution provider with appropriate logging
  std::string log_message = "Creating OpenVINO EP for devices " + device_type_;
  OrtStatus* status = ort_api.Logger_LogMessage(logger,
                                                OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
                                                log_message.c_str(), ORT_FILE, __LINE__, __FUNCTION__);
  if (status != nullptr) {
    return status;
  }

  bool is_meta_device_factory = IsMetaDeviceFactory();
  if (!is_meta_device_factory && num_devices != 1) {
    return ort_api.CreateStatus(ORT_INVALID_ARGUMENT, "Non meta device OpenVINOEP's must only specify a single device. E.g. GPU, GPU.1, CPU, NPU");
  }

  std::string ov_device_string;
  if (is_meta_device_factory) {
    // Build up a meta device string based on the devices that are passed in. E.g. AUTO:NPU,GPU.0,CPU
    ov_device_string = device_type_;
    ov_device_string += ":";
  }

  for (size_t i = 0; i < num_devices; i++) {
    if (i > 0) {
      ov_device_string += ",";
    }

    const char* value = ort_api.GetKeyValue(ep_metadata[i], ov_device_key_);
    if (!value) {
      return ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                  std::format("Ep device missing expected metadata key {}", ov_device_key_).c_str());
    }
    ov_device_string += value;
  }

  // Create a new OpenVINO execution provider with this factory's device type
  auto ov_ep = std::make_unique<OpenVINOEpPlugin>(*this, ep_name_, *session_options, *logger, ov_device_string, ov_core_);
  *ep = ov_ep.release();
  return nullptr;
}

OrtStatus* OpenVINOEpPluginFactory::CreateAllocator(const OrtMemoryInfo* memory_info,
                                                    const OrtKeyValuePairs* allocator_options,
                                                    OrtAllocator** allocator) noexcept {
  // Not optional to implement atm.
  (void)memory_info;
  (void)allocator_options;
  *allocator = nullptr;
  return nullptr;
}

void OpenVINOEpPluginFactory::ReleaseAllocator(OrtAllocator* allocator) noexcept {
  // Not optional to implement atm.
  (void)allocator;
}

OrtStatus* OpenVINOEpPluginFactory::CreateDataTransfer(OrtDataTransferImpl** data_transfer) noexcept {
  // Not optional to implement atm.
  *data_transfer = nullptr;
  return nullptr;
}

extern "C" {
//
// Public symbols
//
OrtStatus* CreateEpFactories(const char* registration_name, const OrtApiBase* ort_api_base,
                             OrtEpFactory** factories, size_t max_factories, size_t* num_factories) {
  InitCxxApi(*ort_api_base);
  const ApiPtrs api_ptrs{Ort::GetApi(), Ort::GetEpApi()};

  // Get available devices from OpenVINO
  auto ov_core = ov_core_singleton::Get();
  std::vector<std::string> supported_factories = {""};
  const auto& meta_devices = OpenVINOEpPluginFactory::GetOvMetaDevices();
  supported_factories.insert(supported_factories.end(), meta_devices.begin(), meta_devices.end());

  const size_t required_factories = supported_factories.size();
  if (max_factories < required_factories) {
    return Ort::Status(std::format("Not enough space to return EP factories. Need at least {} factories.", required_factories).c_str(), ORT_INVALID_ARGUMENT);
  }

  size_t factory_index = 0;
  for (const auto& device_name : supported_factories) {
    // Create a factory for this specific device
    factories[factory_index++] = new OpenVINOEpPluginFactory(registration_name, api_ptrs, device_name, ov_core);
  }

  *num_factories = factory_index;
  return nullptr;
}

OrtStatus* ReleaseEpFactory(OrtEpFactory* factory) {
  delete factory;
  return nullptr;
}
}
