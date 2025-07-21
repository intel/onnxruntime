// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include <algorithm>
#include <cctype>
#include <map>
#include <set>

#include <utility>
#include "core/providers/shared_library/provider_api.h"
#include "core/providers/openvino/openvino_provider_factory.h"
#include "core/providers/openvino/openvino_execution_provider.h"
#include "core/providers/openvino/openvino_provider_factory_creator.h"
#include "core/providers/openvino/contexts.h"
#include "core/providers/openvino/backend_utils.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "nlohmann/json.hpp"
#include "core/providers/openvino/openvino_parser_utils.h"

namespace onnxruntime {
namespace openvino_ep {
void ParseConfigOptions(ProviderInfo& pi) {
  if (pi.config_options == nullptr)
    return;

  pi.so_disable_cpu_ep_fallback = pi.config_options->GetConfigOrDefault(kOrtSessionOptionsDisableCPUEPFallback, "0") == "1";
  pi.so_context_enable = pi.config_options->GetConfigOrDefault(kOrtSessionOptionEpContextEnable, "0") == "1";
  pi.so_context_embed_mode = pi.config_options->GetConfigOrDefault(kOrtSessionOptionEpContextEmbedMode, "0") == "1";
  pi.so_share_ep_contexts = pi.config_options->GetConfigOrDefault(kOrtSessionOptionShareEpContexts, "0") == "1";
  pi.so_context_file_path = pi.config_options->GetConfigOrDefault(kOrtSessionOptionEpContextFilePath, "");

  if (pi.so_share_ep_contexts) {
    ov::AnyMap map;
    map["NPU_COMPILATION_MODE_PARAMS"] = "enable-wd-blockarg-input=true compute-layers-with-higher-precision=Sqrt,Power,ReduceSum";
    pi.load_config["NPU"] = std::move(map);
  }
}

void* ParseUint64(const ProviderOptions& provider_options, std::string option_name) {
  if (provider_options.contains(option_name)) {
    uint64_t number = std::strtoull(provider_options.at(option_name).data(), nullptr, 16);
    return reinterpret_cast<void*>(number);
  } else {
    return nullptr;
  }
}

bool ParseBooleanOption(const ProviderOptions& provider_options, std::string option_name) {
  if (provider_options.contains(option_name)) {
    const auto& value = provider_options.at(option_name);
    if (value == "true" || value == "True") {
      return true;
    } else if (value == "false" || value == "False") {
      return false;
    } else {
      ORT_THROW("[ERROR] [OpenVINO-EP] ", option_name, " should be a boolean.\n");
    }
  }
  return false;
}

std::string ParseDeviceType(std::shared_ptr<OVCore> ov_core, const ProviderOptions& provider_options) {
  std::set<std::string> supported_device_types = {"CPU", "GPU", "NPU"};
  std::set<std::string> supported_device_modes = {"AUTO", "HETERO", "MULTI"};
  std::vector<std::string> devices_to_check;
  std::string selected_device;
  std::vector<std::string> luid_list;
  std::string device_mode = "";
  std::map<std::string, std::string> ov_luid_map;

  if (provider_options.contains("device_type")) {
    selected_device = provider_options.at("device_type");
    std::erase(selected_device, ' ');
    if (selected_device == "AUTO") return selected_device;

    if (auto delimit = selected_device.find(":"); delimit != std::string::npos) {
      device_mode = selected_device.substr(0, delimit);
      if (supported_device_modes.contains(device_mode)) {
        const auto& devices = selected_device.substr(delimit + 1);
        devices_to_check = split(devices, ',');
        ORT_ENFORCE(devices_to_check.size() > 0, "Mode AUTO/HETERO/MULTI should have devices listed based on priority");
      } else {
        ORT_THROW("[ERROR] [OpenVINO] Invalid device_type is selected. Supported modes are AUTO/HETERO/MULTI");
      }
    } else {
      devices_to_check.push_back(selected_device);
    }
  } else {
    // Take default behavior from project configuration
#if defined OPENVINO_CONFIG_CPU
    selected_device = "CPU";
    LOGS_DEFAULT(INFO) << "[OpenVINO-EP] Choosing Device: " << selected_device;
    return selected_device;
#elif defined OPENVINO_CONFIG_GPU
    selected_device = "GPU";
    LOGS_DEFAULT(INFO) << "[OpenVINO-EP] Choosing Device: " << selected_device;
    return selected_device;
#elif defined OPENVINO_CONFIG_NPU
    selected_device = "NPU";
    LOGS_DEFAULT(INFO) << "[OpenVINO-EP] Choosing Device: " << selected_device;
    return selected_device;
#elif defined OPENVINO_CONFIG_HETERO || defined OPENVINO_CONFIG_MULTI || defined OPENVINO_CONFIG_AUTO
    selected_device = DEVICE_NAME;

    // Add sub-devices to check-list
    int delimit = selected_device.find(":");
    const auto& devices = selected_device.substr(delimit + 1);
    devices_to_check = split(devices, ',');
#endif
  }

  // Get the LUID passed from the provider option in a comma separated string list
  // Compare each of the LUID's against the LUID obtained using ov property and map with the right device
  if (provider_options.contains("device_luid")) {
    std::string luid_str = provider_options.at("device_luid");
    std::erase(luid_str, ' ');
    luid_list = split(luid_str, ',');
  }

  for (auto device : devices_to_check) {
    bool device_found = false;
    // Check deprecated device format (CPU_FP32, GPU.0_FP16, etc.) and remove the suffix in place
    // Suffix will be parsed in ParsePrecision
    if (auto delimit = device.find("_"); delimit != std::string::npos) {
      device = device.substr(0, delimit);
    }
    // Just the device name without .0, .1, etc. suffix
    auto device_prefix = device;
    // Check if device index is appended (.0, .1, etc.), if so, remove it
    if (auto delimit = device_prefix.find("."); delimit != std::string::npos)
      device_prefix = device_prefix.substr(0, delimit);
    if (supported_device_types.contains(device_prefix)) {
      try {
        std::vector<std::string> available_devices = ov_core->GetAvailableDevices(device_prefix);
        // Here we need to find the full device name (with .idx, but without _precision)
        if (std::find(std::begin(available_devices), std::end(available_devices), device) != std::end(available_devices))
          device_found = true;
        if (!device_found) {
          ORT_THROW("[ERROR] [OpenVINO] Device ", device, " is not available");
        }
        if (device_prefix != "CPU" && luid_list.size() > 0) {
          for (const auto& dev : available_devices) {
            ov::device::LUID ov_luid = OVCore::Get()->core.get_property(dev, ov::device::luid);
            std::stringstream ov_luid_str;
            ov_luid_str << ov_luid;
            ov_luid_map.emplace(ov_luid_str.str(), dev);
          }
        }
      } catch (const char* msg) {
        ORT_THROW(msg);
      }
    }
  }
  if (luid_list.size() > 0) {
    std::string ov_luid_devices;
    for (const auto& luid_str : luid_list) {
      if (ov_luid_map.contains(luid_str)) {
        std::string ov_dev = ov_luid_map.at(luid_str);
        std::string ov_dev_strip = split(ov_dev, '.')[0];
        if (std::find(std::begin(devices_to_check), std::end(devices_to_check), ov_dev) != std::end(devices_to_check) ||
            std::find(std::begin(devices_to_check), std::end(devices_to_check), ov_dev_strip) != std::end(devices_to_check)) {
          if (!ov_luid_devices.empty()) ov_luid_devices = ov_luid_devices + ",";
          ov_luid_devices = ov_luid_devices + ov_dev;
        } else {
          ORT_THROW(" LUID : ", ov_dev, " does not match with device_type : ", selected_device);
        }
      } else {
        ORT_THROW(provider_options.at("device_luid"), " does not exist for the selected device_type : ", selected_device);
      }
    }
    if (!device_mode.empty()) {
      selected_device = device_mode + ":" + ov_luid_devices;
      for (const auto& dev_str : devices_to_check) {
        const auto default_dev = split(dev_str, '.')[0];

        if (ov_luid_devices.find(default_dev) == std::string::npos)
          selected_device = selected_device + "," + dev_str;
      }
    } else {
      selected_device = std::move(ov_luid_devices);
    }
  }

  LOGS_DEFAULT(INFO) << "[OpenVINO-EP] Choosing Device: " << selected_device;
  return selected_device;
}

void ParseProviderOptions([[maybe_unused]] ProviderInfo& result, [[maybe_unused]] const ProviderOptions& config_options) {}

// Initializes a ProviderInfo struct from a ProviderOptions map and a ConfigOptions map.
static void ParseProviderInfo(const ProviderOptions& provider_options,
                              const ConfigOptions* config_options,
                              /*output*/ ProviderInfo& pi) {
  pi.config_options = config_options;

  // Lambda function to check for invalid keys and throw an error
  auto validateKeys = [&]() {
    for (const auto& pair : provider_options) {
      if (pi.valid_provider_keys.find(pair.first) == pi.valid_provider_keys.end()) {
        ORT_THROW("Invalid provider_option key: " + pair.first);
      }
    }
  };
  validateKeys();

  std::string bool_flag = "";

  // Minor optimization: we'll hold an OVCore reference to ensure we don't create a new core between ParseDeviceType and
  // (potential) SharedContext creation.
  auto ov_core = OVCore::Get();
  pi.device_type = ParseDeviceType(std::move(ov_core), provider_options);

  if (provider_options.contains("device_id")) {
    std::string dev_id = provider_options.at("device_id").data();
    LOGS_DEFAULT(WARNING) << "[OpenVINO] The options 'device_id' is deprecated. "
                          << "Upgrade to set deice_type and precision session options.\n";
    if (dev_id == "CPU" || dev_id == "GPU" || dev_id == "NPU") {
      pi.device_type = std::move(dev_id);
    } else {
      ORT_THROW("[ERROR] [OpenVINO] Unsupported device_id is selected. Select from available options.");
    }
  }
  if (provider_options.contains("cache_dir")) {
    pi.cache_dir = provider_options.at("cache_dir");
  }

  pi.precision = OpenVINOParserUtils::ParsePrecision(provider_options, pi.device_type, "precision");

  if (provider_options.contains("reshape_input")) {
    pi.reshape = OpenVINOParserUtils::ParseInputShape(provider_options.at("reshape_input"));
  }

  if (provider_options.contains("load_config")) {
    auto parse_config = [&](const std::string& config_str) -> std::map<std::string, ov::AnyMap> {
      // If the config string is empty, return an empty map and skip processing
      if (config_str.empty()) {
        LOGS_DEFAULT(WARNING) << "Empty OV Config Map passed. Skipping load_config option parsing.\n";
        return {};
      }

      std::stringstream input_str_stream(config_str);
      std::map<std::string, ov::AnyMap> target_map;

      try {
        nlohmann::json json_config = nlohmann::json::parse(input_str_stream);

        if (!json_config.is_object()) {
          ORT_THROW("Invalid JSON structure: Expected an object at the root.");
        }

        for (auto& [key, value] : json_config.items()) {
          ov::AnyMap inner_map;
          std::set<std::string> valid_ov_devices = {"CPU", "GPU", "NPU", "AUTO", "HETERO", "MULTI"};
          // Ensure the key is one of "CPU", "GPU", or "NPU"
          if (valid_ov_devices.find(key) == valid_ov_devices.end()) {
            LOGS_DEFAULT(WARNING) << "Unsupported device key: " << key << ". Skipping entry.\n";
            continue;
          }

          // Ensure that the value for each device is an object (PROPERTY -> VALUE)
          if (!value.is_object()) {
            ORT_THROW("Invalid JSON structure: Expected an object for device properties.");
          }

          for (auto& [inner_key, inner_value] : value.items()) {
            if (inner_value.is_string()) {
              inner_map[inner_key] = inner_value.get<std::string>();
            } else if (inner_value.is_number_integer()) {
              inner_map[inner_key] = inner_value.get<int64_t>();
            } else if (inner_value.is_number_float()) {
              inner_map[inner_key] = inner_value.get<double>();
            } else if (inner_value.is_boolean()) {
              inner_map[inner_key] = inner_value.get<bool>();
            } else {
              LOGS_DEFAULT(WARNING) << "Unsupported JSON value type for key: " << inner_key << ". Skipping key.";
            }
          }
          target_map[key] = std::move(inner_map);
        }
      } catch (const nlohmann::json::parse_error& e) {
        // Handle syntax errors in JSON
        ORT_THROW("JSON parsing error: " + std::string(e.what()));
      } catch (const nlohmann::json::type_error& e) {
        // Handle invalid type accesses
        ORT_THROW("JSON type error: " + std::string(e.what()));
      } catch (const std::exception& e) {
        ORT_THROW("Error parsing load_config Map: " + std::string(e.what()));
      }
      return target_map;
    };

    pi.load_config = parse_config(provider_options.at("load_config"));
  }

  pi.context = ParseUint64(provider_options, "context");
#if defined(IO_BUFFER_ENABLED)
  // a valid context must be provided to enable IO Buffer optimizations
  if (pi.context == nullptr) {
#undef IO_BUFFER_ENABLED
#define IO_BUFFER_ENABLED = 0
    LOGS_DEFAULT(WARNING) << "Context is not set. Disabling IO Buffer optimization";
  }
#endif

  if (provider_options.contains("num_of_threads")) {
    if (!std::all_of(provider_options.at("num_of_threads").begin(),
                     provider_options.at("num_of_threads").end(), ::isdigit)) {
      ORT_THROW("[ERROR] [OpenVINO-EP] Number of threads should be a number. \n");
    }
    pi.num_of_threads = std::stoi(provider_options.at("num_of_threads"));
    if (pi.num_of_threads <= 0) {
      pi.num_of_threads = 1;
      LOGS_DEFAULT(WARNING) << "[OpenVINO-EP] The value for the key 'num_threads' should be in the positive range.\n "
                            << "Executing with num_threads=1";
    }
  }

  if (provider_options.contains("model_priority")) {
    pi.model_priority = provider_options.at("model_priority").data();
    std::vector<std::string> supported_priorities({"LOW", "MEDIUM", "HIGH", "DEFAULT"});
    if (std::find(supported_priorities.begin(), supported_priorities.end(),
                  pi.model_priority) == supported_priorities.end()) {
      pi.model_priority = "DEFAULT";
      LOGS_DEFAULT(WARNING) << "[OpenVINO-EP] The value for the key 'model_priority' "
                            << "is not one of LOW, MEDIUM, HIGH, DEFAULT. "
                            << "Executing with model_priorty=DEFAULT";
    }
  }

  if (provider_options.contains("num_streams")) {
    pi.num_streams = std::stoi(provider_options.at("num_streams"));
    if (pi.num_streams <= 0) {
      pi.num_streams = 1;
      LOGS_DEFAULT(WARNING) << "[OpenVINO-EP] The value for the key 'num_streams' should be in the range of 1-8.\n "
                            << "Executing with num_streams=1";
    }
  }
  try {
    pi.enable_opencl_throttling = ParseBooleanOption(provider_options, "enable_opencl_throttling");

    pi.enable_qdq_optimizer = ParseBooleanOption(provider_options, "enable_qdq_optimizer");

    pi.enable_causallm = ParseBooleanOption(provider_options, "enable_causallm");

    pi.disable_dynamic_shapes = ParseBooleanOption(provider_options, "disable_dynamic_shapes");
  } catch (std::string msg) {
    ORT_THROW(msg);
  }

  // Should likely account for meta devices as well, but for now keep the current behavior.
  bool target_devices_support_dynamic_shapes =
      pi.device_type.find("GPU") != std::string::npos ||
      pi.device_type.find("CPU") != std::string::npos ||
      (pi.device_type.find("NPU") != std::string::npos &&
       pi.enable_causallm);

  pi.disable_dynamic_shapes = !target_devices_support_dynamic_shapes;
}

struct OpenVINOProviderFactory : IExecutionProviderFactory {
  OpenVINOProviderFactory(ProviderInfo provider_info, std::shared_ptr<SharedContext> shared_context)
      : provider_info_(std::move(provider_info)), shared_context_(std::move(shared_context)) {}

  ~OpenVINOProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override {
    ParseConfigOptions(provider_info_);
    return std::make_unique<OpenVINOExecutionProvider>(provider_info_, shared_context_);
  }

  // Called by InferenceSession when registering EPs. Allows creation of an EP instance that is initialized with
  // session-level configurations.
  std::unique_ptr<IExecutionProvider> CreateProvider(const OrtSessionOptions& session_options,
                                                     const OrtLogger& session_logger) override {
    const ConfigOptions& config_options = session_options.GetConfigOptions();
    const std::unordered_map<std::string, std::string>& config_options_map = config_options.GetConfigOptionsMap();

    // The implementation of the SessionOptionsAppendExecutionProvider C API function automatically adds EP options to
    // the session option configurations with the key prefix "ep.<lowercase_ep_name>.".
    // Extract those EP options into a new "provider_options" map.
    std::string lowercase_ep_name = kOpenVINOExecutionProvider;
    std::transform(lowercase_ep_name.begin(), lowercase_ep_name.end(), lowercase_ep_name.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });

    std::string key_prefix = "ep.";
    key_prefix += lowercase_ep_name;
    key_prefix += ".";

    std::unordered_map<std::string, std::string> provider_options;
    for (const auto& [key, value] : config_options_map) {
      if (key.rfind(key_prefix, 0) == 0) {
        provider_options[key.substr(key_prefix.size())] = value;
      }
    }

    ProviderInfo provider_info = provider_info_;
    ParseProviderInfo(provider_options, &config_options, provider_info);
    ParseConfigOptions(provider_info);

    auto ov_ep = std::make_unique<OpenVINOExecutionProvider>(provider_info, shared_context_);
    ov_ep->SetLogger(reinterpret_cast<const logging::Logger*>(&session_logger));
    return ov_ep;
  }

 private:
  ProviderInfo provider_info_;
  std::shared_ptr<SharedContext> shared_context_;
};

struct ProviderInfo_OpenVINO_Impl : ProviderInfo_OpenVINO {
  std::vector<std::string> GetAvailableDevices() const override {
    return OVCore::Get()->GetAvailableDevices();
  }
};

struct OpenVINO_Provider : Provider {
  void* GetInfo() override { return &info_; }

  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(const void* void_params) override {
    if (void_params == nullptr) {
      LOGS_DEFAULT(ERROR) << "[OpenVINO EP] Passed NULL options to CreateExecutionProviderFactory()";
      return nullptr;
    }

    std::array<void*, 2> pointers_array = *reinterpret_cast<const std::array<void*, 2>*>(void_params);
    const ProviderOptions provider_options = *reinterpret_cast<ProviderOptions*>(pointers_array[0]);
    const ConfigOptions* config_options = reinterpret_cast<ConfigOptions*>(pointers_array[1]);

    ProviderInfo pi;
    ParseProviderInfo(provider_options, config_options, pi);

    return std::make_shared<OpenVINOProviderFactory>(pi, SharedContext::Get());
  }

  Status CreateIExecutionProvider(const OrtHardwareDevice* const* devices,
                                  const OrtKeyValuePairs* const* ep_metadata,
                                  size_t num_devices,
                                  ProviderOptions& provider_options,
                                  const OrtSessionOptions& session_options,
                                  const OrtLogger& logger,
                                  std::unique_ptr<IExecutionProvider>& ep) override {
    if (num_devices == 0) {
      return Status(common::ONNXRUNTIME, ORT_EP_FAIL, "OpenVINO EP requires at least one device.");
    }

    const ConfigOptions* config_options = &session_options.GetConfigOptions();

    std::array<const void*, 2> configs_array = {&provider_options, config_options};
    const void* arg = reinterpret_cast<const void*>(&configs_array);

    auto ep_factory = CreateExecutionProviderFactory(arg);
    if (!ep_factory) {
      return Status(common::ONNXRUNTIME, ORT_EP_FAIL, "Failed to create OpenVINO EP factory.");
    }

    ep = ep_factory->CreateProvider(session_options, logger);
    if (!ep) {
      return Status(common::ONNXRUNTIME, ORT_EP_FAIL, "Failed to create OpenVINO EP instance.");
    }

    return Status::OK();
  }

  void Initialize() override {
  }

  void Shutdown() override {
  }

 private:
  ProviderInfo_OpenVINO_Impl info_;
};  // OpenVINO_Provider

}  // namespace openvino_ep
}  // namespace onnxruntime

extern "C" {

ORT_API(onnxruntime::Provider*, GetProvider) {
  static onnxruntime::openvino_ep::OpenVINO_Provider g_provider;
  return &g_provider;
}
}

#include "core/framework/error_code_helper.h"
#include "onnxruntime_config.h"  // for ORT_VERSION

// OrtEpApi infrastructure to be able to use the OpenVINO EP as an OrtEpFactory for auto EP selection.
struct OpenVINOEpFactory : OrtEpFactory {
  OpenVINOEpFactory(const OrtApi& ort_api_in,
                    const char* ep_name,
                    OrtHardwareDeviceType hw_type,
                    const char* ov_device_type)
      : ort_api{ort_api_in}, ep_name{ep_name}, ort_hw_device_type{hw_type}, ov_device_type{ov_device_type} {
    GetName = GetNameImpl;
    GetVendor = GetVendorImpl;
    GetVersion = GetVersionImpl;
    GetSupportedDevices = GetSupportedDevicesImpl;
    CreateEp = CreateEpImpl;
    ReleaseEp = ReleaseEpImpl;
  }

  // Returns the name for the EP. Each unique factory configuration must have a unique name.
  // Ex: a factory that supports GPU should have a different name than a factory that supports CPU.
  static const char* GetNameImpl(const OrtEpFactory* this_ptr) {
    const auto* factory = static_cast<const OpenVINOEpFactory*>(this_ptr);
    return factory->ep_name.c_str();
  }

  static const char* GetVendorImpl(const OrtEpFactory* this_ptr) {
    const auto* factory = static_cast<const OpenVINOEpFactory*>(this_ptr);
    return factory->vendor.c_str();
  }

  static const char* ORT_API_CALL GetVersionImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
    return ORT_VERSION;
  }

  // Creates and returns OrtEpDevice instances for all OrtHardwareDevices that this factory supports.
  // OpenVINO EP can support multiple device types (CPU, GPU, NPU, etc.) but each factory instance
  // is configured for a specific device type to enable proper hardware matching.
  static OrtStatus* GetSupportedDevicesImpl(OrtEpFactory* this_ptr,
                                            const OrtHardwareDevice* const* devices,
                                            size_t num_devices,
                                            OrtEpDevice** ep_devices,
                                            size_t max_ep_devices,
                                            size_t* p_num_ep_devices) {
    size_t& num_ep_devices = *p_num_ep_devices;
    auto* factory = static_cast<OpenVINOEpFactory*>(this_ptr);

    for (size_t i = 0; i < num_devices && num_ep_devices < max_ep_devices; ++i) {
      const OrtHardwareDevice& device = *devices[i];
      if (factory->ort_api.HardwareDevice_Type(&device) == factory->ort_hw_device_type &&
          factory->ort_api.HardwareDevice_VendorId(&device) == factory->vendor_id) {
        OrtKeyValuePairs* ep_options = nullptr;
        factory->ort_api.CreateKeyValuePairs(&ep_options);
        factory->ort_api.AddKeyValuePair(ep_options, "device_type", factory->ov_device_type.c_str());
        ORT_API_RETURN_IF_ERROR(
            factory->ort_api.GetEpApi()->CreateEpDevice(factory, &device, nullptr, ep_options,
                                                        &ep_devices[num_ep_devices++]));
      }
    }

    return nullptr;
  }

  static OrtStatus* CreateEpImpl(OrtEpFactory* /*this_ptr*/,
                                 _In_reads_(num_devices) const OrtHardwareDevice* const* /*devices*/,
                                 _In_reads_(num_devices) const OrtKeyValuePairs* const* /*ep_metadata*/,
                                 _In_ size_t /*num_devices*/,
                                 _In_ const OrtSessionOptions* /*session_options*/,
                                 _In_ const OrtLogger* /*logger*/,
                                 _Out_ OrtEp** /*ep*/) {
    return onnxruntime::CreateStatus(ORT_INVALID_ARGUMENT, "OpenVINO EP factory does not support this method.");
  }

  static void ReleaseEpImpl(OrtEpFactory* /*this_ptr*/, OrtEp* /*ep*/) {
    // no-op as we never create an EP here.
  }

  const OrtApi& ort_api;
  const std::string ep_name;              // EP name
  const std::string vendor{"Intel"};  // EP vendor name

  // Intel vendor ID. Refer to the ACPI ID registry (search Intel): https://uefi.org/ACPI_ID_List
  const uint32_t vendor_id{0x8086}; // Intel vendor ID
  const OrtHardwareDeviceType ort_hw_device_type;  // Supported OrtHardwareDevice
  const std::string ov_device_type;                // OpenVINO device type (CPU, GPU, NPU, etc.)
};

extern "C" {
//
// Public symbols
//
OrtStatus* CreateEpFactories(const char* /*registration_name*/, const OrtApiBase* ort_api_base,
                             OrtEpFactory** factories, size_t max_factories, size_t* num_factories) {
  const OrtApi* ort_api = ort_api_base->GetApi(ORT_API_VERSION);

  // Create factories for different OpenVINO device types
  std::vector<std::unique_ptr<OrtEpFactory>> created_factories;

  // CPU factory - most common and always available
  created_factories.push_back(
      std::make_unique<OpenVINOEpFactory>(*ort_api,
                                          onnxruntime::kOpenVINOExecutionProvider,
                                          OrtHardwareDeviceType_CPU, "CPU"));

  // GPU factory - for Intel integrated graphics
  created_factories.push_back(
      std::make_unique<OpenVINOEpFactory>(*ort_api,
                                          onnxruntime::kOpenVINOExecutionProvider,
                                          OrtHardwareDeviceType_GPU, "GPU"));

  // NPU factory - for Intel Neural Processing Units
  created_factories.push_back(
      std::make_unique<OpenVINOEpFactory>(*ort_api,
                                          onnxruntime::kOpenVINOExecutionProvider,
                                          OrtHardwareDeviceType_NPU, "NPU"));

  size_t factories_to_create = std::min(created_factories.size(), max_factories);

  if (factories_to_create == 0) {
    return ort_api->CreateStatus(ORT_INVALID_ARGUMENT,
                                 "Not enough space to return EP factory. Need at least one.");
  }

  for (size_t i = 0; i < factories_to_create; ++i) {
    factories[i] = created_factories[i].release();
  }

  *num_factories = factories_to_create;

  return nullptr;
}

OrtStatus* ReleaseEpFactory(OrtEpFactory* factory) {
  delete static_cast<OpenVINOEpFactory*>(factory);
  return nullptr;
}
}
