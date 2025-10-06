// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/providers/openvino/ov_telemetry.h"

#ifdef _WIN32
#if !BUILD_OPENVINO_EP_STATIC_LIB
#include <windows.h>
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 26440)
#endif
#include <TraceLoggingProvider.h>
#include <winmeta.h>
#include <sstream>
#include "core/platform/windows/TraceLoggingConfig.h"

// UTF-8 workarounds
#ifdef _TlgPragmaUtf8Begin
#undef _TlgPragmaUtf8Begin
#define _TlgPragmaUtf8Begin
#endif
#ifdef _TlgPragmaUtf8End
#undef _TlgPragmaUtf8End
#define _TlgPragmaUtf8End
#endif

TRACELOGGING_DEFINE_PROVIDER(
    ov_telemetry_provider_handle,
    "Microsoft.ML.ONNXRuntime.OpenVINO",
    (0xb5a8c2e1, 0x4d7f, 0x4a3b, 0x9c, 0x2e, 0x1f, 0x8e, 0x5a, 0x6b, 0x7c, 0x9d),
    TraceLoggingOptionMicrosoftTelemetry());

#ifdef _MSC_VER
#pragma warning(pop)
#endif
#endif // !BUILD_OPENVINO_EP_STATIC_LIB

namespace onnxruntime {
namespace openvino_ep {

#if !BUILD_OPENVINO_EP_STATIC_LIB
std::mutex OVTelemetry::mutex_;
std::mutex OVTelemetry::provider_change_mutex_;
uint32_t OVTelemetry::global_register_count_ = 0;
bool OVTelemetry::enabled_ = true;
UCHAR OVTelemetry::level_ = 0;
UINT64 OVTelemetry::keyword_ = 0;
std::vector<const OVTelemetry::EtwInternalCallback*> OVTelemetry::callbacks_;
std::mutex OVTelemetry::callbacks_mutex_;
#endif

OVTelemetry::OVTelemetry() {
#if !BUILD_OPENVINO_EP_STATIC_LIB
  std::lock_guard<std::mutex> lock(mutex_);
  if (global_register_count_ == 0) {
    HRESULT hr = TraceLoggingRegisterEx(ov_telemetry_provider_handle, ORT_TL_EtwEnableCallback, nullptr);
    if (SUCCEEDED(hr)) {
      global_register_count_ += 1;
    }
  }
#endif
}

OVTelemetry::~OVTelemetry() {
#if !BUILD_OPENVINO_EP_STATIC_LIB
  std::lock_guard<std::mutex> lock(mutex_);
  if (global_register_count_ > 0) {
    global_register_count_ -= 1;
    if (global_register_count_ == 0) {
      TraceLoggingUnregister(ov_telemetry_provider_handle);
    }
  }
  std::lock_guard<std::mutex> lock_callbacks(callbacks_mutex_);
  callbacks_.clear();
#endif
}

OVTelemetry& OVTelemetry::Instance() {
  static OVTelemetry instance;
  return instance;
}

bool OVTelemetry::IsEnabled() const {
#if BUILD_OPENVINO_EP_STATIC_LIB
  return false; // Simplified for now
#else
  std::lock_guard<std::mutex> lock(provider_change_mutex_);
  return enabled_;
#endif
}

UCHAR OVTelemetry::Level() const {
#if BUILD_OPENVINO_EP_STATIC_LIB
  return 0;
#else
  std::lock_guard<std::mutex> lock(provider_change_mutex_);
  return level_;
#endif
}

UINT64 OVTelemetry::Keyword() const {
#if BUILD_OPENVINO_EP_STATIC_LIB
  return 0;
#else
  std::lock_guard<std::mutex> lock(provider_change_mutex_);
  return keyword_;
#endif
}

std::string OVTelemetry::SerializeOptionsMap(const std::unordered_map<std::string, std::string>& options) const {
  std::stringstream ss;
  bool first = true;
  for (const auto& [key, value] : options) {
    if (!first) ss << ", ";
    ss << key << "=" << value;
    first = false;
  }
  return ss.str();
}

void OVTelemetry::LogProviderInit(uint32_t session_id, const std::string& device_type, const std::string& precision) const {
#if !BUILD_OPENVINO_EP_STATIC_LIB
  TraceLoggingWrite(
      ov_telemetry_provider_handle, "OVProviderInit",
      TraceLoggingKeyword(ov_keywords::OV_PROVIDER),
      TraceLoggingLevel(WINEVENT_LEVEL_INFO),
      TraceLoggingUInt32(session_id, "session_id"),
      TraceLoggingString(device_type.c_str(), "device_type"),
      TraceLoggingString(precision.c_str(), "precision"));
#endif
}

void OVTelemetry::LogProviderShutdown(uint32_t session_id) const {
#if !BUILD_OPENVINO_EP_STATIC_LIB
  TraceLoggingWrite(
      ov_telemetry_provider_handle, "OVProviderShutdown",
      TraceLoggingKeyword(ov_keywords::OV_PROVIDER),
      TraceLoggingLevel(WINEVENT_LEVEL_INFO),
      TraceLoggingUInt32(session_id, "session_id"));
#endif
}

void OVTelemetry::LogSessionOptions(uint32_t session_id, const std::unordered_map<std::string, std::string>& session_options) const {
#if !BUILD_OPENVINO_EP_STATIC_LIB
  std::string options_str = SerializeOptionsMap(session_options);
  TraceLoggingWrite(
      ov_telemetry_provider_handle, "OVSessionOptions",
      TraceLoggingKeyword(ov_keywords::OV_SESSION | ov_keywords::OV_OPTIONS),
      TraceLoggingLevel(WINEVENT_LEVEL_INFO),
      TraceLoggingUInt32(session_id, "session_id"),
      TraceLoggingUInt32(static_cast<uint32_t>(session_options.size()), "options_count"),
      TraceLoggingString(options_str.c_str(), "session_options"));
#endif
}

void OVTelemetry::LogProviderOptions(uint32_t session_id, const std::unordered_map<std::string, std::string>& provider_options) const {
#if !BUILD_OPENVINO_EP_STATIC_LIB
  std::string options_str = SerializeOptionsMap(provider_options);
  TraceLoggingWrite(
      ov_telemetry_provider_handle, "OVProviderOptions",
      TraceLoggingKeyword(ov_keywords::OV_PROVIDER | ov_keywords::OV_OPTIONS),
      TraceLoggingLevel(WINEVENT_LEVEL_INFO),
      TraceLoggingUInt32(session_id, "session_id"),
      TraceLoggingUInt32(static_cast<uint32_t>(provider_options.size()), "options_count"),
      TraceLoggingString(options_str.c_str(), "provider_options"));
#endif
}

void OVTelemetry::LogSessionCreation(uint32_t session_id, const std::string& model_path, const std::string& openvino_version) const {
#if !BUILD_OPENVINO_EP_STATIC_LIB
  TraceLoggingWrite(
      ov_telemetry_provider_handle, "OVSessionCreation",
      TraceLoggingKeyword(ov_keywords::OV_SESSION),
      TraceLoggingLevel(WINEVENT_LEVEL_INFO),
      TraceLoggingUInt32(session_id, "session_id"),
      TraceLoggingString(model_path.c_str(), "model_path"),
      TraceLoggingString(openvino_version.c_str(), "openvino_version"));
#endif
}

void OVTelemetry::LogSessionDestruction(uint32_t session_id) const {
#if !BUILD_OPENVINO_EP_STATIC_LIB
  TraceLoggingWrite(
      ov_telemetry_provider_handle, "OVSessionDestruction",
      TraceLoggingKeyword(ov_keywords::OV_SESSION),
      TraceLoggingLevel(WINEVENT_LEVEL_INFO),
      TraceLoggingUInt32(session_id, "session_id"));
#endif
}

void OVTelemetry::LogCapabilityDetection(uint32_t session_id, uint32_t node_count, bool wholly_supported,
                                        bool has_external_weights, const std::string& device_type) const {
#if !BUILD_OPENVINO_EP_STATIC_LIB
  TraceLoggingWrite(
      ov_telemetry_provider_handle, "OVCapabilityDetection",
      TraceLoggingKeyword(ov_keywords::OV_CAPABILITY),
      TraceLoggingLevel(WINEVENT_LEVEL_INFO),
      TraceLoggingUInt32(session_id, "session_id"),
      TraceLoggingUInt32(node_count, "node_count"),
      TraceLoggingBool(wholly_supported, "wholly_supported"),
      TraceLoggingBool(has_external_weights, "has_external_weights"),
      TraceLoggingString(device_type.c_str(), "device_type"));
#endif
}

void OVTelemetry::LogCompileStart(uint32_t session_id, uint32_t fused_node_count, const std::string& device_type,
                                 const std::string& precision) const {
#if !BUILD_OPENVINO_EP_STATIC_LIB
  TraceLoggingWrite(
      ov_telemetry_provider_handle, "OVCompileStart",
      TraceLoggingKeyword(ov_keywords::OV_COMPILATION),
      TraceLoggingLevel(WINEVENT_LEVEL_INFO),
      TraceLoggingUInt32(session_id, "session_id"),
      TraceLoggingUInt32(fused_node_count, "fused_node_count"),
      TraceLoggingString(device_type.c_str(), "device_type"),
      TraceLoggingString(precision.c_str(), "precision"));
#endif
}

void OVTelemetry::LogCompileEnd(uint32_t session_id, bool success, const std::string& error_message,
                               int64_t compile_duration_ms) const {
#if !BUILD_OPENVINO_EP_STATIC_LIB
  TraceLoggingWrite(
    ov_telemetry_provider_handle, "OVCompileEnd",
    TraceLoggingKeyword(ov_keywords::OV_COMPILATION | ov_keywords::OV_PERFORMANCE),
    TraceLoggingLevel(WINEVENT_LEVEL_VERBOSE),
    TraceLoggingUInt32(session_id, "session_id"),
    TraceLoggingBool(success, "success"),
    TraceLoggingString(error_message.c_str(), "error_message"),
    TraceLoggingInt64(compile_duration_ms, "compile_duration_ms"));
#endif
}

void OVTelemetry::LogComputeStart(uint32_t session_id, const std::string& subgraph_name, const std::string& device_type) const {
#if !BUILD_OPENVINO_EP_STATIC_LIB
  TraceLoggingWrite(
      ov_telemetry_provider_handle, "OVComputeStart",
      TraceLoggingKeyword(ov_keywords::OV_EXECUTION),
      TraceLoggingLevel(WINEVENT_LEVEL_VERBOSE),
      TraceLoggingUInt32(session_id, "session_id"),
      TraceLoggingString(subgraph_name.c_str(), "subgraph_name"),
      TraceLoggingString(device_type.c_str(), "device_type"));
#endif
}

void OVTelemetry::LogComputeEnd(uint32_t session_id, int64_t duration_microseconds, const std::string& subgraph_name,
                               bool success, const std::string& error_message) const {
#if !BUILD_OPENVINO_EP_STATIC_LIB
  TraceLoggingWrite(
    ov_telemetry_provider_handle, "OVComputeEnd",
    TraceLoggingKeyword(ov_keywords::OV_EXECUTION | ov_keywords::OV_PERFORMANCE),
    TraceLoggingLevel(WINEVENT_LEVEL_VERBOSE),
    TraceLoggingUInt32(session_id, "session_id"),
    TraceLoggingInt64(duration_microseconds, "duration_microseconds"),
    TraceLoggingString(subgraph_name.c_str(), "subgraph_name"),
    TraceLoggingBool(success, "success"),
    TraceLoggingString(error_message.c_str(), "error_message"));
#endif
}

void OVTelemetry::LogBackendManagerEvent(uint32_t session_id, const std::string& event_type, const std::string& details) const {
#if !BUILD_OPENVINO_EP_STATIC_LIB
  TraceLoggingWrite(
      ov_telemetry_provider_handle, "OVBackendEvent",
      TraceLoggingKeyword(ov_keywords::OV_BACKEND),
      TraceLoggingLevel(WINEVENT_LEVEL_VERBOSE),
      TraceLoggingUInt32(session_id, "session_id"),
      TraceLoggingString(event_type.c_str(), "event_type"),
      TraceLoggingString(details.c_str(), "details"));
#endif
}

void OVTelemetry::LogDeviceSelection(uint32_t session_id, const std::string& requested_device,
                                    const std::string& actual_device, const std::string& selection_reason) const {
#if !BUILD_OPENVINO_EP_STATIC_LIB
  TraceLoggingWrite(
      ov_telemetry_provider_handle, "OVDeviceSelection",
      TraceLoggingKeyword(ov_keywords::OV_PROVIDER | ov_keywords::OV_OPTIONS),
      TraceLoggingLevel(WINEVENT_LEVEL_INFO),
      TraceLoggingUInt32(session_id, "session_id"),
      TraceLoggingString(requested_device.c_str(), "requested_device"),
      TraceLoggingString(actual_device.c_str(), "actual_device"),
      TraceLoggingString(selection_reason.c_str(), "selection_reason"));
#endif
}

void OVTelemetry::LogError(uint32_t session_id, const std::string& error_category, const std::string& error_message,
                          const std::string& function_name, int line_number) const {
#if !BUILD_OPENVINO_EP_STATIC_LIB
  TraceLoggingWrite(
      ov_telemetry_provider_handle, "OVError",
      TraceLoggingKeyword(ov_keywords::OV_ERROR),
      TraceLoggingLevel(WINEVENT_LEVEL_ERROR),
      TraceLoggingUInt32(session_id, "session_id"),
      TraceLoggingString(error_category.c_str(), "error_category"),
      TraceLoggingString(error_message.c_str(), "error_message"),
      TraceLoggingString(function_name.c_str(), "function_name"),
      TraceLoggingInt32(line_number, "line_number"));
#endif
}

void OVTelemetry::RegisterInternalCallback(const EtwInternalCallback& callback) {
#if BUILD_OPENVINO_EP_STATIC_LIB
  // Handle static case if needed
#else
  std::lock_guard<std::mutex> lock_callbacks(callbacks_mutex_);
  callbacks_.push_back(&callback);
#endif
}

void OVTelemetry::UnregisterInternalCallback(const EtwInternalCallback& callback) {
#if BUILD_OPENVINO_EP_STATIC_LIB
  // Handle static case if needed
#else
  std::lock_guard<std::mutex> lock_callbacks(callbacks_mutex_);
  auto new_end = std::remove_if(callbacks_.begin(), callbacks_.end(),
                               [&callback](const EtwInternalCallback* ptr) {
                                 return ptr == &callback;
                               });
  callbacks_.erase(new_end, callbacks_.end());
#endif
}

#if !BUILD_OPENVINO_EP_STATIC_LIB
void NTAPI OVTelemetry::ORT_TL_EtwEnableCallback(
    _In_ LPCGUID SourceId, _In_ ULONG IsEnabled, _In_ UCHAR Level, _In_ ULONGLONG MatchAnyKeyword,
    _In_ ULONGLONG MatchAllKeyword, _In_opt_ PEVENT_FILTER_DESCRIPTOR FilterData, _In_opt_ PVOID CallbackContext) {
  std::lock_guard<std::mutex> lock(provider_change_mutex_);
  enabled_ = (IsEnabled != 0);
  level_ = Level;
  keyword_ = MatchAnyKeyword;
  InvokeCallbacks(SourceId, IsEnabled, Level, MatchAnyKeyword, MatchAllKeyword, FilterData, CallbackContext);
}

void OVTelemetry::InvokeCallbacks(LPCGUID SourceId, ULONG IsEnabled, UCHAR Level, ULONGLONG MatchAnyKeyword,
                                 ULONGLONG MatchAllKeyword, PEVENT_FILTER_DESCRIPTOR FilterData, PVOID CallbackContext) {
  std::lock_guard<std::mutex> lock_callbacks(callbacks_mutex_);
  for (const auto& callback : callbacks_) {
    (*callback)(SourceId, IsEnabled, Level, MatchAnyKeyword, MatchAllKeyword, FilterData, CallbackContext);
  }
}
#endif

} // namespace openvino_ep
} // namespace onnxruntime

#endif // defined(_WIN32)
