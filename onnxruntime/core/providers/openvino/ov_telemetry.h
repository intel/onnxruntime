// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#ifdef _WIN32
#include <windows.h>
#if !BUILD_OPENVINO_EP_STATIC_LIB
#include <TraceLoggingProvider.h>
#endif
#include <functional>
#include <mutex>
#include <string>
#include <vector>
#include <unordered_map>
// #include "core/common/logging/logging.h"

#if !BUILD_OPENVINO_EP_STATIC_LIB
TRACELOGGING_DECLARE_PROVIDER(ov_telemetry_provider_handle);
#endif

namespace onnxruntime {
namespace openvino_ep {

class OVTelemetry {
 public:
  static OVTelemetry& Instance();
  bool IsEnabled() const;
  unsigned char Level() const;
  UINT64 Keyword() const;

  // Provider lifecycle events
  void LogProviderInit(uint32_t session_id, const std::string& device_type, const std::string& precision) const;
  void LogProviderShutdown(uint32_t session_id) const;

  // Session and Provider Options logging
  void LogSessionOptions(uint32_t session_id, const std::unordered_map<std::string, std::string>& session_options) const;
  void LogProviderOptions(uint32_t session_id, const std::unordered_map<std::string, std::string>& provider_options) const;
  void LogSessionCreation(uint32_t session_id, const std::string& model_path, const std::string& openvino_version) const;
  void LogSessionDestruction(uint32_t session_id) const;

  // Core OpenVINO EP events
  void LogCapabilityDetection(uint32_t session_id, uint32_t node_count, bool wholly_supported,
                             bool has_external_weights, const std::string& device_type) const;
  void LogCompileStart(uint32_t session_id, uint32_t fused_node_count, const std::string& device_type,
                      const std::string& precision) const;
  void LogCompileEnd(uint32_t session_id, bool success, const std::string& error_message,
                    int64_t compile_duration_ms = 0) const;
  void LogComputeStart(uint32_t session_id, const std::string& subgraph_name, const std::string& device_type) const;
  void LogComputeEnd(uint32_t session_id, int64_t duration_microseconds, const std::string& subgraph_name,
                    bool success = true, const std::string& error_message = "") const;

  // Backend and configuration events
  void LogBackendManagerEvent(uint32_t session_id, const std::string& event_type, const std::string& details) const;
  void LogDeviceSelection(uint32_t session_id, const std::string& requested_device,
                         const std::string& actual_device, const std::string& selection_reason) const;
  void LogError(uint32_t session_id, const std::string& error_category, const std::string& error_message,
               const std::string& function_name = "", int line_number = 0) const;

  using EtwInternalCallback = std::function<void(
      LPCGUID SourceId, ULONG IsEnabled, UCHAR Level, ULONGLONG MatchAnyKeyword, ULONGLONG MatchAllKeyword,
      PEVENT_FILTER_DESCRIPTOR FilterData, PVOID CallbackContext)>;

  static void RegisterInternalCallback(const EtwInternalCallback& callback);
  static void UnregisterInternalCallback(const EtwInternalCallback& callback);

 private:
  OVTelemetry();
  ~OVTelemetry();
  OVTelemetry(const OVTelemetry&) = delete;
  OVTelemetry& operator=(const OVTelemetry&) = delete;
  OVTelemetry(OVTelemetry&&) = delete;
  OVTelemetry& operator=(OVTelemetry&&) = delete;

  std::string SerializeOptionsMap(const std::unordered_map<std::string, std::string>& options) const;

#if !BUILD_OPENVINO_EP_STATIC_LIB
  static std::mutex mutex_;
  static uint32_t global_register_count_;
  static bool enabled_;
  static std::vector<const EtwInternalCallback*> callbacks_;
  static std::mutex callbacks_mutex_;
  static std::mutex provider_change_mutex_;
  static UCHAR level_;
  static ULONGLONG keyword_;

  static void InvokeCallbacks(LPCGUID SourceId, ULONG IsEnabled, UCHAR Level, ULONGLONG MatchAnyKeyword,
                             ULONGLONG MatchAllKeyword, PEVENT_FILTER_DESCRIPTOR FilterData, PVOID CallbackContext);

  static void NTAPI ORT_TL_EtwEnableCallback(
      _In_ LPCGUID SourceId, _In_ ULONG IsEnabled, _In_ UCHAR Level, _In_ ULONGLONG MatchAnyKeyword,
      _In_ ULONGLONG MatchAllKeyword, _In_opt_ PEVENT_FILTER_DESCRIPTOR FilterData, _In_opt_ PVOID CallbackContext);
#endif
};

// Keywords for comprehensive OpenVINO EP tracing
namespace ov_keywords {
  constexpr uint64_t OV_PROVIDER = 0x1;
  constexpr uint64_t OV_SESSION = 0x2;
  constexpr uint64_t OV_OPTIONS = 0x4;
  constexpr uint64_t OV_CAPABILITY = 0x8;
  constexpr uint64_t OV_COMPILATION = 0x10;
  constexpr uint64_t OV_EXECUTION = 0x20;
  constexpr uint64_t OV_BACKEND = 0x40;
  constexpr uint64_t OV_PERFORMANCE = 0x80;
  constexpr uint64_t OV_ERROR = 0x100;
}

} // namespace openvino_ep
} // namespace onnxruntime

#endif // defined(_WIN32)
