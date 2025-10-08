// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#ifdef _WIN32
#include <windows.h>
#if !BUILD_OPENVINO_EP_STATIC_LIB
#include <TraceLoggingProvider.h>
#include <winmeta.h>
#endif
#include <functional>
#include <mutex>
#include <string>
#include <vector>
#include <sstream>
#include <unordered_map>
#include "core/providers/openvino/contexts.h"
#include "nlohmann/json.hpp"

#if !BUILD_OPENVINO_EP_STATIC_LIB
TRACELOGGING_DECLARE_PROVIDER(ov_telemetry_provider_handle);
#endif

namespace onnxruntime {
namespace openvino_ep {

namespace ov_keywords {
  constexpr uint64_t OV_PROVIDER   = 0x1;
  constexpr uint64_t OV_SESSION    = 0x2;
  constexpr uint64_t OV_OPTIONS    = 0x4;
  constexpr uint64_t OV_CAPABILITY = 0x8;
  constexpr uint64_t OV_COMPILATION= 0x10;
  constexpr uint64_t OV_EXECUTION  = 0x20;
  constexpr uint64_t OV_BACKEND    = 0x40;
  constexpr uint64_t OV_PERFORMANCE= 0x80;
  constexpr uint64_t OV_ERROR      = 0x100;
}

class OVTelemetry {
 public:
  static OVTelemetry& Instance();
  bool IsEnabled() const;
  unsigned char Level() const;
  UINT64 Keyword() const;

  // Comprehensive provider options logging
  void LogAllProviderOptions(uint32_t session_id, const SessionContext& ctx) const;
  void LogAllSessionOptions(uint32_t session_id, const SessionContext& ctx) const;

  // Individual logging methods to avoid template issues
  void LogSessionDestruction(uint32_t session_id) const;
  void LogProviderShutdown(uint32_t session_id) const;
  void LogCompileStart(uint32_t session_id, uint32_t fused_node_count, const std::string& device_type, const std::string& precision) const;
  void LogCompileEnd(uint32_t session_id, bool success, const std::string& error_message, int64_t duration_ms) const;

  using EtwInternalCallback = std::function<void(
    LPCGUID, ULONG, UCHAR, ULONGLONG, ULONGLONG, PEVENT_FILTER_DESCRIPTOR, PVOID)>;
  static void RegisterInternalCallback(const EtwInternalCallback& callback);
  static void UnregisterInternalCallback(const EtwInternalCallback& callback);

 private:
  OVTelemetry();
  ~OVTelemetry();
  OVTelemetry(const OVTelemetry&) = delete;
  OVTelemetry& operator=(const OVTelemetry&) = delete;
  OVTelemetry(OVTelemetry&&) = delete;
  OVTelemetry& operator=(OVTelemetry&&) = delete;

  // Helper functions for complex serialization
  std::string SerializeLoadConfig(const SessionContext& ctx) const;
  std::string SerializeReshapeConfig(const SessionContext& ctx) const;
  std::string SerializeLayoutConfig(const SessionContext& ctx) const;

#if !BUILD_OPENVINO_EP_STATIC_LIB
  static std::mutex mutex_;
  static uint32_t global_register_count_;
  static bool enabled_;
  static std::vector<const EtwInternalCallback*> callbacks_;
  static std::mutex callbacks_mutex_;
  static std::mutex provider_change_mutex_;
  static UCHAR level_;
  static ULONGLONG keyword_;

  static void InvokeCallbacks(LPCGUID, ULONG, UCHAR, ULONGLONG, ULONGLONG, PEVENT_FILTER_DESCRIPTOR, PVOID);
  static void NTAPI ORT_TL_EtwEnableCallback(_In_ LPCGUID, _In_ ULONG, _In_ UCHAR, _In_ ULONGLONG,
                                             _In_ ULONGLONG, _In_opt_ PEVENT_FILTER_DESCRIPTOR, _In_opt_ PVOID);
#endif
};

// Direct macro definitions to avoid template conflicts
#define OV_LOG_SESSION_CREATION(session_id, model_path, version) \
  do { \
    if (OVTelemetry::Instance().IsEnabled()) { \
      TraceLoggingWrite(ov_telemetry_provider_handle, "OVSessionCreation", \
                       TraceLoggingKeyword(ov_keywords::OV_SESSION), \
                       TraceLoggingLevel(5), \
                       TraceLoggingUInt32(session_id, "session_id"), \
                       TraceLoggingString(model_path.c_str(), "model_path"), \
                       TraceLoggingString(version.c_str(), "openvino_version")); \
    } \
  } while(0)

#define OV_LOG_PROVIDER_INIT(session_id, device_type, precision) \
  do { \
    if (OVTelemetry::Instance().IsEnabled()) { \
      TraceLoggingWrite(ov_telemetry_provider_handle, "OVProviderInit", \
                       TraceLoggingKeyword(ov_keywords::OV_PROVIDER), \
                       TraceLoggingLevel(5), \
                       TraceLoggingUInt32(session_id, "session_id"), \
                       TraceLoggingString(device_type.c_str(), "device_type"), \
                       TraceLoggingString(precision.c_str(), "precision")); \
    } \
  } while(0)

#define OV_LOG_CAPABILITY_DETECTION(session_id, node_count, wholly_supported, has_external_weights, device_type) \
  do { \
    if (OVTelemetry::Instance().IsEnabled()) { \
      TraceLoggingWrite(ov_telemetry_provider_handle, "OVCapabilityDetection", \
                       TraceLoggingKeyword(ov_keywords::OV_CAPABILITY), \
                       TraceLoggingLevel(5), \
                       TraceLoggingUInt32(session_id, "session_id"), \
                       TraceLoggingUInt32(node_count, "node_count"), \
                       TraceLoggingBool(wholly_supported, "wholly_supported"), \
                       TraceLoggingBool(has_external_weights, "has_external_weights"), \
                       TraceLoggingString(device_type.c_str(), "device_type")); \
    } \
  } while(0)

#define OV_LOG_COMPILE_START(session_id, fused_node_count, device_type, precision) \
  OVTelemetry::Instance().LogCompileStart(session_id, fused_node_count, device_type, precision)

#define OV_LOG_COMPILE_END(session_id, success, error_message, duration_ms) \
  OVTelemetry::Instance().LogCompileEnd(session_id, success, error_message, duration_ms)

} // namespace openvino_ep
} // namespace onnxruntime

#endif // defined(_WIN32)
