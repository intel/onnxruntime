// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#ifdef _WIN32
#include <windows.h>
#include <TraceLoggingProvider.h>
#include <winmeta.h>

#include <functional>
#include <mutex>
#include <string>
#include <vector>
#include <sstream>
#include <unordered_map>
#include <optional>
#include <algorithm>
#include "core/providers/openvino/contexts.h"
#include "nlohmann/json.hpp"

TRACELOGGING_DECLARE_PROVIDER(ov_telemetry_provider_handle);

namespace onnxruntime {
namespace openvino_ep {

namespace ov_keywords {
  constexpr uint64_t OV_PROVIDER   = 0x1;
  constexpr uint64_t OV_SESSION    = 0x2;
  constexpr uint64_t OV_OPTIONS    = 0x3;
}

class OVTelemetry {
 public:
  static OVTelemetry& Instance();
  bool IsEnabled() const;
  unsigned char Level() const;
  UINT64 Keyword() const;

  void LogAllProviderOptions(uint32_t session_id, const SessionContext& ctx) const;
  void LogAllSessionOptions(uint32_t session_id, const SessionContext& ctx) const;

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
  std::string SerializeReshapeInputConfig(const SessionContext& ctx) const;
  std::string SerializeLayoutConfig(const SessionContext& ctx) const;

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

};

} // namespace openvino_ep
} // namespace onnxruntime

#endif // defined(_WIN32)
