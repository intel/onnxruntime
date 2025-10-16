// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/providers/openvino/ov_tracing.h"

#ifdef _WIN32
#include <windows.h>
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 26440)
#endif
#include <TraceLoggingProvider.h>
#include <winmeta.h>
#include "core/platform/windows/TraceLoggingConfig.h"

TRACELOGGING_DEFINE_PROVIDER(
    ov_tracing_provider_handle,
    "Intel.ML.ONNXRuntime.OpenVINO",
    // {"b5a8c2e1-4d7f-4a3b-9c2e-1f8e5a6b7c9d"}
    (0xb5a8c2e1, 0x4d7f, 0x4a3b, 0x9c, 0x2e, 0x1f, 0x8e, 0x5a, 0x6b, 0x7c, 0x9d),
    TraceLoggingOptionMicrosoftTelemetry());

#ifdef _MSC_VER
#pragma warning(pop)
#endif

namespace {
std::string EscapeJsonString(const std::string& input) {
  std::string escaped;
  // Reserve extra space for escaping
  escaped.reserve(input.size() + input.size() / 5);

  for (char c : input) {
    switch (c) {
      case '\"':
        escaped += "\\\"";
        break;
      case '\\':
        escaped += "\\\\";
        break;
      case '\b':
        escaped += "\\b";
        break;
      case '\f':
        escaped += "\\f";
        break;
      case '\n':
        escaped += "\\n";
        break;
      case '\r':
        escaped += "\\r";
        break;
      case '\t':
        escaped += "\\t";
        break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char unicode_escape[7];
          sprintf_s(unicode_escape, sizeof(unicode_escape), "\\u%04x", static_cast<unsigned char>(c));
          escaped += unicode_escape;
        } else {
          escaped += c;
        }
        break;
    }
  }
  return escaped;
}
}  // namespace

namespace onnxruntime {
namespace openvino_ep {

std::mutex OVTracing::mutex_;
std::mutex OVTracing::provider_change_mutex_;
uint32_t OVTracing::global_register_count_ = 0;
bool OVTracing::enabled_ = true;
UCHAR OVTracing::level_ = 0;
UINT64 OVTracing::keyword_ = 0;
std::vector<const OVTracing::EtwInternalCallback*> OVTracing::callbacks_;
std::mutex OVTracing::callbacks_mutex_;

OVTracing::OVTracing() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (global_register_count_ == 0) {
    HRESULT hr = TraceLoggingRegisterEx(ov_tracing_provider_handle, ORT_TL_EtwEnableCallback, nullptr);
    if (SUCCEEDED(hr)) {
      global_register_count_ += 1;
    }
  }
}

OVTracing::~OVTracing() {
  // Clean up TraceLogging, only hold mutex_
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (global_register_count_ > 0) {
      global_register_count_ -= 1;
      if (global_register_count_ == 0) {
        TraceLoggingUnregister(ov_tracing_provider_handle);
      }
    }
  }

  // Clean up callbacks, only hold callbacks_mutex_
  {
    std::lock_guard<std::mutex> lock_callbacks(callbacks_mutex_);
    callbacks_.clear();
  }
}

OVTracing& OVTracing::Instance() {
  static OVTracing instance;
  return instance;
}

bool OVTracing::IsEnabled() const {
  std::lock_guard<std::mutex> lock(provider_change_mutex_);
  return enabled_;
}

UCHAR OVTracing::Level() const {
  std::lock_guard<std::mutex> lock(provider_change_mutex_);
  return level_;
}

UINT64 OVTracing::Keyword() const {
  std::lock_guard<std::mutex> lock(provider_change_mutex_);
  return keyword_;
}

void OVTracing::LogAllRuntimeOptions(uint32_t session_id, const SessionContext& ctx) const {
  if (!IsEnabled()) return;

  std::ostringstream opts;
  opts << "{";
  bool first = true;

  // Always log SDK version first
  opts << "\"openvino_sdk_version\":\"" << ctx.openvino_sdk_version << "\"";
  first = false;

  // Log all runtime options (session options contain everything including provider options)
  for (const auto& [key, value] : ctx.runtime_config.options) {
    if (!value.empty()) {
      if (!first) opts << ",";
      opts << "\"" << key << "\":\"" << EscapeJsonString(value) << "\"";
      first = false;
    }
  }

  opts << "}";

  TraceLoggingWrite(ov_tracing_provider_handle, "OVEPRuntimeOptions",
                    TraceLoggingLevel(WINEVENT_LEVEL_VERBOSE),
                    TraceLoggingUInt32(session_id, "session_id"),
                    TraceLoggingString(opts.str().c_str(), "runtime_options"));
}

void OVTracing::RegisterInternalCallback(const EtwInternalCallback& callback) {
  std::lock_guard<std::mutex> lock_callbacks(callbacks_mutex_);
  callbacks_.push_back(&callback);
}

void OVTracing::UnregisterInternalCallback(const EtwInternalCallback& callback) {
  std::lock_guard<std::mutex> lock_callbacks(callbacks_mutex_);
  auto new_end = std::remove_if(callbacks_.begin(), callbacks_.end(),
                                [&callback](const EtwInternalCallback* ptr) {
                                  return ptr == &callback;
                                });
  callbacks_.erase(new_end, callbacks_.end());
}

void NTAPI OVTracing::ORT_TL_EtwEnableCallback(
    _In_ LPCGUID SourceId, _In_ ULONG IsEnabled, _In_ UCHAR Level, _In_ ULONGLONG MatchAnyKeyword,
    _In_ ULONGLONG MatchAllKeyword, _In_opt_ PEVENT_FILTER_DESCRIPTOR FilterData, _In_opt_ PVOID CallbackContext) {
  std::lock_guard<std::mutex> lock(provider_change_mutex_);
  enabled_ = (IsEnabled != 0);
  level_ = Level;
  keyword_ = MatchAnyKeyword;
  InvokeCallbacks(SourceId, IsEnabled, Level, MatchAnyKeyword, MatchAllKeyword, FilterData, CallbackContext);
}

void OVTracing::InvokeCallbacks(LPCGUID SourceId, ULONG IsEnabled, UCHAR Level, ULONGLONG MatchAnyKeyword,
                                ULONGLONG MatchAllKeyword, PEVENT_FILTER_DESCRIPTOR FilterData, PVOID CallbackContext) {
  std::lock_guard<std::mutex> lock_callbacks(callbacks_mutex_);
  for (const auto& callback : callbacks_) {
    (*callback)(SourceId, IsEnabled, Level, MatchAnyKeyword, MatchAllKeyword, FilterData, CallbackContext);
  }
}

}  // namespace openvino_ep
}  // namespace onnxruntime

#endif  // defined(_WIN32)
