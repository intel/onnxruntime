// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/providers/openvino/ov_telemetry.h"

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
    ov_telemetry_provider_handle,
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

template <typename T>
void AddOptionalValue(std::ostringstream& json, const std::string& key, const T& value, const T& default_value, bool& first) {
  if (value != default_value) {
    if (!first) json << ",";
    json << "\"" << key << "\":";
    if constexpr (std::is_same_v<T, std::string>) {
      json << "\"" << EscapeJsonString(value) << "\"";
    } else if constexpr (std::is_same_v<T, std::filesystem::path>) {
      json << "\"" << EscapeJsonString(value.string()) << "\"";
    } else if constexpr (std::is_same_v<T, bool>) {
      json << (value ? "true" : "false");
    } else {
      json << value;
    }
    first = false;
  }
}
}  // namespace

namespace onnxruntime {
namespace openvino_ep {

std::mutex OVTelemetry::mutex_;
std::mutex OVTelemetry::provider_change_mutex_;
uint32_t OVTelemetry::global_register_count_ = 0;
bool OVTelemetry::enabled_ = true;
UCHAR OVTelemetry::level_ = 0;
UINT64 OVTelemetry::keyword_ = 0;
std::vector<const OVTelemetry::EtwInternalCallback*> OVTelemetry::callbacks_;
std::mutex OVTelemetry::callbacks_mutex_;

OVTelemetry::OVTelemetry() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (global_register_count_ == 0) {
    HRESULT hr = TraceLoggingRegisterEx(ov_telemetry_provider_handle, ORT_TL_EtwEnableCallback, nullptr);
    if (SUCCEEDED(hr)) {
      global_register_count_ += 1;
    }
  }
}

OVTelemetry::~OVTelemetry() {
  // Clean up TraceLogging, only hold mutex_
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (global_register_count_ > 0) {
      global_register_count_ -= 1;
      if (global_register_count_ == 0) {
        TraceLoggingUnregister(ov_telemetry_provider_handle);
      }
    }
  }

  // Clean up callbacks, only hold callbacks_mutex_
  {
    std::lock_guard<std::mutex> lock_callbacks(callbacks_mutex_);
    callbacks_.clear();
  }
}

OVTelemetry& OVTelemetry::Instance() {
  static OVTelemetry instance;
  return instance;
}

bool OVTelemetry::IsEnabled() const {
  std::lock_guard<std::mutex> lock(provider_change_mutex_);
  return enabled_;
}

UCHAR OVTelemetry::Level() const {
  std::lock_guard<std::mutex> lock(provider_change_mutex_);
  return level_;
}

UINT64 OVTelemetry::Keyword() const {
  std::lock_guard<std::mutex> lock(provider_change_mutex_);
  return keyword_;
}

std::string OVTelemetry::SerializeLoadConfig(const SessionContext& ctx) const {
  if (ctx.load_config.empty()) return "{}";

  std::ostringstream json;
  json << "{";
  bool first_device = true;

  for (const auto& [device, anymap] : ctx.load_config) {
    if (!first_device) json << ",";
    json << "\"" << device << "\":{";

    bool first_entry = true;
    for (const auto& [key, value] : anymap) {
      if (!first_entry) json << ",";
      json << "\"" << key << "\":";

      // Use ov::Any's type checking and extraction capabilities
      if (value.is<std::string>()) {
        std::string str_val = value.as<std::string>();
        json << "\"" << EscapeJsonString(str_val) << "\"";
      } else if (value.is<int>()) {
        json << value.as<int>();
      } else if (value.is<int64_t>()) {
        json << value.as<int64_t>();
      } else if (value.is<uint64_t>()) {
        json << value.as<uint64_t>();
      } else if (value.is<bool>()) {
        json << (value.as<bool>() ? "true" : "false");
      } else if (value.is<float>()) {
        json << value.as<float>();
      } else if (value.is<double>()) {
        json << value.as<double>();
      } else {
        // Use ov::Any's print method for unknown types
        std::ostringstream temp;
        value.print(temp);
        std::string val_str = temp.str();
        json << "\"" << EscapeJsonString(val_str) << "\"";
      }
      first_entry = false;
    }
    json << "}";
    first_device = false;
  }
  json << "}";
  return json.str();
}

std::string OVTelemetry::SerializeReshapeInputConfig(const SessionContext& ctx) const {
  if (ctx.reshape.empty()) return "{}";

  std::ostringstream json;
  json << "{";
  bool first = true;

  for (const auto& [tensor_name, partial_shape] : ctx.reshape) {
    if (!first) json << ",";
    json << "\"" << tensor_name << "\":[";

    bool first_dim = true;
    // Use ov::PartialShape's iterator interface to iterate through dimensions
    for (const auto& dimension : partial_shape) {
      if (!first_dim) json << ",";

      if (dimension.is_dynamic()) {
        // Handle dynamic dimensions
        if (dimension.get_interval().has_upper_bound()) {
          // Dynamic dimension with bounds: {min..max}
          json << "{\"min\":" << dimension.get_min_length()
               << ",\"max\":" << dimension.get_max_length() << "}";
        } else {
          // Fully dynamic dimension
          json << "\"dynamic\"";
        }
      } else {
        // Static dimension - use get_length() for ov::Dimension
        json << dimension.get_length();
      }
      first_dim = false;
    }
    json << "]";
    first = false;
  }
  json << "}";
  return json.str();
}

std::string OVTelemetry::SerializeLayoutConfig(const SessionContext& ctx) const {
  if (ctx.layout.empty()) return "{}";

  std::ostringstream json;
  json << "{";
  bool first = true;

  for (const auto& [tensor_name, ov_layout] : ctx.layout) {
    if (!first) json << ",";
    json << "\"" << tensor_name << "\":";

    // Use ov::Layout's to_string() method for proper string representation
    std::string layout_str = ov_layout.to_string();
    json << "\"" << EscapeJsonString(layout_str) << "\"";
    first = false;
  }
  json << "}";
  return json.str();
}

void OVTelemetry::LogAllProviderOptions(uint32_t session_id, const SessionContext& ctx) const {
  if (!IsEnabled()) return;

  std::ostringstream opts;
  opts << "{";
  bool first = true;

  // Only log non-default values
  AddOptionalValue(opts, "device_type", ctx.device_type, std::string(""), first);
  AddOptionalValue(opts, "precision", ctx.precision, std::string(""), first);
  AddOptionalValue(opts, "disable_dynamic_shapes", ctx.disable_dynamic_shapes, false, first);
  AddOptionalValue(opts, "enable_qdq_optimizer", ctx.enable_qdq_optimizer, false, first);
  AddOptionalValue(opts, "enable_causallm", ctx.enable_causallm, false, first);

  if (!ctx.cache_dir.empty()) {
    if (!first) opts << ",";
    opts << "\"cache_dir\":\"" << EscapeJsonString(ctx.cache_dir.string()) << "\"";
    first = false;
  }

  // Load configuration
  std::string load_config_json = SerializeLoadConfig(ctx);
  if (load_config_json != "{}") {
    if (!first) opts << ",";
    opts << "\"load_config\":" << load_config_json;
    first = false;
  }

  // Reshape configuration
  std::string reshape_json = SerializeReshapeInputConfig(ctx);
  if (reshape_json != "{}") {
    if (!first) opts << ",";
    opts << "\"reshape_input\":" << reshape_json;
    first = false;
  }

  // Layout configuration
  std::string layout_json = SerializeLayoutConfig(ctx);
  if (layout_json != "{}") {
    if (!first) opts << ",";
    opts << "\"layout\":" << layout_json;
    first = false;
  }

  opts << "}";

  // Log only if there are provider options available
  if (opts.str() != "{}") {
    TraceLoggingWrite(ov_telemetry_provider_handle, "OVEPProviderOptions",
                      TraceLoggingKeyword(ov_keywords::OV_PROVIDER | ov_keywords::OV_OPTIONS),
                      TraceLoggingLevel(5),
                      TraceLoggingUInt32(session_id, "session_id"),
                      TraceLoggingString(opts.str().c_str(), "provider_options"));
  }
}

void OVTelemetry::LogAllSessionOptions(uint32_t session_id, const SessionContext& ctx) const {
  if (!IsEnabled()) return;

  std::ostringstream sopts;
  sopts << "{";
  bool first = true;

  // Always log SDK version
  sopts << "\"openvino_sdk_version\":\"" << ctx.openvino_sdk_version << "\"";
  first = false;

  // Only log session options if they're non-default
  AddOptionalValue(sopts, "ep.context_enable", ctx.so_context_enable, false, first);
  AddOptionalValue(sopts, "session.disable_cpu_ep_fallback", ctx.so_disable_cpu_ep_fallback, false, first);
  AddOptionalValue(sopts, "ep.context_embed_mode", ctx.so_context_embed_mode, false, first);
  AddOptionalValue(sopts, "ep.share_ep_contexts", ctx.so_share_ep_contexts, false, first);
  AddOptionalValue(sopts, "ep.stop_share_ep_contexts", ctx.so_stop_share_ep_contexts, false, first);
  AddOptionalValue(sopts, "ep.context_file_path", ctx.so_context_file_path, std::filesystem::path(), first);

  sopts << "}";

  TraceLoggingWrite(ov_telemetry_provider_handle, "OVEPSessionOptions",
                    TraceLoggingKeyword(ov_keywords::OV_SESSION | ov_keywords::OV_OPTIONS),
                    TraceLoggingLevel(5),
                    TraceLoggingUInt32(session_id, "session_id"),
                    TraceLoggingString(sopts.str().c_str(), "session_options"));
}

void OVTelemetry::RegisterInternalCallback(const EtwInternalCallback& callback) {
  std::lock_guard<std::mutex> lock_callbacks(callbacks_mutex_);
  callbacks_.push_back(&callback);
}

void OVTelemetry::UnregisterInternalCallback(const EtwInternalCallback& callback) {
  std::lock_guard<std::mutex> lock_callbacks(callbacks_mutex_);
  auto new_end = std::remove_if(callbacks_.begin(), callbacks_.end(),
                                [&callback](const EtwInternalCallback* ptr) {
                                  return ptr == &callback;
                                });
  callbacks_.erase(new_end, callbacks_.end());
}

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

}  // namespace openvino_ep
}  // namespace onnxruntime

#endif  // defined(_WIN32)
