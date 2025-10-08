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
#include "core/platform/windows/TraceLoggingConfig.h"

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
  return false;
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
        // Escape quotes in string
        std::string escaped;
        for (char c : str_val) {
          if (c == '\"' || c == '\\') escaped.push_back('\\');
          escaped.push_back(c);
        }
        json << "\"" << escaped << "\"";
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

        // Escape quotes in the printed string
        std::string escaped;
        for (char c : val_str) {
          if (c == '\"' || c == '\\') escaped.push_back('\\');
          escaped.push_back(c);
        }
        json << "\"" << escaped << "\"";
      }
      first_entry = false;
    }
    json << "}";
    first_device = false;
  }
  json << "}";
  return json.str();
}


std::string OVTelemetry::SerializeReshapeConfig(const SessionContext& ctx) const {
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

    // Escape quotes in the layout string if any
    std::string escaped;
    for (char c : layout_str) {
      if (c == '\"' || c == '\\') escaped.push_back('\\');
      escaped.push_back(c);
    }

    json << "\"" << escaped << "\"";
    first = false;
  }
  json << "}";
  return json.str();
}


void OVTelemetry::LogAllProviderOptions(uint32_t session_id, const SessionContext& ctx) const {
#if !BUILD_OPENVINO_EP_STATIC_LIB
  if (!IsEnabled()) return;

  std::ostringstream opts;
  opts << "{"
       << "\"device_type\":\"" << ctx.device_type << "\","
       << "\"precision\":\"" << ctx.precision << "\","
       << "\"num_of_threads\":" << ctx.num_of_threads << ","
       << "\"model_priority\":\"" << ctx.model_priority << "\","
       << "\"num_streams\":" << ctx.num_streams << ","
       << "\"enable_opencl_throttling\":" << (ctx.enable_opencl_throttling ? "true" : "false") << ","
       << "\"disable_dynamic_shapes\":" << (ctx.disable_dynamic_shapes ? "true" : "false") << ","
       << "\"enable_qdq_optimizer\":" << (ctx.enable_qdq_optimizer ? "true" : "false") << ","
       << "\"enable_causallm\":" << (ctx.enable_causallm ? "true" : "false") << ","
       << "\"cache_dir\":\"" << ctx.cache_dir.string() << "\","
       << "\"context\":" << (ctx.context ? "\"set\"" : "null") << ","
       << "\"load_config\":" << SerializeLoadConfig(ctx) << ","
       << "\"reshape\":" << SerializeReshapeConfig(ctx) << ","
       << "\"layout\":" << SerializeLayoutConfig(ctx)
       << "}";

  TraceLoggingWrite(ov_telemetry_provider_handle, "OVProviderOptionsComplete",
                   TraceLoggingKeyword(ov_keywords::OV_PROVIDER | ov_keywords::OV_OPTIONS),
                   TraceLoggingLevel(5),
                   TraceLoggingUInt32(session_id, "session_id"),
                   TraceLoggingString(opts.str().c_str(), "provider_options"));
#endif
}

void OVTelemetry::LogAllSessionOptions(uint32_t session_id, const SessionContext& ctx) const {
#if !BUILD_OPENVINO_EP_STATIC_LIB
  if (!IsEnabled()) return;

  std::ostringstream sopts;
  sopts << "{"
        << "\"onnx_model_path_name\":\"" << ctx.onnx_model_path_name.string() << "\","
        << "\"onnx_opset_version\":" << ctx.onnx_opset_version << ","
        << "\"wholly_supported_graph\":" << (ctx.is_wholly_supported_graph ? "true" : "false") << ","
        << "\"has_external_weights\":" << (ctx.has_external_weights ? "true" : "false") << ","
        << "\"openvino_sdk_version\":\"" << ctx.openvino_sdk_version << "\","
        << "\"so_context_enable\":" << (ctx.so_context_enable ? "true" : "false") << ","
        << "\"so_disable_cpu_ep_fallback\":" << (ctx.so_disable_cpu_ep_fallback ? "true" : "false") << ","
        << "\"so_context_embed_mode\":" << (ctx.so_context_embed_mode ? "true" : "false") << ","
        << "\"so_share_ep_contexts\":" << (ctx.so_share_ep_contexts ? "true" : "false") << ","
        << "\"so_context_file_path\":\"" << ctx.so_context_file_path.string() << "\""
        << "}";

  TraceLoggingWrite(ov_telemetry_provider_handle, "OVSessionOptionsComplete",
                   TraceLoggingKeyword(ov_keywords::OV_SESSION | ov_keywords::OV_OPTIONS),
                   TraceLoggingLevel(5),
                   TraceLoggingUInt32(session_id, "session_id"),
                   TraceLoggingString(sopts.str().c_str(), "session_options"));
#endif
}

void OVTelemetry::LogSessionDestruction(uint32_t session_id) const {
#if !BUILD_OPENVINO_EP_STATIC_LIB
  if (!IsEnabled()) return;
  TraceLoggingWrite(ov_telemetry_provider_handle, "OVSessionDestruction",
                   TraceLoggingKeyword(ov_keywords::OV_SESSION),
                   TraceLoggingLevel(5),
                   TraceLoggingUInt32(session_id, "session_id"));
#endif
}

void OVTelemetry::LogProviderShutdown(uint32_t session_id) const {
#if !BUILD_OPENVINO_EP_STATIC_LIB
  if (!IsEnabled()) return;
  TraceLoggingWrite(ov_telemetry_provider_handle, "OVProviderShutdown",
                   TraceLoggingKeyword(ov_keywords::OV_PROVIDER),
                   TraceLoggingLevel(5),
                   TraceLoggingUInt32(session_id, "session_id"));
#endif
}

void OVTelemetry::LogCompileStart(uint32_t session_id, uint32_t fused_node_count, const std::string& device_type, const std::string& precision) const {
#if !BUILD_OPENVINO_EP_STATIC_LIB
  if (!IsEnabled()) return;
  TraceLoggingWrite(ov_telemetry_provider_handle, "OVCompileStart",
                   TraceLoggingKeyword(ov_keywords::OV_COMPILATION),
                   TraceLoggingLevel(5),
                   TraceLoggingUInt32(session_id, "session_id"),
                   TraceLoggingUInt32(fused_node_count, "fused_node_count"),
                   TraceLoggingString(device_type.c_str(), "device_type"),
                   TraceLoggingString(precision.c_str(), "precision"));
#endif
}

void OVTelemetry::LogCompileEnd(uint32_t session_id, bool success, const std::string& error_message, int64_t duration_ms) const {
#if !BUILD_OPENVINO_EP_STATIC_LIB
  if (!IsEnabled()) return;
  TraceLoggingWrite(ov_telemetry_provider_handle, "OVCompileEnd",
                   TraceLoggingKeyword(ov_keywords::OV_COMPILATION | ov_keywords::OV_PERFORMANCE),
                   TraceLoggingLevel(4),
                   TraceLoggingUInt32(session_id, "session_id"),
                   TraceLoggingBool(success, "success"),
                   TraceLoggingString(error_message.c_str(), "error_message"),
                   TraceLoggingInt64(duration_ms, "compile_duration_ms"));
#endif
}

void OVTelemetry::RegisterInternalCallback(const EtwInternalCallback& callback) {
#if BUILD_OPENVINO_EP_STATIC_LIB
#else
  std::lock_guard<std::mutex> lock_callbacks(callbacks_mutex_);
  callbacks_.push_back(&callback);
#endif
}

void OVTelemetry::UnregisterInternalCallback(const EtwInternalCallback& callback) {
#if BUILD_OPENVINO_EP_STATIC_LIB
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
