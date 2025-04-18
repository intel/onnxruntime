#pragma once

#include <string_view>

#include "openvino/core/version.hpp"

#define STRINGIFY(s) STRING(s)
#define STRING(s) #s

namespace onnxruntime {
namespace openvino_ep {

namespace constants {

constexpr inline std::string_view metadata_bin_name{"weight_info_map.bin"};
constexpr inline uint32_t max_device_available{8};

namespace ov_version {
constexpr inline uint32_t major{OPENVINO_VERSION_MAJOR};
constexpr inline uint32_t minor{OPENVINO_VERSION_MINOR};
constexpr inline std::string_view name{STRINGIFY(OPENVINO_VERSION_MAJOR) "." STRINGIFY(OPENVINO_VERSION_MINOR)};
constexpr inline float number{OPENVINO_VERSION_MAJOR + 0.1f * OPENVINO_VERSION_MINOR + 0.01f * OPENVINO_VERSION_PATCH};
}  // namespace ov_version

}  // namespace constants
}  // namespace openvino_ep
}  // namespace onnxruntime
