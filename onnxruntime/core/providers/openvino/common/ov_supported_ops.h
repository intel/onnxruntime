// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include <cstdint>
#include <string>
#include <string_view>
#include <set>
#include <variant>

namespace onnxruntime {
namespace openvino_ep {

struct OpenVINOVersion {
  OpenVINOVersion() = default;
  OpenVINOVersion(uint32_t major, uint32_t minor)
      : version_((static_cast<uint64_t>(major) << 32) | minor) {}

  static OpenVINOVersion GetBuiltVersion();
  uint32_t major_version() const { return static_cast<uint32_t>(version_ >> 32); }
  uint32_t minor_version() const { return static_cast<uint32_t>(version_ & 0xFFFFFFFF); }

  bool operator==(const OpenVINOVersion& other) const {
    return version_ == other.version_;
  }

  bool operator<(const OpenVINOVersion& other) const {
    return version_ < other.version_;
  }

  bool operator>=(const OpenVINOVersion& other) const {
    return version_ >= other.version_;
  }

 private:
  uint64_t version_{0};
};

struct SupportedOps {
  bool
  IsOpSupported(std::string_view op_type, std::string_view domain) const {
    return ops_.find(std::make_pair(op_type, domain)) != ops_.end();
  }

  bool IsEpContextNode(std::string_view op_type, std::string_view domain) const {
    return op_type == "EPContext" && domain == "com.microsoft";
  }

  static SupportedOps& Get() {
    static SupportedOps instance(OpenVINOVersion::GetBuiltVersion());
    return instance;
  }

 private:
  SupportedOps() = default;
  SupportedOps(const OpenVINOVersion& ov_version);
  OpenVINOVersion version_;
  std::set<std::pair<std::string_view, std::string_view>> ops_;
};

}  // namespace openvino_ep
}  // namespace onnxruntime
