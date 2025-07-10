// Copyright (C) Intel Corporation
// Licensed under the MIT License

#pragma once

#include <string>
#include <cstdint>

#include "ov_provider.h"

namespace onnxruntime {
namespace openvino_ep {

struct EpContextNode : ApiPtrs {
  size_t num_nodes{0};
  int64_t main_context{1};
  std::string ep_cache_context;
  int64_t embed_mode{1};
  std::string ep_sdk_version;
  std::string onnx_model_filename;
  std::string hardware_architecture;
  std::string partition_name;
  std::string source;
  std::string notes;
  int64_t max_size{0};

  enum class EpContextType {
    Native,
    OV_IR,
  };
  EpContextType type_{EpContextType::Native};

  EpContextNode(ApiPtrs apis, const OrtNode* node)
      : ApiPtrs(apis) {
    // Helper lambda to extract attribute values

    auto get_attr_int64 = [&](const char* name, int64_t default_val) -> int64_t {
      int64_t val = default_val;
      const OrtOpAttr* attr = nullptr;
      const OrtOpAttrType type = ORT_OP_ATTR_INT;

      OrtStatus* status = ort_api.Node_GetAttributeByName(node, name, &attr);
      if (status) {
        ort_api.ReleaseStatus(status);
        return val;
      }

      size_t size_read = 0;
      status = ort_api.ReadOpAttr(attr, type, &val, sizeof(val), &size_read);
      if (status || size_read != sizeof(val)) {
        ort_api.ReleaseStatus(status);
        return val;
      }

      return val;
    };

    auto get_attr_string = [&](const char* name) -> std::string {
      std::string val{};
      const OrtOpAttr* attr = nullptr;
      const OrtOpAttrType type = ORT_OP_ATTR_STRING;

      OrtStatus* status = ort_api.Node_GetAttributeByName(node, name, &attr);
      if (status) {
        ort_api.ReleaseStatus(status);
        return val;
      }
      size_t required_size = 0;
      status = ort_api.ReadOpAttr(attr, type, nullptr, 0, &required_size);
      if (status) {
        // expect it to fail
        ort_api.ReleaseStatus(status);
      }

      val.resize(required_size);
      status = ort_api.ReadOpAttr(attr, type, val.data(), val.size(), &required_size);
      if (status) {
        ort_api.ReleaseStatus(status);
        return val;
      }

      return val;
    };

    main_context = get_attr_int64("main_context", 1);
    ep_cache_context = get_attr_string("ep_cache_context");
    embed_mode = get_attr_int64("embed_mode", 1);
    ep_sdk_version = get_attr_string("ep_sdk_version");
    onnx_model_filename = get_attr_string("onnx_model_filename");
    hardware_architecture = get_attr_string("hardware_architecture");
    partition_name = get_attr_string("partition_name");
    source = get_attr_string("source");
    notes = get_attr_string("notes");
    max_size = get_attr_int64("max_size", 0);
    type_ = EpContextType::Native;
  }
};

}  // namespace openvino_ep
}  // namespace onnxruntime
