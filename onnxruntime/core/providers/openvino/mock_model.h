// Copyright (C) Intel Corporation
// Licensed under the MIT License

#pragma once

#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {
namespace openvino_ep {

struct mock_model {
  mock_model(const OrtApi& api, OrtGraph* graph) : api{api}, graph{graph} {}
  ~mock_model();

  const OrtApi &api{nullptr};
  OrtGraph* graph{nullptr};
};

mock_model build_model(const OrtApi& api, bool use_constant_node = true);

}
}  // namespace onnxruntime