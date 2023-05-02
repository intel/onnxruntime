// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/shared_library/provider_api.h"
#include "contrib_ops/cpu/transformers/beam_search.h"

namespace onnxruntime {

class SessionState;

namespace contrib {

namespace openvino_ep {

class BeamSearch final : public onnxruntime::contrib::transformers::BeamSearch {
 public:
  BeamSearch(const OpKernelInfo& info);

  Status Compute(OpKernelContext* context) const override;

 private:
  Status ComputeInternal(OpKernelContext* context) const;
};

}  // namespace openvino
}  // namespace contrib
}  // namespace onnxruntime
