// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/openvino/beam_search.h"

namespace onnxruntime {
namespace contrib {
namespace openvino_ep {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      BeamSearch,                                                 \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kOpenVINOExecutionProvider,                                 \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      BeamSearch);

REGISTER_KERNEL_TYPED(float)

BeamSearch::BeamSearch(const OpKernelInfo& info)
    : onnxruntime::contrib::transformers::BeamSearch(info) {
 
}

Status BeamSearch::ComputeInternal(OpKernelContext* context) const {
  return onnxruntime::contrib::transformers::BeamSearch::Compute(context);
}

Status BeamSearch::Compute(OpKernelContext* context) const {
  auto s = ComputeInternal(context);
  return s;
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
