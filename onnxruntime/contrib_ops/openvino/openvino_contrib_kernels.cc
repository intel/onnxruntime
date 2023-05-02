// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/openvino/openvino_contrib_kernels.h"

using namespace onnxruntime::common;

namespace onnxruntime {
namespace contrib {
namespace openvino_ep {

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kOpenVINOExecutionProvider, kMSDomain, 1, float, BeamSearch);

template <>
KernelCreateInfo BuildKernelCreateInfo<void>() {
  KernelCreateInfo info;
  return info;
}

Status RegisterOpenVINOContribKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
    BuildKernelCreateInfo<void>,  // default entry to avoid the list become empty after ops-reducing
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kOpenVINOExecutionProvider, kMSDomain, 1, float, BeamSearch)>
  };

  for (auto& function_table_entry : function_table) {
    KernelCreateInfo info = function_table_entry();
    if (info.kernel_def != nullptr) {  // filter disabled entries where type is void
      ORT_RETURN_IF_ERROR(kernel_registry.Register(std::move(info)));
      //return kernel_registry.Register(std::move(info));
    }
  }

  return Status::OK();
}

}  // namespace openvino 
}  // namespace contrib
}  // namespace onnxruntime

