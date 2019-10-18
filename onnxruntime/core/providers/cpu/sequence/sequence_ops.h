// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {

class SequenceLength final : public OpKernel {
 public:
  SequenceLength(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

class SequenceAt final : public OpKernel {
 public:
  SequenceAt(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

class SequenceEmpty final : public OpKernel {
 public:
  SequenceEmpty(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;

 private:
  int64_t dtype_{};
};

class SequenceInsert final : public OpKernel {
 public:
  SequenceInsert(const OpKernelInfo& info) : OpKernel(info) {
  }
  Status Compute(OpKernelContext* context) const override;
};

class SequenceErase final : public OpKernel {
 public:
  SequenceErase(const OpKernelInfo& info) : OpKernel(info) {
  }
  Status Compute(OpKernelContext* context) const override;
};

class SequenceConstruct final : public OpKernel {
 public:
  SequenceConstruct(const OpKernelInfo& info) : OpKernel(info) {
  }
  Status Compute(OpKernelContext* context) const override;
};
}  //namespace onnxruntime
