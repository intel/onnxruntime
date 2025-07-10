// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include <vector>
#include <string_view>

#include "ov_supported_ops.h"
#include "openvino/core/version.hpp"

namespace onnxruntime {
namespace openvino_ep {

OpenVINOVersion OpenVINOVersion::GetBuiltVersion() {
  return OpenVINOVersion(OPENVINO_VERSION_MAJOR, OPENVINO_VERSION_MINOR);
}

struct SupportedOp {
  SupportedOp() = default;
  SupportedOp(const OpenVINOVersion& min_version, const std::string_view op_type, const std::string_view domain)
      : min_version(min_version), op_type(op_type), domain(domain) {
  }

  OpenVINOVersion min_version;
  std::string_view op_type;
  std::string_view domain;
};

constexpr const char* kDefaultDomain = "";
constexpr const char* kOpenvinoDomain = "org.openvinotoolkit";
constexpr const char* kMicrosoftDomain = "com.microsoft";

// Static list of all supported operations across all OpenVINO versions
static const std::vector<SupportedOp> kAllSupportedOps = {
    // Standard ONNX operations
    {OpenVINOVersion(2024, 4), "Abs", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Acos", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Acosh", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "AdaptiveAvgPool2d", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Add", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Affine", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "And", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "ArgMax", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "ArgMin", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Asin", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Asinh", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Atan", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Atanh", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "ATen", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "AveragePool", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "BatchNormalization", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "BitShift", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "BitwiseAnd", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "BitwiseNot", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "BitwiseOr", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "BitwiseXor", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "BlackmanWindow", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Cast", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "CastLike", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Ceil", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Celu", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Clip", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Compress", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Concat", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Constant", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "ConstantFill", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "ConstantOfShape", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Conv", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "ConvInteger", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "ConvTranspose", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Cos", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Cosh", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Crop", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "CumSum", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "DepthToSpace", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "DequantizeLinear", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "DFT", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Div", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Dropout", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "DynamicQuantizeLinear", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Einsum", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Elu", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Equal", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Erf", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Exp", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Expand", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "EyeLike", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Flatten", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Floor", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Gather", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "GatherElements", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "GatherND", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Gelu", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Gemm", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "GlobalAveragePool", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "GlobalLpPool", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "GlobalMaxPool", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Greater", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "GreaterOrEqual", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "GridSample", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "GroupNormalization", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "GRU", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "HammingWindow", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "HardSigmoid", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "HardSwish", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Hardmax", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Identity", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "If", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "ImageScaler", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "InstanceNormalization", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "IsFinite", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "IsInf", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "IsNaN", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "LayerNormalization", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "LeakyRelu", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Less", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "LessOrEqual", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Log", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "LogSoftmax", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Loop", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "LpNormalization", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "LRN", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "LSTM", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "MatMul", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "MatMulInteger", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Max", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "MaxPool", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "MaxRoiPool", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Mean", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "MeanVarianceNormalization", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Min", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Mish", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "MMCVRoIAlignRotated", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Mod", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Mul", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Multinomial", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Neg", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "NMSRotated", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "NonMaxSuppression", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "NonZero", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Not", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "OneHot", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Or", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Pad", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Pow", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "PRelu", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "QLinearConv", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "QLinearMatMul", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "QuantizeLinear", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "RandomNormal", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "RandomNormalLike", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "RandomUniform", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "RandomUniformLike", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Range", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Reciprocal", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "ReduceLogSum", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "ReduceLogSumExp", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "ReduceL1", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "ReduceL2", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "ReduceMax", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "ReduceMean", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "ReduceMin", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "ReduceProd", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "ReduceSum", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "ReduceSumSquare", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Relu", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Reshape", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Resize", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "ReverseSequence", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "RNN", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "RoiAlign", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Round", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Scan", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "ScatterElements", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "ScatterND", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Selu", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Shape", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Shrink", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Sigmoid", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Sign", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Sin", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Sinh", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Size", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Slice", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Softmax", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Softplus", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Softsign", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "SpaceToDepth", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Split", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Sqrt", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Squeeze", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "STFT", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Sub", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Sum", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Tan", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Tanh", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "ThresholdedRelu", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Tile", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "TopK", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Transpose", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Trilu", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Unique", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Unsqueeze", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Upsample", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Where", kDefaultDomain},
    {OpenVINOVersion(2024, 4), "Xor", kDefaultDomain},

    // Deprecated ONNX operations (already included above: Affine, Crop, Upsample)
    {OpenVINOVersion(2024, 4), "Scatter", kDefaultDomain},

    // Custom operations - org.openvinotoolkit domain
    {OpenVINOVersion(2024, 4), "DeformableConv2D", kOpenvinoDomain},
    {OpenVINOVersion(2024, 4), "DetectionOutput", kOpenvinoDomain},
    {OpenVINOVersion(2024, 4), "ExperimentalDetectronDetectionOutput", kOpenvinoDomain},
    {OpenVINOVersion(2024, 4), "ExperimentalDetectronGenerateProposalsSingleImage", kOpenvinoDomain},
    {OpenVINOVersion(2024, 4), "ExperimentalDetectronGroupNorm", kOpenvinoDomain},
    {OpenVINOVersion(2024, 4), "ExperimentalDetectronPriorGridGenerator", kOpenvinoDomain},
    {OpenVINOVersion(2024, 4), "ExperimentalDetectronROIFeatureExtractor", kOpenvinoDomain},
    {OpenVINOVersion(2024, 4), "ExperimentalDetectronTopKROIs", kOpenvinoDomain},
    {OpenVINOVersion(2024, 4), "FakeQuantize", kOpenvinoDomain},
    {OpenVINOVersion(2024, 4), "GroupNorm", kOpenvinoDomain},
    {OpenVINOVersion(2024, 4), "Normalize", kOpenvinoDomain},
    {OpenVINOVersion(2024, 4), "PriorBox", kOpenvinoDomain},
    {OpenVINOVersion(2024, 4), "PriorBoxClustered", kOpenvinoDomain},
    {OpenVINOVersion(2024, 4), "Swish", kOpenvinoDomain},

    // Custom operations - com.microsoft domain
    {OpenVINOVersion(2024, 4), "Attention", kMicrosoftDomain},
    {OpenVINOVersion(2024, 4), "Bias_Add", kMicrosoftDomain},
    {OpenVINOVersion(2024, 4), "BiasGelu", kMicrosoftDomain},
    {OpenVINOVersion(2024, 4), "Dynamic_Quantize_MatMul", kMicrosoftDomain},
    {OpenVINOVersion(2024, 4), "EmbedLayerNormalization", kMicrosoftDomain},
    {OpenVINOVersion(2024, 4), "Fast_Gelu", kMicrosoftDomain},
    {OpenVINOVersion(2024, 4), "Fused_Conv", kMicrosoftDomain},
    {OpenVINOVersion(2024, 4), "FusedGemm", kMicrosoftDomain},
    {OpenVINOVersion(2024, 4), "FusedMatMul", kMicrosoftDomain},
    {OpenVINOVersion(2024, 4), "MatMulIntegerToFloat", kMicrosoftDomain},
    {OpenVINOVersion(2024, 4), "MatMulNBits", kMicrosoftDomain},
    {OpenVINOVersion(2024, 4), "Pad", kMicrosoftDomain},
    {OpenVINOVersion(2024, 4), "QLinearActivation", kMicrosoftDomain},
    {OpenVINOVersion(2024, 4), "QLinearAdd", kMicrosoftDomain},
    {OpenVINOVersion(2024, 4), "QLinearMul", kMicrosoftDomain},
    {OpenVINOVersion(2024, 4), "QuickGelu", kMicrosoftDomain},
    {OpenVINOVersion(2024, 4), "Range", kMicrosoftDomain},
    {OpenVINOVersion(2024, 4), "SimplifiedLayerNormalization", kMicrosoftDomain},
    {OpenVINOVersion(2024, 4), "SkipLayerNormalization", kMicrosoftDomain},
    {OpenVINOVersion(2024, 4), "SkipSimplifiedLayerNormalization", kMicrosoftDomain},

    // OVEP supported operators
    {OpenVINOVersion(0, 0), "EPContext", kMicrosoftDomain},

};  // namespace openvino_ep

SupportedOps::SupportedOps(const OpenVINOVersion& ov_version)
    : version_(ov_version) {
  // Filter operations based on the provided OpenVINO version
  for (const auto& op : kAllSupportedOps) {
    if (ov_version >= op.min_version) {
      ops_.insert({op.op_type, op.domain});
    }
  }
}

}  // namespace openvino_ep
}  // namespace onnxruntime
