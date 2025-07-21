// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include <unordered_set>
#include <string>
#include <vector>
#include <utility>
#include <map>
#include <set>

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/openvino/backend_utils.h"
#include "core/providers/openvino/backend_manager.h"
#include "core/providers/openvino/ov_interface.h"
#include "core/providers/openvino/ov_versions/data_ops.h"
#include "core/providers/openvino/ov_versions/capability.h"
#include "core/providers/openvino/ov_versions/utils.h"

#if defined(_MSC_VER)
#pragma warning(disable : 4244 4245 5208)
#elif __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
// #include <ngraph/ngraph.hpp>
// #include <ngraph/frontend/onnx_import/onnx.hpp>
#if defined(_MSC_VER)
#pragma warning(default : 4244 4245)
#elif __GNUC__
#pragma GCC diagnostic pop
#endif

namespace onnxruntime {
namespace openvino_ep {

// Ops which are supported only in models(as intermediate nodes) and not in unit tests
std::set<std::string> ops_supported_only_in_model = {
    "Add",
    "Cast",
    "Celu",
    "Concat",
    "ConstantOfShape",
    "DequantizeLinear",
    "Dropout",
    "Einsum",
    "Exp",
    "Expand",
    "EyeLike",
    "GatherElements",
    "GatherND",
    "GridSample",
    "Identity",
    "LayerNormalization",
    "Loop",
    "LSTM",
    "NonMaxSuppression",
    "NonZero",
    "Not",
    "OneHot",
    "Pad",
    "QuantizeLinear",
    "RandomNormalLike",
    "Range",
    "ReduceMin",
    "Resize",
    "Round",
    "Shape",
    "Slice",
    "Split",
    "Tile",
    "TopK",
    "Trilu"};

// Ops which are supported as functions (as composite ops)
std::set<std::string> ops_supported_as_function = {
    "LessOrEqual",
    "GreaterOrEqual",
    "LayerNormalization",
    "Celu"};

std::vector<SupportedOp> supported_op_mode = {
{"Abs", V_2024_4, {"All"}},
{"Acos", V_2024_4, {"All"}},
{"Acosh", V_2024_4, {"All"}},
{"AdaptiveAvgPool2d", V_2024_4, {"All"}},
{"Add", V_2024_4, {"All"}},
{"Affine", V_2024_4, {"All"}},
{"And", V_2024_4, {"All"}},
{"ArgMax", V_2024_4, {"All"}},
{"ArgMin", V_2024_4, {"All"}},
{"Asin", V_2024_4, {"All"}},
{"Asinh", V_2024_4, {"All"}},
{"Atan", V_2024_4, {"All"}},
{"Atanh", V_2024_4, {"All"}},
{"ATen", V_2024_4, {"All"}},
{"AveragePool", V_2024_4, {"All"}},
{"BatchNormalization", V_2024_4, {"All"}},
{"BitShift", V_2024_4, {"All"}},
{"BitwiseAnd", V_2024_4, {"All"}},
{"BitwiseNot", V_2024_4, {"All"}},
{"BitwiseOr", V_2024_4, {"All"}},
{"BitwiseXor", V_2024_4, {"All"}},
{"BlackmanWindow", V_2024_4, {"All"}},
{"Cast", V_2024_4, {"All"}},
{"CastLike", V_2024_4, {"All"}},
{"Ceil", V_2024_4, {"All"}},
{"Celu", V_2024_4, {"All"}},
{"Clip", V_2024_4, {"All"}},
{"Compress", V_2024_4, {"All"}},
{"Concat", V_2024_4, {"All"}},
{"Constant", V_2024_4, {"All"}},
{"ConstantFill", V_2024_4, {"All"}},
{"ConstantOfShape", V_2024_4, {"All"}},
{"Conv", V_2024_4, {"All"}},
{"ConvInteger", V_2024_4, {"All"}},
{"ConvTranspose", V_2024_4, {"All"}},
{"Cos", V_2024_4, {"All"}},
{"Cosh", V_2024_4, {"All"}},
{"Crop", V_2024_4, {"All"}},
{"CumSum", V_2024_4, {"All"}},
{"DepthToSpace", V_2024_4, {"All"}},
{"DequantizeLinear", V_2024_4, {"All"}},
{"DFT", V_2024_4, {"All"}},
{"Div", V_2024_4, {"All"}},
{"Dropout", V_2024_4, {"All"}},
{"DynamicQuantizeLinear", V_2024_4, {"All"}},
{"Einsum", V_2024_4, {"All"}},
{"Elu", V_2024_4, {"All"}},
{"Equal", V_2024_4, {"All"}},
{"Erf", V_2024_4, {"All"}},
{"Exp", V_2024_4, {"All"}},
{"Expand", V_2024_4, {"All"}},
{"EyeLike", V_2024_4, {"All"}},
{"Flatten", V_2024_4, {"All"}},
{"Floor", V_2024_4, {"All"}},
{"Gather", V_2024_4, {"All"}},
{"GatherElements", V_2024_4, {"All"}},
{"GatherND", V_2024_4, {"All"}},
{"Gelu", V_2024_4, {"All"}},
{"Gemm", V_2024_4, {"All"}},
{"GlobalAveragePool", V_2024_4, {"All"}},
{"GlobalLpPool", V_2024_4, {"All"}},
{"GlobalMaxPool", V_2024_4, {"All"}},
{"Greater", V_2024_4, {"All"}},
{"GreaterOrEqual", V_2024_4, {"All"}},
{"GridSample", V_2024_4, {"All"}},
{"GroupNormalization", V_2024_4, {"All"}},
{"GRU", V_2024_4, {"All"}},
{"HammingWindow", V_2024_4, {"All"}},
{"HardSigmoid", V_2024_4, {"All"}},
{"HardSwish", V_2024_4, {"All"}},
{"Hardmax", V_2024_4, {"All"}},
{"Identity", V_2024_4, {"All"}},
{"If", V_2024_4, {"All"}},
{"ImageScaler", V_2024_4, {"All"}},
{"InstanceNormalization", V_2024_4, {"All"}},
{"IsFinite", V_2024_4, {"All"}},
{"IsInf", V_2024_4, {"All"}},
{"IsNaN", V_2024_4, {"All"}},
{"LayerNormalization", V_2024_4, {"All"}},
{"LeakyRelu", V_2024_4, {"All"}},
{"Less", V_2024_4, {"All"}},
{"LessOrEqual", V_2024_4, {"All"}},
{"Log", V_2024_4, {"All"}},
{"LogSoftmax", V_2024_4, {"All"}},
{"Loop", V_2024_4, {"All"}},
{"LpNormalization", V_2024_4, {"All"}},
{"LRN", V_2024_4, {"All"}},
{"LSTM", V_2024_4, {"All"}},
{"MatMul", V_2024_4, {"All"}},
{"MatMulInteger", V_2024_4, {"All"}},
{"Max", V_2024_4, {"All"}},
{"MaxPool", V_2024_4, {"All"}},
{"MaxRoiPool", V_2024_4, {"All"}},
{"Mean", V_2024_4, {"All"}},
{"MeanVarianceNormalization", V_2024_4, {"All"}},
{"Min", V_2024_4, {"All"}},
{"Mish", V_2024_4, {"All"}},
{"MMCVRoIAlignRotated", V_2024_4, {"All"}},
{"Mod", V_2024_4, {"All"}},
{"Mul", V_2024_4, {"All"}},
{"Multinomial", V_2024_4, {"All"}},
{"Neg", V_2024_4, {"All"}},
{"NMSRotated", V_2024_4, {"All"}},
{"NonMaxSuppression", V_2024_4, {"All"}},
{"NonZero", V_2024_4, {"All"}},
{"Not", V_2024_4, {"All"}},
{"OneHot", V_2024_4, {"All"}},
{"Or", V_2024_4, {"All"}},
{"Pad", V_2024_4, {"All"}},
{"Pow", V_2024_4, {"All"}},
{"PRelu", V_2024_4, {"All"}},
{"QLinearConv", V_2024_4, {"All"}},
{"QLinearMatMul", V_2024_4, {"All"}},
{"QuantizeLinear", V_2024_4, {"All"}},
{"RandomNormal", V_2024_4, {"All"}},
{"RandomNormalLike", V_2024_4, {"All"}},
{"RandomUniform", V_2024_4, {"All"}},
{"RandomUniformLike", V_2024_4, {"All"}},
{"Range", V_2024_4, {"All"}},
{"Reciprocal", V_2024_4, {"All"}},
{"ReduceLogSum", V_2024_4, {"All"}},
{"ReduceLogSumExp", V_2024_4, {"All"}},
{"ReduceL1", V_2024_4, {"All"}},
{"ReduceL2", V_2024_4, {"All"}},
{"ReduceMax", V_2024_4, {"All"}},
{"ReduceMean", V_2024_4, {"All"}},
{"ReduceMin", V_2024_4, {"All"}},
{"ReduceProd", V_2024_4, {"All"}},
{"ReduceSum", V_2024_4, {"All"}},
{"ReduceSumSquare", V_2024_4, {"All"}},
{"Relu", V_2024_4, {"All"}},
{"Reshape", V_2024_4, {"All"}},
{"Resize", V_2024_4, {"All"}},
{"ReverseSequence", V_2024_4, {"All"}},
{"RNN", V_2024_4, {"All"}},
{"RoiAlign", V_2024_4, {"All"}},
{"Round", V_2024_4, {"All"}},
{"Scan", V_2024_4, {"All"}},
{"ScatterElements", V_2024_4, {"All"}},
{"ScatterND", V_2024_4, {"All"}},
{"Selu", V_2024_4, {"All"}},
{"Shape", V_2024_4, {"All"}},
{"Shrink", V_2024_4, {"All"}},
{"Sigmoid", V_2024_4, {"All"}},
{"Sign", V_2024_4, {"All"}},
{"Sin", V_2024_4, {"All"}},
{"Sinh", V_2024_4, {"All"}},
{"Size", V_2024_4, {"All"}},
{"Slice", V_2024_4, {"All"}},
{"Softmax", V_2024_4, {"All"}},
{"Softplus", V_2024_4, {"All"}},
{"Softsign", V_2024_4, {"All"}},
{"SpaceToDepth", V_2024_4, {"All"}},
{"Split", V_2024_4, {"All"}},
{"Sqrt", V_2024_4, {"All"}},
{"Squeeze", V_2024_4, {"All"}},
{"STFT", V_2024_4, {"All"}},
{"Sub", V_2024_4, {"All"}},
{"Sum", V_2024_4, {"All"}},
{"Tan", V_2024_4, {"All"}},
{"Tanh", V_2024_4, {"All"}},
{"ThresholdedRelu", V_2024_4, {"All"}},
{"Tile", V_2024_4, {"All"}},
{"TopK", V_2024_4, {"All"}},
{"Transpose", V_2024_4, {"All"}},
{"Trilu", V_2024_4, {"All"}},
{"Unique", V_2024_4, {"All"}},
{"Unsqueeze", V_2024_4, {"All"}},
{"Upsample", V_2024_4, {"All"}},
{"Where", V_2024_4, {"All"}},
{"Xor", V_2024_4, {"All"}},
{"Scatter", V_2024_4, {"All"}},
{"DeformableConv2D", V_2024_4, {"All"}},
{"DetectionOutput", V_2024_4, {"All"}},
{"ExperimentalDetectronDetectionOutput", V_2024_4, {"All"}},
{"ExperimentalDetectronGenerateProposalsSingleImage", V_2024_4, {"All"}},
{"ExperimentalDetectronGroupNorm", V_2024_4, {"All"}},
{"ExperimentalDetectronPriorGridGenerator", V_2024_4, {"All"}},
{"ExperimentalDetectronROIFeatureExtractor", V_2024_4, {"All"}},
{"ExperimentalDetectronTopKROIs", V_2024_4, {"All"}},
{"FakeQuantize", V_2024_4, {"All"}},
{"GroupNorm", V_2024_4, {"All"}},
{"Normalize", V_2024_4, {"All"}},
{"PriorBox", V_2024_4, {"All"}},
{"PriorBoxClustered", V_2024_4, {"All"}},
{"Swish", V_2024_4, {"All"}},
{"Attention", V_2024_4, {"All"}},
{"Bias_Add", V_2024_4, {"All"}},
{"BiasGelu", V_2024_4, {"All"}},
{"Dynamic_Quantize_MatMul", V_2024_4, {"All"}},
{"EmbedLayerNormalization", V_2024_4, {"All"}},
{"Fast_Gelu", V_2024_4, {"All"}},
{"Fused_Conv", V_2024_4, {"All"}},
{"FusedGemm", V_2024_4, {"All"}},
{"FusedMatMul", V_2024_4, {"All"}},
{"MatMulIntegerToFloat", V_2024_4, {"All"}},
{"MatMulNBits", V_2024_4, {"All"}},
{"QLinearActivation", V_2024_4, {"All"}},
{"QLinearAdd", V_2024_4, {"All"}},
{"QLinearMul", V_2024_4, {"All"}},
{"QuickGelu", V_2024_4, {"All"}},
{"SimplifiedLayerNormalization", V_2024_4, {"All"}},
{"SkipLayerNormalization", V_2024_4, {"All"}},
{"SkipSimplifiedLayerNormalization", V_2024_4, {"All"}}
};

void DataOps::populate_types_supported() {
  supported_types_initializer_.insert(
      std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_BOOL));
  supported_types_initializer_.insert(
      std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT));
  supported_types_initializer_.insert(
      std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32));
  supported_types_initializer_.insert(
      std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT64));
  supported_types_initializer_.insert(
      std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT16));
  supported_types_initializer_.insert(
      std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT16));
  supported_types_initializer_.insert(
      std::make_pair(V_2021_1, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16));
  supported_types_initializer_.insert(
      std::make_pair(V_2021_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8));
  supported_types_initializer_.insert(
      std::make_pair(V_2021_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8));
  supported_types_initializer_.insert(
      std::make_pair(V_2024_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT4));
  supported_types_initializer_.insert(
      std::make_pair(V_2024_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT4));

  supported_types_npu_.insert(
      std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_BOOL));
  supported_types_npu_.insert(
      std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT));
  supported_types_npu_.insert(
      std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8));
  supported_types_npu_.insert(
      std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8));
  supported_types_npu_.insert(
      std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT16));
  supported_types_npu_.insert(
      std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT16));
  supported_types_npu_.insert(
      std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32));
  supported_types_npu_.insert(
      std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT64));
  supported_types_npu_.insert(
      std::make_pair(V_2021_1, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16));
  supported_types_npu_.insert(
      std::make_pair(V_2024_3, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT8E4M3FN));
  supported_types_npu_.insert(
      std::make_pair(V_2024_3, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT8E4M3FNUZ));
  supported_types_npu_.insert(
      std::make_pair(V_2024_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT4));
  supported_types_npu_.insert(
      std::make_pair(V_2024_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT4));

  supported_types_cpu_.insert(
      std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_BOOL));
  supported_types_cpu_.insert(
      std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT));
  supported_types_cpu_.insert(
      std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32));
  supported_types_cpu_.insert(
      std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT16));
  supported_types_cpu_.insert(
      std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT16));
  supported_types_cpu_.insert(
      std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8));
  supported_types_cpu_.insert(
      std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8));
  supported_types_cpu_.insert(
      std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT64));
  supported_types_cpu_.insert(
      std::make_pair(V_2022_2, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16));
  supported_types_cpu_.insert(
      std::make_pair(V_2024_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT4));
  supported_types_cpu_.insert(
      std::make_pair(V_2024_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT4));

  supported_types_gpu_.insert(
      std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT));
  supported_types_gpu_.insert(
      std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32));
  supported_types_gpu_.insert(
      std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT64));
  supported_types_gpu_.insert(
      std::make_pair(V_2021_1, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16));
  supported_types_gpu_.insert(
      std::make_pair(V_2021_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8));
  supported_types_gpu_.insert(
      std::make_pair(V_2021_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8));
  supported_types_gpu_.insert(
      std::make_pair(V_2022_1, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_BOOL));
  supported_types_gpu_.insert(
      std::make_pair(V_2024_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT4));
  supported_types_gpu_.insert(
      std::make_pair(V_2024_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT4));
}

void DataOps::populate_op_mode_supported() {
  no_dimension_supported_.push_back({"Add", V_2022_1, {"All"}});
  no_dimension_supported_.push_back({"And", V_2022_1, {"All"}});
  no_dimension_supported_.push_back({"Cast", V_2020_4, {"All"}});
  no_dimension_supported_.push_back({"Ceil", V_2021_4, {"All"}});
  no_dimension_supported_.push_back({"Clip", V_2022_1, {"All"}});
  no_dimension_supported_.push_back({"Div", V_2020_4, {"All"}});
  no_dimension_supported_.push_back({"DequantizeLinear", V_2021_4, {"All"}});
  no_dimension_supported_.push_back({"Equal", V_2022_1, {"CPU"}});
  no_dimension_supported_.push_back({"Equal", V_2023_0, {"GPU"}});
  no_dimension_supported_.push_back({"Expand", V_2023_3, {"CPU"}});
  no_dimension_supported_.push_back({"Expand", V_2024_3, {"CPU", "GPU"}});
  no_dimension_supported_.push_back({"Floor", V_2020_4, {"All"}});
  no_dimension_supported_.push_back({"Gather", V_2020_4, {"All"}});
  no_dimension_supported_.push_back({"Greater", V_2024_4, {"All"}});
  no_dimension_supported_.push_back({"Identity", V_2023_0, {"All"}});
  no_dimension_supported_.push_back({"If", V_2022_3, {"CPU", "GPU"}});
  no_dimension_supported_.push_back({"Less", V_2022_1, {"CPU"}});
  no_dimension_supported_.push_back({"Loop", V_2021_4, {"All"}});
  no_dimension_supported_.push_back({"Max", V_2024_4, {"All"}});
  no_dimension_supported_.push_back({"Min", V_2020_4, {"All"}});
  no_dimension_supported_.push_back({"Mul", V_2020_4, {"All"}});
  no_dimension_supported_.push_back({"Neg", V_2023_0, {"CPU", "GPU"}});
  no_dimension_supported_.push_back({"Pow", V_2023_0, {"CPU", "GPU"}});
  no_dimension_supported_.push_back({"QuantizeLinear", V_2021_4, {"All"}});
  no_dimension_supported_.push_back({"Range", V_2021_2, {"All"}});
  no_dimension_supported_.push_back({"ReduceMax", V_2021_4, {"All"}});
  no_dimension_supported_.push_back({"ReduceMin", V_2021_4, {"All"}});
  no_dimension_supported_.push_back({"ReduceProd", V_2022_1, {"CPU", "GPU"}});
  no_dimension_supported_.push_back({"Reshape", V_2022_1, {"All"}});
  no_dimension_supported_.push_back({"Shape", V_2022_1, {"GPU"}});
  no_dimension_supported_.push_back({"Shape", V_2023_0, {"CPU"}});
  no_dimension_supported_.push_back({"Sqrt", V_2023_0, {"All"}});
  no_dimension_supported_.push_back({"Squeeze", V_2020_4, {"All"}});
  no_dimension_supported_.push_back({"Sub", V_2020_4, {"All"}});
  no_dimension_supported_.push_back({"Unsqueeze", V_2020_4, {"All"}});
  no_dimension_supported_.push_back({"Where", V_2021_2, {"All"}});

  subgraph_supported_.push_back({"Cast", V_2020_4, {"All"}});
  subgraph_supported_.push_back({"Concat", V_2020_4, {"All"}});
  subgraph_supported_.push_back({"Div", V_2021_1, {"CPU"}});
  subgraph_supported_.push_back({"Gather", V_2020_4, {"All"}});
  subgraph_supported_.push_back({"Identity", V_2021_1, {"CPU"}});
  subgraph_supported_.push_back({"Mul", V_2020_4, {"All"}});
  subgraph_supported_.push_back({"Sub", V_2021_1, {"CPU"}});
  subgraph_supported_.push_back({"Transpose", V_2020_4, {"All"}});
  subgraph_supported_.push_back({"Unsqueeze", V_2020_4, {"All"}});

  // populate unsupportedmode_t
  {
    UnsupportedOpMode obj = {{V_2024_1, V_2024_2, V_2024_3, V_2024_4, V_2024_5, V_2024_6, V_2025_0, V_2025_1, V_2025_2},
                             [this](const Node* node, const InitializedTensorSet&) {
                               // If the Input of ReduceMax op is UINT8, it is rejected (Due to output mismatch)
                               for (size_t i = 0; i < node->InputDefs().size(); i++) {
                                 if ((node->InputDefs()[i]->TypeAsProto()->tensor_type().elem_type() ==
                                      ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8) ||
                                     (node->InputDefs()[i]->TypeAsProto()->tensor_type().elem_type() ==
                                      ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8))
                                   return true;
                               }
                               return false;
                             }};
    op_list_.insert({"ReduceMax", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2023_1, V_2023_2, V_2023_3, V_2024_0, V_2024_1, V_2024_2,
                              V_2024_3, V_2024_4, V_2024_5, V_2024_6, V_2025_0, V_2025_1,
                              V_2025_2},
                             [this](const Node* node, const InitializedTensorSet&) {
                               const auto& input_args = node->InputDefs();
                               const auto& input_arg = (input_args.size() > 1) ? input_args[1] : input_args[0];
                               auto shape = input_arg->Shape();
                               // Reshape op with empty dim is Rejected for Myriad
                               // [TODO] Is this condition required anymore with Myriad removed?
                               if (shape != nullptr) {
                                 for (const auto& dim : input_arg->Shape()->dim()) {
                                   if (utils::HasDimValue(dim) && dim.dim_value() == 0)
                                     return true;
                                 }
                               }
                               return false;
                             }};
    op_list_.insert({"Reshape", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2023_1, V_2023_2, V_2023_3, V_2024_0, V_2024_1, V_2024_2,
                              V_2024_3, V_2024_4, V_2024_5, V_2024_6, V_2025_0, V_2025_1,
                              V_2025_2},
                             [this](const Node* node, const InitializedTensorSet&) {
                               // If the operator is unsqueeze
                               // If axes is an input, then we cannot produce a static graph.
                               // Conversion fails in convert_function_to_cnn_network.
                               for (size_t i = 0; i < node->InputDefs().size(); i++) {
                                 if (node->InputDefs()[i]->Name() == "axes") {
                                   return true;
                                 }
                               }
                               return (!this->dimension_unsupported(node));
                             }};
    op_list_.insert({"Unsqueeze", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2023_1, V_2023_2, V_2023_3, V_2024_0, V_2024_1, V_2024_2, V_2024_3, V_2024_4, V_2024_5,
                              V_2024_6, V_2025_0, V_2025_1, V_2025_2},
                             [this](const Node* node, const InitializedTensorSet&) {
                               // check for attributes
                               auto& upsample_attr = node->GetAttributes();
                               if (upsample_attr.count("scales") > 0) {
                                 auto& upsample_arg = upsample_attr.at("scales");
                                 auto float_size = upsample_arg.floats_size();
                                 if (float_size > 2 &&
                                     (upsample_arg.floats(0) != 1.f || upsample_arg.floats(1) != 1.f)) {
                                   return true;
                                 }
                               }

                               // check for input dimensions
                               const auto& x_arg = node->InputDefs()[0];
                               auto shape = x_arg->Shape();
                               if (shape != nullptr) {
                                 // input tensor rank cannot be of one dimension
                                 if (shape->dim_size() == 1 || shape->dim_size() == 4) {
                                   return true;
                                 }
                               }
                               // x_arg supports only float, int8 and float16 type
                               if ((x_arg->TypeAsProto()->tensor_type().elem_type() ==
                                    ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT) ||
                                   (x_arg->TypeAsProto()->tensor_type().elem_type() ==
                                    ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8) ||
                                   (x_arg->TypeAsProto()->tensor_type().elem_type() ==
                                    ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16)) {
                                 return false;
                               } else {
                                 return true;
                               }
                             }};
    op_list_.insert({"Upsample", obj});
  }
}

bool DataOps::op_is_supported(std::string name, std::vector<SupportedOp>& op_list) {
  bool auto_support = false;
  bool multi_support = false;
  for (size_t i = 0; i < op_list.size(); i++) {
    if (op_list[i].optype == name) {
      if (op_list[i].version <= version_id_) {
        auto it = op_list[i].device_type.begin();
        while (it != op_list[i].device_type.end()) {
          // status variable is set to True if it's Hetero/Multi/Auto device type
          bool status = false;

          // The operator to be marked true, it should be supported by either of the devices specified with HETERO
          if (device_id_.find("HETERO") == 0) {
            status = true;
            if (device_id_.find(*it) != std::string::npos || (*it == "All")) {
              return true;
            }
          }

          // The operator to be marked true, it should be supported by all the devices specified with MULTI/AUTO
          if (device_id_.find("MULTI") == 0) {
            status = true;
            if ((*it == "All") || device_id_.find(*it) != std::string::npos) {
              multi_support = true;
            }
          }
          // The operator to be marked true, it should be supported by atleast CPU device specified with AUTO
          if (device_id_.find("AUTO") == 0) {
            if (std::string(*it).find("CPU") == std::string::npos) {
              auto_support = false;
            } else if ((*it == "All") || (device_id_.find(*it) != std::string::npos)) {
              auto_support = true;
            }
          }
          // if device supported is all then we support it
          if (*it == "All") {
            return true;
          }
          // check for device supported
          if (status == false) {
            if (device_id_.find(*it) != std::string::npos) {
              return true;
            }
          }
          it++;
        }
      }
    }
  }
  if (device_id_.find("AUTO") == 0 && auto_support == true) {
    return true;
  }
  if (device_id_.find("MULTI") == 0 && multi_support == true) {
    return true;
  }
  return false;
}

bool DataOps::type_is_supported(const NodeArg* node_arg, bool is_initializer) {
  const auto* type_proto = node_arg->TypeAsProto();
  if (!type_proto) {
#ifndef NDEBUG
    if (openvino_ep::backend_utils::IsDebugEnabled()) {
      std::cout << "Node is not a proto " << std::endl;
    }
#endif
    return false;
  }

  if (is_initializer) {
    auto dtype = type_proto->tensor_type().elem_type();
    for (auto const& var : supported_types_initializer_) {
      if ((var.first <= version_id_) &&
          (var.second == dtype)) {
        return true;
      }
    }

#ifndef NDEBUG
    if (openvino_ep::backend_utils::IsDebugEnabled()) {
      std::cout << "Initializer Data Type is not supported" << std::endl;
    }
#endif
    return false;
  } else {
    auto dtype = type_proto->tensor_type().elem_type();

    if (device_id_.find("HETERO") != std::string::npos ||
        device_id_.find("MULTI") != std::string::npos || device_id_.find("AUTO") != std::string::npos) {
      for (auto const& var : supported_types_npu_) {
        if ((var.first <= version_id_) &&
            (var.second == dtype)) {
          return true;
        }
      }

#ifndef NDEBUG
      if (openvino_ep::backend_utils::IsDebugEnabled()) {
        std::cout << "I/O data type is not supported" << std::endl;
      }
#endif
      return false;

    } else if (device_id_ == "CPU") {
      for (auto const& var : supported_types_cpu_) {
        if ((var.first <= version_id_) &&
            (var.second == dtype)) {
          return true;
        }
      }
#ifndef NDEBUG
      if (openvino_ep::backend_utils::IsDebugEnabled()) {
        std::cout << "I/O data type is not supported" << std::endl;
      }
#endif
      return false;

    } else if (device_id_ == "GPU") {
      for (auto const& var : supported_types_gpu_) {
        if ((var.first <= version_id_) &&
            (var.second == dtype)) {
          return true;
        }
        // experimentally for GPU and qdq stripping mode allow int16 types
        if (npu_qdq_optimizer_enabled_ && (dtype == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT16 || dtype == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT16))
          return true;
      }
#ifndef NDEBUG
      if (openvino_ep::backend_utils::IsDebugEnabled()) {
        std::cout << "I/O data type is not supported" << std::endl;
      }
#endif
      return false;
    }
    return true;
  }
}

bool DataOps::unsupported_op_mode(const Node* node, bool& has_external_weights_) {
  bool result = false;
  const auto& optype = node->OpType();
  const auto& initializers = graph_viewer_.GetAllInitializedTensors();

  for (const auto& tensor_pair : initializers) {
    const ONNX_NAMESPACE::TensorProto* tensor_proto = tensor_pair.second;
    // Check if the tensor exists and if it has an external data location
    if (tensor_proto && tensor_proto->has_data_location() &&
        tensor_proto->data_location() == ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL) {
      has_external_weights_ = true;
      break;
    }
  }

  auto iter = op_list_.equal_range(optype);
  for (auto it = iter.first; it != iter.second; ++it) {
    auto ob = it->second;
    if (std::find(ob.ver.begin(), ob.ver.end(), version_id_) != ob.ver.end()) {
      return ob.func(node, initializers);
    }
  }
  return result;
}

bool DataOps::dimension_unsupported(const Node* node) {
  auto node_inputs = node->InputDefs();
  size_t input_dims = 0;
  if (node_inputs[0]->Shape() == nullptr) {
    return true;
  } else {
    input_dims = node_inputs[0]->Shape()->dim_size();
    if (node->OpType().find("Pool") != std::string::npos) {
      if (input_dims != 4 && input_dims != 5)
        return false;
    }
    /*
    if (node->OpType() == "Unsqueeze") {
      auto& attributes = node->GetAttributes();
      int64_t axes_size = attributes.count("axes") > 0 ? attributes.at("axes").ints().size() : 0;
      if (device_id_.find("GPU") != std::string::npos) {
        if (axes_size == 0)
          return true;
      }
      if (input_dims + axes_size > 5 || axes_size == 0) {
        return false;
      }
    }
    */

    if (node->OpType() == "ReduceSum") {
      auto& attributes = node->GetAttributes();
      int64_t axes_size = attributes.count("axes") > 0 ? attributes.at("axes").ints().size() : 0;
      if (device_id_.find("GPU") != std::string::npos) {
        if (axes_size == 0)
          return true;
      }
      if (axes_size == 0)
        return false;
    }
  }
  return true;
}

bool DataOps::node_is_supported(const NodeIndex node_idx, bool& has_external_weights_) {
  const auto& node = graph_viewer_.GetNode(node_idx);
  const auto& optype = node->OpType();

#ifndef NDEBUG
  if (openvino_ep::backend_utils::IsDebugEnabled()) {
    std::cout << "Node " << optype << std::endl;
  }
#endif

  const auto& domain = node->Domain();

  /*
  0. Check if node is in the unsupported list
  1. Check input and output data types are supported.
  2. Check if there is unsupported dimension in input and output shapes
  3. Check Op is supported
   3a. Check if Op is of known unsupported modes (edge cases). If yes return false right away.
   3b. If above is not true, check if the op is available in nGraph.
  */

  // Check 0
  if (!op_is_supported(optype, supported_op_mode)) {
#ifndef NDEBUG
    if (openvino_ep::backend_utils::IsDebugEnabled()) {
      std::cout << "Node is not in the supported ops list" << std::endl;
    }
#endif
    return false;
  }

  // Check 1
//   bool are_types_supported = true;

//   node->ForEachDef([this, &are_types_supported](const NodeArg& node_arg, bool is_input) {
//     bool is_initializer = false;
//     if (is_input) {
//       if (this->graph_viewer_.IsConstantInitializer(node_arg.Name(), true))
//         is_initializer = true;
//     }
//     bool is_supported = type_is_supported(&node_arg, is_initializer);
//     are_types_supported &= is_supported;
//   });

//   if (!are_types_supported) {
// #ifndef NDEBUG
//     if (openvino_ep::backend_utils::IsDebugEnabled()) {
//       std::cout << "DType is not supported" << std::endl;
//     }
// #endif
//     return false;
//   }

//   // Check 2

//   bool has_unsupported_dimension = false;
//   node->ForEachDef([&has_unsupported_dimension, this, &optype, &node](const NodeArg& node_arg, bool is_input) {
//     if (is_input) {
//       if (this->graph_viewer_.IsConstantInitializer(node_arg.Name(), true))
//         return;
//     }
//     auto shape = node_arg.Shape();
//     if (shape != nullptr) {
//       // Can't have no dimensions
//       if (shape->dim_size() == 0) {
//         if (op_is_supported(optype, no_dimension_supported_)) {
//           return;
//         }
//         // Special handling for the "Pad" operator
//         if (optype == "Pad") {
//           bool is_quantized = false;
//           // Detect a quantized model by checking for a DequantizeLinear input
//           for (Node::NodeConstIterator it_dq = node->InputNodesBegin(); it_dq != node->InputNodesEnd(); ++it_dq) {
//             const auto& DQ = &*it_dq;
//             if (DQ->OpType() == "DequantizeLinear") {
//               is_quantized = true;
//               break;
//             }
//           }
//           if (is_quantized) {
//             // For quantized Pad ops when the QDQ optimizer is disabled,
//             // bypass the unsupported dimension check to ensure 'pad_value' is constant
//             if (!npu_qdq_optimizer_enabled_) {
// #ifndef NDEBUG
//               if (openvino_ep::backend_utils::IsDebugEnabled()) {
//                 // Pad Op with DQ inputs gets optimized in the downstream,
//                 // so mark those no dim quantized Pad ops supported here
//                 std::cout << "QDQ optimizer disabled; quantized Pad op detected (DequantizeLinear present), so marking those no dim quantized Pad ops as supported" << std::endl;
//               }
// #endif
//             }
//             return;
//           }
//         }
//         // For ops that haven't been handled above, mark as unsupported dim
//         has_unsupported_dimension = true;
//         return;
//       } else {
//         // Zero dimension check
//         for (const auto& dim : shape->dim()) {
//           if (utils::HasDimValue(dim) && dim.dim_value() == 0) {
//             if (((device_id_.find("CPU") != std::string::npos) || (device_id_.find("GPU") != std::string::npos)) &&
//                 ((optype == "Expand") || (optype == "Equal") ||
//                  (optype == "Slice") || (optype == "Concat") ||
//                  (optype == "Shape") || (optype == "Cast") ||
//                  (optype == "Resize"))) {
//               return;
//             }
//             has_unsupported_dimension = true;
//             return;
//           }
//         }
//       }
//     }
//   });
//   if (has_unsupported_dimension) {
// #ifndef NDEBUG
//     if (openvino_ep::backend_utils::IsDebugEnabled()) {
//       std::cout << "Dimension check failed" << std::endl;
//     }
// #endif

//     return false;
//   }

//   // Check 3a
//   if (domain == kOnnxDomain && unsupported_op_mode(node, has_external_weights_)) {
//     if (optype == "GatherElements") {
//       return true;
//     }
// #ifndef NDEBUG
//     if (openvino_ep::backend_utils::IsDebugEnabled()) {
//       std::cout << "Failed in unsupported op mode" << std::endl;
//     }
// #endif
//     return false;
//   }

  return true;
}

std::vector<NodeIndex> DataOps::GetUnsupportedNodeIndices(std::unordered_set<std::string>& ng_required_initializers,
                                                          bool& has_external_weights_) {
  std::vector<NodeIndex> unsupported_nodes_idx;

  for (const auto& node_idx : graph_viewer_.GetNodesInTopologicalOrder()) {
    if (node_is_supported(node_idx, has_external_weights_)) {
      // Collect inputs that are initializers
      graph_viewer_.GetNode(node_idx)->ForEachDef([&ng_required_initializers, this](const NodeArg& node_arg,
                                                                                    bool is_input) {
            if (is_input && this->graph_viewer_.GetAllInitializedTensors().count(node_arg.Name())) {
                ng_required_initializers.insert(node_arg.Name());
              } },
                                                  true);
    } else {
      unsupported_nodes_idx.push_back(node_idx);
    }
  }
  return unsupported_nodes_idx;
}

bool DataOps::IsOpSupportedOnlyInModel(std::string name) {
  return ops_supported_only_in_model.find(name) != ops_supported_only_in_model.end();
}

bool DataOps::SpecialConditionForClusterSizeOne(std::unordered_set<std::string>& ng_required_initializers,
                                                const Node* node) {
  if (node->OpType() == "Reshape") {
    const auto& shape_arg = node->InputDefs()[1];
    if (ng_required_initializers.find(shape_arg->Name()) == ng_required_initializers.end()) {
      return true;
    }
  } else if (node->OpType() == "Expand") {
    // nGraph only supports constant shape input values
    const auto& output = node->OutputDefs()[0];
    if (output->TypeAsProto()->tensor_type().elem_type() !=
        ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16)
      return true;
  } else if (node->OpType() == "RoiAlign") {
    using onnx_dtype = ONNX_NAMESPACE::TensorProto_DataType;

    onnx_dtype input_0_data_type =
        (ONNX_NAMESPACE::TensorProto_DataType)node->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
    onnx_dtype input_1_data_type =
        (ONNX_NAMESPACE::TensorProto_DataType)node->InputDefs()[1]->TypeAsProto()->tensor_type().elem_type();
    onnx_dtype input_2_data_type =
        (ONNX_NAMESPACE::TensorProto_DataType)node->InputDefs()[2]->TypeAsProto()->tensor_type().elem_type();
    onnx_dtype output_data_type =
        (ONNX_NAMESPACE::TensorProto_DataType)node->OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type();

    if ((input_0_data_type != onnx_dtype::TensorProto_DataType_FLOAT16) ||
        (input_1_data_type != onnx_dtype::TensorProto_DataType_FLOAT16) ||
        (input_2_data_type != onnx_dtype::TensorProto_DataType_FLOAT) ||
        (output_data_type != onnx_dtype::TensorProto_DataType_FLOAT16))
      return true;
  }
  return false;
}

bool DataOps::DoNotOmitSubGraph(const std::string& name) {
  return op_is_supported(name, subgraph_supported_);
}

bool DataOps::InsertNode(const std::string& optype) {
  if (optype == "TopK" || optype == "NonZero") {
    return true;
  }
  return false;
}

}  // namespace openvino_ep
}  // namespace onnxruntime
