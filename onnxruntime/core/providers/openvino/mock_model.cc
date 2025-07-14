// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include <vector>
#include <numeric>
#include <string_view>

#include "core/graph/constants.h"
#include "mock_model.h"
#include <span>

#define BREAK_ON_ERROR(ort_status) \
  if (ort_status) break;

namespace onnxruntime {
namespace openvino_ep {

// Create OrtNode using the C API
OrtNode* CreateNode(const OrtModelEditorApi& api,
                    const char* operator_name, const char* node_name,
                    const std::span<const char*> input_names,
                    const std::span<const char*> output_names,
                    const std::span<OrtOpAttr*> attributes = {},
                    const char* domain_name = onnxruntime::kOnnxDomain) {
  OrtNode* node = nullptr;
  auto status = api.CreateNode(operator_name, domain_name, node_name,
                               input_names.data(), input_names.size(),
                               output_names.data(), output_names.size(),
                               attributes.data(), attributes.size(),
                               &node);
  return (status == nullptr) ? node : nullptr;
}

// convenience func to convert initalizer lists to gsl::span
OrtNode* CreateNode(const OrtModelEditorApi& api,
                    const char* operator_name, const char* node_name,
                    const std::initializer_list<const char*> input_names,
                    const std::initializer_list<const char*> output_names,
                    const std::initializer_list<OrtOpAttr*> attributes = {},
                    const char* domain_name = onnxruntime::kOnnxDomain) {
  std::vector<const char*> inputs(input_names);
  std::vector<const char*> outputs(output_names);
  std::vector<OrtOpAttr*> attrs(attributes);
  return CreateNode(api, operator_name, node_name, inputs, outputs, attrs, domain_name);
}

mock_model build_model(const OrtApi& api, bool use_constant_node) {
  OrtGraph* graph = nullptr;

  // return void so we can use assert_* in the lambda
  do {
    auto& model_editor_api = *api.GetModelEditorApi();
    BREAK_ON_ERROR(model_editor_api.CreateGraph(&graph));

    //
    // Create OrtModel with a Gemm. X input is 3x4, Y input is 4x8, Z output is 3x8.
    // X is model input. Y is initializer.
    // Set the alpha attribute of the Gemm node to 2.0 to test attribute handling.
    //

    // model input
    OrtTensorTypeAndShapeInfo* tensor_type_info = nullptr;
    std::vector<int64_t> input_dims = {3, 4};
    // can use api.SetSymbolicDimensions to set symbolic dimensions.
    // the input array should have the same rank as the call to SetDimensions.
    // e.g. call SetDimensions with {-1, 3, 2} and SetSymbolicDimensions with {"N", nullptr, nullptr} to create
    //      a shape of {"N", 3, 2}

    BREAK_ON_ERROR(api.CreateTensorTypeAndShapeInfo(&tensor_type_info));
    BREAK_ON_ERROR(api.SetTensorElementType(tensor_type_info, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
    BREAK_ON_ERROR(api.SetDimensions(tensor_type_info, input_dims.data(), input_dims.size()));

    OrtTypeInfo* input_type_info = nullptr;
    BREAK_ON_ERROR(model_editor_api.CreateTensorTypeInfo(tensor_type_info, &input_type_info));
    api.ReleaseTensorTypeAndShapeInfo(tensor_type_info);  // input_type_info took a copy

    // create ValueInfo and release the type info as CreateValueInfo takes a copy.
    OrtValueInfo* input_value_info = nullptr;
    BREAK_ON_ERROR(model_editor_api.CreateValueInfo("X", input_type_info, &input_value_info));
    api.ReleaseTypeInfo(input_type_info);  // input_value_info took a copy
    tensor_type_info = nullptr;

    // model outputs
    OrtTypeInfo* output_type_info = nullptr;
    std::vector<int64_t> output_dims = {3, 8};

    BREAK_ON_ERROR(api.CreateTensorTypeAndShapeInfo(&tensor_type_info));
    BREAK_ON_ERROR(api.SetTensorElementType(tensor_type_info, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
    BREAK_ON_ERROR(api.SetDimensions(tensor_type_info, output_dims.data(), output_dims.size()));

    BREAK_ON_ERROR(model_editor_api.CreateTensorTypeInfo(tensor_type_info, &output_type_info));
    api.ReleaseTensorTypeAndShapeInfo(tensor_type_info);  // input_type_info took a copy

    OrtValueInfo* output_value_info = nullptr;
    BREAK_ON_ERROR(model_editor_api.CreateValueInfo("Z", output_type_info, &output_value_info));
    api.ReleaseTypeInfo(output_type_info);

    std::vector<OrtValueInfo*> graph_inputs = {input_value_info};
    std::vector<OrtValueInfo*> graph_outputs = {output_value_info};
    BREAK_ON_ERROR(model_editor_api.SetGraphInputs(graph, graph_inputs.data(), graph_inputs.size()));
    BREAK_ON_ERROR(model_editor_api.SetGraphOutputs(graph, graph_outputs.data(), graph_outputs.size()));
    input_value_info = nullptr;  // graph now owns the input/output values
    output_value_info = nullptr;

    //
    // Gemm node
    //

    OrtOpAttr* alpha_attr = nullptr;
    float alpha_value = 2.0;
    BREAK_ON_ERROR(api.CreateOpAttr("alpha", &alpha_value, 1, OrtOpAttrType::ORT_OP_ATTR_FLOAT, &alpha_attr));

    std::vector<const char*> node_input_names = {"X", "Y"};
    auto gemm_output_name = use_constant_node ? "Z_temp" : "Z";
    auto node_output_names = std::vector<const char*>{use_constant_node ? "Z_temp" : "Z"};
    std::vector<OrtOpAttr*> node_attributes{alpha_attr};
    OrtNode* node = CreateNode(model_editor_api, "Gemm", "Gemm1", node_input_names, node_output_names, node_attributes);
    alpha_attr = nullptr;  // Node now owns

    BREAK_ON_ERROR(model_editor_api.AddNodeToGraph(graph, node));
    node = nullptr;  // graph now owns node

    // Y input
    // As it's 128 bytes it could either be allocated using CreateTensorAsOrtValue or use existing memory.
    // Under 128 bytes must use CreateTensorAsOrtValue.
    std::vector<int64_t> y_dims = {4, 8};

    std::vector<float> y_values(32);
    std::iota(y_values.begin(), y_values.end(), 1.0f);

    OrtMemoryInfo* ort_meminfo{nullptr};
    BREAK_ON_ERROR(api.CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &ort_meminfo));

    // create an initializer for the Y input. add to `weights` so the memory remains valid.
    OrtValue* y_tensor = nullptr;
    BREAK_ON_ERROR(api.CreateTensorWithDataAsOrtValue(ort_meminfo,
                                                      y_values.data(), y_values.size() * sizeof(y_values[0]),
                                                      y_dims.data(), y_dims.size(),
                                                      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                                      &y_tensor));

    BREAK_ON_ERROR(model_editor_api.AddInitializerToGraph(graph, "Y", y_tensor, /*data is external*/ true));
    y_tensor = nullptr;  // graph now owns

    if (use_constant_node) {
      // Test that a Constant node is converted to an initializer

      // create Constant nodes for min/max to limit output range
      OrtOpAttr* min_attr = nullptr;
      float min = 400.0f;
      BREAK_ON_ERROR(api.CreateOpAttr("value", &min, sizeof(min), ORT_OP_ATTR_FLOAT, &min_attr));
      node = CreateNode(model_editor_api, "Constant", "clip_min", {}, {"min"}, {min_attr});
      BREAK_ON_ERROR(model_editor_api.AddNodeToGraph(graph, node));
      node = nullptr;  // graph now owns node

      OrtOpAttr* max_attr = nullptr;
      float max = 900.0f;
      BREAK_ON_ERROR(api.CreateOpAttr("value", &max, sizeof(max), ORT_OP_ATTR_FLOAT, &max_attr));
      node = CreateNode(model_editor_api, "Constant", "clip_max", {}, {"max"}, {max_attr});
      BREAK_ON_ERROR(model_editor_api.AddNodeToGraph(graph, node));
      node = nullptr;  // graph now owns node

      node = CreateNode(model_editor_api, "Clip", "Clip1", {gemm_output_name, "min", "max"}, {"Z"});
      BREAK_ON_ERROR(model_editor_api.AddNodeToGraph(graph, node));
      node = nullptr;  // graph now owns node
    }

    //std::vector<const char*> domain_names = {onnxruntime::kOnnxDomain};
    //std::vector<int> opset_versions = {18};
    //OrtModel* model = nullptr;
    //BREAK_ON_ERROR(model_editor_api.CreateModel(domain_names.data(), opset_versions.data(), domain_names.size(),
    //                                            &model));
    //BREAK_ON_ERROR(model_editor_api.AddGraphToModel(model, graph));
    //graph = nullptr;  // model now owns
  } while (0);

  return {api, graph};
}

mock_model::~mock_model() {
  api.ReleaseGraph(graph);
}

}  // namespace openvino_ep
}  // namespace onnxruntime