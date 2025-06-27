// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include <vector>
#include <numeric>
#include <string_view>

//#define ORT_RUNTIME_CLASS(X) \
//  struct Ort##X;             \
//  typedef struct Ort##X Ort##X
//
// typedef enum OrtErrorCode {
//  ORT_OK,
//  ORT_FAIL,
//  ORT_INVALID_ARGUMENT,
//  ORT_NO_SUCHFILE,
//  ORT_NO_MODEL,
//  ORT_ENGINE_ERROR,
//  ORT_RUNTIME_EXCEPTION,
//  ORT_INVALID_PROTOBUF,
//  ORT_MODEL_LOADED,
//  ORT_NOT_IMPLEMENTED,
//  ORT_INVALID_GRAPH,
//  ORT_EP_FAIL,
//  ORT_MODEL_LOAD_CANCELED,
//  ORT_MODEL_REQUIRES_COMPILATION,
//} OrtErrorCode;
//
// struct OrtStatus {
//  OrtErrorCode code;
//  char msg[1];  // a null-terminated string
//};
//
// #ifdef _MSC_VER
// typedef _Return_type_success_(return == 0) OrtStatus* OrtStatusPtr;
// #else
// typedef OrtStatus* OrtStatusPtr;
// #endif
//
// #define ORT_API_CALL _stdcall
//
// struct OrtKernelContext;
//
// #ifndef _Frees_ptr_opt_
// #define _Frees_ptr_opt_ _SAL_L_Source_(_Frees_ptr_opt_, (), _Pre_maybenull_ _Post_ptr_invalid_ __drv_freesMem(Mem))
// #endif
//
//// XXX: Unfortunately, SAL annotations are known to not work with function pointers
// #d efine ORT_API2_STATUS(NAME, ...) \
//  _Check_return_ _Ret_maybenull_ OrtStatusPtr(ORT_API_CALL* NAME)(__VA_ARGS__) NO_EXCEPTION ORT_MUST_USE_RESULT
//
// #ifndef __has_feature
// #define __has_feature(x) 0
// #endif
//
// #if ((__cplusplus >= 201103L) || (_MSC_VER >= 1900) || (defined(__has_feature) && __has_feature(cxx_noexcept)))
// #define NO_EXCEPTION noexcept
// #else
// #define NO_EXCEPTION throw()
// #endif
//
// #define ORT_MUST_USE_RESULT
//
// ORT_RUNTIME_CLASS(Graph);
// ORT_RUNTIME_CLASS(Node);
// ORT_RUNTIME_CLASS(HardwareDevice);
// ORT_RUNTIME_CLASS(EpDevice);
// ORT_RUNTIME_CLASS(KeyValuePairs);
// ORT_RUNTIME_CLASS(SessionOptions);
// ORT_RUNTIME_CLASS(Logger);
//
// #define ORT_CLASS_RELEASE(X) void(ORT_API_CALL * Release##X)(_Frees_ptr_opt_ Ort##X * input)
//
// #d efine ORT_API_T(RETURN_TYPE, NAME, ...) \
//  RETURN_TYPE(ORT_API_CALL* NAME)(__VA_ARGS__) NO_EXCEPTION
//
// struct OrtApi;
// typedef struct OrtApi OrtApi;
//
// struct OrtApiBase {
//   /** \brief Get a pointer to the requested version of the ::OrtApi
//    *
//    * \param[in] version Must be ::ORT_API_VERSION
//    * \return The ::OrtApi for the version requested, nullptr will be returned if this version is unsupported, for example when using a runtime
//    *   older than the version created with this header file.
//    *
//    * One can call GetVersionString() to get the version of the Onnxruntime library for logging
//    * and error reporting purposes.
//    */
//   const OrtApi*(ORT_API_CALL* GetApi)(uint32_t version)NO_EXCEPTION;
//
//   /** \brief Returns a null terminated string of the version of the Onnxruntime library (eg: "1.8.1")
//    *
//    *  \return UTF-8 encoded version string. Do not deallocate the returned buffer.
//    */
//   const char*(ORT_API_CALL* GetVersionString)(void)NO_EXCEPTION;
// };
//
// typedef struct OrtApiBase OrtApiBase;
//
// #include "core/session/onnxruntime_ep_c_api.h"

#include "core/session/onnxruntime_c_api.h"
#include "core/graph/constants.h"
#include <span>

#define BREAK_ON_ERROR(ort_status) \
  if (ort_status) break;

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

OrtModel* build_model(bool use_constant_node, OrtApi& api, OrtModelEditorApi& model_editor_api) {
  OrtModel* model = nullptr;

  // return void so we can use assert_* in the lambda
  do {
    OrtGraph* graph = nullptr;
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

    // create an initializer for the Y input. add to `weights` so the memory remains valid.
    OrtValue* y_tensor = nullptr;
    BREAK_ON_ERROR(api.CreateTensorWithDataAsOrtValue(nullptr,
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

    std::vector<const char*> domain_names = {onnxruntime::kOnnxDomain};
    std::vector<int> opset_versions = {18};
    BREAK_ON_ERROR(model_editor_api.CreateModel(domain_names.data(), opset_versions.data(), domain_names.size(),
                                                &model));
    BREAK_ON_ERROR(model_editor_api.AddGraphToModel(model, graph));
    graph = nullptr;  // model now owns
  } while (0);

  return model;
}
