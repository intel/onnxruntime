// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>

#include "ov_compute.h"
#include "ov_ep_context.h"

using namespace onnxruntime::openvino_ep;

namespace onnxruntime {
namespace openvino_ep {

OvComputeInfo::OvComputeInfo(ApiPtrs apis, ov::Core& ov_core) : ApiPtrs(apis), ov_core_(ov_core) {
  ort_version_supported = ORT_API_VERSION;
  OrtNodeComputeInfo::CreateState = CreateStateImpl;
  OrtNodeComputeInfo::Compute = ComputeImpl;
  OrtNodeComputeInfo::ReleaseState = ReleaseStateImpl;
}

OrtStatus* OvComputeInfo::Init(const std::string& ov_device, OnnxIOMapping io_mapping, EpContextNode ep_context_node) {
  ov::AnyMap configs = {}; /* no configs yet*/

  switch (ep_context_node.type_) {
    case EpContextNode::EpContextType::Native:
      if (ep_context_node.embed_mode == 1) {
        std::istringstream model_stream(std::move(ep_context_node.ep_cache_context));
        compiled_model_ = ov_core_.import_model(model_stream, ov_device, configs);
      } else {
        // How do I get the path to the graph? Graph API doesn't seem to have the source?
        std::filesystem::path ep_cache_context = ep_context_node.ep_cache_context;
        std::ifstream model_stream(ep_cache_context, std::ios_base::binary | std::ios_base::in);
        compiled_model_ = ov_core_.import_model(model_stream, ov_device, configs);
      }
      break;
    default:
      return Ort::Status("Unsupported EpContextType", ORT_INVALID_ARGUMENT).release();
  }

  infer_request_pool_ = std::make_unique<InferRequestPool>(compiled_model_, 1, [](InferRequestPool::OVInferRequestPtr&) {});
  onnx_to_ov_bindings_ = std::make_unique<OnnxToOvNetworkBindings>(compiled_model_, io_mapping, SessionContext{});

  return nullptr;
}

OrtStatus* OvComputeInfo::Init(const std::string& ov_device, OnnxIOMapping io_mapping, std::unique_ptr<onnx::GraphProto> graph_proto) {
  ov::AnyMap configs = {}; /* no configs yet*/

  auto model_proto = std::make_unique<onnx::ModelProto>();
  model_proto->set_allocated_graph(graph_proto.release());
  model_proto->set_ir_version(onnx::IR_VERSION);
  model_proto->set_producer_name("onnxruntime_ov_provider_plugin");
  model_proto->set_producer_version(OVEP_PLUGIN_VERSION);

  std::string model = model_proto->SerializeAsString();
  model_proto.reset();

  compiled_model_ = ov_core_.compile_model(std::move(model), {}, ov_device, configs);
  infer_request_pool_ = std::make_unique<InferRequestPool>(compiled_model_, 1, [](InferRequestPool::OVInferRequestPtr&) {});
  onnx_to_ov_bindings_ = std::make_unique<OnnxToOvNetworkBindings>(compiled_model_, io_mapping, SessionContext{});

  return nullptr;
}

OrtStatus* OvComputeInfo::Compute(void* /*compute_state*/,
                                  OrtKernelContext* kernel_context) {
  auto guarded_infer_req = infer_request_pool_->getRequest();
  auto& infer_request = guarded_infer_req.infer_request_;

  Ort::KernelContext context(kernel_context);

  if (onnx_to_ov_bindings_->has_dynamic_io_) {
    // Dynamic shape inference

    // We don't know the output shapes so we need to get the outputs from the infer request and copy them into the ort
    // tensors instead of binding them to the infer request directly.

    // Bind inputs
    for (const auto& input_info : onnx_to_ov_bindings_->network_inputs_) {
      // Set the input shape based on the input tensor from ort
      auto tensor = context.GetInput(input_info.onnx_index);
      auto ort_shape = tensor.GetTensorTypeAndShapeInfo().GetShape();
      auto input_shape = ParameterShape(ort_shape);

      infer_request->SetTensor(input_info.name,
                               input_info.type,
                               input_shape,
                               const_cast<void*>(tensor.GetTensorRawData()));
    }

    // Run Inference
    infer_request->Infer();

    // Copy outputs
    for (const auto& output_info : onnx_to_ov_bindings_->network_outputs_) {
      auto ov_tensor = infer_request->ov().get_tensor(output_info.name);
      auto output_shape = ParameterShape::ToOrtShape(ov_tensor.get_shape());
      auto ort_tensor = context.GetOutput(output_info.onnx_index, output_shape);

      RETURN_IF(ov_tensor.get_byte_size() == ort_tensor.GetTensorSizeInBytes(),
                ort_api,
                std::format("Output tensor size mismatch for {}", output_info.name).c_str());

      std::memcpy(ort_tensor.GetTensorMutableRawData(),
                  ov_tensor.data(),
                  ov_tensor.get_byte_size());
    }
  } else {
    // Static shape inference

    // Bind inputs
    for (const auto& input_info : onnx_to_ov_bindings_->network_inputs_) {
      infer_request->SetTensor(input_info.name,
                               input_info.type,
                               input_info.shape,
                               const_cast<void*>(context.GetInput(input_info.onnx_index).GetTensorRawData()));
    }

    // Bind outputs
    for (const auto& output_info : onnx_to_ov_bindings_->network_outputs_) {
      infer_request->SetTensor(output_info.name,
                               output_info.type,
                               output_info.shape,
                               context.GetOutput(output_info.onnx_index, output_info.shape).GetTensorMutableRawData());
    }

    // Run Inference
    infer_request->Infer();
  }

  return nullptr;
}

OrtStatus* OvComputeInfo::CreateState(OrtNodeComputeContext* compute_context,
                                      void** compute_state) {
  // Dummy implementation: set compute_state to nullptr
  (void)compute_context;
  *compute_state = nullptr;
  return nullptr;
}

void OvComputeInfo::ReleaseState(void* compute_state) {
  // Dummy implementation: do nothing
  (void)compute_state;
}

}  // namespace openvino_ep
}  // namespace onnxruntime
