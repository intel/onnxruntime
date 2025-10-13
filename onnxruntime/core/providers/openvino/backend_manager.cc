// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include <array>
#include <algorithm>
#include <cassert>
#include <fstream>
#include <regex>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <istream>

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/openvino/backend_manager.h"
#include "core/providers/openvino/backend_utils.h"
#include "core/providers/openvino/contexts.h"
#include "core/providers/openvino/ibackend.h"
#include "core/providers/openvino/ov_interface.h"
#include "core/providers/openvino/ov_versions/capability.h"
#include "core/providers/openvino/qdq_transformations/qdq_stripping.h"
#include "core/providers/openvino/qdq_transformations/qdq_scales_fix.h"
#include "../../framework/tensorprotoutils.h"

namespace onnxruntime {
namespace openvino_ep {

SessionContext& BackendManager::GetSessionContext() {
  return session_context_;
}

ov::CompiledModel BackendManager::GetOVCompiledModel() {
  if (concrete_backend_)
    return concrete_backend_->GetOVCompiledModel();
  return ov::CompiledModel();
}

BackendManager::BackendManager(SessionContext& session_context,
                               SharedContext& shared_context,
                               const onnxruntime::Node& fused_node,
                               const onnxruntime::GraphViewer& subgraph,
                               const logging::Logger& logger,
                               EPCtxHandler& ep_ctx_handle) : ep_ctx_handle_(ep_ctx_handle),
                                                              session_context_(session_context),
                                                              shared_context_{shared_context} {
  subgraph_context_.is_ep_ctx_graph = ep_ctx_handle_.CheckForOVEPCtxNodeInGraph(subgraph);
  // If the graph contains a OVIR wrapped node, we check if it has matching xml file name attribute
  subgraph_context_.is_ep_ctx_ovir_encapsulated = ep_ctx_handle_.CheckEPCacheContextAttribute(subgraph,
                                                                                              session_context_.onnx_model_path_name.filename().replace_extension("xml").string());

  subgraph_context_.model_precision = [&](const GraphViewer& graph_viewer) {
    // return empty if graph has no inputs or if types are not one of FP32/FP16
    // else assume the type of the first input
    if (graph_viewer.GetInputs().empty()) {
      return "";
    } else {
      auto input_type = graph_viewer.GetInputs()[0]->TypeAsProto()->tensor_type().elem_type();
      if (session_context_.precision == "ACCURACY" &&
          session_context_.device_type.find("GPU") != std::string::npos) {
        if (input_type == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT) {
          return "FP32";
        } else if (input_type == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16) {
          return "FP16";
        }
      }
    }
    return "";
  }(subgraph);

  // Save the indexes of graph inputs among fused_node's inputDefs
  // (which also contains initializers).
  for (uint32_t index = 0; const auto& node : subgraph.GetInputs()) {
    if (subgraph.GetGraph().GetConsumerNodes(node->Name()).size() == 0) {
      continue;  // Skip if the input is a dangling node
    }
    subgraph_context_.input_names.insert({node->Name(), index++});
  }

  for (uint32_t index = 0; const auto& node : subgraph.GetOutputs()) {
    subgraph_context_.output_names.insert({node->Name(), index++});
  }

  subgraph_context_.subgraph_name = fused_node.Name();

  ptr_stream_t model_stream;
  std::unique_ptr<onnx::ModelProto> model_proto;
  if (subgraph_context_.is_ep_ctx_graph) {
    if (!session_context_.reshape.empty()) {
      std::string exception_str =
          "[OpenVINO-EP] Bounded dynamic model execution using provider option reshape_input is not supported for OVEP EPContext model";
      ORT_THROW(exception_str);
    }
    if (subgraph_context_.is_ep_ctx_ovir_encapsulated) {
      model_stream = ep_ctx_handle_.GetModelBlobStream(session_context_.onnx_model_path_name.replace_extension("xml").string(), subgraph);
    } else {
      model_stream = ep_ctx_handle_.GetModelBlobStream(session_context_.so_context_file_path, subgraph);
    }

  } else {
    model_proto = GetModelProtoFromFusedNode(fused_node, subgraph, logger);
  }
  std::string device_type = session_context_.device_type;

  auto& sw = shared_context_.shared_weights;
  if (session_context_.so_share_ep_contexts && !sw.metadata.empty()) {
    std::filesystem::path weight_filename = session_context_.onnx_model_path_name.parent_path();
    if (sw.external_weight_filename.empty()) {
      // Reasonable assumption that all metadata entries have the same external file location
      sw.external_weight_filename = sw.metadata.begin()->second.location;
    }
    weight_filename /= sw.external_weight_filename;
    std::ifstream weight_file(weight_filename);

    ORT_ENFORCE(weight_file, "Initializer file not found: ", weight_filename.string());
    if (!sw.mapped_weights) {
      sw.mapped_weights = std::make_unique<SharedContext::SharedWeights::WeightsFile>(weight_filename);
    }
    backend_utils::CreateOVTensors(session_context_.device_type, sw.metadata, *sw.mapped_weights);
  }

  if (ModelHasSymbolicInputDims(subgraph)) {
    subgraph_context_.has_dynamic_input_shape = true;
    LOGS_DEFAULT(INFO) << "[OpenVINO-EP] Model has symbolic input dims";
    if ((!session_context_.disable_dynamic_shapes &&
         (session_context_.device_type.find("CPU") != std::string::npos ||
          session_context_.device_type.find("GPU") != std::string::npos ||
          (session_context_.device_type.find("NPU") != std::string::npos &&
           session_context_.enable_causallm))) ||
        (subgraph_context_.is_ep_ctx_graph)) {
      LOGS_DEFAULT(INFO) << "[OpenVINO-EP] Starting backend initialization. "
                         << "Creating backend Dynamic Shapes";
      try {
        concrete_backend_ = BackendFactory::MakeBackend(model_proto,
                                                        session_context_,
                                                        subgraph_context_,
                                                        shared_context_,
                                                        model_stream);
      } catch (std::string const& msg) {
        ORT_THROW(msg);
      }
      LOGS_DEFAULT(INFO) << "[OpenVINO-EP] "
                         << "Backend created for graph " << subgraph_context_.subgraph_name;
    } else {
      // Only cache model_proto in session context to rewrite the model with input shapes at runtime.
      // For dynamic backend creation
      model_proto_ = std::move(model_proto);
    }
  } else {
    LOGS_DEFAULT(INFO) << "[OpenVINO-EP] Model has concrete input dims. "
                       << "Initializing backend for graph "
                       << subgraph_context_.subgraph_name;

    subgraph_context_.has_dynamic_input_shape = false;

    // OV NPU plugin is supported with fallback to OV CPU upon compilation failures.
    try {
      concrete_backend_ = BackendFactory::MakeBackend(model_proto,
                                                      session_context_,
                                                      subgraph_context_,
                                                      shared_context_,
                                                      model_stream);
    } catch (const OnnxRuntimeException& ex) {
      std::string exception_str = ex.what();

      if (session_context_.device_type.find("NPU") != std::string::npos &&
          exception_str.find("intel_npu") != std::string::npos) {
        // Handle NPU device related errors
#ifndef NDEBUG
        std::string suffix = session_context_.so_disable_cpu_ep_fallback ?
          "\nModel failed to compile on NPU. Enable CPU fallback or try another device.\n" :
          "\nModel needs to be recompiled\n";
        ORT_THROW(exception_str + suffix);
#else
        std::string error_message = "UNKNOWN NPU ERROR";
        std::string error_code = "code 0x0";
        std::regex error_message_pattern(R"(\bZE_\w*\b)");
        std::regex error_code_pattern("code 0x[0-9a-fA-F]+");
        std::smatch matches;
        if (std::regex_search(exception_str, matches, error_message_pattern)) {
          error_message = matches[0];
        }
        if (std::regex_search(exception_str, matches, error_code_pattern)) {
          error_code = matches[0];
        }
        std::string suffix = session_context_.so_disable_cpu_ep_fallback ?
          "\nModel failed to compile on NPU. Enable CPU fallback or try another device.\n" :
          "\nModel needs to be recompiled\n";
        throw std::runtime_error(error_message + ", " + error_code + suffix);
#endif
      } else {
        ORT_THROW(exception_str);
      }
    }
  }
  if (session_context_.so_context_enable &&
      (subgraph_context_.is_ep_ctx_ovir_encapsulated || !subgraph_context_.is_ep_ctx_graph)) {
    if (concrete_backend_) {
      auto status = onnxruntime::openvino_ep::BackendManager::ExportCompiledBlobAsEPCtxNode(subgraph);
      if (!status.IsOK()) {
        ORT_THROW(status);
      }
    } else {
      ORT_THROW("[OpenVINO-EP] Cannot export compiled blob as EPCtx Node: Backend not initialized.");
    }
  }
}

// Call EPContext model exporter here if the provider option for exporting
// precompiled blob is set. If that's the case:
// By default, create model in embed mode where the blob stream is exported as data within
// the EPContext node.
Status BackendManager::ExportCompiledBlobAsEPCtxNode(const onnxruntime::GraphViewer& graph_body_viewer) {
  if (session_context_.disable_dynamic_shapes && subgraph_context_.has_dynamic_input_shape) {
    std::string exception_str =
        "Exporting dynamically compiled models at runtime is not supported. "
        "Cannot export blobs of dynamic models that request static shape inference. "
        "To export this model, set disable_dynamic_shapes to False";
    ORT_THROW(exception_str);
  }

  // If embed_mode, then pass on the serialized blob
  // If not embed_mode, dump the blob here and only pass on the path to the blob
  std::string model_blob_str;
  auto compiled_model = concrete_backend_->GetOVCompiledModel();
  if (session_context_.so_context_embed_mode) {  // Internal blob
    std::ostringstream model_blob_stream;
    compiled_model.export_model(model_blob_stream);
    model_blob_str = std::move(model_blob_stream).str();
    if (model_blob_str.empty()) {
      ORT_THROW("Model blob stream is empty after exporting the compiled model.");
    }
  } else {  // External blob
    // Build name by combining EpCtx model name (if available) and subgraph name. Model
    // name is not available in when creating a session from memory
    auto name = session_context_.so_context_file_path.stem().string();
    if (name.empty() && !graph_body_viewer.ModelPath().empty()) {
      name = graph_body_viewer.ModelPath().stem().string();
    }
    ORT_ENFORCE(!name.empty());
    name += "_" + subgraph_context_.subgraph_name;

    std::filesystem::path blob_filename = session_context_.so_context_file_path;
    if (blob_filename.empty()) {
      blob_filename = session_context_.onnx_model_path_name;
    }
    blob_filename = blob_filename.parent_path() / (name + ".blob");
    std::ofstream blob_file(blob_filename,
                            std::ios::out | std::ios::trunc | std::ios::binary);
    if (!blob_file) {
      std::ostringstream err_msg;
      err_msg << "Unable to open file for epctx model dump: " << blob_filename;
      ORT_THROW(err_msg.str());
    }
    compiled_model.export_model(blob_file);
    model_blob_str = blob_filename.filename().string();
  }

  ORT_RETURN_IF_ERROR(ep_ctx_handle_.AddOVEPCtxNodeToGraph(graph_body_viewer,
                                                           subgraph_context_.subgraph_name,
                                                           session_context_.so_context_embed_mode,
                                                           std::move(model_blob_str)));

  return Status::OK();
}

bool BackendManager::ModelHasBatchedInputs(const ONNX_NAMESPACE::ModelProto& model_proto) const {
  bool has_batched_inputs = true;

  for (const auto& [name, index] : subgraph_context_.input_names) {
    auto& input = model_proto.graph().input(index);

    // Batch-process only raw image inputs (NCHW or NHWC layouts)
    auto& shape = input.type().tensor_type().shape();
    if (shape.dim_size() != 4) {
      has_batched_inputs = false;
      break;
    }

    if (shape.dim(0).value_case() == shape.dim(0).kDimValue) {
      has_batched_inputs = false;
      break;
    }

    for (int dim_index = 1; dim_index < 4; dim_index++) {
      if (shape.dim(dim_index).value_case() != shape.dim(0).kDimValue) {
        has_batched_inputs = false;
        break;
      }
    }
    if (!has_batched_inputs) {
      break;
    }
  }
  return has_batched_inputs;
}

bool BackendManager::ModelHasSymbolicInputDims(const onnxruntime::GraphViewer& subgraph) const {
  const auto& graph_inputs = subgraph.GetInputs();

  // First validate shapes if provided by user
  bool shapes_valid = true;
  if (!session_context_.reshape.empty()) {
    try {
      ValidateInputShapes(session_context_.reshape, graph_inputs);
    } catch (const std::exception& e) {
      LOGS_DEFAULT(ERROR) << "[OpenVINO-EP] Shape validation failed: " << e.what();
      session_context_.reshape.clear();  // Clear the shape map as it's invalid
      shapes_valid = false;
    }
  }

  // Count dynamic inputs and check if reshape covers all of them
  size_t dynamic_input_count = 0;
  bool all_dynamic_inputs_covered = true;

  for (const auto* input : graph_inputs) {
    // Skip dangling inputs (no consumers)
    if (subgraph.GetGraph().GetConsumerNodes(input->Name()).empty()) {
      continue;
    }

    // Check if input has dynamic dimensions
    bool has_dynamic_dim = false;

    // Case 1: Completely undefined shape
    if (input->Shape() == nullptr) {
      has_dynamic_dim = true;
    }
    // Case 2: Shape defined but with symbolic dimensions
    else {
      for (const auto& dim : input->Shape()->dim()) {
        if (dim.value_case() != dim.kDimValue) {
          has_dynamic_dim = true;
          break;
        }
      }
    }

    // If dynamic, count it and check if reshape covers it
    if (has_dynamic_dim) {
      dynamic_input_count++;

      // Check if this dynamic input is covered by reshape input
      if (!session_context_.reshape.empty() &&
          session_context_.reshape.find(input->Name()) == session_context_.reshape.end()) {
        all_dynamic_inputs_covered = false;
        LOGS_DEFAULT(WARNING) << "[OpenVINO-EP] reshape_input is provided but doesn't cover dynamic input: "
                              << input->Name();
      }
    }
  }

  const bool has_symbolic_dims = (dynamic_input_count > 0);

  // Early return if no reshape input provided
  if (session_context_.reshape.empty()) {
    return has_symbolic_dims;  // Return based on whether model has symbolic dims
  }

  // For dynamic models with incomplete reshape coverage, clear shapes
  if (has_symbolic_dims && !all_dynamic_inputs_covered) {
    session_context_.reshape.clear();
    LOGS_DEFAULT(WARNING) << "reshape_input does not cover all dynamic dimensions, "
                          << "ignoring all provided shapes";
    return true;  // Model is dynamic
  }

  // If shapes are valid with complete coverage for dynamic model, treat as concrete
  if (has_symbolic_dims && shapes_valid && all_dynamic_inputs_covered) {
    LOGS_DEFAULT(INFO) << "All dynamic dimensions successfully covered by reshape_input";
    return false;  // Model is now effectively static with concrete shapes
  }

  return has_symbolic_dims;  // Return dynamic status based on symbolic dimensions
}

// Check to see if the graph is QDQ
static bool IsQDQGraph(const onnxruntime::GraphViewer& graph_viewer) {
  std::unordered_set<std::string> qdq_ops = {"QuantizeLinear", "DequantizeLinear"};
  const auto& node_indices = graph_viewer.GetNodesInTopologicalOrder();

  for (size_t i = 0; i < node_indices.size(); i++) {
    gsl::not_null<const onnxruntime::Node*> node(graph_viewer.GetNode(node_indices[i]));
    if (qdq_ops.find(node->OpType()) != qdq_ops.end()) {
      return true;
    }
  }
  return false;
}

static bool IsModelBF16(const onnxruntime::GraphViewer& graph_viewer) {
  const auto& node_indices = graph_viewer.GetNodesInTopologicalOrder();
  for (std::size_t i = 0; i < node_indices.size(); i++) {
    gsl::not_null<const onnxruntime::Node*> node(graph_viewer.GetNode(node_indices[i]));
    for (auto& output : node->OutputDefs()) {
      if (output->ToProto().type().tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16)
        return true;
    }
  }
  return false;
}

static bool Is16BitTensor(const onnxruntime::NodeArg* node_arg) {
  const auto* type_proto = node_arg ? node_arg->TypeAsProto() : nullptr;
  return type_proto && type_proto->has_tensor_type() &&
         (type_proto->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_UINT16 ||
          type_proto->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_INT16);
}

// Check to see if the graph has Q/DQ nodes with int16 or uint16 quantization
static bool IsQDQGraphWithUint16OrInt16(const onnxruntime::GraphViewer& graph_viewer) {
  std::unordered_set<std::string> qdq_ops = {"QuantizeLinear", "DequantizeLinear"};
  const auto& node_indices = graph_viewer.GetNodesInTopologicalOrder();

  for (size_t i = 0; i < node_indices.size(); i++) {
    gsl::not_null<const onnxruntime::Node*> node(graph_viewer.GetNode(node_indices[i]));

    if (qdq_ops.find(node->OpType()) != qdq_ops.end()) {
      const auto& input_defs = node->InputDefs();

      if (node->OpType() == "DequantizeLinear") {
        // DequantizeLinear: [quantized_input, scale, zero_point] -> [float_output]
        // Check quantized input tensor and optional zero point
        if (Is16BitTensor(input_defs.empty() ? nullptr : input_defs[0]) ||
            (input_defs.size() >= 3 && Is16BitTensor(input_defs[2]))) {
          return true;
        }
      } else if (node->OpType() == "QuantizeLinear") {
        // QuantizeLinear: [float_input, scale, zero_point] -> [quantized_output]
        const auto& output_defs = node->OutputDefs();
        if (Is16BitTensor(output_defs.empty() ? nullptr : output_defs[0]) ||
            (input_defs.size() >= 3 && Is16BitTensor(input_defs[2]))) {
          return true;
        }
      }
    }
  }
  return false;
}

static void DumpOpenVINOEPModel([[maybe_unused]] const std::filesystem::path& onnx_model_path_name,
                                [[maybe_unused]] ONNX_NAMESPACE::ModelProto* model_proto,
                                [[maybe_unused]] const onnxruntime::Node& fused_node) {
#ifndef RELEASE
  if (openvino_ep::backend_utils::IsDebugEnabled()) {
    auto model_name = onnx_model_path_name.empty() ? "unknown.onnx" : onnx_model_path_name.filename();

    const auto& subgraph_name = fused_node.Name();
    size_t dash = subgraph_name.find_last_of("-");
    if (dash != std::string::npos) {
      auto new_name = model_name.stem().string() + subgraph_name.substr(dash, std::string::npos);
      model_name.replace_filename(new_name);
      model_name.replace_extension(".onnx");
    }

    std::fstream dump(model_name, std::ios::out | std::ios::trunc | std::ios::binary);
    model_proto->SerializeToOstream(dump);
  }
#endif
}

// this is a helper function to set the data fields, it duplicates ExternalDataInfo::SetExternalLocationToProto
// but we cannot use that function as it is not part of public provider api.
static void SetExternalDataFields(ONNX_NAMESPACE::TensorProto* proto_init, const void* data_ptr, int64_t data_size) {
  static constexpr const char* ORT_INTERNAL_MEM_INITIALIZER = "*/_ORT_MEM_ADDR_/*";
  auto* external_data = proto_init->mutable_external_data();
  bool found_location = false, found_offset = false, found_length = false;
  const int ext_data_size = external_data->size();
  proto_init->set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation::TensorProto_DataLocation_EXTERNAL);

  for (int j = 0; j < ext_data_size; ++j) {
    auto& ext_entry = external_data->at(j);
    auto& key = *ext_entry.mutable_key();
    if (key == "location") {
      *ext_entry.mutable_value() = ORT_INTERNAL_MEM_INITIALIZER;
      found_location = true;
    } else if (key == "offset") {
      *ext_entry.mutable_value() = std::to_string(reinterpret_cast<uintptr_t>(data_ptr));
      found_offset = true;
    } else if (key == "length") {
      *ext_entry.mutable_value() = std::to_string(data_size);
      found_length = true;
    }
  }

  if (!found_location) {
    auto* new_entry = external_data->Add();
    *new_entry->mutable_key() = "location";
    *new_entry->mutable_value() = ORT_INTERNAL_MEM_INITIALIZER;
  }
  if (!found_offset) {
    auto* new_entry = external_data->Add();
    *new_entry->mutable_key() = "offset";
    *new_entry->mutable_value() = std::to_string(reinterpret_cast<uintptr_t>(data_ptr));
  }
  if (!found_length) {
    auto* new_entry = external_data->Add();
    *new_entry->mutable_key() = "length";
    *new_entry->mutable_value() = std::to_string(data_size);
  }
}

static void ReadExternalDataFields(const ONNX_NAMESPACE::TensorProto* src_init, std::string& location, size_t& offset, size_t& length) {
  // Remove constness as we need to use mutable_external_data() to get the entries to read.
  // The entries themselves are not modified...
  auto& mutable_proto = *const_cast<ONNX_NAMESPACE::TensorProto*>(src_init);
  auto* entry_protos = mutable_proto.mutable_external_data();
  for (int i = 0; i < entry_protos->size(); i++) {
    auto& string_entry_proto{entry_protos->at(i)};
    const auto& pb_key{*(string_entry_proto.mutable_key())};
    const auto& pb_value{*(string_entry_proto.mutable_value())};
    if (pb_key == "location") {
      location = pb_value;
    } else if (pb_key == "offset") {
      const auto res = std::from_chars(pb_value.data(), pb_value.data() + pb_value.size(), offset);
      if (res.ec != std::errc()) {
        std::ostringstream err_msg;
        err_msg << "External data in memory has invalid offset field: "
                << src_init->name() << "], location: " << location
                << ", offset: " << pb_value;
        ORT_THROW(err_msg.str());
      }
    } else if (pb_key == "length") {
      const auto res = std::from_chars(pb_value.data(), pb_value.data() + pb_value.size(), length);
      if (res.ec != std::errc()) {
        std::ostringstream err_msg;
        err_msg << "External data in memory has invalid length field: "
                << src_init->name() << "], location: " << location
                << ", length: " << pb_value;
        ORT_THROW(err_msg.str());
      }
    }
  }
}

std::unique_ptr<ONNX_NAMESPACE::ModelProto>
BackendManager::GetModelProtoFromFusedNode(const onnxruntime::Node& fused_node,
                                           const onnxruntime::GraphViewer& subgraph,
                                           const logging::Logger& logger) const {
  std::chrono::time_point<std::chrono::high_resolution_clock> model_proto_create_start_, model_proto_create_end_;
  if (openvino_ep::backend_utils::IsDebugEnabled()) {
    model_proto_create_start_ = std::chrono::high_resolution_clock::now();
  }

  auto print_model_proto_duration = [&]() {
    if (openvino_ep::backend_utils::IsDebugEnabled()) {
      model_proto_create_end_ = std::chrono::high_resolution_clock::now();
      auto model_proto_create_duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              model_proto_create_end_ - model_proto_create_start_)
              .count();
      LOGS_DEFAULT(INFO) << "[OpenVINO-EP] Model Proto creation took: " << model_proto_create_duration << " ms.";
    }
  };

  [[maybe_unused]] bool enable_ovep_qdq_optimizer = session_context_.enable_qdq_optimizer && IsQDQGraph(subgraph);
  [[maybe_unused]] std::optional<bool> enable_compiler_qdq_optimization = queryOVProperty("NPU_QDQ_OPTIMIZATION", session_context_.device_type);
#if (((OPENVINO_VERSION_MAJOR == 2025) && (OPENVINO_VERSION_MINOR > 0)) || (OPENVINO_VERSION_MAJOR > 2025))
  if (session_context_.device_type.find("NPU") != std::string::npos && session_context_.enable_qdq_optimizer) {
    if (enable_compiler_qdq_optimization.has_value() && enable_compiler_qdq_optimization.value()) {
      LOGS_DEFAULT(INFO) << "[OpenVINO-EP]: Compiler QDQ optimization pass is enabled";
      OVCore::Get()->core.set_property("NPU", {ov::intel_npu::qdq_optimization(true)});
      // disabling OVEP qdq stripping
      // at this stage provider option "enable_qdq_optimizer" is still true but OVEP stripping is (disabled) false
      // as compiler stripping is enabled
      enable_ovep_qdq_optimizer = false;
    } else {
      LOGS_DEFAULT(INFO) << "[OpenVINO-EP]: OVEP QDQ optimization pass is enabled";
    }
  }
#endif

  // Check if the graph is QDQ and has int16 or uint16 quantization
  // If so, we will apply the QDQ scales fix transformation (for GPU device only)
  bool is_qdq_graph_uint16_or_int16 = IsQDQGraphWithUint16OrInt16(subgraph);

  const auto& onnx_model_path_name = subgraph.ModelPath();
  // QDQ stripping enabled only for the NPU and experimentally on the GPU
  if ((session_context_.device_type.find("NPU") != std::string::npos) &&
      (enable_ovep_qdq_optimizer || session_context_.so_share_ep_contexts)) {
    std::unique_ptr<onnxruntime::Model> model;
    Status status = CreateModelWithStrippedQDQNodes(subgraph, logger, session_context_.so_share_ep_contexts, enable_ovep_qdq_optimizer, model, shared_context_.shared_weights);
    auto model_proto = model->ToProto();
    model_proto->set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
    print_model_proto_duration();
    DumpOpenVINOEPModel(onnx_model_path_name, model_proto.get(), fused_node);
    ORT_ENFORCE(status.IsOK(), status.ErrorMessage());
    return model_proto;
  } else if ((session_context_.device_type.find("GPU") != std::string::npos) &&
             is_qdq_graph_uint16_or_int16) {
    // Create a copy of the model
    std::unique_ptr<onnxruntime::Model> model;
    Status status = qdq_scales_fix::Transform(subgraph, logger, model);
    auto model_proto = model->ToProto();
    model_proto->set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
    print_model_proto_duration();
    DumpOpenVINOEPModel(onnx_model_path_name, model_proto.get(), fused_node);
    ORT_ENFORCE(status.IsOK(), status.ErrorMessage());
    return model_proto;
  } else if (IsModelBF16(subgraph)) {
    LOGS_DEFAULT(INFO) << "[OpenVINO-EP] OVEP bfloat16->float16 optimization pass is enabled";
    std::unique_ptr<onnxruntime::Model> model;
    Status status = bfloat16_fix::Transform(subgraph, logger, model);
    auto model_proto = model->ToProto();
    model_proto->set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
    print_model_proto_duration();
    DumpOpenVINOEPModel(onnx_model_path_name, model_proto.get(), fused_node);
    ORT_ENFORCE(status.IsOK(), status.ErrorMessage());
    return model_proto;
  } else {
    LOGS_DEFAULT(INFO) << "[OpenVINO-EP] OVEP QDQ optimization pass is disabled";

    // scan ext initializers:
    std::unordered_map<std::string, std::pair<size_t, size_t>> external_initializers_offset_and_length;
    std::string tempLocation;
    size_t extInitializerTotalSize = 0;
    if (session_context_.has_external_weights) {
      auto allInitializers = subgraph.GetAllInitializedTensors();
      for (auto& [name, tp] : allInitializers) {
        if (utils::HasExternalDataInMemory(*tp)) {
          size_t offset = 0;
          size_t length = 0;
          ReadExternalDataFields(tp, tempLocation, offset, length);
          extInitializerTotalSize += length;
          external_initializers_offset_and_length[name] = {offset, length};
        }
      }
    }

    // when we have external weights in memory, the model proto will actually embed those
    // and bloat the serialized string. We can avoid that by not including the data in the proto
    // but then we have to update those initializers and set the external_data fields to mem_addr tag...
    // proto is limited to 2GB, but let's use 32MB as threshold to be conservative and still gain some memory reductions.
#if (((OPENVINO_VERSION_MAJOR == 2025) && (OPENVINO_VERSION_MINOR > 3)) || (OPENVINO_VERSION_MAJOR > 2025))
    constexpr size_t MAX_EMBEDDED_INITIALIZER_SIZE = 1024 * 1024 * 32;
    const bool include_initializer_data_in_proto = !(session_context_.has_external_weights &&
                                                     external_initializers_offset_and_length.size() > 1 &&
                                                     extInitializerTotalSize >= MAX_EMBEDDED_INITIALIZER_SIZE);
#else
    const bool include_initializer_data_in_proto = true;
#endif


    auto model = subgraph.CreateModel(logger);
    auto model_proto = model->ToProto();
    model_proto->set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
    subgraph.ToProto(*model_proto->mutable_graph(), /*include_initializers*/true,
                     /*include_outer_scope_args*/true, /*execution_order*/0, /*include_initializer_data*/include_initializer_data_in_proto);

    print_model_proto_duration();

    if (!include_initializer_data_in_proto) {
      LOGS(logger, INFO) << "Initializer data is not included in the model proto. Updating metadata..., total size " << extInitializerTotalSize / (1024 * 1024) << " MB in " << external_initializers_offset_and_length.size() << " initializers";
      auto* graph_proto = model_proto->mutable_graph();
      auto* proto_initializers = graph_proto->mutable_initializer();

      std::unordered_map<std::string, ONNX_NAMESPACE::TensorProto*> proto_initializer_map;
      for (int i = 0, n = proto_initializers->size(); i < n; ++i) {
        auto& proto_init = proto_initializers->at(i);
        proto_initializer_map[proto_init.name()] = &proto_init;
      }

      for (const auto& [name, src_init] : subgraph.GetAllInitializedTensors()) {
        auto it = proto_initializer_map.find(name);
        if (it == proto_initializer_map.end())
          continue;

        auto* proto_init = it->second;

        // If the proto initializer is missing data, fill it in
        if (!proto_init->has_raw_data() && src_init->has_raw_data()) {
          *proto_init->mutable_raw_data() = src_init->raw_data();
        }

        // Only set in-memory external_data fields if the data is in memory
        if (src_init->has_raw_data()) {
          LOGS(logger, VERBOSE) << "In-memory initializer RAW: "
                                << src_init->name()
                                << ", data_type: " << src_init->data_type()
                                << ", raw_data size: " << src_init->raw_data().size();
          if (src_init->raw_data().size() > 0)
              SetExternalDataFields(proto_init, src_init->raw_data().data(), src_init->raw_data().size());
          else
              LOGS(logger, VERBOSE) << "Initializer has empty raw_data: skipping initializer '" << src_init->name() << "'...";
        } else if (onnxruntime::utils::HasExternalDataInMemory(*src_init)) {
          auto it_ext = external_initializers_offset_and_length.find(name);
          if (it_ext == external_initializers_offset_and_length.end()) {
            std::ostringstream err_msg;
            err_msg << "Initializer marked as external in memory but missing offset/length info: " << src_init->name();
            ORT_THROW(err_msg.str());
          }
          const size_t offset = it_ext->second.first;
          const size_t length = it_ext->second.second;

          LOGS(logger, VERBOSE) << "In-memory initializer EXT: " << src_init->name() << ", size: " << length;

          SetExternalDataFields(proto_init, (const void*)offset, length);
        } else {
          LOGS(logger, VERBOSE) << "File-based initializer: " << src_init->name() << ", data_type: " << src_init->data_type();
        }
      }
    }

    DumpOpenVINOEPModel(onnx_model_path_name, model_proto.get(), fused_node);

    return model_proto;
  }
}

std::vector<std::vector<int64_t>> GetInputTensorShapes(const Ort::KernelContext& context) {
  const auto input_count = context.GetInputCount();
  std::vector<std::vector<int64_t>> input_shapes;
  input_shapes.reserve(input_count);
  for (size_t i = 0; i < input_count; i++) {
    auto input_tensor = context.GetInput(i);
    auto tensor_shape = input_tensor.GetTensorTypeAndShapeInfo().GetShape();
    input_shapes.push_back(std::move(tensor_shape));
  }
  return input_shapes;
}

std::string MakeMapKeyString(const std::vector<std::vector<int64_t>>& shapes,
                             const std::string& device_type) {
  std::string key;
  key += device_type;
  key += "|";  // separator
  for (auto shape : shapes) {
    for (auto dim : shape) {
      std::ostringstream o;
      o << dim;
      key += o.str();
      key += ",";
    }
    key += "|";
  }
  return key;
}

std::unique_ptr<ONNX_NAMESPACE::ModelProto>
BackendManager::ReWriteInputShapeInfo(const ONNX_NAMESPACE::ModelProto& model_proto,
                                      const std::vector<std::vector<int64_t>>& input_shapes) {
  auto model_copy = ONNX_NAMESPACE::ModelProto::Create();
  std::string proto_str;
  model_proto.SerializeToString(proto_str);
  model_copy->ParseFromString(proto_str);
  auto graph_proto = model_copy->mutable_graph();

  for (size_t i = 0, limit = input_shapes.size(); i < limit; i++) {
    auto g_in_shape = graph_proto->mutable_input(static_cast<int>(i))
                          ->mutable_type()
                          ->mutable_tensor_type()
                          ->mutable_shape();
    g_in_shape->clear_dim();
    const auto& shape = input_shapes[i];
    for (size_t dim = 0, end = shape.size(); dim < end; dim++) {
      g_in_shape->add_dim()->set_dim_value(shape[dim]);
    }
  }
  return model_copy;
}

std::shared_ptr<ONNX_NAMESPACE::ModelProto>
BackendManager::ReWriteBatchDimWithOne(const ONNX_NAMESPACE::ModelProto& model_proto) {
  auto model_copy = std::shared_ptr<ONNX_NAMESPACE::ModelProto>(ONNX_NAMESPACE::ModelProto::Create());
  std::string proto_str;
  model_proto.SerializeToString(proto_str);
  model_copy->ParseFromString(proto_str);
  auto graph_proto = model_copy->mutable_graph();

  for (int i = 0; i < graph_proto->input_size(); i++) {
    ONNX_NAMESPACE::TensorShapeProto* g_in_shape =
        graph_proto->mutable_input(static_cast<int>(i))
            ->mutable_type()
            ->mutable_tensor_type()
            ->mutable_shape();
    g_in_shape->mutable_dim(0)->clear_dim_value();
    g_in_shape->mutable_dim(0)->set_dim_value(1);
  }
  return model_copy;
}

void BackendManager::ValidateInputShapes(const reshape_t& shapes,
                                         const std::vector<const NodeArg*>& graph_inputs) const {
  for (const auto& [tensor_name, requested_shape] : shapes) {
    // Find matching input in graph
    const NodeArg* graph_input = nullptr;
    for (const auto* input : graph_inputs) {
      if (input->Name() == tensor_name) {
        graph_input = input;
        break;
      }
    }

    if (!graph_input) {
      ORT_THROW("Input '" + tensor_name + "' specified in reshape_input does not exist in the graph");
    }

    const ONNX_NAMESPACE::TensorShapeProto* graph_shape = graph_input->Shape();
    if (!graph_shape) {
      ORT_THROW("Graph input '" + tensor_name + "' has no shape information");
    }

    // Check dimensions count matches
    size_t graph_dim_count = graph_shape->dim_size();
    size_t requested_dim_count = requested_shape.get_max_shape().size();

    if (graph_dim_count != requested_dim_count) {
      ORT_THROW("Dimensions mismatch for input '" + tensor_name +
                "': graph expects " + std::to_string(graph_dim_count) +
                " dimensions but reshape_input specifies " +
                std::to_string(requested_dim_count) + " dimensions");
    }
  }
}

void BackendManager::Compute(OrtKernelContext* context) {
  Ort::KernelContext ctx(context);
  std::chrono::high_resolution_clock::time_point start_compute, end_compute;

#ifdef OPENVINO_FIL_ENABLED
  static bool fil_enabled = true;
  if (fil_enabled) {
    start_compute = std::chrono::high_resolution_clock::now();
    LOGS_DEFAULT(INFO) << "Start Compute";
  }
#endif

  // if disable_dynamic_shapes is set to true then execution of dynamic model is done
  // by rewriting the model to static shaped model at runtime based on input shape.
  // disable_dynamic_shapes should be set for devices that don't support dynamic shapes.
  bool need_dynamic_backend = subgraph_context_.has_dynamic_input_shape &&
                              session_context_.disable_dynamic_shapes && !subgraph_context_.is_ep_ctx_graph;

  if (!need_dynamic_backend) {
    concrete_backend_->Infer(context);
  } else {
    std::vector<std::vector<int64_t>> tensor_shapes = GetInputTensorShapes(ctx);
    auto key = MakeMapKeyString(tensor_shapes, session_context_.device_type);
    std::shared_ptr<IBackend> dynamic_backend;

    {
      std::unique_lock<std::mutex> lock(mutex_);
      auto it = backend_map_.find(key);
      if (it != backend_map_.end()) {
        dynamic_backend = it->second;
      }
    }

    if (!dynamic_backend) {
      ptr_stream_t model_stream;
      LOGS_DEFAULT(INFO) << "[OpenVINO-EP] "
                         << "Creating dynamic backend for key: " << key;
      LOGS_DEFAULT(INFO) << "[OpenVINO-EP] "
                         << "Backend created for graph " << subgraph_context_.subgraph_name;
      auto modelproto_with_concrete_shapes = ReWriteInputShapeInfo(*model_proto_, tensor_shapes);
      try {
        dynamic_backend = BackendFactory::MakeBackend(modelproto_with_concrete_shapes,
                                                      session_context_,
                                                      subgraph_context_,
                                                      shared_context_,
                                                      model_stream);
      } catch (const OnnxRuntimeException& ex) {
        // Build option disables fallback to CPU on compilation failures with NPU.
#if defined(OPENVINO_DISABLE_NPU_FALLBACK)
        LOGS_DEFAULT(WARNING) << "Model compilation failed at OV NPU.";
        ORT_THROW(ex.what());
#else
        if (session_context_.device_type.find("NPU") != std::string::npos &&
            !session_context_.so_disable_cpu_ep_fallback) {
          LOGS_DEFAULT(WARNING) << ex.what();
          LOGS_DEFAULT(WARNING) << "Model compilation failed at OV NPU."
                                << "Falling back to OV CPU for execution";
          session_context_.device_type = "CPU";
          session_context_.precision = "FP32";
          key = MakeMapKeyString(tensor_shapes, session_context_.device_type);
          try {
            dynamic_backend = BackendFactory::MakeBackend(modelproto_with_concrete_shapes,
                                                          session_context_,
                                                          subgraph_context_,
                                                          shared_context_,
                                                          model_stream);
          } catch (std::string const& msg) {
            ORT_THROW(msg);
          }
        } else {
          std::string exception_str = ex.what();
          if (session_context_.so_disable_cpu_ep_fallback){
            std::string error_message = "UNKNOWN NPU ERROR";
            std::string error_code = "code 0x0";
            std::regex error_message_pattern(R"(\bZE_\w*\b)");
            std::regex error_code_pattern("code 0x[0-9a-fA-F]+");
            std::smatch matches;
            if (std::regex_search(exception_str, matches, error_message_pattern)) {
              error_message = matches[0];
            }
            if (std::regex_search(exception_str, matches, error_code_pattern)) {
              error_code = matches[0];
            }
            std::string suffix = "\nModel failed to compile on NPU. Enable CPU fallback or try another device.\n" ;
            throw std::runtime_error(error_message + ", " + error_code + suffix);
          }
          else{
            ORT_THROW(exception_str);
          }
        }
#endif
      }
      std::unique_lock<std::mutex> lock(mutex_);
      backend_map_.insert({key, dynamic_backend});
    }

    dynamic_backend->Infer(context);
  }
#ifdef OPENVINO_FIL_ENABLED
  if (fil_enabled) {
    end_compute = std::chrono::high_resolution_clock::now();
    LOGS_DEFAULT(INFO) << "End Compute";
    std::chrono::duration<double> compute_time = end_compute - start_compute;
    std::cout << "Compute Time: " << compute_time.count() << " s" << std::endl;
    fil_enabled = false;  // calculating compute time for first run only
  }
#endif
}

void BackendManager::ShutdownBackendManager() {
  std::unique_lock<std::mutex> lock(mutex_);
  backend_map_.clear();
  concrete_backend_.reset();
}

void BackendManager::RewindKVCache(size_t index) {
  if (concrete_backend_) {
    concrete_backend_->RewindKVCache(index);
  }
}

}  // namespace openvino_ep
}  // namespace onnxruntime
