// Copyright (C) Intel Corporation
// Licensed under the MIT License
#include <filesystem>
#include <utility>
#include <string>
#include <memory>
#include <vector>
#include "core/providers/shared_library/provider_api.h"
#include "core/providers/openvino/openvino_execution_provider.h"
#include "core/providers/openvino/contexts.h"
#include "core/providers/openvino/backend_manager.h"
#include "core/providers/openvino/onnx_ctx_model_helper.h"
#include "core/providers/openvino/ov_versions/capability.h"
#include "core/providers/openvino/qdq_transformations/qdq_stripping.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "openvino/core/version.hpp"
#ifdef USE_OVEP_NPU_MEMORY
#include "core/providers/openvino/ov_allocator.h"
#endif

namespace onnxruntime {
namespace openvino_ep {

std::optional<fs::path> GetExternalWeightFilename(const std::vector<IExecutionProvider::FusedNodeAndGraph>& fused_nodes);

// Parking this code here for now before it's moved to the factory
#if defined OPENVINO_CONFIG_HETERO || defined OPENVINO_CONFIG_MULTI || defined OPENVINO_CONFIG_AUTO
static std::vector<std::string> parseDevices(const std::string& device_string,
                                             const std::vector<std::string>& available_devices) {
  std::string comma_separated_devices = device_string;
  if (comma_separated_devices.find(":") != std::string::npos) {
    comma_separated_devices = comma_separated_devices.substr(comma_separated_devices.find(":") + 1);
  }
  auto devices = split(comma_separated_devices, ',');
  if (devices.size() < 2) {
    print_build_options();
    ORT_THROW("Invalid device string: " + device_string);
  }
  std::set<std::string> dev_options = {"CPU", "GPU", "NPU"};

  for (auto& device : available_devices) {
    if (dev_options.find(device) == dev_options.end()) {
      auto dev_options_update = dev_options.emplace(device);
    }
  }

  for (const std::string& dev : devices) {
    if (!std::count(dev_options.begin(), dev_options.end(), dev)) {
      print_build_options();
      ORT_THROW("Invalid device string: " + device_string);
    }
  }
  return devices;
}
#endif

OpenVINOExecutionProvider::OpenVINOExecutionProvider(const ProviderInfo& info, std::shared_ptr<SharedContext> shared_context)
    : IExecutionProvider{onnxruntime::kOpenVINOExecutionProvider},
      session_context_(info, *GetLogger()),
      shared_context_{shared_context} {
  InitProviderOrtApi();
}

OpenVINOExecutionProvider::~OpenVINOExecutionProvider() {
  for (auto& backend_manager : backend_managers_) {
    backend_manager.ShutdownBackendManager();
  }
  backend_managers_.clear();
  shared_context_.reset();
}

std::vector<std::unique_ptr<ComputeCapability>>
OpenVINOExecutionProvider::GetCapability(const GraphViewer& graph_viewer,
                                         const IKernelLookup& /*kernel_lookup*/,
                                         const GraphOptimizerRegistry& /* graph_optimizer_registry */,
                                         IResourceAccountant* /* resource_accountant */) const {
  std::vector<std::unique_ptr<ComputeCapability>> result;

  // Enable CI Logs
  if (!(GetEnvironmentVar("ORT_OPENVINO_ENABLE_CI_LOG").empty())) {
    std::cout << "In the OpenVINO EP" << std::endl;
  }
  openvino_ep::GetCapability obj(session_context_.ep_ctx_handler,
                                 graph_viewer,
                                 session_context_.device_type,
                                 session_context_.enable_qdq_optimizer);
  result = obj.Execute();
  session_context_.is_wholly_supported_graph = obj.IsWhollySupportedGraph();
  session_context_.has_external_weights = obj.HasExternalWeights();
  return result;
}

common::Status OpenVINOExecutionProvider::Compile(
    const std::vector<FusedNodeAndGraph>& fused_nodes,
    std::vector<NodeComputeInfo>& node_compute_funcs) {
  Status status = Status::OK();

  if (!fused_nodes.empty()) {
    // Assume these properties are constant for all the model subgraphs, otherwise move to SubGraphContext
    const auto& graph_body_viewer_0 = fused_nodes[0].filtered_graph.get();
    session_context_.onnx_model_path_name = graph_body_viewer_0.ModelPath().string();
    session_context_.onnx_opset_version =
        graph_body_viewer_0.DomainToVersionMap().at(kOnnxDomain);
  }

  // Weights can be internal (in source model) or external (in context binary
  // or separate file). If external this block prepares the external weights
  // input stream.
  fs::path external_weights_full_path;
  if (auto filename = GetExternalWeightFilename(fused_nodes)) {
    // Model is using external weights
    auto temp_weights_full_path = session_context_.onnx_model_path_name.parent_path() / filename.value();

    // Initialize external weights with fully qualified path
    ORT_ENFORCE(fs::exists(temp_weights_full_path),
                "Error: Failed to locate weight file at ",
                temp_weights_full_path.string());

    external_weights_full_path = temp_weights_full_path;
  }

  auto& shared_weight_info = shared_context_->shared_weight_info;
  std::unique_ptr<EPCtxBinReader> ep_ctx_bin_reader;
  std::unique_ptr<EPCtxBinWriter> ep_ctx_bin_writer;

  if (session_context_.so_share_ep_contexts && shared_context_->context_binary_file_path.empty()) {
    // Store output context_binary_file_path to use it in subsequent sessions
    shared_context_->context_binary_file_path = session_context_.onnx_model_path_name.stem().string() + "_openvino.bin";
  }

  if (session_context_.so_context_enable) {
    session_context_.ep_ctx_bin_writer = std::make_unique<EPCtxBinWriter>(session_context_.ep_ctx_handler,
                                                                          shared_context_->context_binary_file_path,
                                                                          external_weights_full_path,
                                                                          shared_weight_info);
  }

  for (const FusedNodeAndGraph& fused_node_graph : fused_nodes) {
    // During backend creation, we check if user wants to use precompiled blob onnx model or the original model
    // For precompiled blob, directly load the model instead of compiling the model
    // For original model, check if the user wants to export a model with pre-compiled blob
    auto& backend_manager = backend_managers_.emplace_back(session_context_,
                                                           *shared_context_,
                                                           fused_node_graph,
                                                           external_weights_full_path);

    struct OpenVINOEPFunctionState {
      AllocateFunc allocate_func = nullptr;
      DestroyFunc destroy_func = nullptr;
      AllocatorHandle allocator_handle = nullptr;
      BackendManager& backend_manager;
    };

    NodeComputeInfo compute_info;
    compute_info.create_state_func =
        [&backend_manager](ComputeContext* context, FunctionState* state) {
          OpenVINOEPFunctionState* p = new OpenVINOEPFunctionState{
              .allocate_func = context->allocate_func,
              .destroy_func = context->release_func,
              .allocator_handle = context->allocator_handle,
              .backend_manager = backend_manager};
          *state = static_cast<FunctionState>(p);
          return 0;
        };

    compute_info.compute_func = [](FunctionState state, const OrtApi* /* api */, OrtKernelContext* context) {
      auto function_state = static_cast<OpenVINOEPFunctionState*>(state);
      try {
        function_state->backend_manager.Compute(context);
      } catch (const std::exception& ex) {
        return common::Status(common::ONNXRUNTIME, common::FAIL, ex.what());
      }
      return Status::OK();
    };

    compute_info.release_state_func =
        [](FunctionState state) {
          if (state) {
            OpenVINOEPFunctionState* function_state = static_cast<OpenVINOEPFunctionState*>(state);
            delete function_state;
          }
        };

    node_compute_funcs.push_back(std::move(compute_info));

    if (!status.IsOK()) {
      break;
    }
  }

  // Release session objects meant to have a lifetime only during Compile()
  session_context_.ep_ctx_bin_writer.reset();
  session_context_.ep_ctx_bin_reader.reset();

  if (session_context_.so_stop_share_ep_contexts) {
    shared_context_->clear();
  }

  return status;
}

#ifdef USE_OVEP_NPU_MEMORY
std::vector<AllocatorPtr> OpenVINOExecutionProvider::CreatePreferredAllocators() {
  if (session_context_.device_type.find("NPU") != std::string::npos) {
    AllocatorCreationInfo npu_allocator_info{
        [this](OrtDevice::DeviceId device_id) {
          return std::make_unique<OVRTAllocator>(
              OVCore::Get()->core,
              OrtDevice::NPU,
              device_id,
              OpenVINO_RT_NPU);
        },
        0,
    };

    // fill in allocator
    return std::vector<AllocatorPtr>{CreateAllocator(npu_allocator_info)};
  } else {
    return std::vector<AllocatorPtr>{};
  }
}
#endif

common::Status OpenVINOExecutionProvider::SetEpDynamicOptions(gsl::span<const char* const> keys,
                                                              gsl::span<const char* const> values) {
  std::string workload_type = "";
  // Ensure the number of keys and values match
  if (keys.size() != values.size()) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Mismatched keys and values sizes.");
  }

  for (size_t i = 0; i < keys.size(); ++i) {
    std::string key = keys[i];
    std::string value = values[i];

    if (key == kOrtEpDynamicOptionsWorkloadType) {
      if (value == "Efficient") {
        workload_type = "EFFICIENT";
      } else if (value == "Default") {
        workload_type = "DEFAULT";
      } else {
        LOGS_DEFAULT(WARNING) << "Unknown workload_type - ignoring " << key << "/" << value;
        LOGS_DEFAULT(WARNING) << "Supported types are 'Efficient' and 'Default' \n";
      }
      if (workload_type != "") {
        LOGS_DEFAULT(INFO) << "SetEpDynamicOptions - modifying: " << key << "/" << value;
        for (auto& backend : backend_managers_) {
          ov::CompiledModel& ov_compiled_model = backend.GetOVCompiledModel();
          ov_compiled_model.set_property(ov::workload_type(workload_type));
        }
      }
    } else {
      // Handle unknown options
      LOGS_DEFAULT(WARNING) << "Unknown key/value pair - ignoring " << key << "/" << value;
    }
  }
  return Status::OK();
}

const InlinedVector<const Node*> OpenVINOExecutionProvider::GetEpContextNodes() const {
  return session_context_.ep_ctx_handler.GetEPCtxNodes();
}

// Returns the location string from the first external initializer nodes found or nullopt if none found
std::optional<fs::path> GetExternalWeightFilename(const std::vector<IExecutionProvider::FusedNodeAndGraph>& fused_nodes) {
  auto get_external_location = [](const ONNX_NAMESPACE::TensorProto& proto) -> std::optional<std::string> {
    using mutable_proto_t = ONNX_NAMESPACE::TensorProto*;
    auto& mutable_proto = *const_cast<mutable_proto_t>(&proto);
    auto* entry_protos = mutable_proto.mutable_external_data();

    if (proto.has_data_location() && proto.data_location() == ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL) {
      for (int i = 0; i < entry_protos->size(); i++) {
        auto& string_entry_proto{entry_protos->at(i)};
        const auto& pb_key{*(string_entry_proto.mutable_key())};
        const auto& pb_value{*(string_entry_proto.mutable_value())};
        if (pb_key == "location") {
          return std::make_optional<std::string>(pb_value);
        }
      }
    }

    return std::nullopt;
  };

  for (const auto& fused_node_graph : fused_nodes) {
    const GraphViewer& graph = fused_node_graph.filtered_graph;
    // Handle constant initializers
    auto& initializers = graph.GetAllInitializedTensors();
    for (const auto& it : initializers) {
      if (auto result = get_external_location(*it.second)) {
        return result;
      }
    }

    // Handle outer-scope constant initializers
    for (auto& node_idx : graph.GetNodesInTopologicalOrder()) {
      const auto& node = graph.GetNode(node_idx);
      for (const auto& input : node->InputDefs()) {
        if (graph.IsConstantInitializer(input->Name(), true)) {
          const auto& initializer_tensor = *graph.GetConstantInitializer(input->Name(), true);

          if (auto result = get_external_location(initializer_tensor)) {
            return result;
          }
        }
      }
    }
  }

  return std::nullopt;
}

}  // namespace openvino_ep
}  // namespace onnxruntime
