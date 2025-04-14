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
      session_context_(info),
      shared_context_{shared_context},
      ep_ctx_handle_{session_context_.openvino_sdk_version, *GetLogger()} {
  InitProviderOrtApi();
}

OpenVINOExecutionProvider::~OpenVINOExecutionProvider() {
  for (auto& backend_manager : backend_managers_) {
    backend_manager.ShutdownBackendManager();
  }
  backend_managers_.clear();
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
  openvino_ep::GetCapability obj(ep_ctx_handle_,
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
  auto& logger = *GetLogger();
  Status status = Status::OK();
  auto &sb = shared_context_.shared_weights.shared_bin_file;
  if (!fused_nodes.empty()) {
    // Assume these properties are constant for all the model subgraphs, otherwise move to SubGraphContext
    const auto& graph_body_viewer_0 = fused_nodes[0].filtered_graph.get();
    session_context_.onnx_model_path_name = graph_body_viewer_0.ModelPath().string();
    session_context_.onnx_opset_version =
        graph_body_viewer_0.DomainToVersionMap().at(kOnnxDomain);

    if (session_context_.so_share_ep_contexts){
      if(session_context_.so_context_file_path.empty()) {
        sb.shared_bin_filename = session_context_.onnx_model_path_name.parent_path() / "metadata.bin";
      } else {
        sb.shared_bin_filename = session_context_.so_context_file_path.parent_path() / "metadata.bin";
      }
      std::cout << sb.shared_bin_filename << std::endl;
      sb.openBinFile(sb.shared_bin_filename);
    }
  }


  // Temporary code to read metadata before it moves to the .bin

  auto& subgraph_metadata = shared_context_.shared_weights.subgraph_metadata;
  auto& metadata = shared_context_.shared_weights.metadata;
  if (session_context_.so_share_ep_contexts) {
    // Metadata is always read from model location, this could be a source or epctx model
    // std::ifstream file(sb.shared_bin_filename, std::ios::binary);
    auto &header = shared_context_.shared_weights.header_;
    auto &footer = shared_context_.shared_weights.footer_;
    if(sb.bin_file_.is_open()) {
      auto header_size = sizeof(SharedContext::SharedWeights::Header);
      std::cout << " sb.bin_size_ " << sb.bin_size_ << std::endl;
      std::cout << " header_size " << header_size << std::endl;
      if(sb.bin_size_ > header_size){
        sb.bin_file_.read(reinterpret_cast<char*>(&header), header_size);
        std::cout << " Footer offset from header = " << header.footer_offset << std::endl;
      }
      std::cout << " file position after reading header " << sb.bin_file_.tellp() << std::endl;
      auto footer_size = sizeof(SharedContext::SharedWeights::Footer);
      std::cout << " footer_size " << footer_size  << std::endl;
      if(header.footer_offset < sb.bin_size_ && footer_size <= sb.bin_size_ &&
        (header.footer_offset <= sb.bin_size_ - footer_size)) {
        sb.bin_file_.seekp(header.footer_offset, std::ios::beg);
        sb.bin_file_.read(reinterpret_cast<char*>(&footer), footer_size);
        std::cout << " subgraph metadata offset from footer = " << footer.subgraph_offset << std::endl;
        std::cout << " subgraph metadata length from footer = " << footer.subgraph_length << std::endl;
        std::cout << " metadata offset from footer = " << footer.metadata_offset << std::endl;
        std::cout << " metadata length from footer = " << footer.metadata_length << std::endl;
      }
      if (footer.subgraph_offset < sb.bin_size_ && footer.subgraph_length <= sb.bin_size_ &&
        (footer.subgraph_offset <= sb.bin_size_ - footer.subgraph_length)) {
        sb.bin_file_.seekp(footer.subgraph_offset, std::ios::beg);
        shared_context_.shared_weights.subgraph_metadata_.readSubgraphDataFromBinaryFile(shared_context_, subgraph_metadata);
        for (const auto& [key, value] : subgraph_metadata){
          std::cout << key.name << std::endl;
          std::cout << value.epctx_offset << std::endl;
          std::cout << value.epctx_length << std::endl;
        }
      }
      if (footer.metadata_offset < sb.bin_size_ && footer.metadata_length <= sb.bin_size_ &&
        (footer.metadata_offset <= sb.bin_size_ - footer.metadata_length)) {
        sb.bin_file_.seekp(footer.metadata_offset, std::ios::beg);
        shared_context_.shared_weights.metadata_.readMetadataFromBinaryFile(shared_context_, metadata);
        for (const auto& [key, value] : metadata){
          std::cout << key.name << std::endl;
          std::cout << value.location << std::endl;
          std::cout << value.data_offset << std::endl;
          std::cout << value.element_type << std::endl;
          std::cout << value.size << std::endl;
          for (const auto& dim : value.dimensions) {
            std::cout << dim << ", ";
          }
          std::cout << std::endl;
      }
      // exit(1);
    }
    }
  }

  struct OpenVINOEPFunctionState {
    AllocateFunc allocate_func = nullptr;
    DestroyFunc destroy_func = nullptr;
    AllocatorHandle allocator_handle = nullptr;
    BackendManager& backend_manager;
  };

  auto &header = shared_context_.shared_weights.header_;
  if(sb.bin_file_.is_open()) {
    sb.bin_file_.write(reinterpret_cast<char*>(&header), sizeof(SharedContext::SharedWeights::Header));
  }
  std::cout << "Current offset after header = " << sb.bin_file_.tellp() << std::endl;
  for (const FusedNodeAndGraph& fused_node_graph : fused_nodes) {
    const GraphViewer& graph_body_viewer = fused_node_graph.filtered_graph;
    const Node& fused_node = fused_node_graph.fused_node;

    NodeComputeInfo compute_info;

    // During backend creation, we check if user wants to use precompiled blob onnx model or the original model
    // For precompiled blob, directly load the model instead of compiling the model
    // For original model, check if the user wants to export a model with pre-compiled blob

    auto& backend_manager = backend_managers_.emplace_back(session_context_,
                                                           *shared_context_,
                                                           fused_node,
                                                           graph_body_viewer,
                                                           logger,
                                                           ep_ctx_handle_);

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

  if (session_context_.so_share_ep_contexts) {
    auto footer = shared_context_.shared_weights.footer_;
    auto &bin_file = sb.bin_file_;
    if (bin_file.is_open()) {
      footer.subgraph_offset = bin_file.tellp();
      // bin_file << subgraph_metadata;
      shared_context_.shared_weights.subgraph_metadata_.writeSubgraphDataToBinaryFile(shared_context_, subgraph_metadata);
      footer.metadata_offset = bin_file.tellp();
      footer.subgraph_length = static_cast<size_t>(footer.metadata_offset - footer.subgraph_offset);
      // footer.subgraph_length = subgraph_metadata.size();
      // bin_file << std::endl;
      shared_context_.shared_weights.metadata_.writeMetadataToBinaryFile(shared_context_, metadata);
      header.footer_offset = bin_file.tellp();
      footer.metadata_length = static_cast<size_t>(header.footer_offset - footer.metadata_offset);
      // footer.metadata_length = metadata.size();
      // bin_file << std::endl;

      std::cout << " header.footer_offset " << header.footer_offset << std::endl;
      std::cout << " subgraph metadata offset from footer = " << footer.subgraph_offset << std::endl;
      std::cout << " subgraph metadata length from footer = " << footer.subgraph_length << std::endl;
      std::cout << " metadata offset from footer = " << footer.metadata_offset << std::endl;
      std::cout << " metadata length from footer = " << footer.metadata_length << std::endl;
      // Write footer to the bin file
      bin_file.write(reinterpret_cast<char*>(&footer), sizeof(SharedContext::SharedWeights::Footer));
      // Update header with Footer offset at the end
      // bin_file.seekp(sizeof(uint32_t), std::ios::beg);
      bin_file.seekp(0, std::ios::beg);
      std::cout << " file ptr after header  seekp = " << bin_file.tellp() << std::endl;
      std::cout << " header.footer offset = " << header.footer_offset << std::endl;
      bin_file.write(reinterpret_cast<char*>(&header), sizeof(SharedContext::SharedWeights::Header));
      // bin_file << header.footer_offset;
      std::cout << " file ptr after updating header = " << bin_file.tellp() << std::endl;
    }
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
  return ep_ctx_handle_.GetEPCtxNodes();
}

}  // namespace openvino_ep
}  // namespace onnxruntime
