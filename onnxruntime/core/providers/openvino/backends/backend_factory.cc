// Copyright (C) Intel Corporation
// Licensed under the MIT License


#include <memory>
#include "core/providers/shared_library/provider_api.h"
#include "core/providers/openvino/contexts.h"
#include "core/providers/openvino/ibackend.h"
#include "core/providers/openvino/backends/basic_backend.h"


namespace onnxruntime {
namespace openvino_ep {


std::shared_ptr<IBackend>
BackendFactory::MakeBackend(std::unique_ptr<ONNX_NAMESPACE::ModelProto>& model_proto,
                            SessionContext& session_context,
                            const SubGraphContext& subgraph_context,
                            SharedContext& shared_context,
                            ptr_stream_t& model_stream) {
  LOGS_DEFAULT(INFO) << "[OpenVINO-EP] DEBUG: BackendFactory::MakeBackend entered";
  LOGS_DEFAULT(INFO) << "[OpenVINO-EP] DEBUG: Device type: " << session_context.device_type;

  std::string type = session_context.device_type;

  if (type == "CPU" || type.find("GPU") != std::string::npos ||
      type.find("NPU") != std::string::npos ||
      type.find("HETERO") != std::string::npos ||
      type.find("MULTI") != std::string::npos ||
      type.find("AUTO") != std::string::npos) {

    LOGS_DEFAULT(INFO) << "[OpenVINO-EP] DEBUG: Device type '" << type << "' matched, creating BasicBackend";

    std::shared_ptr<IBackend> concrete_backend_;

    try {
      LOGS_DEFAULT(INFO) << "[OpenVINO-EP] DEBUG: About to call std::make_shared<BasicBackend>";
      LOGS_DEFAULT(INFO) << "[OpenVINO-EP] DEBUG: Subgraph name: " << subgraph_context.subgraph_name;

      concrete_backend_ = std::make_shared<BasicBackend>(model_proto,
                                                          session_context,
                                                          subgraph_context,
                                                          shared_context,
                                                          model_stream);

      LOGS_DEFAULT(INFO) << "[OpenVINO-EP] DEBUG: std::make_shared<BasicBackend> completed successfully";

    } catch (const std::string& msg) {
      LOGS_DEFAULT(ERROR) << "[OpenVINO-EP] DEBUG: Caught string exception in BackendFactory: " << msg;
      ORT_THROW(msg);
    } catch (const char* msg) {
      LOGS_DEFAULT(ERROR) << "[OpenVINO-EP] DEBUG: Caught const char* exception in BackendFactory: " << msg;
      ORT_THROW(std::string(msg));
    } catch (const std::exception& ex) {
      LOGS_DEFAULT(ERROR) << "[OpenVINO-EP] DEBUG: Caught std::exception in BackendFactory: " << ex.what();
      throw;
    } catch (...) {
      LOGS_DEFAULT(ERROR) << "[OpenVINO-EP] DEBUG: Caught unknown exception type in BackendFactory";
      throw;
    }

    LOGS_DEFAULT(INFO) << "[OpenVINO-EP] DEBUG: Returning concrete_backend_ from BackendFactory";
    LOGS_DEFAULT(INFO) << "[OpenVINO-EP] DEBUG: concrete_backend_ is " << (concrete_backend_ ? "NOT NULL" : "NULL");

    return concrete_backend_;

  } else {
    LOGS_DEFAULT(ERROR) << "[OpenVINO-EP] DEBUG: Unknown backend type: " << type;
    ORT_THROW("[OpenVINO-EP] Backend factory error: Unknown backend type: " + type);
  }
}


}  // namespace openvino_ep
}  // namespace onnxruntime
