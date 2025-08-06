// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// onnxruntime dependencies
#include <core/session/onnxruntime_c_api.h>
#include <random>
#include "command_args_parser.h"
#include "performance_runner.h"
#include <google/protobuf/stubs/common.h>

using namespace onnxruntime;
const OrtApi* g_ort = NULL;

#ifdef _WIN32
int real_main(int argc, wchar_t* argv[]) {
#else
int real_main(int argc, char* argv[]) {
#endif
  auto registration_name = "OpenVINOExecutionProvider";
  g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  perftest::PerformanceTestConfig test_config;
  if (!perftest::CommandLineParser::ParseArguments(test_config, argc, argv)) {
    perftest::CommandLineParser::ShowUsage();
    return -1;
  }
  Ort::Env env{nullptr};
  {
    bool failed = false;
    ORT_TRY {
      OrtLoggingLevel logging_level = test_config.run_config.f_verbose
                                          ? ORT_LOGGING_LEVEL_VERBOSE
                                          : ORT_LOGGING_LEVEL_WARNING;
      env = Ort::Env(logging_level, "Default");

      const ORTCHAR_T* library_path = ORT_TSTR("onnxruntime_providers_openvino.dll");
      env.RegisterExecutionProviderLibrary(registration_name, library_path);
    }
    ORT_CATCH(const Ort::Exception& e) {
      ORT_HANDLE_EXCEPTION([&]() {
        fprintf(stderr, "Error creating environment: %s \n", e.what());
        failed = true;
      });
    }

    if (failed) {
      env.UnregisterExecutionProviderLibrary(registration_name);
      return -1;
    }
  }
  std::random_device rd;
  {
    perftest::PerformanceRunner perf_runner(env, test_config, rd);

    // Exit if user enabled -n option so that user can measure session creation time
    if (test_config.run_config.exit_after_session_creation) {
      perf_runner.LogSessionCreationTime();
      // perf_runner destructor will be called when we exit this scope
    } else {
      auto status = perf_runner.Run();
      if (!status.IsOK()) {
        printf("Run failed:%s\n", status.ErrorMessage().c_str());
        env.UnregisterExecutionProviderLibrary(registration_name);
        return -1;
      }
      perf_runner.SerializeResult();
    }
  }
 env.UnregisterExecutionProviderLibrary(registration_name);

  return 0;
}

#ifdef _WIN32
int wmain(int argc, wchar_t* argv[]) {
#else
int main(int argc, char* argv[]) {
#endif
  int retval = -1;
  ORT_TRY {
    retval = real_main(argc, argv);
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      fprintf(stderr, "%s\n", ex.what());
      retval = -1;
    });
  }

  ::google::protobuf::ShutdownProtobufLibrary();

  return retval;
}
