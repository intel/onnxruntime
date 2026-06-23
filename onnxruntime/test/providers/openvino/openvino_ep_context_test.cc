// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <filesystem>
#include <string>

#include "core/framework/provider_options.h"
#include "core/framework/tensor_shape.h"
#include "core/common/float16.h"

#include "test/util/include/test_utils.h"
#include "test/util/include/test/test_environment.h"
#include "test/util/include/default_providers.h"
#include "test/unittest_util/qdq_test_utils.h"

#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/session/inference_session.h"
#include "core/graph/model_saving_options.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::logging;

extern std::unique_ptr<Ort::Env> ort_env;

class OVEPEPContextTests : public ::testing::Test {
};

namespace onnxruntime {
namespace test {

// Test if folder path given to ep_context_file_path throws an error
TEST_F(OVEPEPContextTests, OVEPEPContextFolderPath) {
  Ort::SessionOptions sessionOptions;
  std::unordered_map<std::string, std::string> ov_options;

  // The below line could fail the test in non NPU platforms.Commenting it out so that the device used for building OVEP will be used.
  // ov_options["device_type"] = "NPU";

  const std::unordered_map<std::string, int> domain_to_version = {{"", 13}, {kMSDomain, 1}};

  auto& logging_manager = DefaultLoggingManager();
  logging_manager.SetDefaultLoggerSeverity(logging::Severity::kERROR);

  onnxruntime::Model model("OVEP_Test_Model", false, ModelMetaData(), PathString(),
                           IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                           logging_manager.DefaultLogger());

  ASSERT_STATUS_OK(model.MainGraph().Resolve());

  // Serialize the model to a string.
  std::string model_data;
  model.ToProto().SerializeToString(&model_data);

  const auto model_data_span = AsByteSpan(model_data.data(), model_data.size());

  const std::string ep_context_file_path = "./ep_context_folder_path/";

  sessionOptions.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
  sessionOptions.AddConfigEntry(kOrtSessionOptionEpContextFilePath, ep_context_file_path.c_str());
  sessionOptions.AppendExecutionProvider_OpenVINO_V2(ov_options);

  try {
    Ort::Session session(*ort_env, model_data_span.data(), model_data_span.size(), sessionOptions);
    FAIL();  // Should not get here!
  } catch (const Ort::Exception& excpt) {
    ASSERT_EQ(excpt.GetOrtErrorCode(), ORT_INVALID_ARGUMENT);
    ASSERT_THAT(excpt.what(), testing::HasSubstr("context_file_path should not point to a folder."));
  }
}

namespace {

// Layout must match header_t in core/providers/openvino/ov_bin_manager.cc.
struct OvBinHeader {
  uint64_t magic;
  uint64_t version;
  uint64_t header_size;
  uint64_t bson_start_offset;
  uint64_t bson_size;
};

// "OVEP_BIN" in little-endian. Must match kMagicNumber in ov_bin_manager.cc.
constexpr uint64_t kOvBinMagic = 0x4E49425F5045564FULL;

// Builds the byte payload of an OVEP_BIN blob containing only a header. The
// header is valid (correct magic/version/size) but advertises a BSON region
// whose size is far larger than the actual payload, simulating a corrupted or
// malicious EP-context cache.
std::string MakeBinBlobWithOversizedBson(uint64_t bson_size) {
  OvBinHeader header{};
  header.magic = kOvBinMagic;
  header.version = 1;  // BinVersion::current
  header.header_size = sizeof(OvBinHeader);
  header.bson_start_offset = sizeof(OvBinHeader);
  header.bson_size = bson_size;

  return std::string(reinterpret_cast<const char*>(&header), sizeof(header));
}

// Serializes a synthetic model containing a single embedded-mode EPContext node
// whose "source" is the OpenVINO EP and whose "ep_cache_context" carries the
// supplied OVEP_BIN payload. Loading this model drives BinManager::Deserialize.
std::string MakeEmbeddedEPContextModel(const std::string& ep_cache_context) {
  ModelProto model;
  model.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  auto* opset = model.add_opset_import();
  opset->set_domain("");
  opset->set_version(13);
  auto* ms_opset = model.add_opset_import();
  ms_opset->set_domain(kMSDomain);
  ms_opset->set_version(1);

  auto* graph = model.mutable_graph();
  graph->set_name("OVEP_BinDeserialize_Test");

  auto* input = graph->add_input();
  input->set_name("input");
  auto* input_type = input->mutable_type()->mutable_tensor_type();
  input_type->set_elem_type(TensorProto_DataType_FLOAT);
  input_type->mutable_shape()->add_dim()->set_dim_value(1);
  input_type->mutable_shape()->add_dim()->set_dim_value(3);

  auto* output = graph->add_output();
  output->set_name("output");
  auto* output_type = output->mutable_type()->mutable_tensor_type();
  output_type->set_elem_type(TensorProto_DataType_FLOAT);
  output_type->mutable_shape()->add_dim()->set_dim_value(1);
  output_type->mutable_shape()->add_dim()->set_dim_value(3);

  auto* node = graph->add_node();
  node->set_op_type("EPContext");
  node->set_domain(kMSDomain);
  node->set_name("ep_context_node");
  node->add_input("input");
  node->add_output("output");

  auto* attr_embed = node->add_attribute();
  attr_embed->set_name("embed_mode");
  attr_embed->set_type(AttributeProto_AttributeType_INT);
  attr_embed->set_i(1);

  auto* attr_main = node->add_attribute();
  attr_main->set_name("main_context");
  attr_main->set_type(AttributeProto_AttributeType_INT);
  attr_main->set_i(1);

  auto* attr_cache = node->add_attribute();
  attr_cache->set_name("ep_cache_context");
  attr_cache->set_type(AttributeProto_AttributeType_STRING);
  attr_cache->set_s(ep_cache_context);

  auto* attr_source = node->add_attribute();
  attr_source->set_name("source");
  attr_source->set_type(AttributeProto_AttributeType_STRING);
  attr_source->set_s("OpenVINOExecutionProvider");

  auto* attr_partition = node->add_attribute();
  attr_partition->set_name("partition_name");
  attr_partition->set_type(AttributeProto_AttributeType_STRING);
  attr_partition->set_s("OVEP_BinDeserialize_Test");

  std::string model_data;
  model.SerializeToString(&model_data);
  return model_data;
}

}  // namespace

// Regression test for the unbounded-allocation hardening in
// BinManager::DeserializeImpl. A crafted EP-context blob advertises a BSON
// region whose size dwarfs the actual payload. Deserialization must reject it
// with a bounded, descriptive error instead of attempting a huge allocation.
TEST_F(OVEPEPContextTests, OVEPBinDeserializeRejectsOversizedBson) {
  Ort::SessionOptions session_options;
  std::unordered_map<std::string, std::string> ov_options;
  // Empty options -> use the device the OVEP build targets (mirrors other tests
  // in this file). Skip the test entirely if no OpenVINO device is available.
  try {
    session_options.AppendExecutionProvider_OpenVINO_V2(ov_options);
  } catch (const Ort::Exception&) {
    GTEST_SKIP() << "OpenVINO device not available on this machine";
  }

  auto& logging_manager = DefaultLoggingManager();
  logging_manager.SetDefaultLoggerSeverity(logging::Severity::kERROR);

  // bson_start_offset (40) + bson_size (1 TiB) is far beyond the ~40 byte blob.
  const std::string malicious_blob = MakeBinBlobWithOversizedBson(uint64_t{1} << 40);
  const std::string model_data = MakeEmbeddedEPContextModel(malicious_blob);

  try {
    Ort::Session session(*ort_env, model_data.data(), model_data.size(), session_options);
    FAIL() << "Expected deserialization of an oversized-BSON blob to throw.";
  } catch (const Ort::Exception& excpt) {
    // The new bounds check produces "BSON region out of bounds ...", which
    // BinManager::Deserialize wraps with a "Could not deserialize" message.
    // Asserting on "out of bounds" confirms the bounds check fired (rather than
    // a std::bad_alloc/length_error from an unbounded allocation attempt).
    ASSERT_THAT(excpt.what(), testing::HasSubstr("out of bounds"));
  }
}

}  // namespace test
}  // namespace onnxruntime
