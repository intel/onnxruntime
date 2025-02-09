#pragma once

#include <list>
#include <map>
#include <set>
#include <string>
#include <utility>

#include "core/framework/provider_options.h"
#include "core/providers/openvino/contexts.h"

namespace onnxruntime {
namespace openvino_ep {

class OpenVINOParserUtils {
 public:
  static std::string ParsePrecision(const ProviderOptions& provider_options,
                                    std::string& device_type,
                                    const std::string& option_name);

  static shape_t ParseInputShape(const std::string& reshape_input_definition);

};

}  // namespace openvino_ep
}  // namespace onnxruntime
