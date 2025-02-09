#include <algorithm>
#include "core/providers/openvino/openvino_parser_utils.h"
#include "core/providers/shared_library/provider_api.h"

namespace onnxruntime {
namespace openvino_ep {


shape_t OpenVINOParserUtils::ParseInputShape(const std::string& reshape_input_definition) {
  shape_t parsed_shape_map;
  std::string unparsed_definition = reshape_input_definition;

  while (!unparsed_definition.empty()) {
    // Find the next shape definition brakcet
    auto shape_start_bracket = unparsed_definition.find_first_of('[');
    if (shape_start_bracket == std::string::npos) {
      ORT_THROW("Malformed input: missing opening bracket '[' in: " + unparsed_definition);
    }
    // Extract the tensor name
    std::string tensor_name = unparsed_definition.substr(0, shape_start_bracket);
    // Remove the leading/trailing whitespaces
    tensor_name.erase(0, tensor_name.find_first_not_of("\t"));
    tensor_name.erase(tensor_name.find_last_not_of("\t") + 1);

    if (tensor_name.empty()) {
      ORT_THROW("Empty tensor name provided in rehsape_input parameter");
    }

    // Closing bracket for current shape definition
    auto shape_end_bracket = unparsed_definition.find_first_of(']', shape_start_bracket);

    if (shape_end_bracket == std::string::npos || shape_end_bracket < shape_start_bracket) {
      ORT_THROW("Missing closing bracket ']' for tensor: " + tensor_name);
    }

    // Extract shape dimensions string
    std::string shape_dimension_str = unparsed_definition.substr(shape_start_bracket + 1,
                                                                 shape_end_bracket - shape_start_bracket - 1);
    std::vector<ov::Dimension> dimension_values;
    std::stringstream dimension_stream(shape_dimension_str);
    std::string dimension_token;

    while (std::getline(dimension_stream, dimension_token, ',')) {
      // Remove leading/trailing whitespaces
      dimension_token.erase(0, dimension_token.find_first_not_of("\t"));
      dimension_token.erase(dimension_token.find_last_not_of("\t") + 1);

      // Check if dimension is a range
      size_t range_separator_pos = dimension_token.find("..");
      if (range_separator_pos != std::string::npos) {
        std::string range_start_str = dimension_token.substr(0, range_separator_pos);
        std::string range_end_str = dimension_token.substr(range_separator_pos + 2);

        // Remove leading/trailing spaced
        range_start_str.erase(0, range_start_str.find_first_not_of("\t"));
        range_start_str.erase(range_start_str.find_last_not_of("\t") + 1);
        range_end_str.erase(0, range_end_str.find_first_not_of("\t"));
        range_end_str.erase(range_end_str.find_last_not_of("\t") + 1);

        if (range_start_str.empty() || range_end_str.empty() ||
            !std::all_of(range_start_str.begin(), range_start_str.end(), ::isdigit) ||
            !std::all_of(range_end_str.begin(), range_end_str.end(), ::isdigit)) {
          ORT_THROW("Invalid dimension range format: " + dimension_token + " for tensor: " + tensor_name);
        }

        int range_start = std::stoi(range_start_str);
        int range_end = std::stoi(range_end_str);

        if (range_start > range_end) {
          ORT_THROW("Invalid dimension range (start > end) for tensor: " + tensor_name);
        }

        dimension_values.emplace_back(ov::Dimension(range_start, range_end));
      } else {
        // Handle single dimension value
        if (dimension_token.empty() ||
            !std::all_of(dimension_token.begin(), dimension_token.end(), ::isdigit)) {
          ORT_THROW("Invalid dimension value: " + dimension_token + " for tensor: " + tensor_name);
        }
        dimension_values.emplace_back(std::stoi(dimension_token));
      }
    }

    // Store parsed shape in result map
    parsed_shape_map[tensor_name] = ov::PartialShape(dimension_values);
    // Update reminaing unparsed string
    unparsed_definition = unparsed_definition.substr(shape_end_bracket + 1);
    if (!unparsed_definition.empty() && unparsed_definition.front() == ',') {
      unparsed_definition = unparsed_definition.substr(1);
    }
    // Remove leading whitespaces
    unparsed_definition.erase(0, unparsed_definition.find_first_not_of("\t"));
  }
  return parsed_shape_map;
}

std::string OpenVINOParserUtils::ParsePrecision(const ProviderOptions& provider_options,
                                                std::string& device_type,
                                                const std::string& option_name) {
  using DeviceName = std::string;
  using DefaultValue = std::string;
  using ValidValues = std::list<std::string>;
  using foo = std::pair<DefaultValue, ValidValues>;
  using ParserHelper = std::map<DeviceName, foo>;

  ParserHelper helper = {
      {"GPU", {"FP16", {"FP16", "FP32", "ACCURACY"}}},
      {"NPU", {"FP16", {"FP16", "ACCURACY"}}},
      {"CPU", {"FP32", {"FP32", "ACCURACY"}}},
  };

  std::set<std::string> deprecated_device_types = {
      "CPU_FP32", "GPU_FP32", "GPU.0_FP32", "GPU.1_FP32", "GPU_FP16",
      "GPU.0_FP16", "GPU.1_FP16"};

  bool is_composite = device_type.find(':') != std::string::npos;  // FOR devices AUTO:,HETERO:,MULTI:

  if (provider_options.contains(option_name)) {
    const auto& precision = provider_options.at(option_name);

    if (is_composite) {
      std::set<std::string> allowed_precisions = {"FP16", "FP32", "ACCURACY"};
      if (allowed_precisions.contains(precision)) {
        return precision;
      } else {
        ORT_THROW("[ERROR] [OpenVINO] Unsupported inference precision is selected. ",
                  precision, ".\n");
      }
    } else {
      if (helper.contains(device_type)) {
        auto const& valid_values = helper[device_type].second;

        if (precision == "ACCURACY") {
          return valid_values.back();  // Return highest supported precision
        } else {
          if (std::find(valid_values.begin(), valid_values.end(), precision) != valid_values.end()) {
            return precision;  // Return precision selected if valid
          } else {
            auto value_iter = valid_values.begin();
            std::string valid_values_joined = *value_iter;
            // Append 2nd and up, if only one then ++value_iter is same as end()
            for (++value_iter; value_iter != valid_values.end(); ++value_iter) {
              valid_values_joined += ", " + *value_iter;
            }

            ORT_THROW("[ERROR] [OpenVINO] Unsupported inference precision is selected. ",
                      device_type, " only supports", valid_values_joined, ".\n");
          }
        }
      } else if (deprecated_device_types.contains(device_type)) {
        LOGS_DEFAULT(WARNING)
            << "[OpenVINO] Selected 'device_type' " + device_type + " is deprecated. \n"
            << "Update the 'device_type' to specified types 'CPU', 'GPU', 'GPU.0', "
            << "'GPU.1', 'NPU' or from HETERO/MULTI/AUTO options and set 'precision' separately. \n";
        auto delimit = device_type.find("_");
        device_type = device_type.substr(0, delimit);
        return device_type.substr(delimit + 1);
      } else {
        ORT_THROW("[ERROR] [OpenVINO] Unsupported device type provided: ",
                  device_type, "\n");
      }
    }
  } else {
    if (device_type.find("NPU") != std::string::npos || device_type.find("GPU") != std::string::npos) {
      return "FP16";
    } else if (device_type.find("CPU") != std::string::npos) {
      return "FP32";
    } else {
      ORT_THROW("[ERROR] [OpenVINO] Unsupported device is selected", device_type, "\n");
    }
  }
}

}  // namespace openvino_ep
}  // namespace onnxruntime
