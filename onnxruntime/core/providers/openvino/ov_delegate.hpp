// Copyright (C) Intel Corporation
// Licensed under the MIT License

#pragma once
#include "core/session/onnxruntime_c_api.h"
#include "core/common/common.h"
#include "openvino/frontend/graph_iterator.hpp"
#include <vector>

namespace onnxruntime {
namespace openvino_ep {

class ort_graph_decoder : public ov::frontend::DecoderBase {
 public:
  ort_graph_decoder(const OrtApi api, const OrtNode* node) : api_{api}, node_{node} {
    size_t num_attributes{0};
    if ((api_.Node_GetNumAttributes(node_, &num_attributes) != nullptr) &&
        (num_attributes > 0)) {
      attributes_.resize(num_attributes);
      api_.Node_GetAttributes(node_, attributes_.data(), num_attributes);
    }

    const char* op_name{nullptr};
    if ((api_.Node_GetName(node_, &op_name) == nullptr) &&
        (op_name != nullptr)) {
      op_name_ = op_name;
    }

    const char* op_type{nullptr};
    if ((api_.Node_GetOperatorType(node_, &op_type) == nullptr) &&
        (op_type != nullptr)) {
      op_type_ = op_type;
    }
  }

  ov::Any get_attribute(const std::string& name) const override final {
    return nullptr;
  }

  size_t get_input_size() const override final {
    size_t size{0};
    if (api_.Node_GetNumInputs(node_, &size) == nullptr) {
      return size;
    }
    return 0;
  }

  void get_input_node(size_t input_port_idx,
                      std::string& producer_name,
                      std::string& producer_output_port_name,
                      size_t& producer_output_port_index) const override final {
    // Get the parent graph of this node
    const OrtGraph* parent_graph = nullptr;
    if (api_.Node_GetGraph(node_, &parent_graph) != nullptr || parent_graph == nullptr) {
      producer_name.clear();
      producer_output_port_name.clear();
      producer_output_port_index = 0;
      return;
    }

    // Get the input value info for the given input port
    size_t num_inputs = 0;
    if (api_.Node_GetNumInputs(node_, &num_inputs) != nullptr || input_port_idx >= num_inputs) {
      producer_name.clear();
      producer_output_port_name.clear();
      producer_output_port_index = 0;
      return;
    }

    std::vector<const OrtValueInfo*> inputs(num_inputs, nullptr);
    if (api_.Node_GetInputs(node_, inputs.data(), num_inputs) != nullptr) {
      producer_name.clear();
      producer_output_port_name.clear();
      producer_output_port_index = 0;
      return;
    }

    const OrtValueInfo* input_value = inputs[input_port_idx];
    if (input_value == nullptr) {
      producer_name.clear();
      producer_output_port_name.clear();
      producer_output_port_index = 0;
      return;
    }

    // Get the producer node and output index for this input
    const OrtNode* prod_node = nullptr;
    size_t prod_output_index = 0;
    if (api_.ValueInfo_GetValueProducer(input_value, &prod_node, &prod_output_index) != nullptr || prod_node == nullptr) {
      producer_name.clear();
      producer_output_port_name.clear();
      producer_output_port_index = 0;
      return;
    }

    // Get the producer node's name
    const char* prod_node_name = nullptr;
    if (api_.Node_GetName(prod_node, &prod_node_name) != nullptr || prod_node_name == nullptr) {
      producer_name.clear();
    } else {
      producer_name = prod_node_name;
    }

    // Get the output name for the producer node's output port
    size_t num_outputs = 0;
    if (api_.Node_GetNumOutputs(prod_node, &num_outputs) != nullptr || prod_output_index >= num_outputs) {
      producer_output_port_name.clear();
    } else {
      std::vector<const OrtValueInfo*> outputs(num_outputs, nullptr);
      if (api_.Node_GetOutputs(prod_node, outputs.data(), num_outputs) == nullptr && outputs[prod_output_index]) {
        const char* output_name = nullptr;
        if (api_.GetValueInfoName(outputs[prod_output_index], &output_name) == nullptr && output_name) {
          producer_output_port_name = output_name;
        } else {
          producer_output_port_name.clear();
        }
      } else {
        producer_output_port_name.clear();
      }
    }

    producer_output_port_index = prod_output_index;
  }

  const std::string& get_op_type() const override final {
    return op_type_;
  }

  const std::string& get_op_name() const override final {
    return op_name_;
  }

  ~ort_graph_decoder() {}

 private:
  const OrtApi api_;
  const OrtNode* node_{nullptr};
  std::string op_name_;
  std::string op_type_;
  std::vector<const OrtOpAttr*> attributes_;
  // OpTypeByName
};

class ort_graph_delegate : public ov::frontend::GraphIterator {
 public:
  ort_graph_delegate(const OrtApi& api, const OrtGraph* graph) : api_{api}, graph_{graph} {
    ORT_ENFORCE(graph != nullptr);

    size_t num_nodes{0};
    if ((api_.Graph_GetNumNodes(graph_, &num_nodes) == nullptr) &&
        (num_nodes > 0)) {
      nodes_.resize(num_nodes, nullptr);
      if (auto result = api_.Graph_GetNodes(graph_, nodes_.data(), num_nodes); result == nullptr) {
        reset();
      } else {
        printf("Error: %s\n", api_.GetErrorMessage(result));
      }
    }
    ORT_ENFORCE(nodes_.size() > 0);
  }

  /// \brief Get a number of operation nodes in the graph
  size_t size() const override final {
    return nodes_.size();
  }

  /// \brief Set iterator to the start position
  void reset() override final {
    iter_ = nodes_.begin();
  }

  /// \brief Move to the next node in the graph
  void next() override final {
    iter_++;
  }

  /// \brief Returns true if iterator goes out of the range of available nodes
  bool is_end() const override final {
    return iter_ == nodes_.end();
  }

  /// \brief Return a pointer to a decoder of the current node
  std::shared_ptr<ov::frontend::DecoderBase> get_decoder() const override final {
    ov::frontend::DecoderBase* ptr = new ort_graph_decoder(api_, *iter_);
    return std::shared_ptr<ov::frontend::DecoderBase>((ov::frontend::DecoderBase*)ptr);
  }

  /// \brief Checks if the main model graph contains a function of the requested name in the library
  /// Returns GraphIterator to this function and nullptr, if it does not exist
  std::shared_ptr<GraphIterator> get_body_graph_iterator(const std::string& func_name) const override final {
    return nullptr;
  }

  /// \brief Returns a vector of input names in the original order
  std::vector<std::string> get_input_names() const override final {
    size_t num_inputs{0};
    if ((api_.Graph_GetNumInputs(graph_, &num_inputs) == nullptr) &&
        (num_inputs > 0)) {
      std::vector<const OrtValueInfo*> inputs(num_inputs, nullptr);
      if (api_.Graph_GetInputs(graph_, inputs.data(), num_inputs) == nullptr) {
        std::vector<std::string> input_names(num_inputs);
        for (const auto input : inputs) {
          const char* name{nullptr};
          if ((api_.GetValueInfoName(input, &name) == nullptr) &&
              (name != nullptr)) {
            input_names.emplace_back(name);
          }
        }
        return input_names;
      }
    }
    return {};
  }

  /// \brief Returns a vector of output names in the original order
  std::vector<std::string> get_output_names() const override final {
    size_t num_outputs{0};
    if ((api_.Graph_GetNumOutputs(graph_, &num_outputs) == nullptr) &&
        (num_outputs > 0)) {
      std::vector<const OrtValueInfo*> outputs(num_outputs, nullptr);
      if (api_.Graph_GetOutputs(graph_, outputs.data(), num_outputs) == nullptr) {
        std::vector<std::string> output_names(num_outputs);
        for (const auto output : outputs) {
          const char* name{nullptr};
          if ((api_.GetValueInfoName(output, &name) == nullptr) &&
              (name != nullptr)) {
            output_names.emplace_back(name);
          }
        }
        return output_names;
      }
    }
    return {};
  }

  /// \brief Returns a map from internal tensor name to (user-defined) external name for inputs
  std::map<std::string, std::string> get_input_names_map() const override final {
    auto input_names = get_input_names();
    std::map<std::string, std::string> result;
    for (const auto& name : input_names) {
      result.insert({name, name});
    }
    return result;
  }

  /// \brief Returns a map from internal tensor name to (user-defined) external name for outputs
  std::map<std::string, std::string> get_output_names_map() const override final {
    auto output_names = get_output_names();
    std::map<std::string, std::string> result;
    for (const auto& name : output_names) {
      result.insert({name, name});
    }
    return result;
  }

 private:
  const OrtApi api_;
  const OrtGraph* graph_{nullptr};
  using ort_nodes_t = std::vector<const OrtNode*>;
  ort_nodes_t nodes_;
  ort_nodes_t::const_iterator iter_;
};

}  // namespace openvino_ep
}  // namespace onnxruntime