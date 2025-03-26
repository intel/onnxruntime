// Copyright (C) Intel Corporation
// Licensed under the MIT License

#pragma once

// for make stateful utility function(s)
#include "openvino/pass/manager.hpp"
#include "openvino/pass/make_stateful.hpp"
#include "openvino/opsets/opset13.hpp"

static inline void logBasicModelInfo(const std::shared_ptr<const ov::Model>& model) {
  std::cout << "Model name: " << model->get_friendly_name() << std::endl;

  // Dump information about model inputs/outputs
  auto inputs = model->inputs();
  auto outputs = model->outputs();

  std::cout << "\tInputs: " << std::endl;
  for (const ov::Output<const ov::Node>& input : inputs) {
    const std::string name = input.get_any_name();
    const ov::element::Type type = input.get_element_type();
    const ov::PartialShape shape = input.get_partial_shape();
    const ov::Layout layout = ov::layout::get_layout(input);

    std::cout << "\t\t" << name << ", " << type << ", " << shape << ", " << layout.to_string() << std::endl;
  }

  std::cout << "\tOutputs: " << std::endl;
  for (const ov::Output<const ov::Node>& output : outputs) {
    const std::string name = output.get_any_name();
    const ov::element::Type type = output.get_element_type();
    const ov::PartialShape shape = output.get_partial_shape();
    const ov::Layout layout = ov::layout::get_layout(output);

    std::cout << "\t\t" << name << ", " << type << ", " << shape << ", " << layout.to_string() << std::endl;
  }

  return;
}

static inline bool model_has_input_output_names(std::shared_ptr<ov::Model> model, const std::string& name_to_match) {
  for (const ov::Output<ov::Node>& input : model->inputs()) {
    auto& names = input.get_names();

    for (auto& name : names) {
      if (name == name_to_match) {
        return true;
      }
    }
  }

  for (const ov::Output<ov::Node>& output : model->outputs()) {
    auto& names = output.get_names();
    for (auto& name : names) {
      if (name == name_to_match) {
        return true;
      }
    }
  }

  return false;
}

static void fuse_cache_reorder(std::shared_ptr<ov::Model> ov_model,
                               std::vector<std::string>& not_kv_inputs,
                               const std::vector<std::string>& key_value_input_names,
                               int gather_dim) {
  if (model_has_input_output_names(ov_model, "beam_idx")) {
    throw std::runtime_error("Model already has fused cache");
  }

  std::string main_input_name = "inputs_embeds";
  if (model_has_input_output_names(ov_model, "input_ids")) {
    main_input_name = "input_ids";
  }

  auto input_batch = ov_model->input(main_input_name).get_partial_shape()[0];

  auto beam_idx = std::make_shared<ov::opset13::Parameter>(ov::element::i32, ov::PartialShape({input_batch}));
  beam_idx->set_friendly_name("beam_idx");
  beam_idx->output(0).get_tensor().add_names({"beam_idx"});
  ov_model->add_parameters({beam_idx});
  not_kv_inputs.push_back(beam_idx->get_friendly_name());

  // Go over all cache parameters and fuse _reorder_cache with indices provided by the new parameter beam_idx
  for (const auto& input_name : key_value_input_names) {
    auto parameter_output_port = ov_model->input(input_name);
    auto consumers = parameter_output_port.get_target_inputs();

    auto gather_op =
        std::make_shared<ov::opset13::Gather>(parameter_output_port,
                                              beam_idx,
                                              ov::opset13::Constant::create(ov::element::i64, {}, {gather_dim}));

    // Replace the source output for all consumers of the input tensor
    for (auto& consumer : consumers) {
      consumer.replace_source_output(gather_op->output(0));
    }
  }

  // Validate the modified model
  ov_model->validate_nodes_and_infer_types();
}

static void make_stateful(std::shared_ptr<ov::Model>& ov_model,
                          const std::vector<std::string>& key_value_input_names,
                          const std::vector<std::string>& key_value_output_names) {
  std::map<std::string, std::string> input_output_map;

  // Create mapping for KV-cache inputs and outputs
  for (size_t i = 0; i < key_value_input_names.size(); ++i) {
    input_output_map[key_value_input_names[i]] = key_value_output_names[i];
  }

  // Apply the transformation to make the model stateful
  ov::pass::Manager manager;
  manager.register_pass<ov::pass::MakeStateful>(input_output_map);
  manager.run_passes(ov_model);
}

// Converted to C++ from here:
// https://github.com/huggingface/optimum-intel/blob/main/optimum/exporters/openvino/stateful.py#L281
static void patch_stateful_decoder(std::shared_ptr<ov::Model> model) {
  std::vector<std::string> key_value_input_names;
  std::vector<std::string> not_kv_inputs;
  for (const ov::Output<ov::Node>& input : model->inputs()) {
    auto& names = input.get_names();

    bool found = false;
    for (auto& name : names) {
      if (name.find("key_values") != std::string::npos) {
        key_value_input_names.push_back(name);
        found = true;
        break;
      }
    }

    if (!found) {
      not_kv_inputs.push_back(input.get_any_name());
    }
  }

  std::vector<std::string> key_value_output_names;
  for (const ov::Output<ov::Node>& output : model->outputs()) {
    auto& names = output.get_names();
    for (auto& name : names) {
      if (name.find("present") != std::string::npos) {
        key_value_output_names.push_back(name);
        break;
      }
    }
  }

  if (key_value_input_names.empty() || key_value_output_names.empty()) {
    std::cout << "no key_value_input_names or key_value_output_names found" << std::endl;
    return;
  }

  // By default, batch is the 0 - th but chatglm uses 1 - st dimension as batch
  // TODO: Deduce from a model via ordinal reshape(? ) and topology
  // batch_dim = 1 if config.model_type == "chatglm" and not hasattr(config, "rope_ratio") else 0
  auto batch_dim = 0;

  fuse_cache_reorder(model, not_kv_inputs, key_value_input_names, batch_dim);

  make_stateful(model, key_value_input_names, key_value_output_names);
}

// Some other utility functions copied from OpenVINO GenAI
static bool has_op_with_type(const std::shared_ptr<const ov::Model>& function, const std::string& type_name) {
  for (const auto& op : function->get_ops()) {
    if (op->get_type_name() == type_name) {
      return true;
    }
  }
  return false;
}

static std::tuple<std::shared_ptr<ov::Node>, int64_t> find_llm_matmul(const std::shared_ptr<ov::Model>& model) {
  auto last_node = model->output(0).get_node()->input_value(0).get_node_shared_ptr();
  std::shared_ptr<ov::Node> matmul = ov::as_type_ptr<ov::op::v0::MatMul>(last_node);

  // in case of PA all tokens are moved to batch dimension and we have to slice / gather accordingly
  const bool pa_based_model = has_op_with_type(model, "PagedAttentionExtension");
  int64_t slice_gather_dim = pa_based_model ? 0 : 1;

  // There are several patterns for matmul we are looking for:
  // Matmul -> Result
  // Matmul -> Add -> Result
  // Matmul -> Transpose -> Result
  // MatMul -> Divide -> Tanh -> Multiply -> Result
  if (!matmul) {
    if (auto add = ov::as_type_ptr<ov::op::v1::Add>(last_node)) {
      matmul = ov::as_type_ptr<ov::op::v0::MatMul>(add->input_value(0).get_node_shared_ptr());
    } else if (auto transpose = ov::as_type_ptr<ov::op::v1::Transpose>(last_node)) {
      matmul = ov::as_type_ptr<ov::op::v0::MatMul>(transpose->input_value(0).get_node_shared_ptr());
      auto order = ov::as_type_ptr<ov::op::v0::Constant>(transpose->input_value(1).get_node_shared_ptr())->get_axis_vector_val();
      slice_gather_dim = order[slice_gather_dim];
    } else if (auto multiply = ov::as_type_ptr<ov::op::v1::Multiply>(last_node)) {
      if (auto tanh = ov::as_type_ptr<ov::op::v0::Tanh>(multiply->input_value(0).get_node_shared_ptr())) {
        if (auto divide = ov::as_type_ptr<ov::op::v1::Divide>(tanh->input_value(0).get_node_shared_ptr())) {
          matmul = as_type_ptr<ov::op::v0::MatMul>(divide->input_value(0).get_node_shared_ptr());
        }
      }
    }
  }
  return std::make_tuple(matmul, slice_gather_dim);
}

static void apply_slice_before_matmul_transformation(std::shared_ptr<ov::Model> model) {
  std::shared_ptr<ov::Node> matmul = nullptr;
  int64_t slice_gather_dim = -1;
  std::tie(matmul, slice_gather_dim) = find_llm_matmul(model);

  if (matmul && matmul->input(0).get_partial_shape().rank().get_length() == 3) {
    auto start = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});
    auto stop = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-2});
    auto step = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});
    auto axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{slice_gather_dim});
    auto slice = std::make_shared<ov::op::v8::Slice>(matmul->input_value(0), start, stop, step, axis);
    matmul->input(0).replace_source_output(slice);
  }
}

static void update_config(ov::AnyMap& config, const std::pair<std::string, ov::Any>& pair) {
  if (config.count(pair.first) == 0) {
    config.insert(pair);
  }
}

static std::optional<ov::Any> pop_option(ov::AnyMap& config, const std::string& option_name) {
  if (auto it = config.find(option_name); it != config.end()) {
    std::optional<ov::Any> found = std::make_optional(it->second);
    config.erase(it);
    return found;
  }
  return std::nullopt;
}

static void rename_key(ov::AnyMap& config, const std::string& old_key, const std::string& new_key) {
  if (config.count(old_key) != 0) {
    auto opt_value = pop_option(config, old_key);
    config[new_key] = opt_value.value();
  }
}

struct KVAxesPosition {
  size_t batch;
  size_t seq_len;
};

KVAxesPosition get_kv_axes_pos(std::shared_ptr<const ov::Model> model) {
  // sequence length axis in key/values tensors, for most cases [BATCH_SIZE, num_kv_heads, seq_len, head_size],
  // therefore usually seq_length_axis = 2 and batch = 0
  KVAxesPosition kv_pos{0u, 2u};

  // "ReadValue" node is KV cache representation in stateful model
  std::string kv_node_type_name = std::string(ov::op::v6::ReadValue::get_type_info_static().name);

  for (const auto op : model->get_ops()) {
    // check input size, as in LoRA adapters case it could be 0
    if (op->get_type_name() != kv_node_type_name || op->get_input_size() < 1) {
      continue;
    }

    // Shape example: [-1,4,0,64]
    auto shape = op->get_input_partial_shape(0);

    for (int64_t i = 0; i < shape.rank().get_length(); i++) {
      // Find axis = 0. This would be sequence length axis.
      if (shape[i] == 0) {
        kv_pos.seq_len = i;
      } else if (shape[i].is_dynamic()) {
        // Dynamic axis is a batch
        kv_pos.batch = i;
      }
    }
    break;
  }

  return kv_pos;
}

struct KVDesc {
  uint32_t max_prompt_len;
  uint32_t min_response_len;
};

static void update_npu_config(ov::AnyMap& config,
                              const std::shared_ptr<ov::Model>& model,
                              const KVAxesPosition& kv_pos,
                              const KVDesc& kv_desc) {
  update_config(config, {"NPU_USE_NPUW", "YES"});
  update_config(config, {"NPUW_LLM", "YES"});

  update_config(config, {"NPUW_LLM_BATCH_DIM", kv_pos.batch});
  update_config(config, {"NPUW_LLM_SEQ_LEN_DIM", kv_pos.seq_len});

  update_config(config, {"NPUW_LLM_MAX_PROMPT_LEN", kv_desc.max_prompt_len});
  update_config(config, {"NPUW_LLM_MIN_RESPONSE_LEN", kv_desc.min_response_len});

  rename_key(config, "++PREFILL_CONFIG", "++NPUW_LLM_PREFILL_CONFIG");
  rename_key(config, "++GENERATE_CONFIG", "++NPUW_LLM_GENERATE_CONFIG");
  rename_key(config, "PREFILL_CONFIG", "NPUW_LLM_PREFILL_CONFIG");
  rename_key(config, "PREFILL_HINT", "NPUW_LLM_PREFILL_HINT");
  rename_key(config, "GENERATE_CONFIG", "NPUW_LLM_GENERATE_CONFIG");
  rename_key(config, "GENERATE_HINT", "NPUW_LLM_GENERATE_HINT");
}

static std::optional<ov::Any> pop_option_new(ov::AnyMap& config, const std::string& option_name) {
  if (auto it = config.find(option_name); it != config.end()) {
    std::optional<ov::Any> found = std::make_optional(it->second);
    config.erase(it);
    return found;
  }
  return std::nullopt;
}

static std::optional<uint32_t> pop_int_and_cast(ov::AnyMap& config, const std::string& key) {
  auto anyopt = pop_option_new(config, key);
  if (anyopt.has_value()) {
    const auto any = anyopt.value();
    int64_t value;
    // NB: Integer value coming from python has int64_t datatype
    if (any.is<int64_t>()) {
      value = any.as<int64_t>();
    } else if (any.is<int>()) {
      value = any.as<int>();
    } else {
      OPENVINO_THROW("Failed to extract " + key + ". Type mismatch: expected types: int or int64_t");
    }
    if (value < 0) {
      OPENVINO_THROW(key + " cannot be negative!");
    }
    return std::make_optional(static_cast<uint32_t>(value));
  }
  return std::nullopt;
}
