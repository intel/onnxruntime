// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include "contrib_ops/cpu/bert/attention_base.h"
#include "contrib_ops/cpu/bert/attention_common.h"
#include "contrib_ops/cpu/bert/attention_helper.h"
#include "contrib_ops/cpu/bert/attention_parameters.h"
#include "core/common/common.h"
#include "core/common/safeint.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

class GQAAttentionBase {
 protected:
  GQAAttentionBase(const OpKernelInfo& info, bool has_local) {
    int64_t num_heads = 0;
    ORT_ENFORCE(info.GetAttr("num_heads", &num_heads).IsOK() && num_heads > 0);
    num_heads_ = static_cast<int>(num_heads);

    int64_t kv_num_heads = 0;
    ORT_ENFORCE(info.GetAttr("kv_num_heads", &kv_num_heads).IsOK() && kv_num_heads > 0);
    kv_num_heads_ = static_cast<int>(kv_num_heads);

    scale_ = info.GetAttrOrDefault<float>("scale", 0.0f);
    softcap_ = info.GetAttrOrDefault<float>("softcap", 0.0f);
    do_rotary_ = info.GetAttrOrDefault<int64_t>("do_rotary", 0) == 1;
    rotary_interleaved_ = info.GetAttrOrDefault<int64_t>("rotary_interleaved", 0) == 1;
    use_smooth_softmax_ = info.GetAttrOrDefault<int64_t>("smooth_softmax", 0) == 1;
    local_window_size_ = has_local ? static_cast<int>(info.GetAttrOrDefault<int64_t>("local_window_size", -1)) : -1;
  }

  int num_heads_;     // number of attention heads of Q
  int kv_num_heads_;  // number of attention heads of K or V
  float scale_;       // the scaling factor applied before softmax
  float softcap_;
  bool do_rotary_;  // whether or not to use rotary embeddings
  bool rotary_interleaved_;
  int local_window_size_;

  bool use_smooth_softmax_;

  template <typename T>
  Status ApplyAttention(const T* Q,                                 // Q data with shape BxNxSxH
                        const T* K,                                 // K data with shape BxN_kvxSxH
                        const T* V,                                 // V data with shape BxN_kvxSxH
                        const Tensor* attention_bias,               // Attention bias to add to QxK'
                        const Tensor* past_key,                     // past K input tensor (if not using past state)
                        const Tensor* past_value,                   // past V input tensor (if not using past state)
                        Tensor* output,                             // output tensor
                        Tensor* present_key,                        // present K output tensor (if separating present KV)
                        Tensor* present_value,                      // present V output tensor (if separating present KV)
                        const Tensor* seqlens_k,                    // past sequence lengths tensor
                        GroupQueryAttentionParameters& parameters,  // attention parameters
                        AllocatorPtr allocator,                     // allocator for temporary tensors
                        OpKernelContext* context) const {
    const bool is_prompt = parameters.is_first_prompt;
    const int batch_size = parameters.batch_size;
    const int sequence_length = parameters.sequence_length;
    const int head_size = parameters.head_size;
    const int hidden_size = parameters.hidden_size;
    const bool packed_qkv = parameters.is_packed_qkv;

    auto* tp = context->GetOperatorThreadPool();

    int seqlen_past_kv_cache = 0;
    if (past_key != nullptr && past_value != nullptr) {
      seqlen_past_kv_cache = static_cast<int>(past_key->Shape().GetDims()[2]);
    }
    int seqlen_present_kv_cache = static_cast<int>(present_key->Shape().GetDims()[2]);

    size_t bytes = SafeInt<size_t>(batch_size) * num_heads_ * sequence_length * seqlen_present_kv_cache * sizeof(float);
    auto attention_probs = allocator->Alloc(bytes);
    BufferUniquePtr scratch_buffer(attention_probs, BufferDeleter(allocator));

    const T* past_key_data = past_key != nullptr ? past_key->Data<T>() : nullptr;
    T* present_key_data = present_key != nullptr ? present_key->MutableData<T>() : nullptr;
    const T* past_value_data = past_value != nullptr ? past_value->Data<T>() : nullptr;
    T* present_value_data = present_value != nullptr ? present_value->MutableData<T>() : nullptr;

    const T* attention_bias_data = attention_bias != nullptr ? attention_bias->Data<T>() : nullptr;
    auto attention_bias_shape = attention_bias != nullptr ? attention_bias->Shape().GetDims() : gsl::span<const int64_t>{};

    bool past_present_share_buffer = past_key_data == present_key_data && past_value_data == present_value_data;

    const T* k = packed_qkv ? Q + num_heads_ * sequence_length * head_size : K;

    ComputeAttentionProbs(static_cast<float*>(attention_probs), Q, k, seqlens_k->Data<int32_t>(), attention_bias_data,
                          batch_size, sequence_length, attention_bias_shape, seqlen_past_kv_cache, seqlen_present_kv_cache,
                          head_size, past_key_data, present_key_data, past_present_share_buffer, packed_qkv, is_prompt,
                          tp, allocator);

    const T* v = packed_qkv ? Q + (num_heads_ + kv_num_heads_) * sequence_length * head_size : V;
    ComputeVxAttentionScore(output->MutableData<T>(), static_cast<float*>(attention_probs), v, seqlens_k->Data<int32_t>(),
                            batch_size, sequence_length, seqlen_past_kv_cache, seqlen_present_kv_cache, head_size,
                            hidden_size, past_value_data, present_value_data, past_present_share_buffer, packed_qkv,
                            is_prompt, tp, allocator);

    return Status::OK();
  }

 private:
  // Helper function to compute the attention probs. It does 2 things:
  //  attention_probs(B, N, S, T) = 1/sqrt(H) x Q(B, N, S, H) x K'(B, N, T, H -> B, N, H, T)
  //  attention_probs(B, N, S, T) = Softmax(attention_probs)
  // If T is float32, U is float32. If T is float16, U could be float16 or float32.
  template <typename T, typename U>
  void ComputeAttentionProbs(U* attention_probs,                                   // output buffer with size BxNxSxT
                             const T* Q,                                           // Q data. Its size is BxNxSxH
                             const T* K,                                           // k data. Its size is BxNxLxH
                             const int32_t* seqlens_k,                             // total - 1 sequence lengths tensor
                             const T* attention_bias,                              // optional attention bias
                             const size_t batch_size,                              // batch size of self-attention
                             const size_t sequence_length,                         // sequence length of self-attention (S)
                             const gsl::span<const int64_t> attention_bias_shape,  // shape of the attention bias
                             const size_t past_buffer_sequence_length,             // sequence length of past state
                             const size_t present_buffer_sequence_length,          // sequence length of present state
                             const size_t head_size,                               // head size of self-attention
                             const T* past_key,                                    // past key only
                             T* present_key,                                       // present key only
                             const bool past_present_share_buffer,                 // whether present key and value share the same buffer
                             const bool packed_qkv,                                // whether Q, K, V are packed
                             const bool is_prompt,                                 // whether it is prompt
                             ThreadPool* tp,                                       // thread pool
                             AllocatorPtr allocator) const {                       // allocator for temporary buffer
    // Essential validation
    if (!attention_probs || !Q || !K || !seqlens_k ||
        batch_size == 0 || sequence_length == 0 || head_size == 0 || present_buffer_sequence_length == 0) {
      return;
    }

    const ptrdiff_t packed_batch_stride =
        packed_qkv ? SafeInt<ptrdiff_t>(num_heads_ + 2 * kv_num_heads_) * sequence_length * head_size
                   : SafeInt<ptrdiff_t>(0);
    const size_t kv_num_heads_factor = num_heads_ / kv_num_heads_;
    const size_t q_input_chunk_length = sequence_length * head_size;                      // S x H
    const size_t kv_input_chunk_length = sequence_length * head_size;                     // L x H
    const size_t past_buff_chunk_length = past_buffer_sequence_length * head_size;        // L x H
    const size_t present_buff_chunk_length = present_buffer_sequence_length * head_size;  // T x H

    if (!past_present_share_buffer && present_key) {
      size_t total_elements = batch_size * kv_num_heads_ * present_buffer_sequence_length * head_size;
      if constexpr (std::is_same_v<T, float>) {
        memset((void*)present_key, 0, total_elements * sizeof(T));
      } else {
        std::fill_n(present_key, total_elements, T(0.0f));
      }
    }

    const size_t loop_len = batch_size * num_heads_;
    const float alpha = scale_ == 0.0f ? 1.0f / sqrt(static_cast<float>(head_size)) : scale_;

    TensorOpCost unit_cost;
    const ptrdiff_t probs_matrix_bytes =
        SafeInt<ptrdiff_t>(sequence_length) * present_buffer_sequence_length * sizeof(T);
    unit_cost.compute_cycles =
        static_cast<double>(SafeInt<ptrdiff_t>(2) * sequence_length * head_size * present_buffer_sequence_length);
    unit_cost.bytes_loaded =
        static_cast<double>((sequence_length + present_buffer_sequence_length) * head_size * sizeof(T));
    unit_cost.bytes_stored = static_cast<double>(probs_matrix_bytes);

    unit_cost.bytes_loaded += static_cast<double>(probs_matrix_bytes);
    unit_cost.bytes_stored += static_cast<double>(probs_matrix_bytes);

    ThreadPool::TryParallelFor(tp, loop_len, unit_cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
      for (std::ptrdiff_t i = begin; i != end; ++i) {
        const size_t batch_index = i / num_heads_;
        const size_t head_index = i % num_heads_;

        // Essential bounds check
        if (batch_index >= batch_size || head_index >= static_cast<size_t>(num_heads_)) {
          continue;
        }

        const size_t total_seqlen = static_cast<size_t>(seqlens_k[batch_index]) + 1;
        const size_t past_seqlen = is_prompt ? 0 : total_seqlen - sequence_length;  // Assume no padding sequence length
        const size_t past_chunk_length = past_seqlen * head_size;

        const ptrdiff_t output_offset = SafeInt<ptrdiff_t>(i) * sequence_length * present_buffer_sequence_length;
        U* output = attention_probs + output_offset;

        // Compute attention bias offset based on the batch and head indexes
        // Attention bias is of shape (B or 1, H or 1, S, T) so handle broadcasting
        const T* attention_bias_thread = nullptr;
        bool is_bias_broadcasted = false;

        if (attention_bias != nullptr && attention_bias_shape.size() >= 4) {
          const ptrdiff_t attention_total_seqlen = static_cast<ptrdiff_t>(attention_bias_shape[3]);
          is_bias_broadcasted = (attention_total_seqlen == 1);

          const ptrdiff_t attention_matrix_size = sequence_length * attention_total_seqlen;
          ptrdiff_t attention_bias_offset = 0;

          if (attention_bias_shape[0] != 1 && batch_index < static_cast<size_t>(attention_bias_shape[0])) {
            attention_bias_offset += SafeInt<ptrdiff_t>(batch_index) * attention_bias_shape[1] * attention_matrix_size;
          }
          if (attention_bias_shape[1] != 1 && head_index < static_cast<size_t>(attention_bias_shape[1])) {
            attention_bias_offset += SafeInt<ptrdiff_t>(head_index) * attention_matrix_size;
          }

          const size_t total_bias_elements = attention_bias_shape[0] * attention_bias_shape[1] *
                                            attention_bias_shape[2] * attention_bias_shape[3];
          if (attention_bias_offset >= 0 && static_cast<size_t>(attention_bias_offset) < total_bias_elements) {
            attention_bias_thread = attention_bias + attention_bias_offset;
          }
        }

        const T* k;
        if (packed_qkv) {
          k = K + packed_batch_stride * batch_index + kv_input_chunk_length * (head_index / kv_num_heads_factor);
        } else {
          k = K + kv_input_chunk_length * (i / kv_num_heads_factor);
        }

        if (present_key) {
          k = ConcatStateChunkGQA(past_key, k, present_key, present_buff_chunk_length, past_buff_chunk_length,
                                  past_chunk_length, kv_input_chunk_length, past_present_share_buffer,
                                  i / kv_num_heads_factor);
          // Essential: Check if ConcatStateChunkGQA returned null
          if (!k) {
            continue;
          }
        }

        const T* q;
        if (packed_qkv) {
          q = Q + packed_batch_stride * batch_index + q_input_chunk_length * head_index;
        } else {
          q = Q + q_input_chunk_length * i;
        }

        // Essential: Validate pointers before matrix multiplication
        if (!q || !k || !output) {
          continue;
        }

        // Matrix multiplication Q*K^T
        const size_t output_elements = sequence_length * present_buffer_sequence_length;
        if constexpr (std::is_same_v<U, float>) {
          std::memset(output, 0, output_elements * sizeof(U));
        } else {
          std::fill_n(output, output_elements, U(0.0f));
        }

        for (size_t s = 0; s < sequence_length; ++s) {
          for (size_t t = 0; t < total_seqlen && t < present_buffer_sequence_length; ++t) {
            float dot_product = 0.0f;
            for (size_t h = 0; h < head_size; ++h) {
              float q_val, k_val;
              if constexpr (std::is_same<T, float>::value) {
                q_val = q[s * head_size + h];
                k_val = k[t * head_size + h];
              } else {
                q_val = static_cast<const MLFloat16*>(q)[s * head_size + h].ToFloat();
                k_val = static_cast<const MLFloat16*>(k)[t * head_size + h].ToFloat();
              }
              dot_product += q_val * k_val;
            }
            const size_t output_idx = s * present_buffer_sequence_length + t;
            if (output_idx < output_elements) {
              if constexpr (std::is_same<U, float>::value) {
                output[output_idx] = alpha * dot_product;
              } else {
                output[output_idx] = MLFloat16(alpha * dot_product);
              }
            }
          }
        }

        // Apply softmax with bias
        U* output_softmax = output;
        for (size_t seq = 0; seq < sequence_length; seq++) {
          const size_t seq_causal_length = past_seqlen + seq + 1;
          const bool should_apply_local_window = local_window_size_ >= 0 &&
                                                 seq_causal_length > static_cast<size_t>(local_window_size_) + 1;
          const size_t start_offset = should_apply_local_window ? seq_causal_length - local_window_size_ - 1 : 0;
          const size_t window_size = should_apply_local_window ? local_window_size_ + 1 : seq_causal_length;

          // Validate window bounds
          if (start_offset >= present_buffer_sequence_length || window_size == 0) {
            output_softmax += present_buffer_sequence_length;
            continue;
          }

          const size_t safe_window_size = std::min(window_size, present_buffer_sequence_length - start_offset);

          // Local window masking
          if (should_apply_local_window) {
            for (size_t total_seq_id = 0; total_seq_id < start_offset; total_seq_id++) {
              if constexpr (std::is_same<U, float>::value) {
                output_softmax[total_seq_id] = 0.f;
              } else {
                output_softmax[total_seq_id] = MLFloat16::FromBits(static_cast<uint16_t>(0));
              }
            }
          }

          // Softcap
          if (softcap_ > 0.f && safe_window_size > 0) {
            ComputeAttentionSoftcapInplace(output_softmax + start_offset, static_cast<int>(safe_window_size),
                                           static_cast<U>(softcap_));
          }

          // Attention bias application
          if (attention_bias_thread != nullptr && safe_window_size > 0) {
            if constexpr (std::is_same_v<U, T>) {
              for (size_t idx = 0; idx < safe_window_size; ++idx) {
                const size_t output_idx = start_offset + idx;
                if (output_idx < present_buffer_sequence_length) {
                  if (is_bias_broadcasted) {
                    output_softmax[output_idx] += attention_bias_thread[0];
                  } else {
                    const size_t bias_idx = start_offset + idx;
                    if (bias_idx < static_cast<size_t>(attention_bias_shape[3])) {
                      output_softmax[output_idx] += attention_bias_thread[bias_idx];
                    }
                  }
                }
              }
            } else {
              static_assert(std::is_same_v<U, float> && std::is_same_v<T, MLFloat16>);
              for (size_t idx = 0; idx < safe_window_size; ++idx) {
                const size_t output_idx = start_offset + idx;
                if (output_idx < present_buffer_sequence_length) {
                  float bias_val = 0.0f;
                  if (is_bias_broadcasted) {
                    bias_val = static_cast<const MLFloat16*>(attention_bias_thread)[0].ToFloat();
                  } else {
                    const size_t bias_idx = start_offset + idx;
                    if (bias_idx < static_cast<size_t>(attention_bias_shape[3])) {
                      bias_val = static_cast<const MLFloat16*>(attention_bias_thread)[bias_idx].ToFloat();
                    }
                  }
                  output_softmax[output_idx] += bias_val;
                }
              }
            }
          }

          // Softmax computation
          if (safe_window_size > 0) {
            if (use_smooth_softmax_) {
              ComputeSmoothSoftmaxInplace(output_softmax + start_offset, 1, static_cast<int>(safe_window_size), nullptr);
            } else {
              ComputeAttentionSoftmaxInplace(output_softmax + start_offset, 1, static_cast<int>(safe_window_size), nullptr);
            }
          }

          // Causal masking
          for (size_t total_seq_id = seq_causal_length; total_seq_id < total_seqlen && total_seq_id < present_buffer_sequence_length; total_seq_id++) {
            if constexpr (std::is_same<U, float>::value) {
              output_softmax[total_seq_id] = 0.f;
            } else {
              output_softmax[total_seq_id] = MLFloat16::FromBits(static_cast<uint16_t>(0));
            }
          }

          output_softmax += present_buffer_sequence_length;
          if (attention_bias_thread != nullptr && !is_bias_broadcasted) {
            attention_bias_thread += attention_bias_shape[3];
          }
        }
      }
    });
  }

  template <typename T, typename U>
  void ComputeVxAttentionScore(T* output, const U* attention_probs, const T* V, const int32_t* seqlens_k,
                               const size_t batch_size, const size_t sequence_length, const size_t past_buffer_sequence_length,
                               const size_t present_buffer_sequence_length, const size_t head_size, const size_t hidden_size,
                               const T* past_value, T* present_value, const bool past_present_share_buffer, const bool packed_qkv,
                               const bool is_prompt, ThreadPool* tp, AllocatorPtr allocator) const {
    if (!output || !attention_probs || !V || !seqlens_k ||
        batch_size == 0 || sequence_length == 0 || head_size == 0) {
      return;
    }

    const size_t total = batch_size * num_heads_ * sequence_length * head_size;
    if constexpr (std::is_same_v<T, float>) {
      memset(output, 0, total * sizeof(T));
    } else {
      std::fill_n(output, total, T(0.0f));
    }
  }
};

}  // namespace contrib
}  // namespace onnxruntime
