/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_CONVOLUTION_HELPER_FUNCTIONS_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_CONVOLUTION_HELPER_FUNCTIONS_H_

#include <algorithm>
#include <cstddef>
#include <string>
#include <type_traits>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/shlo/data_type.h"
#include "tensorflow/lite/experimental/shlo/dispatch.h"
#include "tensorflow/lite/experimental/shlo/ops/dot_general.h"
#include "tensorflow/lite/experimental/shlo/ops/util.h"
#include "tensorflow/lite/experimental/shlo/quantize.h"
#include "tensorflow/lite/experimental/shlo/quantized_tensor_element_type.h"
#include "tensorflow/lite/experimental/shlo/shape.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

namespace shlo_ref {

bool IsUnique(const int64_t& batch_dimension, const int64_t& feature_dimension,
              const absl::Span<int64_t>& operand) {
  std::unordered_set<int64_t> seen_elements;
  if (!seen_elements.insert(batch_dimension).second) {
    return false;
  }
  if (!seen_elements.insert(feature_dimension).second) {
    return false;
  }
  for (int64_t element : operand) {
    if (!seen_elements.insert(element).second) {
      return false;
    }
  }
  return true;
}

bool IsInRange(const int64_t& batch_dimension, const int64_t& feature_dimension,
               const absl::Span<int64_t>& operand, size_t N) {
  auto is_in_range = [N](int64_t v) { return v >= 0 && v < N; };
  if (!is_in_range(batch_dimension) || !is_in_range(feature_dimension)) {
    return false;
  }
  return absl::c_all_of(operand, is_in_range);
}

bool IsGreaterThanZero(const absl::Span<int64_t>& operand) {
  return absl::c_all_of(operand, [](int64_t x) { return x > 0; });
}

bool CheckOutputSpacial(ConvolutionOp& op, const Tensor& lhs, const Tensor& rhs,
                        const Tensor& output) {
  const int64_t* padding_buffer =
      op.attributes.padding.GetDataAs<DataType::kSI64>();
  int64_t is_empty_window;
  int64_t expected_output_shape;
  int64_t dilated_lhs_shape;
  int64_t padded_lhs_shape;
  int64_t dilated_rhs_shape;
  for (size_t i = 0; i < output.Rank() - 2; ++i) {
    dilated_lhs_shape = lhs.shape().Dim(static_cast<Axis>(
                            op.attributes.input_spatial_dimensions[i])) == 0
                            ? 0
                            : (lhs.shape().Dim(static_cast<Axis>(
                                   op.attributes.input_spatial_dimensions[i])) -
                               1) * op.attributes.lhs_dilation[i] +
                                  1;
    padded_lhs_shape =
        dilated_lhs_shape + padding_buffer[2 * i] + padding_buffer[2 * i + 1];
    dilated_rhs_shape =
        rhs.shape().Dim(
            static_cast<Axis>(op.attributes.kernel_spatial_dimensions[i])) == 0
            ? 0
            : (rhs.shape().Dim(static_cast<Axis>(
                   op.attributes.kernel_spatial_dimensions[i])) -
               1) * op.attributes.rhs_dilation[i] +
                  1;
    is_empty_window =
        padded_lhs_shape == 0 || dilated_rhs_shape > padded_lhs_shape;
    expected_output_shape =
        is_empty_window ? 0
                        : std::floor((padded_lhs_shape - dilated_rhs_shape) /
                                     op.attributes.window_strides[i]) +
                              1;
    if (output.shape().Dim(
            static_cast<Axis>(op.attributes.output_spatial_dimensions[i])) !=
        expected_output_shape) {
      return false;
    }
  }
  return true;
}

bool ContainsDimension(absl::Span<const Axis> dimensions, Axis dimension) {
  return std::find(dimensions.begin(), dimensions.end(), dimension) !=
         dimensions.end();
}

absl::InlinedVector<Axis, kMaxNumDimensions> CalculateResultDimensions(
    size_t rank, absl::Span<const Axis> batching_dimensions,
    absl::Span<const Axis> contracting_dimensions) {
  absl::InlinedVector<Axis, kMaxNumDimensions> result_dims;
  for (size_t i = 0; i < rank; ++i) {
    if (!ContainsDimension(batching_dimensions, i) &&
        !ContainsDimension(contracting_dimensions, i)) {
      result_dims.push_back(i);
    }
  }
  return result_dims;
}

void GenerateIndices(size_t index,
                     absl::InlinedVector<Axis, kMaxNumDimensions>& output_index,
                     const Tensor& output, size_t output_rank) {
  size_t divisor = 1, dim = 0;
  for (size_t i = 0, j = output_rank - 1; i < output_rank; ++i, --j) {
    dim = output.shape().Dim(j);
    output_index[j] = (index / divisor) % dim;
    divisor *= dim;
  }
  return;
}

bool IncrementIndices(const Tensor& lhs,
                      absl::InlinedVector<Axis, kMaxNumDimensions>& lhs_index,
                      absl::InlinedVector<Axis, kMaxNumDimensions>& rhs_index,
                      absl::Span<const Axis>& lhs_contracting_dimensions,
                      absl::Span<const Axis>& rhs_contracting_dimensions,
                      const size_t lhsc_size) {
  if (lhsc_size == 0) return false;
  for (size_t i = lhsc_size - 1; i >= 0; --i) {
    lhs_index[lhs_contracting_dimensions[i]]++;
    rhs_index[rhs_contracting_dimensions[i]]++;
    if (lhs_index[lhs_contracting_dimensions[i]] <
        lhs.shape().Dim(lhs_contracting_dimensions[i]))
      return true;
    if (i == 0) return false;
    lhs_index[lhs_contracting_dimensions[i]] = 0;
    rhs_index[rhs_contracting_dimensions[i]] = 0;
  }
  return true;
}

// DotGeneral op implementation
template <DataType storage_type>
absl::Status DotGeneralImpl(ConvolutionOp& op, const Tensor& lhs,
                            const Tensor& rhs,
                            absl::Span<const Axis> lhs_batching_dimensions,
                            absl::Span<const Axis> rhs_batching_dimensions,
                            absl::Span<const Axis> lhs_contracting_dimensions,
                            absl::Span<const Axis> rhs_contracting_dimensions,
                            Tensor& output) {
  using StorageT = StorageType<storage_type>;

  const StorageT* lhs_data = lhs.GetDataAs<storage_type>();
  const StorageT* rhs_data = rhs.GetDataAs<storage_type>();
  StorageT* output_data = output.GetDataAs<storage_type>();
  const DimensionSize lhs_size = lhs.NumElements();
  const DimensionSize rhs_size = rhs.NumElements();
  const DimensionSize output_size = output.NumElements();
  const size_t lhs_rank = lhs.Rank();
  const size_t rhs_rank = rhs.Rank();
  const size_t output_rank = output.Rank();
  const size_t lhsb_size = lhs_batching_dimensions.size();
  const size_t rhsb_size = rhs_batching_dimensions.size();
  const size_t lhsc_size = lhs_contracting_dimensions.size();
  const size_t rhsc_size = rhs_contracting_dimensions.size();

  absl::InlinedVector<Axis, kMaxNumDimensions> lhs_index;
  lhs_index.resize(lhs_rank);
  absl::InlinedVector<Axis, kMaxNumDimensions> rhs_index;
  rhs_index.resize(rhs_rank);
  absl::InlinedVector<DimensionSize, kMaxNumDimensions> lhs_index_helper;
  lhs_index_helper.resize(lhs_rank);
  absl::InlinedVector<DimensionSize, kMaxNumDimensions> rhs_index_helper;
  rhs_index_helper.resize(rhs_rank);
  absl::InlinedVector<Axis, kMaxNumDimensions> output_index;
  output_index.resize(output_rank);

  DimensionSize lhs_dim_accumulator = 1, rhs_dim_accumulator = 1;
  for (size_t i = 0; i < lhs_rank; ++i) {
    lhs_dim_accumulator *= lhs.shape().Dim(i);
    lhs_index_helper[i] = lhs_size / lhs_dim_accumulator;
  }
  for (size_t i = 0; i < rhs_rank; ++i) {
    rhs_dim_accumulator *= rhs.shape().Dim(i);
    rhs_index_helper[i] = rhs_size / rhs_dim_accumulator;
  }

  StorageT output_element(0);
  DimensionSize lhs_element_index = 0, rhs_element_index = 0;
  for (size_t k = 0; k < output_size; ++k, ++output_data) {
    GenerateIndices(k, output_index, output, output_rank);
    absl::c_fill(lhs_index, 0);
    absl::c_fill(rhs_index, 0);

    DimensionSize result_dim = 0;
    for (size_t i = 0; i < lhsb_size; ++i, ++result_dim) {
      lhs_index[lhs_batching_dimensions[i]] = output_index[result_dim];
      rhs_index[rhs_batching_dimensions[i]] = output_index[result_dim];
    }
    for (size_t i = 0; i < op.lhs_result_dims.size(); ++i, ++result_dim) {
      lhs_index[op.lhs_result_dims[i]] = output_index[result_dim];
    }
    for (size_t i = 0; i < op.rhs_result_dims.size(); ++i, ++result_dim) {
      if (result_dim < output.Rank()) {
        rhs_index[op.rhs_result_dims[i]] = output_index[result_dim];
      }
    }
    output_element = 0;
    while (true) {
      lhs_element_index = 0;
      rhs_element_index = 0;
      for (size_t i = 0; i < lhs_rank; ++i) {
        lhs_element_index += lhs_index[i] * lhs_index_helper[i];
      }
      for (size_t i = 0; i < rhs_rank; ++i) {
        rhs_element_index += rhs_index[i] * rhs_index_helper[i];
      }
      output_element +=
          lhs_data[lhs_element_index] * rhs_data[rhs_element_index];

      if (!IncrementIndices(lhs, lhs_index, rhs_index,
                            lhs_contracting_dimensions,
                            rhs_contracting_dimensions, lhsc_size)) {
        break;
      }
    }
    *output_data = output_element;
  }
  return absl::OkStatus();
}

}  // namespace shlo_ref

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_SHLO_CONVOLUTION_HELPER_FUNCTIONS_H_