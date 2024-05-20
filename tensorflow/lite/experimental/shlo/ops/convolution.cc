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

#include "tensorflow/lite/experimental/shlo/ops/convolution.h"

#include <algorithm>
#include <cstddef>
#include <set>
#include <string>
#include <type_traits>

#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/dispatch.h"
#include "tensorflow/lite/experimental/shlo/ops/convolution_helper_functions.h"
#include "tensorflow/lite/experimental/shlo/ops/unary_elementwise.h"
#include "tensorflow/lite/experimental/shlo/ops/util.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

namespace shlo_ref {

// Constraints Check
absl::Status CheckParameters(ConvolutionOp& op, const Tensor& lhs,
                             const Tensor& rhs, const Tensor& output) {
  if (op.attributes.precision_configs.size() != 2) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution: Size of precision_config must be two.");
  }
  Axis rank = lhs.Rank();
  if (lhs.Rank() != rhs.Rank()) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution: Rank of lhs and rhs must be same");
  }
  if (output.Rank() != lhs.Rank()) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution: Rank of lhs and output must be same");
  }
  if (!lhs.IsQuantized()) {
    SHLO_REF_RETURN_ON_ERROR(
        CheckSameBaselineType(CheckCtx("Convolution"), lhs, rhs));
    SHLO_REF_RETURN_ON_ERROR(
        CheckSameBaselineType(CheckCtx("Convolution"), lhs, output));
  }
  if (op.attributes.window_strides.size() != rank - 2) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution: Size of window_stride must be rank - 2");
  }
  if (!IsGreaterThanZero(op.attributes.window_strides)) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution: The window_stride must be > 0");
  }
  if (op.attributes.padding.shape().Dim(0) != rank - 2 ||
      op.attributes.padding.shape().Dim(1) != 2) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution: Shape of padding must be [rank - 2, 2]");
  }
  if (op.attributes.lhs_dilation.size() != rank - 2) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution: Shape of lhs_dilation must be rank - 2");
  }
  if (!IsGreaterThanZero(op.attributes.lhs_dilation)) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution: The lhs_dilation must be > 0");
  }
  if (op.attributes.rhs_dilation.size() != rank - 2) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution: Shape of rhs_dilation must be rank - 2");
  }
  if (!IsGreaterThanZero(op.attributes.rhs_dilation)) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution: The rhs_dilation must be > 0");
  }
  if (lhs.shape().Dim(static_cast<Axis>(op.attributes.input_batch_dimension)) %
          op.attributes.batch_group_count !=
      0) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution: "
        "Dim(lhs,input_batch_dimension) % batch_group_count = 0");
  }
  if (lhs.shape().Dim(
          static_cast<Axis>(op.attributes.input_feature_dimension)) %
          op.attributes.feature_group_count !=
      0) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution: "
        "Dim(lhs,input_feature_dimension) % (feature_group_count) = 0");
  }
  if (op.attributes.input_spatial_dimensions.size() != rank - 2) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution: "
        "Size of input_spatial_dimensions must be rank - 2");
  }
  if (!IsUnique(op.attributes.input_batch_dimension,
                op.attributes.input_feature_dimension,
                op.attributes.input_spatial_dimensions)) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution: The input_dimensions must be unique");
  }
  if (!IsInRange(op.attributes.input_batch_dimension,
                 op.attributes.input_feature_dimension,
                 op.attributes.input_spatial_dimensions, rank)) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution: The input_dimensions must be >= 0 and < rank");
  }
  if (rhs.shape().Dim(
          static_cast<Axis>(op.attributes.kernel_input_feature_dimension)) !=
      lhs.shape().Dim(
          static_cast<Axis>(op.attributes.input_feature_dimension)) /
          op.attributes.feature_group_count) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution: "
        "Dim(rhs,kernel_input_feature_dimension) = "
        "Dim(lhs,input_feature_dimension) / feature_group_count");
  }
  if (rhs.shape().Dim(
          static_cast<Axis>(op.attributes.kernel_output_feature_dimension)) %
          op.attributes.batch_group_count !=
      0) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution: "
        "Dim(rhs,kernel_output_feature_dimension) % batch_group_count = 0");
  }
  if (rhs.shape().Dim(
          static_cast<Axis>(op.attributes.kernel_output_feature_dimension)) %
          op.attributes.feature_group_count !=
      0) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution: "
        "Dim(rhs,kernel_output_feature_dimension) % (feature_group_count) = 0");
  }
  if (op.attributes.kernel_spatial_dimensions.size() != rank - 2) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution: "
        "Size of kernel_spatial_dimensions must be rank - 2");
  }
  if (!IsUnique(op.attributes.kernel_output_feature_dimension,
                op.attributes.kernel_input_feature_dimension,
                op.attributes.kernel_spatial_dimensions)) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution: "
        "The kernel_dimensions must be unique");
  }
  if (!IsInRange(op.attributes.kernel_output_feature_dimension,
                 op.attributes.kernel_input_feature_dimension,
                 op.attributes.kernel_spatial_dimensions, rank)) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution: The kernel_dimensions must be >= 0 and < rank");
  }
  if (op.attributes.output_spatial_dimensions.size() != rank - 2) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution: "
        "Size of output_spatial_dimensions must be rank - 2");
  }
  if (!IsUnique(op.attributes.output_batch_dimension,
                op.attributes.output_feature_dimension,
                op.attributes.output_spatial_dimensions)) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution: "
        "The output_dimensions must be unique");
  }
  if (!IsInRange(op.attributes.output_batch_dimension,
                 op.attributes.output_feature_dimension,
                 op.attributes.output_spatial_dimensions, rank)) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution: The output_dimensions must be >= 0 and < rank");
  }
  if (op.attributes.feature_group_count <= 0) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution: The feature_group_count must be > 0");
  }
  if (op.attributes.batch_group_count <= 0) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution: The batch_group_count must be > 0");
  }
  if (op.attributes.batch_group_count != 1 &&
      op.attributes.feature_group_count != 1) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution: The batch_group_count == 1 or "
        "feature_group_count == 1");
  }
  if (output.shape().Dim(
          static_cast<Axis>(op.attributes.output_batch_dimension)) !=
      lhs.shape().Dim(static_cast<Axis>(op.attributes.input_batch_dimension)) /
          op.attributes.batch_group_count) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution: "
        "output.shape().Dim(output_batch_dimension) == "
        "lhs.shape().Dim(input_batch_dimension) / batch_group_count");
  }
  if (output.shape().Dim(
          static_cast<Axis>(op.attributes.output_feature_dimension)) !=
      rhs.shape().Dim(
          static_cast<Axis>(op.attributes.kernel_output_feature_dimension))) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution: "
        "output.shape().Dim(output_feature_dimension) == "
        "rhs.shape().Dim(kernel_output_feature_dimension)");
  }
  if (!CheckOutputSpatial(op, lhs, rhs, output)) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution: "
        "output.shape().Dim(spatial_dim) is not properly set");
  }
  if (lhs.IsQuantized() || rhs.IsQuantized() || output.IsQuantized()) {
    if (!(lhs.IsQuantized() && rhs.IsQuantized() && output.IsQuantized())) {
      return absl::FailedPreconditionError(
          "stablehlo.convolution: lhs.IsQuantized() && "
          "rhs.IsQuantized() && output.IsQuantized()");
    }
    if (rhs.IsPerTensorQuantized()) {
      if (!(output.IsPerTensorQuantized())) {
        return absl::FailedPreconditionError(
            "stablehlo.convolution: If is_per_tensor_quantized(rhs), then "
            "is_per_tensor_quantized(output)");
      }
      // This will be replaced by Tensor::ExpressedType()
      if (lhs.quantized_per_tensor_element_type().ExpressedType() !=
          rhs.quantized_per_tensor_element_type().ExpressedType()) {
        return absl::FailedPreconditionError(
            "stablehlo.convolution: The expressed_type of lhs must be the same "
            "as the expressed_type of rhs.");
      }
      if (output.quantized_per_tensor_element_type().ExpressedType() !=
          rhs.quantized_per_tensor_element_type().ExpressedType()) {
        return absl::FailedPreconditionError(
            "stablehlo.convolution: The expressed_type of output must be the "
            "same as the expressed_type of rhs.");
      }
    }
    if (rhs.IsPerAxisQuantized()) {
      if (rhs.quantized_per_axis_element_type().QuantizedDimension() !=
          op.attributes.kernel_output_feature_dimension) {
        return absl::FailedPreconditionError(
            "stablehlo.convolution:  If is_per_axis_quantized(rhs), then "
            "quantization_dimension(rhs) = "
            "op.attributes.kernel_output_feature_dimension");
      }
      // This will be replaced by Tensor::ExpressedType()
      if (lhs.quantized_per_tensor_element_type().ExpressedType() !=
          rhs.quantized_per_axis_element_type().ExpressedType()) {
        return absl::FailedPreconditionError(
            "stablehlo.convolution: The expressed_type of lhs must be the same "
            "as the expressed_type of rhs.");
      }
      if (output.IsPerTensorQuantized()) {
        if (output.quantized_per_tensor_element_type().ExpressedType() !=
            rhs.quantized_per_axis_element_type().ExpressedType()) {
          return absl::FailedPreconditionError(
              "stablehlo.convolution: The expressed_type of output must be the "
              "same as the expressed_type of rhs.");
        }
      }
    }
    if (output.IsPerAxisQuantized()) {
      if (output.quantized_per_axis_element_type().QuantizedDimension() !=
          op.attributes.output_feature_dimension) {
        return absl::FailedPreconditionError(
            "stablehlo.convolution:  If "
            "is_per_axis_quantized(output), then "
            "quantization_dimension(output) = "
            "op.attributes.output_feature_dimension");
      }
      // This will be replaced by Tensor::ExpressedType()
      if (output.quantized_per_axis_element_type().ExpressedType() !=
          rhs.quantized_per_axis_element_type().ExpressedType()) {
        return absl::FailedPreconditionError(
            "stablehlo.convolution: The expressed_type of output must be the "
            "same as the expressed_type of rhs.");
      }
    }
    if (lhs.StorageType() != rhs.StorageType()) {
      return absl::FailedPreconditionError(
          "stablehlo.convolution: "
          "The storage_type of lhs must be the same as storage_type of rhs");
    }
  }
  return absl::OkStatus();
}

template <DataType storage_type>
absl::Status PrepareImpl(ConvolutionOp& op, const Tensor& lhs,
                         const Tensor& rhs, Tensor& output) {
  using StorageT = StorageType<storage_type>;

  SHLO_REF_RETURN_ON_ERROR(CheckParameters(op, lhs, rhs, output));

  // preparing data for transpose
  absl::InlinedVector<Axis, kMaxNumDimensions> lhs_permutation_values(
      lhs.Rank(), 0);
  lhs_permutation_values[0] = op.attributes.input_batch_dimension;
  lhs_permutation_values[1] = op.attributes.input_feature_dimension;
  absl::InlinedVector<DimensionSize, kMaxNumDimensions> lhs_shape_dims(
      lhs.Rank(), 0);
  lhs_shape_dims[0] =
      lhs.shape().Dim(static_cast<Axis>(op.attributes.input_batch_dimension));
  lhs_shape_dims[1] =
      lhs.shape().Dim(static_cast<Axis>(op.attributes.input_feature_dimension));
  for (size_t i = 0; i < lhs.Rank() - 2; ++i) {
    lhs_shape_dims[i + 2] = lhs.shape().Dim(
        static_cast<Axis>(op.attributes.input_spatial_dimensions[i]));
    lhs_permutation_values[i + 2] = op.attributes.input_spatial_dimensions[i];
  }

  op.lhs_transposed_data =
      std::vector<std::byte>(lhs.NumElements() * sizeof(StorageT));
  const Shape lhs_transposed_shape(lhs_shape_dims);
  Tensor lhs_transposed{.type = TensorType{.shape = lhs_transposed_shape,
                                           .element_type = storage_type},
                        .data = op.lhs_transposed_data.data()};

  absl::InlinedVector<Axis, kMaxNumDimensions> rhs_permutation_values(
      rhs.Rank(), 0);
  rhs_permutation_values[0] = op.attributes.kernel_output_feature_dimension;
  rhs_permutation_values[1] = op.attributes.kernel_input_feature_dimension;
  absl::InlinedVector<DimensionSize, kMaxNumDimensions> rhs_shape_dims(
      rhs.Rank(), 0);
  rhs_shape_dims[0] = rhs.shape().Dim(
      static_cast<Axis>(op.attributes.kernel_output_feature_dimension));
  rhs_shape_dims[1] = rhs.shape().Dim(
      static_cast<Axis>(op.attributes.kernel_input_feature_dimension));
  for (size_t i = 0; i < rhs.Rank() - 2; ++i) {
    rhs_shape_dims[i + 2] = rhs.shape().Dim(
        static_cast<Axis>(op.attributes.kernel_spatial_dimensions[i]));
    rhs_permutation_values[i + 2] = op.attributes.kernel_spatial_dimensions[i];
  }

  op.rhs_transposed_data =
      std::vector<std::byte>(rhs.NumElements() * sizeof(StorageT));
  const Shape rhs_transposed_shape(rhs_shape_dims);
  Tensor rhs_transposed{.type = TensorType{.shape = rhs_transposed_shape,
                                           .element_type = storage_type},
                        .data = op.rhs_transposed_data.data()};

  absl::InlinedVector<Axis, kMaxNumDimensions> output_permutation_values(
      output.Rank(), 0);
  output_permutation_values[op.attributes.output_batch_dimension] = 0;
  output_permutation_values[op.attributes.output_feature_dimension] = 1;
  absl::InlinedVector<DimensionSize, kMaxNumDimensions> output_shape_dims(
      output.Rank(), 0);
  output_shape_dims[0] = output.shape().Dim(
      static_cast<Axis>(op.attributes.output_batch_dimension));
  output_shape_dims[1] = output.shape().Dim(
      static_cast<Axis>(op.attributes.output_feature_dimension));
  for (size_t i = 0; i < output.Rank() - 2; ++i) {
    output_shape_dims[i + 2] = output.shape().Dim(
        static_cast<Axis>(op.attributes.output_spatial_dimensions[i]));
    output_permutation_values[op.attributes.output_spatial_dimensions[i]] =
        i + 2;
  }

  op.output_transposed_data =
      std::vector<std::byte>(output.NumElements() * sizeof(StorageT));
  const Shape output_transposed_shape(output_shape_dims);
  Tensor output_transposed{.type = TensorType{.shape = output_transposed_shape,
                                              .element_type = storage_type},
                           .data = op.output_transposed_data.data()};
  // transpose data prepare end

  // preparing data for dot general
  absl::InlinedVector<DimensionSize, kMaxNumDimensions>
      rhs_dot_general_dimensions(rhs_transposed.Rank(), 0);
  size_t rhs_transposed_tensor_size = 1;
  rhs_dot_general_dimensions[0] = 1;
  for (size_t i = 1; i < rhs_transposed.Rank(); ++i) {
    rhs_dot_general_dimensions[i] = rhs_transposed.shape().Dim(i);
    rhs_transposed_tensor_size *= rhs_transposed.shape().Dim(i);
  }
  const Shape rhs_dot_general_shape(rhs_dot_general_dimensions);
  op.rhs_dot_general_data =
      std::vector<std::byte>(rhs_transposed_tensor_size * sizeof(StorageT));
  Tensor rhs_dot_general{.type = TensorType{.shape = rhs_dot_general_shape,
                                            .element_type = storage_type},
                         .data = op.rhs_dot_general_data.data()};

  op.lhs_dot_general_data =
      std::vector<std::byte>(rhs_transposed_tensor_size * sizeof(StorageT));
  Tensor lhs_dot_general{.type = TensorType{.shape = rhs_dot_general_shape,
                                            .element_type = storage_type},
                         .data = op.lhs_dot_general_data.data()};

  absl::InlinedVector<Axis, kMaxNumDimensions>
      lhs_contracting_dimensions_values(lhs_transposed.Rank() - 1);
  for (size_t i = 0; i < lhs_transposed.Rank() - 2; ++i) {
    lhs_contracting_dimensions_values[i] = i + 2;
  }
  lhs_contracting_dimensions_values[lhs_transposed.Rank() - 2] = 1;
  absl::Span<Axis> lhs_contracting_dimensions(
      lhs_contracting_dimensions_values);

  absl::InlinedVector<Axis, kMaxNumDimensions>
      rhs_contracting_dimensions_values(rhs_transposed.Rank() - 1);
  for (size_t i = 0; i < rhs_transposed.Rank() - 2; ++i) {
    rhs_contracting_dimensions_values[i] = i + 2;
  }
  rhs_contracting_dimensions_values[rhs_transposed.Rank() - 2] = 1;
  absl::Span<Axis> rhs_contracting_dimensions(
      rhs_contracting_dimensions_values);

  std::vector<StorageT> dot_general_output_values(1);
  dot_general_output_values[0] = 0;
  op.output_dot_general_data = std::vector<std::byte>(
      reinterpret_cast<std::byte*>(dot_general_output_values.data()),
      reinterpret_cast<std::byte*>(dot_general_output_values.data() +
                                   dot_general_output_values.size()));
  const Shape dot_general_output_shape{{1}};
  Tensor output_dot_general{
      .type = TensorType{.shape = dot_general_output_shape,
                         .element_type = storage_type},
      .data = op.output_dot_general_data.data()};

  absl::Span<Axis> lhs_batching_dimensions;
  absl::Span<Axis> rhs_batching_dimensions;

  const size_t lhs_rank = lhs.Rank();
  const size_t rhs_rank = rhs.Rank();

  op.lhs_result_dimensions = CalculateResultDimensions(
      lhs_rank, lhs_batching_dimensions, lhs_contracting_dimensions);
  op.rhs_result_dimensions = CalculateResultDimensions(
      rhs_rank, rhs_batching_dimensions, rhs_contracting_dimensions);
  // Dot general data prepare end

  op.lhs_permutations = std::move(lhs_permutation_values);
  op.lhs_transposed = std::move(lhs_transposed);
  op.rhs_permutations = std::move(rhs_permutation_values);
  op.rhs_transposed = std::move(rhs_transposed);
  op.output_permutations = std::move(output_permutation_values);
  op.output_transposed = std::move(output_transposed);
  op.lhs_dot_general = std::move(lhs_dot_general);
  op.rhs_dot_general = std::move(rhs_dot_general);
  op.output_dot_general = std::move(output_dot_general);
  op.lhs_contracting_dimensions = std::move(lhs_contracting_dimensions_values);
  op.rhs_contracting_dimensions = std::move(rhs_contracting_dimensions_values);

  return absl::OkStatus();
}

// Convolution
template <DataType storage_type>
absl::Status ConvolutionImpl(ConvolutionOp& op, size_t& output_channel,
                             const Tensor& lhs, const Tensor& rhs,
                             Tensor& output) {
  using StorageT = StorageType<storage_type>;
  const StorageT* lhs_buffer = lhs.GetDataAs<storage_type>();
  const StorageT* rhs_buffer = rhs.GetDataAs<storage_type>();
  StorageT* output_buffer = output.GetDataAs<storage_type>();

  size_t rhs_tensor_size = 1;
  size_t rhs_spatial_size = 1;
  size_t output_spatial_size = 1;
  for (size_t i = 1; i < rhs.Rank(); ++i) {
    rhs_tensor_size *= rhs.shape().Dim(i);
    if (i > 1) {
      output_spatial_size *= output.shape().Dim(i);
      rhs_spatial_size *= rhs.shape().Dim(i);
    }
  }

  Tensor lhs_slice = op.lhs_dot_general;
  Tensor rhs_slice = op.rhs_dot_general;
  Tensor dot_general_output = op.output_dot_general;

  StorageT* lhs_slice_pointer = lhs_slice.GetDataAs<storage_type>();
  StorageT* rhs_slice_pointer = rhs_slice.GetDataAs<storage_type>();

  for (size_t i = 0; i < lhs.shape().Dim(0); ++i) {
    for (size_t j = 0; j < output_spatial_size; ++j) {
      // This will be replaced by tensor GetNdIndex function
      int64_t output_dimensions[output.Rank()];
      size_t output_depth = 1;
      for (size_t m = output.Rank() - 1; m > 1; --m) {
        output_dimensions[m] = (j / output_depth) % output.shape().Dim(m);
        output_depth *= output.shape().Dim(m);
      }
      for (size_t k = 0; k < lhs.shape().Dim(1); ++k) {
        for (size_t l = 0; l < rhs_spatial_size; ++l) {
          // This will be replaced by tensor GetNdIndex function
          int64_t filter_spatials[rhs.Rank() - 2];
          size_t depth = 1;
          for (size_t m = rhs.Rank() - 1; m > 1; --m) {
            filter_spatials[m - 2] = (l / depth) % rhs.shape().Dim(m);
            depth *= rhs.shape().Dim(m);
          }

          // This will be replaced by tensor FlattenIndex function
          int64_t lhs_dimensions[lhs.Rank()];
          lhs_dimensions[0] = i;
          lhs_dimensions[1] = k;
          depth = 1;
          size_t lhs_index = 0;
          for (int64_t m = lhs.Rank() - 1; m >= 0; --m) {
            if (m > 1) {
              lhs_dimensions[m] =
                  output_dimensions[m] * op.attributes.window_strides[m - 2] +
                  filter_spatials[m - 2] * op.attributes.rhs_dilation[m - 2];
            }
            lhs_index += lhs_dimensions[m] * depth;
            depth *= lhs.shape().Dim(m);
          }

          size_t channel_skip = k * rhs_spatial_size;
          lhs_slice_pointer[l + channel_skip] = lhs_buffer[lhs_index];
        }
      }
      for (size_t k = 0; k < rhs.shape().Dim(0); ++k) {
        size_t batch_skip = k * rhs_tensor_size;
        std::copy(rhs_buffer + batch_skip,
                  rhs_buffer + batch_skip + rhs_tensor_size, rhs_slice_pointer);

        absl::Span<Axis> lhs_batching_span;
        absl::Span<Axis> rhs_batching_span;
        absl::Span<Axis> lhs_contracting_span(op.lhs_contracting_dimensions);
        absl::Span<Axis> rhs_contracting_span(op.rhs_contracting_dimensions);
        SHLO_REF_RETURN_ON_ERROR(DotGeneralImpl<storage_type>(
            op, lhs_slice, rhs_slice, lhs_batching_span, rhs_batching_span,
            lhs_contracting_span, rhs_contracting_span, dot_general_output));

        StorageT* dot_general_output_buffer =
            dot_general_output.GetDataAs<storage_type>();

        // This will be replaced by tensor FlattenIndex function
        output_dimensions[0] = i;
        output_dimensions[1] = output_channel + k;
        output_depth = 1;
        size_t output_index = 0;
        for (int64_t m = output.Rank() - 1; m >= 0; --m) {
          output_index += output_dimensions[m] * output_depth;
          output_depth *= output.shape().Dim(m);
        }
        output_buffer[output_index] = dot_general_output_buffer[0];
      }
    }
  }
  output_channel += rhs.shape().Dim(0);

  return absl::OkStatus();
}

template <DataType storage_type>
absl::Status EvaluateImpl(ConvolutionOp& op, const Tensor& lhs,
                          const Tensor& rhs, Tensor& output) {
  SHLO_REF_RETURN_ON_ERROR(
      TransposeImpl<storage_type>(lhs, op.lhs_permutations, op.lhs_transposed));

  SHLO_REF_RETURN_ON_ERROR(
      TransposeImpl<storage_type>(rhs, op.rhs_permutations, op.rhs_transposed));

  size_t output_channel = 0;
  SHLO_REF_RETURN_ON_ERROR(
      ConvolutionImpl<storage_type>(op, output_channel, op.lhs_transposed,
                                    op.rhs_transposed, op.output_transposed));
  SHLO_REF_RETURN_ON_ERROR(TransposeImpl<storage_type>(
      op.output_transposed, op.output_permutations, output));
  return absl::OkStatus();
}

ConvolutionOp Create(const ConvolutionOp::Attributes& attributes) {
  return {.attributes = attributes};
}

absl::Status Prepare(ConvolutionOp& op, const Tensor& lhs, const Tensor& rhs,
                     Tensor& output) {
  DISPATCH_INT_FLOAT(PrepareImpl, lhs.StorageType(), op, lhs, rhs, output);
}

absl::Status Evaluate(ConvolutionOp& op, const Tensor& lhs, const Tensor& rhs,
                      Tensor& output) {
  DISPATCH_INT_FLOAT(EvaluateImpl, output.tensor_element_type(), op, lhs, rhs,
                     output);
}
}  // namespace shlo_ref