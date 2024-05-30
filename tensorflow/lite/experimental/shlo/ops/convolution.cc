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

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/shlo/data_type.h"
#include "tensorflow/lite/experimental/shlo/dispatch.h"
#include "tensorflow/lite/experimental/shlo/ops/convolution_helper_functions.h"
#include "tensorflow/lite/experimental/shlo/ops/util.h"
#include "tensorflow/lite/experimental/shlo/quantize.h"
#include "tensorflow/lite/experimental/shlo/quantized_tensor_element_type.h"
#include "tensorflow/lite/experimental/shlo/shape.h"
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
  const int64_t* padding_buffer =
      op.attributes.padding.GetDataAs<DataType::kSI64>();

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

  // preparing data for padding
  op.pad_input_offset = 0;
  op.pad_output_offset = 0;
  int64_t lhs_padded_spatials[lhs_transposed.Rank() - 2];
  int64_t lhs_padded_tensor_size = 1;
  for (size_t i = lhs_transposed.Rank() - 1; i > 1; --i) {
    lhs_padded_spatials[i - 2] = lhs_transposed.shape().Dim(i) +
                                 (op.attributes.lhs_dilation[i - 2] - 1) *
                                     (lhs_transposed.shape().Dim(i) - 1) +
                                 padding_buffer[2 * (i - 2)] +
                                 padding_buffer[(2 * (i - 2)) + 1];
    lhs_padded_tensor_size *= lhs_padded_spatials[i - 2];
  }

  lhs_padded_tensor_size *=
      lhs_transposed.shape().Dim(0) * lhs_transposed.shape().Dim(1);
  op.lhs_padded_data =
      std::vector<std::byte>(lhs_padded_tensor_size * sizeof(StorageT));
  absl::InlinedVector<DimensionSize, kMaxNumDimensions> lhs_padding_shape_dims(
      lhs_transposed.Rank(), 0);
  lhs_padding_shape_dims[0] = lhs_transposed.shape().Dim(0);
  lhs_padding_shape_dims[1] = lhs_transposed.shape().Dim(1);
  for (size_t i = 0; i < lhs_transposed.Rank() - 2; ++i) {
    lhs_padding_shape_dims[i + 2] =
        static_cast<int64_t>(lhs_padded_spatials[i]);
  }
  const Shape lhs_padding_shape(lhs_padding_shape_dims);
  Tensor lhs_padded{.type = TensorType{.shape = lhs_padding_shape,
                                       .element_type = storage_type},
                    .data = op.lhs_padded_data.data()};
  int64_t pad_output_shape[kMaxNumDimensions];
  std::copy(lhs_padding_shape_dims.data(),
            lhs_padding_shape_dims.data() + lhs_transposed.Rank(),
            pad_output_shape);
  int64_t edge_pad_high[kMaxNumDimensions];
  int64_t edge_pad_low[kMaxNumDimensions];
  int64_t interior_pad[kMaxNumDimensions];
  edge_pad_high[0] = edge_pad_low[0] = interior_pad[0] = 0;
  edge_pad_high[1] = edge_pad_low[1] = interior_pad[1] = 0;
  for (int64_t i = 2; i < lhs_transposed.Rank(); ++i) {
    edge_pad_low[i] = padding_buffer[2 * (i - 2)];
    edge_pad_high[i] = padding_buffer[2 * (i - 2) + 1];
    interior_pad[i] = op.attributes.lhs_dilation[i - 2] - 1;
  }
  int64_t pad_rank = lhs_transposed.Rank();
  int64_t pad_output_dimension_sizes[kMaxNumDimensions];
  pad_output_dimension_sizes[pad_rank - 1] = 1;
  op.pad_output_strides.resize(pad_rank, 0);
  op.pad_output_strides[pad_rank - 1] = interior_pad[pad_rank - 1] + 1;
  for (int64_t i = pad_rank - 2; i >= 0; --i) {
    pad_output_dimension_sizes[i] =
        pad_output_shape[i + 1] * pad_output_dimension_sizes[i + 1];
    op.pad_output_strides[i] =
        pad_output_dimension_sizes[i] * (interior_pad[i] + 1);
  }
  for (int64_t i = 0; i < pad_rank; ++i) {
    op.pad_output_offset +=
        std::max<int64_t>(edge_pad_low[i], 0) * pad_output_dimension_sizes[i];
  }
  op.pad_input_strides.resize(pad_rank, 0);
  op.pad_input_strides[pad_rank - 1] = 1;
  for (int64_t i = pad_rank - 1; i >= 1; --i) {
    op.pad_input_strides[i - 1] =
        lhs_transposed.shape().Dim(i) * op.pad_input_strides[i];
  }
  for (int64_t i = 0; i < pad_rank; ++i) {
    op.pad_input_shape.push_back(
        lhs_transposed.shape().Dim(i) +
        DivNegRoundAwayOrZero(edge_pad_low[i], interior_pad[i] + 1) +
        DivNegRoundAwayOrZero(edge_pad_high[i], interior_pad[i] + 1));
  }

  for (int64_t i = 0; i < pad_rank; ++i) {
    op.pad_input_offset -=
        DivNegRoundAwayOrZero(edge_pad_low[i], interior_pad[i] + 1) *
        op.pad_input_strides[i];
    if (edge_pad_low[i] < 0) {
      int64_t tmp_offset =
          ((interior_pad[i] + 1 + edge_pad_low[i]) % (interior_pad[i] + 1));
      if (tmp_offset < 0) {
        tmp_offset += interior_pad[i] + 1;
      }
      op.pad_output_offset += tmp_offset * pad_output_dimension_sizes[i];
    }
  }
  // padding data prepare end

  // preparing Split data
  int64_t num_splits =
      op.attributes.batch_group_count * op.attributes.feature_group_count;
  int64_t split_dimension = 0;
  for (int64_t i = 0; i < num_splits; ++i) {
    absl::InlinedVector<DimensionSize, kMaxNumDimensions> rhs_split_dims(
        rhs_transposed.Rank(), 0);
    for (size_t i = 0; i < rhs_transposed.Rank(); ++i) {
      if (i == split_dimension) {
        rhs_split_dims[i] = (rhs_transposed.shape().Dim(i) / num_splits);
      } else {
        rhs_split_dims[i] = rhs_transposed.shape().Dim(i);
      }
    }
    const Shape rhs_split_shape(rhs_split_dims);
    op.rhs_splits_data.push_back(std::vector<std::byte>(
        (rhs_transposed.NumElements() / num_splits) * sizeof(StorageT)));
    Tensor rhs_split{.type = TensorType{.shape = rhs_split_shape,
                                        .element_type = storage_type},
                     .data = op.rhs_splits_data.back().data()};
    op.rhs_splits.push_back(rhs_split);
  }

  if (op.attributes.feature_group_count > 1) {
    split_dimension = 1;
  }

  for (int64_t i = 0; i < num_splits; ++i) {
    absl::InlinedVector<DimensionSize, kMaxNumDimensions> lhs_split_dims(
        lhs_padded.Rank(), 0);
    for (size_t i = 0; i < lhs_padded.Rank(); ++i) {
      if (i == split_dimension) {
        lhs_split_dims[i] = (lhs_padded.shape().Dim(i) / num_splits);
      } else {
        lhs_split_dims[i] = lhs_padded.shape().Dim(i);
      }
    }
    const Shape lhs_split_shape(lhs_split_dims);
    op.lhs_splits_data.push_back(std::vector<std::byte>(
        (lhs_padded.NumElements() / num_splits) * sizeof(StorageT)));
    Tensor lhs_split{.type = TensorType{.shape = lhs_split_shape,
                                        .element_type = storage_type},
                     .data = op.lhs_splits_data.back().data()};
    op.lhs_splits.push_back(lhs_split);
  }
  // split data prepare end

  op.lhs_permutations = std::move(lhs_permutation_values);
  op.lhs_transposed = std::move(lhs_transposed);
  op.rhs_permutations = std::move(rhs_permutation_values);
  op.rhs_transposed = std::move(rhs_transposed);
  op.output_permutations = std::move(output_permutation_values);
  op.output_transposed = std::move(output_transposed);
  op.lhs_padded = std::move(lhs_padded);

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

  for (size_t i = 0; i < lhs.shape().Dim(0); ++i) {
    for (size_t j = 0; j < output_spatial_size; ++j) {
      // This will be replaced by tensor GetNdIndex function
      int64_t output_dimensions[output.Rank()];
      size_t output_depth = 1;
      for (size_t m = output.Rank() - 1; m > 1; --m) {
        output_dimensions[m] = (j / output_depth) % output.shape().Dim(m);
        output_depth *= output.shape().Dim(m);
      }
      for (size_t k = 0; k < rhs_spatial_size; ++k) {
        for (size_t l = 0; l < lhs.shape().Dim(1); ++l) {
          // This will be replaced by tensor GetNdIndex function
          int64_t filter_spatials[rhs.Rank() - 2];
          size_t depth = 1;
          for (size_t m = rhs.Rank() - 1; m > 1; --m) {
            filter_spatials[m - 2] = (k / depth) % rhs.shape().Dim(m);
            depth *= rhs.shape().Dim(m);
          }

          // This will be replaced by tensor FlattenIndex function
          int64_t lhs_dimensions[lhs.Rank()];
          lhs_dimensions[0] = i;
          lhs_dimensions[1] = l;
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

          size_t channel_skip = l * rhs_spatial_size;

          for (size_t m = 0; m < rhs.shape().Dim(0); ++m) {
            size_t batch_skip = m * rhs_tensor_size;
            output_dimensions[0] = i;
            output_dimensions[1] = output_channel + m;
            output_depth = 1;
            size_t output_index = 0;
            // This will be replaced by tensor FlattenIndex function
            for (int64_t n = output.Rank() - 1; n >= 0; --n) {
              output_index += output_dimensions[n] * output_depth;
              output_depth *= output.shape().Dim(n);
            }
            output_buffer[output_index] +=
                rhs_buffer[batch_skip + channel_skip + k] *
                lhs_buffer[lhs_index];
          }
        }
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

  PaddingImpl<storage_type>(op, op.lhs_transposed);

  // spliting the lhs and rhs
  size_t output_channel = 0;

  if (op.attributes.feature_group_count > 1) {
    Split<storage_type>(op.lhs_padded, op.attributes.feature_group_count, 1,
                        op.lhs_splits);
    Split<storage_type>(op.rhs_transposed, op.attributes.feature_group_count, 0,
                        op.rhs_splits);

    for (int64_t i = 0; i < op.attributes.feature_group_count; ++i) {
      SHLO_REF_RETURN_ON_ERROR(ConvolutionImpl<storage_type>(
          op, output_channel, op.lhs_splits[i], op.rhs_splits[i],
          op.output_transposed));
    }
    SHLO_REF_RETURN_ON_ERROR(TransposeImpl<storage_type>(
        op.output_transposed, op.output_permutations, output));
    return absl::OkStatus();
  } else if (op.attributes.batch_group_count > 1) {
    Split<storage_type>(op.lhs_padded, op.attributes.batch_group_count, 0,
                        op.lhs_splits);
    Split<storage_type>(op.rhs_transposed, op.attributes.batch_group_count, 0,
                        op.rhs_splits);

    for (int64_t i = 0; i < op.attributes.batch_group_count; ++i) {
      SHLO_REF_RETURN_ON_ERROR(ConvolutionImpl<storage_type>(
          op, output_channel, op.lhs_splits[i], op.rhs_splits[i],
          op.output_transposed));
    }
    SHLO_REF_RETURN_ON_ERROR(TransposeImpl<storage_type>(
        op.output_transposed, op.output_permutations, output));
    return absl::OkStatus();
  }

  SHLO_REF_RETURN_ON_ERROR(
      ConvolutionImpl<storage_type>(op, output_channel, op.lhs_padded,
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