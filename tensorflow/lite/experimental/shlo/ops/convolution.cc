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
#include "tensorflow/lite/experimental/shlo/ops/util.h"
#include "tensorflow/lite/experimental/shlo/quantize.h"
#include "tensorflow/lite/experimental/shlo/quantized_tensor_element_type.h"
#include "tensorflow/lite/experimental/shlo/shape.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

namespace shlo_ref {

// Will be updated with data preparation for padding and other functions
template <DataType storage_type>
absl::Status PrepareImpl(ConvolutionOp& op, const Tensor& lhs,
                         const Tensor& rhs, Tensor& output) {
  using StorageT = StorageType<storage_type>;
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
  size_t output_channel = 0;
  SHLO_REF_RETURN_ON_ERROR(
      ConvolutionImpl<storage_type>(op, output_channel, lhs, rhs, output));
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