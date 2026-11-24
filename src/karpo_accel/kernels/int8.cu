#include <torch/library.h>

#include "quantization/w8a8/cutlass/scaled_mm_c2x_sm80_dispatch.cuh"
#include "cutlass_extensions/epilogue/scaled_mm_epilogues_c2x.hpp"

namespace {
void scaled_mm(torch::Tensor out, const torch::Tensor& a, const torch::Tensor& b,
               const torch::Tensor& row_scales, const torch::Tensor& column_scales) {
  TORCH_CHECK(out.scalar_type() == torch::kBFloat16);
  vllm::cutlass_gemm_sm80_dispatch<int8_t, cutlass::bfloat16_t, vllm::c2x::ScaledEpilogue>(
      out, a, b, row_scales, column_scales);
}
}  // namespace

TORCH_LIBRARY(karpo_ampere, m) {
  m.def("scaled_mm(Tensor(a!) out, Tensor a, Tensor b, Tensor sa, Tensor sb) -> ()");
}

TORCH_LIBRARY_IMPL(karpo_ampere, CUDA, m) {
  m.impl("scaled_mm", scaled_mm);
}
