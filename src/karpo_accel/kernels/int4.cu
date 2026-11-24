#include <torch/library.h>

#include "quantization/w8a8/cutlass/scaled_mm_c2x.cuh"
#include "cutlass_extensions/epilogue/scaled_mm_epilogues_c2x.hpp"

namespace {
namespace evt = cutlass::epilogue::threadblock;
using S4 = cutlass::int4b_t;
using BF16 = cutlass::bfloat16_t;
using cutlass::FloatRoundStyle;
using OutputStride = cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>;

template <class ThreadMap>
struct Dequantize : private vllm::c2x::ScaledEpilogueBase<BF16, ThreadMap> {
  using Base = vllm::c2x::ScaledEpilogueBase<BF16, ThreadMap>;
  using TokenScale = typename Base::template ColOrScalarLoad<float>;
  using ChannelScale = typename Base::template RowOrScalarLoad<float>;
  using MultiplyToFloat = evt::VisitorCompute<cutlass::multiplies, float, float,
                                              FloatRoundStyle::round_to_nearest>;
  using MultiplyToBF16 = evt::VisitorCompute<cutlass::multiplies, BF16, float,
                                             FloatRoundStyle::round_to_nearest>;
  using TokenProduct = evt::Sm80EVT<MultiplyToFloat, typename Base::Accum, TokenScale>;
  using Compute = evt::Sm80EVT<MultiplyToBF16, TokenProduct, ChannelScale>;
  using Store = evt::VisitorAuxStore<ThreadMap, BF16, FloatRoundStyle::round_to_nearest, OutputStride>;
  using Visitor = evt::Sm80EVT<Store, Compute>;

  static typename Visitor::Arguments prepare(const torch::Tensor& token, const torch::Tensor& channel,
                                              BF16* output, int64_t stride) {
    auto left = Base::template args_from_tensor<TokenScale, float>(token);
    auto right = Base::template args_from_tensor<ChannelScale, float>(channel);
    typename TokenProduct::Arguments row_product{{}, left, {}};
    typename Compute::Arguments values{row_product, right, {}};
    typename Store::Arguments destination{output, OutputStride{stride, cute::Int<1>{}, cute::Int<0>{}}};
    return {values, destination};
  }
};

template <int Rows, int WarpRows, int Columns = 128, int WarpColumns = 64, int Depth = 128, int Stages = 4>
struct ScaledInt4 {
  using Tile = cutlass::gemm::GemmShape<Rows, Columns, Depth>;
  using Warp = cutlass::gemm::GemmShape<WarpRows, WarpColumns, 128>;
  using Instruction = cutlass::gemm::GemmShape<16, 8, 64>;
  using ThreadMap = evt::OutputTileThreadLayout<Tile, Warp, float, 4, 1>;
  using Epilogue = Dequantize<ThreadMap>;
  using Kernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
      S4, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 32,
      S4, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 32,
      float, cutlass::layout::RowMajor, 4, int32_t, float,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, Tile, Warp, Instruction,
      typename Epilogue::Visitor, cutlass::gemm::threadblock::ThreadblockSwizzleStreamK,
      Stages, cutlass::arch::OpMultiplyAddSaturate, 1>::GemmKernel;
  using Operation = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

  static void run(torch::Tensor& out, const torch::Tensor& a, const torch::Tensor& b,
                  const torch::Tensor& token, const torch::Tensor& channel) {
    const int m = a.size(0);
    const int n = b.size(0);
    const int k = a.size(1) * 2;
    auto* output = reinterpret_cast<BF16*>(out.data_ptr<at::BFloat16>());
    typename Operation::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel,
        {m, n, k}, 1, Epilogue::prepare(token, channel, output, n),
        a.data_ptr<uint8_t>(), b.data_ptr<uint8_t>(), nullptr, nullptr,
        0, 0, 0, 0, k, k, n, n};
    Operation operation;
    auto workspace = torch::empty(operation.get_workspace_size(arguments), a.options());
    CUTLASS_CHECK(operation.can_implement(arguments));
    CUTLASS_CHECK(operation(arguments, workspace.data_ptr(), at::cuda::getCurrentCUDAStream(a.get_device())));
  }
};

void scaled_mm(torch::Tensor out, const torch::Tensor& a, const torch::Tensor& b,
               const torch::Tensor& token, const torch::Tensor& channel, int64_t tile) {
  TORCH_CHECK(a.dtype() == torch::kUInt8 && b.dtype() == torch::kUInt8 && out.dtype() == torch::kBFloat16);
  TORCH_CHECK(token.dtype() == torch::kFloat32 && channel.dtype() == torch::kFloat32);
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && out.dim() == 2);
  TORCH_CHECK(a.size(1) == b.size(1) && out.size(0) == a.size(0) && out.size(1) == b.size(0));
  TORCH_CHECK(token.numel() == a.size(0) && channel.numel() == b.size(0));
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && out.is_contiguous());
  TORCH_CHECK(token.is_contiguous() && channel.is_contiguous());
  switch (tile) {
    case 64: return ScaledInt4<64, 32>::run(out, a, b, token, channel);
    case 128: return ScaledInt4<128, 64>::run(out, a, b, token, channel);
    default: TORCH_CHECK(false, "Unsupported Karpo scaled INT4 tile");
  }
}
}  // namespace

TORCH_LIBRARY(karpo_ampere_int4_fused, library) {
  library.def("scaled_mm(Tensor(a!) out, Tensor a, Tensor b, Tensor token, Tensor channel, int tile) -> ()");
}

TORCH_LIBRARY_IMPL(karpo_ampere_int4_fused, CUDA, library) {
  library.impl("scaled_mm", scaled_mm);
}
