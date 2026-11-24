import os
from importlib.util import find_spec
from unittest import TestCase, skipUnless


@skipUnless(os.environ.get("KARPO_KERNEL_DIR") and find_spec("torch"), "set KARPO_KERNEL_DIR on an A100")
class KernelTests(TestCase):
    def test_scaled_operators_against_explicit_integer_products(self):
        import torch

        from karpo_accel.build_kernels import load_kernels
        from karpo_accel.projections import apply_rotation, quantize8, quantize_rotated4

        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0):
            self.skipTest("requires SM80")
        load_kernels(os.environ["KARPO_KERNEL_DIR"], "w4a4")
        rng = torch.Generator(device="cuda").manual_seed(2314)
        x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16, generator=rng)
        w = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16, generator=rng)
        out = torch.empty(64, 128, device="cuda", dtype=torch.bfloat16)
        a, sa = quantize8(x)
        b, sb = quantize8(w)
        torch.ops.karpo_ampere.scaled_mm(out, a, b.T, sa[:, None], sb[None, :])
        expected = (a.cpu().float() @ b.cpu().float().T) * sa.cpu()[:, None] * sb.cpu()[None, :]
        torch.testing.assert_close(out.cpu(), expected.bfloat16(), rtol=0.01, atol=0.02)

        def unpack(packed):
            packed = packed.cpu()
            low = (packed << 4).to(torch.int8) >> 4
            high = packed.to(torch.int8) >> 4
            return torch.stack((low, high), dim=-1).flatten(1).float()

        signs = apply_rotation(x.shape[1], x.device)
        a, sa = quantize_rotated4(x, signs)
        b, sb = quantize_rotated4(w, signs)
        torch.ops.karpo_ampere_int4_fused.scaled_mm(out, a, b, sa, sb, 64)
        expected = (unpack(a) @ unpack(b).T) * sa.cpu()[:, None] * sb.cpu()[None, :]
        torch.testing.assert_close(out.cpu(), expected.bfloat16(), rtol=0.01, atol=0.02)
