from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root mean square normalization.

    Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
    Refer to https://arxiv.org/abs/1910.07467
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.variance_epsilon = eps
        self.weight = nn.Parameter(torch.randn(hidden_size, dtype=torch.float16))

    def forward_native(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if residual is not None:
            x = x + residual.to(torch.float32)
            residual = x.to(orig_dtype)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x.to(orig_dtype) * self.weight
        if residual is None:
            return x
        else:
            return x, residual

    # 使用 vllm 的自定义算子进行推理
    def forward_with_vllm_op(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        from vllm import _custom_ops as ops

        if residual is not None:
            ops.fused_add_rms_norm(x, residual, self.weight.data, self.variance_epsilon)
            return x, residual
        out = torch.empty_like(x)
        ops.rms_norm(out, x, self.weight.data, self.variance_epsilon)
        return out

    # 使用 forward_with_flashinfer 的算子进行推理
    def forward_with_flashinfer(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        from flashinfer.norm import (
                fused_add_rmsnorm,
                rmsnorm,
            )

        if residual is not None:
            fused_add_rmsnorm(x, residual, self.weight.data, self.variance_epsilon)
            return x, residual
        out = rmsnorm(x, self.weight.data, self.variance_epsilon)
        return out

if __name__ == "__main__":
    rms_norm_module = RMSNorm(2048).cuda()

    inp = torch.randn(1024, 2048, device="cuda:0", dtype=torch.float16)
    output = rms_norm_module.forward_native(x=inp)
    output_2 = rms_norm_module.forward_with_vllm_op(x=inp)
    output_3 = rms_norm_module.forward_with_flashinfer(x=inp)
    is_close = torch.allclose(output, output_2, rtol=1e-03, atol=1e-04)
    is_close_2 = torch.allclose(output, output_3, rtol=1e-03, atol=1e-04)
    print(f"{is_close}")
    print(f"{is_close_2}")


    inp = torch.randn(1024, 2048, device="cuda:0", dtype=torch.float16)
    res = torch.randn(1024, 2048, device="cuda:0", dtype=torch.float16)
    output, _ = rms_norm_module.forward_native(x=inp.clone(), residual=res.clone())
    output_2, _ = rms_norm_module.forward_with_vllm_op(x=inp.clone(), residual=res.clone())
    output_3, _ = rms_norm_module.forward_with_flashinfer(x=inp.clone(), residual=res.clone())
    is_close = torch.allclose(output, output_2, rtol=1e-3, atol=1e-04)
    is_close_2 = torch.allclose(output, output_3, rtol=1e-03, atol=1e-04)
    print(f"{is_close}")
    print(f"{is_close_2}")
