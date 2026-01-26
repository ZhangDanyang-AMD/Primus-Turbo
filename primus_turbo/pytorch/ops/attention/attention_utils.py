###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from abc import ABC
from typing import List

import torch

from primus_turbo.pytorch.kernels.attention.attention_triton_impl import (
    get_f8_fwd_dtype,
)
from primus_turbo.pytorch.kernels.quantize import convert_to_mxfp8
from primus_turbo.triton.attention.attention_kernel import FIXED_BLOCK_M


def _check_and_convert(t, scale, float8_fw):
    finfo = torch.finfo(float8_fw)
    return (t * scale).clamp(min=finfo.min, max=finfo.max).to(dtype=float8_fw) if t.dtype != float8_fw else t


def block_scaling_node(tensor, use_fp8, BLOCK_M=FIXED_BLOCK_M, float8_dtype=get_f8_fwd_dtype()):
    """
    Used to scale tensor in per-block mode

    Inputs:
        tensor(Tensor): bf16 tensor
        BLOCK_M(int): triton block size
        float8_dtype(Tensor.dtype): float8_dtype

    Output:
        fp8tensor(Tensor): tensor after blockwise quant
        unscale_tensor(Tensor): tensor for unscale quanted tensor from fp8 to bf16
    """
    if use_fp8:
        tensor = tensor.permute(0, 2, 1, 3)  # [B, H, L, D]
        B, H, L, D = tensor.shape
        tensor = tensor.reshape(B, H, L // BLOCK_M, BLOCK_M, D).reshape(B, H, L // BLOCK_M, BLOCK_M * D)
        MAX_E4M3 = torch.finfo(float8_dtype).max
        tensor_max = tensor.abs().max(dim=-1)[0]
        tensor_max = torch.where(tensor_max == 0, MAX_E4M3, tensor_max)
        scale = MAX_E4M3 / tensor_max
        tensor = tensor * scale.reshape(scale.shape + (1,))
        tensor = tensor.clamp(-MAX_E4M3, MAX_E4M3)
        tensor = tensor.to(float8_dtype)
        tensor = tensor.reshape(B, H, L, D).permute(0, 2, 1, 3).contiguous()
        # [B, L, H, D]
        return tensor, 1.0 / scale.to(torch.float32).contiguous()
    else:
        scale = torch.tensor([1.0], device=tensor.device)
        return tensor, scale


def block_scaling_node_mxfp8(
    tensor,
    quant_block_size,
    layout,
    is_2d_block=True,
    float8_dtype_pt=get_f8_fwd_dtype(),
    cu_seqlens=None,
    max_seqlens=None,
):
    if layout == "bhsd":
        tensor_bhsd = tensor
        B, H, S, D = tensor_bhsd.shape

    elif layout == "bshd":
        tensor_bhsd = tensor.permute(0, 2, 1, 3).contiguous()
        B, H, S, D = tensor_bhsd.shape

    elif layout == "thd":
        assert cu_seqlens is not None, "thd layout requires cu_seqlens"
        assert tensor.dim() == 3, f"expected thd tensor shape [T,H,D], got {tensor.shape}"
        T, H, D = tensor.shape
        B = int(cu_seqlens.numel() - 1)

        assert max_seqlens is not None, "thd layout requires max_seqlens"
        if isinstance(max_seqlens, int):
            max_seqlen = int(max_seqlens)
        else:
            max_seqlen = int(max(max_seqlens))

        if is_2d_block:
            padded_max_seqlen = ((max_seqlen + quant_block_size - 1) // quant_block_size) * quant_block_size
        else:
            padded_max_seqlen = max_seqlen

        tensor_bhsd = torch.zeros(
            (B, H, padded_max_seqlen, D),
            device=tensor.device,
            dtype=tensor.dtype,
        )
        for b in range(B):
            s = int(cu_seqlens[b].item())
            e = int(cu_seqlens[b + 1].item())
            L = e - s
            assert L <= max_seqlen, f"sequence length {L} exceeds max_seqlen {max_seqlen}"
            tensor_bhsd[b, :, :L, :] = tensor[s:e].transpose(0, 1)

    else:
        raise AssertionError(f"Got unsupported layout: {layout}")

    quanted_bhsd, scale_bhsd = convert_to_mxfp8(
        tensor_bhsd,
        block_size=quant_block_size,
        axis=-1,
        is_2d_block=is_2d_block,
        float8_dtype_pt=float8_dtype_pt,
    )

    if layout == "bshd":
        return (
            quanted_bhsd.permute(0, 2, 1, 3).contiguous(),
            scale_bhsd.permute(0, 2, 1, 3).contiguous(),
        )

    if layout == "thd":
        qs = []
        ss = []
        for b in range(B):
            s0 = int(cu_seqlens[b].item())
            e0 = int(cu_seqlens[b + 1].item())
            L = e0 - s0

            q_chunk = quanted_bhsd[b, :, :L, :].transpose(0, 1).contiguous()
            qs.append(q_chunk)

            if is_2d_block:
                m_blocks = (L + quant_block_size - 1) // quant_block_size
                s_chunk = scale_bhsd[b, :, :m_blocks, :].transpose(0, 1).contiguous()
            else:
                s_chunk = scale_bhsd[b, :, :L, :].transpose(0, 1).contiguous()

            ss.append(s_chunk)

        return torch.cat(qs, dim=0), torch.cat(ss, dim=0)

    return quanted_bhsd, scale_bhsd


def quant_p_scale_mxfp8():
    mxfp8_fw = get_f8_fwd_dtype()
    p_scale = torch.finfo(mxfp8_fw).max
    print("p_scale", bin(int(p_scale)))
    print("p_scale", int(p_scale))
    if mxfp8_fw == torch.float8_e4m3fn:
        mask_s = 0b1111
        mbits = 3
        s_bias = 7
    else:
        mask_s = 0b11111
        mbits = 2
        s_bias = 15

    hp_ebias = 127

    p_scale = torch.bitwise_right_shift(torch.tensor(p_scale).to(mxfp8_fw).view(torch.uint8), mbits) & mask_s
    p_scale = (p_scale - s_bias + hp_ebias).to(torch.uint32).item()
    return p_scale


def quant_v_get_p_scale(v, use_fp8: bool):
    """
    Get p_scale for quant_v_getp_scale
    """
    if use_fp8:
        range_v = torch.max(torch.abs(v))

        float8_fw = get_f8_fwd_dtype()
        dtype_max = torch.finfo(float8_fw).max

        v_scale = dtype_max / range_v
        p_scale = dtype_max
        v = _check_and_convert(v, v_scale, float8_fw)

    else:
        v_scale = torch.tensor([1.0], device=v.device)
        p_scale = 1.0

    return v, v_scale, p_scale


class AttentionSharder(ABC):
    """AttentionSharder Interface"""

    def shard_cp_input(self, input_tensors: List[torch.Tensor], cp_group) -> List[torch.Tensor]:
        """
        Shard input from whole seq to specific cp rank, the implementation differ from different cp-comm type

        Inputs:
            input_tensors: tensors to shard as [Q, K, V]
            cp_groups: cp communication process group
        """


class All2AllAttentionSharder(AttentionSharder):
    """All2All AttentionSharder Impl"""

    def shard_cp_input(self, input_tensors: List[torch.Tensor], cp_group, seq_dim=1) -> List[torch.Tensor]:
        cp_size = cp_group.size()
        cp_rank = cp_group.rank()

        output_list = []
        for t in input_tensors:
            output_list.append(t.chunk(cp_size, seq_dim)[cp_rank].contiguous())

        return output_list
