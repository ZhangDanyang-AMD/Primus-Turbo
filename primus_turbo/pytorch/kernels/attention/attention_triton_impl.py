###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import math

import torch
import triton
import triton.language as tl
import torch.distributed as dist

_torch_custom_op_wrapper = torch.library.custom_op
from typing import Optional, Tuple

from torch._library import wrap_triton

from primus_turbo.pytorch.core.float8 import float8_e4m3, float8_e5m2
from primus_turbo.triton.attention.attention_kernel import (
    DEBUG,
    FIXED_BLOCK_M,
    FIXED_BLOCK_N,
    USE_FP8E5M2_BWD,
    _bwd_kernel_dkdv,
    _bwd_kernel_dq,
    _bwd_preprocess_use_o,
    attn_fwd,
    get_padded_head_dim,
    get_shape_from_layout,
    get_strides_from_layout,
    get_tl_f8_bwd_dtype,
    is_cdna4,
    philox_offset,
    philox_seed,
)
from primus_turbo.triton.attention.mxfp8_attention_kernel import (
    _bwd_kernel_dkdv_mxfp8,
    _bwd_kernel_dq_mxfp8,
    _bwd_preprocess_use_o_mxfp8,
    attn_fwd_mxfp8,
)

fwd_torch_dtype: tl.constexpr = torch.bfloat16
bwd_torch_dtype: tl.constexpr = torch.bfloat16


def get_f8_fwd_dtype():
    return float8_e4m3


def get_f8_bwd_dtype():
    return float8_e5m2 if USE_FP8E5M2_BWD else float8_e4m3


F8_FWD_MAX: tl.constexpr = torch.finfo(get_f8_fwd_dtype()).max
F8_BWD_MAX: tl.constexpr = torch.finfo(get_f8_bwd_dtype()).max


@_torch_custom_op_wrapper("primus_turbo::attention_triton_forward_impl", mutates_args=(), device_types="cuda")
def attention_triton_forward_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    p_scale: float,
    q_descale: torch.Tensor,
    k_descale: torch.Tensor,
    v_scale: torch.Tensor,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    bias: Optional[torch.Tensor],
    alibi_slopes: Optional[torch.Tensor],
    return_softmax: bool,
    use_fp8: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    assert (
        window_size_left == -1 and window_size_right == -1
    ), "in triton attn kernel, window_size_left and window_size_right must be -1."
    assert q.is_contiguous()
    assert k.is_contiguous()
    assert v.is_contiguous()
    assert q_descale.is_contiguous()
    assert k_descale.is_contiguous()

    layout = "bshd"
    cu_seqlens_q = 0
    cu_seqlens_k = 0
    max_seqlens_q = q.shape[1]
    max_seqlens_k = k.shape[1]
    return_scores = return_softmax
    use_exp2 = True
    if DEBUG:
        print()
        print("attention_forward_triton_impl")
        print("q:", q, q.shape)
        print("k:", k, k.shape)
        print("v:", v, v.shape)
        print("softmax_scale:", softmax_scale)
        print("alibi_slopes:", alibi_slopes)
        print("causal:", causal)
        print("bias:", bias)
        print("dropout_p:", dropout_p)
        print("layout:", layout)
        print("cu_seqlens_q:", cu_seqlens_q)
        print("cu_seqlens_k:", cu_seqlens_k)
        print("max_seqlens_q:", max_seqlens_q)
        print("max_seqlens_k:", max_seqlens_k)
        print("return_scores:", return_scores)
        print("use_exp2:", use_exp2)
        print("use_fp8:", use_fp8)

    o_shape = list(q.shape)
    o_shape[-1] = v.shape[-1]  # output shape should match v's head dim
    o = torch.empty(
        o_shape,
        device=q.device,
        dtype=fwd_torch_dtype if use_fp8 else q.dtype,
        requires_grad=True,
    )

    # check if varlen
    is_varlen = layout == "thd"

    # NOTE: a large bias tensor leads to overflow during pointer arithmetic
    if bias is not None:
        assert bias.numel() < 2**31

    batch, nheads_q, nheads_k, head_size_qk, head_size_v, seqlen_q, seqlen_k = get_shape_from_layout(
        q, k, v, layout, cu_seqlens_q, cu_seqlens_k, max_seqlens_q, max_seqlens_k
    )
    q_strides = get_strides_from_layout(q, layout)
    k_strides = get_strides_from_layout(k, layout)
    v_strides = get_strides_from_layout(v, layout)
    o_strides = get_strides_from_layout(o, layout)

    # Get closest power of 2 over or equal to 32.
    padded_d_model_qk = get_padded_head_dim(head_size_qk)
    padded_d_model_v = get_padded_head_dim(head_size_v)

    grid = (triton.cdiv(max_seqlens_q, FIXED_BLOCK_M), nheads_q, batch)

    if return_scores:
        scores = torch.zeros(
            (batch, nheads_q, max_seqlens_q, max_seqlens_k), device=q.device, dtype=torch.float32
        )
        scores_scaled_shifted = torch.zeros(
            (batch, nheads_q, max_seqlens_q, max_seqlens_k), device=q.device, dtype=torch.float32
        )
        scores_strides = (scores.stride(0), scores.stride(1), scores.stride(2), scores.stride(3))
    else:
        scores = torch.empty([], device=q.device, dtype=torch.float32)
        scores_scaled_shifted = None
        scores_strides = (0, 0, 0, 0)

    # exp_scores is used to validate dropout behavior vs the PyTorch SDPA math backend reference.  We zero this out
    # to give a consistent starting point and then populate it with the output of softmax with the sign bit set according
    # to the dropout mask. The resulting return allows this mask to be fed into the reference implementation for testing
    # only.  This return holds no useful output aside from debugging.
    if return_scores:
        exp_scores = torch.zeros(
            (batch, nheads_q, max_seqlens_q, max_seqlens_k), device=q.device, dtype=torch.float32
        )
    else:
        exp_scores = torch.empty([], device=q.device, dtype=torch.float32)

    # stores LSE the log of the normalization constant / sum of expoential score(unnormalzied probablities)
    if is_varlen:
        softmax_lse = torch.empty((q.shape[0] * 2, nheads_q), device=q.device, dtype=torch.float32)
        stride_lse_m, stride_lse_h = softmax_lse.stride()
        stride_lse_z = 0
    else:
        softmax_lse = torch.empty((batch, nheads_q, max_seqlens_q * 2), device=q.device, dtype=torch.float32)
        stride_lse_z, stride_lse_h, stride_lse_m = softmax_lse.stride()

    if bias is not None:
        bias_strides = (bias.stride(0), bias.stride(1), bias.stride(2), bias.stride(3))
    else:
        bias_strides = (0, 0, 0, 0)

    if alibi_slopes is not None:
        alibi_strides = (alibi_slopes.stride(0), alibi_slopes.stride(1))
    else:
        alibi_strides = (0, 0)

    if use_fp8:
        stride_qdescale_z, stride_qdescale_h, stride_qdescale_m = (
            q_descale.stride(0),
            q_descale.stride(1),
            q_descale.stride(2),
        )
        stride_kdescale_z, stride_kdescale_h, stride_kdescale_m = (
            k_descale.stride(0),
            k_descale.stride(1),
            k_descale.stride(2),
        )
        padded_kscale_block_num = 1 << (stride_kdescale_h - 1).bit_length()

    else:
        stride_qdescale_z, stride_qdescale_h, stride_qdescale_m = None, None, None
        stride_kdescale_z, stride_kdescale_h, stride_kdescale_m = None, None, None
        padded_kscale_block_num = None

    wrap_triton(attn_fwd)[grid](
        q,
        k,
        v,
        bias,
        p_scale,
        q_descale,
        k_descale,
        v_scale,
        use_fp8,
        softmax_scale,
        softmax_lse,
        o,
        *q_strides,
        *k_strides,
        *v_strides,
        *o_strides,
        *bias_strides,
        *alibi_strides,
        *scores_strides,
        stride_lse_z,
        stride_lse_h,
        stride_lse_m,
        stride_qdescale_z,
        stride_qdescale_h,
        stride_qdescale_m,
        stride_kdescale_z,
        stride_kdescale_h,
        stride_kdescale_m,
        padded_kscale_block_num,
        cu_seqlens_q,
        cu_seqlens_k,
        dropout_p=dropout_p,
        philox_seed=philox_seed,
        philox_offset_base=philox_offset,
        scores=scores,
        scores_scaled_shifted=scores_scaled_shifted,
        exp_scores=exp_scores,
        alibi_slopes=alibi_slopes,
        HQ=nheads_q,
        HK=nheads_k,
        ACTUAL_BLOCK_DMODEL_QK=head_size_qk,
        ACTUAL_BLOCK_DMODEL_V=head_size_v,
        MAX_SEQLENS_Q=max_seqlens_q,
        MAX_SEQLENS_K=max_seqlens_k,
        IS_CAUSAL=causal,
        VARLEN=is_varlen,
        BLOCK_DMODEL_QK=padded_d_model_qk,
        BLOCK_DMODEL_V=padded_d_model_v,
        USE_BIAS=False if bias is None else True,
        USE_ALIBI=False if alibi_slopes is None else True,
        ENABLE_DROPOUT=dropout_p > 0.0,
        USE_EXP2=use_exp2,
        RETURN_SCORES=return_scores,
        BLOCK_M=FIXED_BLOCK_M,
        BLOCK_N=FIXED_BLOCK_N,
    )

    return o, softmax_lse, exp_scores


@attention_triton_forward_impl.register_fake
def _attention_triton_forward_impl_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    p_scale: float,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    bias: Optional[torch.Tensor],
    alibi_slopes: Optional[torch.Tensor],
    return_softmax: bool,
    use_fp8: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    o_shape = list(q.shape)
    o_shape[-1] = v.shape[-1]  # output shape should match v's head dim
    o = torch.empty(
        o_shape,
        device=q.device,
        dtype=torch.bfloat16 if use_fp8 else q.dtype,
        requires_grad=True,
    )

    batch_q, max_seqlen_q, nheads_q, head_size_q = q.shape
    batch_k, max_seqlen_k, nheads_k, head_size_k = k.shape

    if return_softmax:
        exp_scores = torch.zeros(
            (batch_q, nheads_q, max_seqlen_q, max_seqlen_k), device=q.device, dtype=torch.float32
        )
    else:
        exp_scores = torch.empty([], device=q.device, dtype=torch.float32)

    softmax_lse = torch.empty((batch_q, nheads_q, max_seqlen_q * 2), device=q.device, dtype=torch.float32)

    return o, softmax_lse, exp_scores


@_torch_custom_op_wrapper(
    "primus_turbo::attention_triton_backward_impl", mutates_args=("dq", "dk", "dv"), device_types="cuda"
)
def attention_triton_backward_impl(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    p_scale: float,
    softmax_lse_delta: torch.Tensor,
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
    cu_seqlens_q: int,
    cu_seqlens_k: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    alibi_slopes: Optional[torch.Tensor],
    use_fp8: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    assert (
        window_size_left == -1 and window_size_right == -1
    ), "in triton attn kernel, window_size_left and window_size_right must be -1."

    use_exp2 = True
    layout = "bshd"
    sequence_parallel = True
    do = do.contiguous()

    if DEBUG:
        print("####################################################")
        print("attention_backward_triton_new_impl")
        print("do:", do, do.shape)
        print("q:", q, q.shape)
        print("k:", k, k.shape)
        print("v:", v, v.shape)
        print("o:", o, o.shape)
        print("softmax_lse:", softmax_lse_delta, softmax_lse_delta.shape)
        print("dq:", dq, dq.shape if dq is not None else None)
        print("dk:", dk, dk.shape if dk is not None else None)
        print("dv:", dv, dv.shape if dv is not None else None)
        print("softmax_scale:", softmax_scale)
        print("alibi_slopes:", alibi_slopes)
        print("causal:", causal)
        print("layout:", layout)
        print("cu_seqlens_q:", cu_seqlens_q)
        print("cu_seqlens_k:", cu_seqlens_k)
        print("max_seqlen_q:", max_seqlen_q)
        print("max_seqlen_k:", max_seqlen_k)
        print("use_exp2:", use_exp2)
        print("sequence_parallel:", sequence_parallel)

    # get strides and shape
    batch, nheads_q, nheads_k, head_size_qk, head_size_v, max_seqlen_q, max_seqlen_k = get_shape_from_layout(
        q, k, v, layout, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k
    )
    q_strides = get_strides_from_layout(q, layout)
    k_strides = get_strides_from_layout(k, layout)
    v_strides = get_strides_from_layout(v, layout)
    o_strides = get_strides_from_layout(o, layout)
    do_strides = get_strides_from_layout(do, layout)
    stride_qz, stride_qh, stride_qm, stride_qk = q_strides
    stride_kz, stride_kh, stride_kn, stride_kk = k_strides
    stride_vz, stride_vh, stride_vn, stride_vk = v_strides
    stride_oz, stride_oh, stride_om, stride_ok = o_strides
    stride_doz, stride_doh, stride_dom, stride_dok = do_strides
    batch_headsize_q = batch * nheads_q
    batch_headsize_k = batch * nheads_k
    is_varlen = layout == "thd"

    assert head_size_qk >= 32 and head_size_v >= 32
    assert head_size_qk % 2 == 0 and head_size_v % 2 == 0
    padded_d_model_qk = get_padded_head_dim(head_size_qk)
    padded_d_model_v = get_padded_head_dim(head_size_v)

    # NOTE: we might need to copy the output tensor if they are not continuous or have other issues
    copy_back = {"dq": False, "dk": False, "dv": False}

    # deal with dq
    if dq is None:
        dq = torch.zeros_like(q, dtype=bwd_torch_dtype)
    else:
        dq_og = dq
        if not dq.is_contiguous():
            dq = dq.contiguous()
            copy_back["dq"] = True

        dq.zero_()
    stride_dq_all = dq.stride()[0]

    # deal with dk, dv
    if (dk is None) or (dv is None):
        dk = torch.zeros_like(k, dtype=bwd_torch_dtype)
        dv = torch.zeros_like(v, dtype=bwd_torch_dtype)
    else:
        if not dk.is_contiguous():
            dk_og = dk
            dk = dk.contiguous()
            copy_back["dk"] = True

        if not dv.is_contiguous():
            dv_og = dv
            dv = dv.contiguous()
            copy_back["dv"] = True

    if DEBUG:
        print("copy_back:", copy_back)

    # assert contigious
    assert do.is_contiguous()
    assert q.is_contiguous()
    assert k.is_contiguous()
    assert v.is_contiguous()
    assert o.is_contiguous()
    assert softmax_lse_delta.is_contiguous()
    assert q_scale.is_contiguous()
    assert k_scale.is_contiguous()

    # # init delta
    # delta = torch.empty_like(softmax_lse)
    if is_varlen:
        stride_lse_delta_m, stride_lse_delta_h = softmax_lse_delta.stride()
        stride_lse_delta_z = 0
    else:
        stride_lse_delta_z, stride_lse_delta_h, stride_lse_delta_m = softmax_lse_delta.stride()

    if use_fp8:
        do_fp8 = torch.empty_like(do, dtype=get_f8_bwd_dtype())
        _shape = (batch, nheads_q, triton.cdiv(max_seqlen_q, FIXED_BLOCK_M))
        do_scale = torch.empty(_shape, dtype=torch.float32, device=q.device)
        stride_descalez, stride_descaleh, stride_descalem = do_scale.stride()
        stride_qscalez, stride_qscaleh, stride_qscalem = q_scale.stride()
        stride_kscalez, stride_kscaleh, stride_kscalem = k_scale.stride()

        padded_doscale_block_num = 1 << (stride_descaleh - 1).bit_length()
        padded_qscale_block_num = 1 << (stride_qscaleh - 1).bit_length()
        padded_kscale_block_num = 1 << (stride_kscaleh - 1).bit_length()

    else:
        do_fp8 = None
        do_scale = None
        stride_descalez, stride_descaleh, stride_descalem = None, None, None
        stride_qscalez, stride_qscaleh, stride_qscalem = None, None, None
        stride_kscalez, stride_kscaleh, stride_kscalem = None, None, None
        padded_doscale_block_num, padded_qscale_block_num, padded_kscale_block_num = None, None, None

    grid_prebwd = (triton.cdiv(max_seqlen_q, FIXED_BLOCK_M), batch_headsize_q)
    wrap_triton(_bwd_preprocess_use_o)[grid_prebwd](
        o,
        do,
        do_fp8,
        do_scale,
        softmax_lse_delta,
        use_fp8,
        stride_oz,
        stride_oh,
        stride_om,
        stride_ok,
        stride_doz,
        stride_doh,
        stride_dom,
        stride_dok,
        stride_lse_delta_z,
        stride_lse_delta_h,
        stride_lse_delta_m,
        stride_descalez,
        stride_descaleh,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        BLOCK_DMODEL_V=padded_d_model_v,
        ACTUAL_BLOCK_DMODEL_V=head_size_v,
        BLOCK_M=FIXED_BLOCK_M,
        N_CTX_Q=max_seqlen_q,
        Z=batch,
        HQ=nheads_q,
        IS_VARLEN=is_varlen,
        F8_BWD_DTYPE=get_tl_f8_bwd_dtype(),
        F8_BWD_MAX=F8_BWD_MAX,
    )

    if DEBUG:
        print("####################################################")
        print("_bwd_kernel inputs")
        print("do:", do, do.shape)
        print("q:", q, q.shape)
        print("k:", k, k.shape)
        print("v:", v, v.shape)
        print("softmax_scale", softmax_scale)
        print("o:", o, o.shape)
        print("dq:", dq, dq.shape)
        print("dk:", dk, dk.shape)
        print("dv:", dv, dv.shape)
        print("L:", softmax_lse_delta, softmax_lse_delta.shape)
        print("stride_qz, stride_qh, stride_qm, stride_qk:", stride_qz, stride_qh, stride_qm, stride_qk)
        print("stride_kz, stride_kh, stride_kn, stride_kk:", stride_kz, stride_kh, stride_kn, stride_kk)
        print("stride_vz, stride_vh, stride_vn, stride_vk:", stride_vz, stride_vh, stride_vn, stride_vk)
        print("batch_q:", batch)
        print("heads_q:", nheads_q)
        print("max_seqlen_q:", max_seqlen_q)
        print("max_seqlen_k:", max_seqlen_k)
        print("BLOCK_DMODEL_QK:", padded_d_model_qk)
        print("BLOCK_DMODEL_V:", padded_d_model_v)
        print("SEQUENCE_PARALLEL:", sequence_parallel)
        print("CAUSAL:", causal)
        print("USE_EXP2:", use_exp2)

    log_p_scale = math.log(p_scale)
    num_block_m = triton.cdiv(max_seqlen_q, FIXED_BLOCK_M)
    grid_bwd = (
        batch_headsize_q,
        triton.cdiv(max_seqlen_q, FIXED_BLOCK_M) if sequence_parallel else 1,
    )
    wrap_triton(_bwd_kernel_dq)[grid_bwd](
        q,
        k,
        v,
        softmax_scale,
        q_scale,
        k_scale,
        v_scale,
        p_scale,
        do_scale,
        o,
        do_fp8 if use_fp8 else do,
        dq,
        dk,
        dv,
        softmax_lse_delta,
        stride_dq_all,
        stride_qz,
        stride_qh,
        stride_qm,
        stride_qk,
        stride_kz,
        stride_kh,
        stride_kn,
        stride_kk,
        stride_vz,
        stride_vh,
        stride_vn,
        stride_vk,
        stride_doz,
        stride_doh,
        stride_dom,
        stride_dok,
        stride_lse_delta_z,
        stride_lse_delta_h,
        stride_lse_delta_m,
        stride_descalez,
        stride_descaleh,
        stride_descalem,
        stride_qscalez,
        stride_qscaleh,
        stride_qscalem,
        stride_kscalez,
        stride_kscaleh,
        stride_kscalem,
        padded_doscale_block_num,
        padded_qscale_block_num,
        padded_kscale_block_num,
        batch,
        nheads_q,
        nheads_k,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        num_block_m=num_block_m,
        BLOCK_M=FIXED_BLOCK_M,
        BLOCK_N=FIXED_BLOCK_N,
        BLOCK_DMODEL_QK=padded_d_model_qk,
        BLOCK_DMODEL_V=padded_d_model_v,
        ACTUAL_BLOCK_DMODEL_QK=head_size_qk,
        ACTUAL_BLOCK_DMODEL_V=head_size_v,
        SEQUENCE_PARALLEL=sequence_parallel,
        CAUSAL=causal,
        USE_EXP2=use_exp2,
        IS_VARLEN=is_varlen,
        USE_FP8=use_fp8,
        log_p_scale=log_p_scale,
        F8_FWD_MAX=F8_FWD_MAX,
    )

    if use_fp8:
        n_groups = nheads_q // nheads_k
        padded_doscale_block_num = 1 << (stride_descaleh - 1).bit_length()
        padded_qscale_block_num = 1 << (stride_qscaleh * n_groups - 1).bit_length()
        padded_kscale_block_num = 1 << (stride_kscaleh * n_groups - 1).bit_length()
    else:
        padded_doscale_block_num = None
        padded_qscale_block_num = None
        padded_kscale_block_num = None

    grid_bwd_dkdv = (
        batch_headsize_k,
        triton.cdiv(max_seqlen_k, FIXED_BLOCK_N) if sequence_parallel else 1,
    )
    wrap_triton(_bwd_kernel_dkdv)[grid_bwd_dkdv](
        q,
        k,
        v,
        softmax_scale,
        q_scale,
        k_scale,
        v_scale,
        p_scale,
        do_scale,
        o,
        do_fp8 if use_fp8 else do,
        dq,
        dk,
        dv,
        softmax_lse_delta,
        stride_dq_all,
        stride_qz,
        stride_qh,
        stride_qm,
        stride_qk,
        stride_kz,
        stride_kh,
        stride_kn,
        stride_kk,
        stride_vz,
        stride_vh,
        stride_vn,
        stride_vk,
        stride_doz,
        stride_doh,
        stride_dom,
        stride_dok,
        stride_lse_delta_z,
        stride_lse_delta_h,
        stride_lse_delta_m,
        stride_descalez,
        stride_descaleh,
        stride_descalem,
        stride_qscalez,
        stride_qscaleh,
        stride_qscalem,
        stride_kscalez,
        stride_kscaleh,
        stride_kscalem,
        padded_doscale_block_num,
        padded_qscale_block_num,
        padded_kscale_block_num,
        batch,
        nheads_q,
        nheads_k,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        num_block_m=num_block_m,
        BLOCK_M=FIXED_BLOCK_M,
        BLOCK_N=FIXED_BLOCK_N,
        BLOCK_DMODEL_QK=padded_d_model_qk,
        BLOCK_DMODEL_V=padded_d_model_v,
        ACTUAL_BLOCK_DMODEL_QK=head_size_qk,
        ACTUAL_BLOCK_DMODEL_V=head_size_v,
        SEQUENCE_PARALLEL=sequence_parallel,
        CAUSAL=causal,
        USE_EXP2=use_exp2,
        IS_VARLEN=is_varlen,
        USE_FP8=use_fp8,
        log_p_scale=log_p_scale,
        F8_FWD_MAX=F8_FWD_MAX,
    )

    if DEBUG:
        print("####################################################")
        print("_bwd_kernel outputs")
        print("dq:", dq, dq.shape)
        print("dk:", dk, dk.shape)
        print("dv:", dv, dv.shape)
        # print("delta:", delta, delta.shape)

    if DEBUG:
        print("####################################################")
        print("attention_prefill_backward_triton_new_impl outputs")
        print("dq:", dq, dq.shape)
        print("dk:", dk, dk.shape)
        print("dv:", dv, dv.shape)
        # print("delta:", delta, delta.shape)
        print("copy_back:", copy_back)

    if copy_back["dq"]:
        dq_og.copy_(dq)
        dq = dq_og
    if copy_back["dk"]:
        dk_og.copy_(dk)
        dk = dk_og
    if copy_back["dv"]:
        dv_og.copy_(dv)
        dv = dv_og

    return dq, dk, dv


@attention_triton_backward_impl.register_fake
def _attention_triton_backward_impl_fake(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    p_scale: float,
    softmax_lse: torch.Tensor,
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
    cu_seqlens_q: int,
    cu_seqlens_k: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    alibi_slopes: Optional[torch.Tensor],
    use_fp8: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dq_out, dk_out, dv_out = (
        torch.empty_like(q, dtype=bwd_torch_dtype),
        torch.empty_like(k, dtype=bwd_torch_dtype),
        torch.empty_like(v, dtype=bwd_torch_dtype),
    )
    return dq_out, dk_out, dv_out


@_torch_custom_op_wrapper(
    "primus_turbo::attention_mxfp8_forward_triton_impl", mutates_args=(), device_types="cuda"
)
def attention_mxfp8_forward_triton_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    p_scale: int,  # p_scale = 127 if not quantize p
    sm_scale: float,
    alibi_slopes: Optional[torch.Tensor],
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    bias: Optional[torch.Tensor],
    dropout_p: float,
    return_softmax: bool,
    use_mxfp8: bool,
    block_m: int = 64,  # block of query seq len
    block_n: int = 64,  # block of key/value seq len
    quant_block_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    assert (
        window_size_left == -1 and window_size_right == -1
    ), "in triton attn kernel, window_size_left and window_size_right must be -1."
    assert is_cdna4(), "mxfp8 is only supported by gfx950 and newer version"
    assert q.is_contiguous()
    assert k.is_contiguous()
    assert v.is_contiguous()
    assert q_scale.is_contiguous()
    assert k_scale.is_contiguous()

    layout = "bshd"
    cu_seqlens_q = 0
    cu_seqlens_k = 0
    max_seqlens_q = q.shape[1]
    max_seqlens_k = k.shape[1]
    return_scores = return_softmax
    use_exp2 = True
    quant_size: int = 32

    if DEBUG and (not dist.is_initialized() or dist.get_rank()==0):
        print()
        print("attention_forward_triton_impl")
        print("q:", q, q.shape)
        print("k:", k, k.shape)
        print("v:", v, v.shape)
        print("sm_scale:", sm_scale)
        print("alibi_slopes:", alibi_slopes)
        print("causal:", causal)
        print("bias:", bias)
        print("dropout_p:", dropout_p)
        print("layout:", layout)
        print("cu_seqlens_q:", cu_seqlens_q)
        print("cu_seqlens_k:", cu_seqlens_k)
        print("max_seqlens_q:", max_seqlens_q)
        print("max_seqlens_k:", max_seqlens_k)
        print("return_scores:", return_scores)
        print("use_exp2:", use_exp2)
        print("use_mxfp8:", use_mxfp8)
        print("BLOCK_M:", block_m)
        print("BLOCK_N:", block_n)
        print("QUANT_BLOCK_SIZE:", quant_block_size)

    o_shape = list(q.shape)
    o_shape[-1] = v.shape[-1]  # output shape should match v's head dim
    o = torch.empty(
        o_shape,
        device=q.device,
        dtype=fwd_torch_dtype if use_mxfp8 else q.dtype,
        requires_grad=True,
    )

    # check if varlen
    is_varlen = layout == "thd"

    # NOTE: a large bias tensor leads to overflow during pointer arithmetic
    if bias is not None:
        assert bias.numel() < 2**31

    batch, nheads_q, nheads_k, head_size_qk, head_size_v, seqlen_q, seqlen_k = get_shape_from_layout(
        q, k, v, layout, cu_seqlens_q, cu_seqlens_k, max_seqlens_q, max_seqlens_k
    )

    assert quant_block_size % quant_size == 0, "quant block must be divided by quant size"
    assert block_m % quant_block_size == 0, "block M in fwd must be divided by quant size"
    assert block_n % quant_block_size == 0, "block N in fwd must be divided by quant size"

    q_strides = get_strides_from_layout(q, layout)
    k_strides = get_strides_from_layout(k, layout)
    v_strides = get_strides_from_layout(v, layout)
    o_strides = get_strides_from_layout(o, layout)

    # Get closest power of 2 over or equal to 32.
    padded_d_model_qk = get_padded_head_dim(head_size_qk)
    padded_d_model_v = get_padded_head_dim(head_size_v)
    assert padded_d_model_qk % quant_block_size == 0, "padded_d_model_qk must be divided by quant size"
    assert padded_d_model_v % quant_block_size == 0, "padded_d_model_v must be divided by quant size"

    grid = (triton.cdiv(max_seqlens_q, block_m), nheads_q, batch)

    if return_scores:
        scores = torch.zeros(
            (batch, nheads_q, max_seqlens_q, max_seqlens_k), device=q.device, dtype=torch.float32
        )
        scores_scaled_shifted = torch.zeros(
            (batch, nheads_q, max_seqlens_q, max_seqlens_k), device=q.device, dtype=torch.float32
        )
        scores_strides = (scores.stride(0), scores.stride(1), scores.stride(2), scores.stride(3))
    else:
        scores = torch.empty([], device=q.device, dtype=torch.float32)
        scores_scaled_shifted = None
        scores_strides = (0, 0, 0, 0)

    # exp_scores is used to validate dropout behavior vs the PyTorch SDPA math backend reference.  We zero this out
    # to give a consistent starting point and then populate it with the output of softmax with the sign bit set according
    # to the dropout mask. The resulting return allows this mask to be fed into the reference implementation for testing
    # only.  This return holds no useful output aside from debugging.
    if return_scores:
        exp_scores = torch.zeros(
            (batch, nheads_q, max_seqlens_q, max_seqlens_k), device=q.device, dtype=torch.float32
        )
    else:
        exp_scores = torch.empty([], device=q.device, dtype=torch.float32)

    # stores LSE the log of the normalization constant / sum of expoential score(unnormalzied probablities)
    if is_varlen:
        softmax_lse = torch.empty((q.shape[0], nheads_q), device=q.device, dtype=torch.float32)
        stride_lse_m, stride_lse_h = softmax_lse.stride()
        stride_lse_z = 0
    else:
        softmax_lse = torch.empty((batch, nheads_q, max_seqlens_q), device=q.device, dtype=torch.float32)
        stride_lse_z, stride_lse_h, stride_lse_m = softmax_lse.stride()

    if bias is not None:
        bias_strides = (bias.stride(0), bias.stride(1), bias.stride(2), bias.stride(3))
    else:
        bias_strides = (0, 0, 0, 0)

    if alibi_slopes is not None:
        alibi_strides = (alibi_slopes.stride(0), alibi_slopes.stride(1))
    else:
        alibi_strides = (0, 0)

    if use_mxfp8:
        stride_qdescale_z, stride_qdescale_h, stride_qdescale_m, stride_qdescale_d = get_strides_from_layout(
            q_scale, layout
        )
        stride_kdescale_z, stride_kdescale_h, stride_kdescale_m, stride_kdescale_d = get_strides_from_layout(
            k_scale, layout
        )
        stride_vdescale_z, stride_vdescale_h, stride_vdescale_m, stride_vdescale_d = get_strides_from_layout(
            v_scale, layout
        )
    else:
        stride_qdescale_z, stride_qdescale_h, stride_qdescale_m, stride_qdescale_d = None, None, None, None
        stride_kdescale_z, stride_kdescale_h, stride_kdescale_m, stride_kdescale_d = None, None, None, None
        stride_vdescale_z, stride_vdescale_h, stride_vdescale_m, stride_vdescale_d = None, None, None, None

    kernel_kwargs = {}
    if padded_d_model_qk % 128 == 0 and block_n % 128 == 0:
        kernel_kwargs["matrix_instr_nonkdim"] = 16

    wrap_triton(attn_fwd_mxfp8)[grid](
        q,
        k,
        v,
        bias,
        p_scale,
        q_scale,
        k_scale,
        v_scale,
        use_mxfp8,
        sm_scale,
        softmax_lse,
        o,
        *q_strides,
        *k_strides,
        *v_strides,
        *o_strides,
        *bias_strides,
        *alibi_strides,
        *scores_strides,
        stride_lse_z,
        stride_lse_h,
        stride_lse_m,
        stride_qdescale_z,
        stride_qdescale_h,
        stride_qdescale_m,
        stride_qdescale_d,
        stride_kdescale_z,
        stride_kdescale_h,
        stride_kdescale_m,
        stride_kdescale_d,
        stride_vdescale_z,
        stride_vdescale_h,
        stride_vdescale_m,
        stride_vdescale_d,
        cu_seqlens_q,
        cu_seqlens_k,
        dropout_p=dropout_p,
        philox_seed=philox_seed,
        philox_offset_base=philox_offset,
        scores=scores,
        scores_scaled_shifted=scores_scaled_shifted,
        exp_scores=exp_scores,
        alibi_slopes=alibi_slopes,
        HQ=nheads_q,
        HK=nheads_k,
        ACTUAL_BLOCK_DMODEL_QK=head_size_qk,
        ACTUAL_BLOCK_DMODEL_V=head_size_v,
        MAX_SEQLENS_Q=max_seqlens_q,
        MAX_SEQLENS_K=max_seqlens_k,
        IS_CAUSAL=causal,
        VARLEN=is_varlen,
        BLOCK_DMODEL_QK=padded_d_model_qk,
        BLOCK_DMODEL_V=padded_d_model_v,
        USE_BIAS=False if bias is None else True,
        USE_ALIBI=False if alibi_slopes is None else True,
        ENABLE_DROPOUT=dropout_p > 0.0,
        USE_EXP2=use_exp2,
        RETURN_SCORES=return_scores,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        QUANT_BLOCK_SIZE=quant_block_size,
        QUANT_SIZE=quant_size,
        **kernel_kwargs,
    )

    return o, softmax_lse, exp_scores


@attention_mxfp8_forward_triton_impl.register_fake
def fake_attention_mxfp8_forward_triton_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    p_scale: int,  # p_scale = 127 if not quantize p
    sm_scale: float,
    alibi_slopes: Optional[torch.Tensor],
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    bias: Optional[torch.Tensor],
    dropout_p: float,
    return_softmax: bool,
    use_mxfp8: bool,
    block_m: int = 64,  # block of query seq len
    block_n: int = 64,  # block of key/value seq len
    quant_block_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    o_shape = list(q.shape)
    o_shape[-1] = v.shape[-1]  # output shape should match v's head dim
    o = torch.empty(
        o_shape,
        device=q.device,
        dtype=fwd_torch_dtype if use_mxfp8 else q.dtype,
        requires_grad=True,
    )

    batch_q, max_seqlen_q, nheads_q, head_size_q = q.shape
    batch_k, max_seqlen_k, nheads_k, head_size_k = k.shape

    if return_softmax:
        exp_scores = torch.zeros(
            (batch_q, nheads_q, max_seqlen_q, max_seqlen_k), device=q.device, dtype=torch.float32
        )
    else:
        exp_scores = torch.empty([], device=q.device, dtype=torch.float32)

    softmax_lse = torch.empty((batch_q, nheads_q, max_seqlen_q * 2), device=q.device, dtype=torch.float32)

    return o, softmax_lse, exp_scores


@_torch_custom_op_wrapper(
    "primus_turbo::attention_triton_mxfp8_backward_triton_impl",
    mutates_args=("dq", "dk", "dv"),
    device_types="cuda",
)
def attention_triton_mxfp8_backward_triton_impl(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
    q_scale: torch.Tensor,  # p_scale = 127 if not quantize p
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    sm_scale: float,
    p_scale: int,
    alibi_slopes: Optional[torch.Tensor],
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    cu_seqlens_q: Optional[int],
    cu_seqlens_k: Optional[int],
    max_seqlen_q: Optional[int],
    max_seqlen_k: Optional[int],
    use_mxfp8: bool,
    block_m_dq_bwd: int = 64,  # block of dq seq len in bwd
    block_n_dq_bwd: int = 64,  # block of dq seq len in bwd
    block_m_dkv_bwd: int = 64,  # block of dkv seq len in bwd
    block_n_dkv_bwd: int = 64,  # block of dkv seq len in bwd
    quant_block_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    assert is_cdna4(), "mxfp8 is only supported by gfx950 and newer version"
    assert (
        window_size_left == -1 and window_size_right == -1
    ), "in triton attn kernel, window_size_left and window_size_right must be -1."

    use_exp2 = True
    layout = "bshd"
    do = do.contiguous()
    quant_size: int = 32

    if DEBUG and (not dist.is_initialized() or dist.get_rank()==0):
        print("####################################################")
        print("attention_backward_triton_new_impl")
        print("do:", do, do.shape)
        print("q:", q, q.shape)
        print("k:", k, k.shape)
        print("v:", v, v.shape)
        print("o:", o, o.shape)
        print("softmax_lse:", softmax_lse, softmax_lse.shape)
        print("dq:", dq, dq.shape if dq is not None else None)
        print("dk:", dk, dk.shape if dk is not None else None)
        print("dv:", dv, dv.shape if dv is not None else None)
        print("sm_scale:", sm_scale)
        print("alibi_slopes:", alibi_slopes)
        print("causal:", causal)
        print("layout:", layout)
        print("cu_seqlens_q:", cu_seqlens_q)
        print("cu_seqlens_k:", cu_seqlens_k)
        print("max_seqlen_q:", max_seqlen_q)
        print("max_seqlen_k:", max_seqlen_k)
        print("use_exp2:", use_exp2)
        print("use_mxfp8:", use_mxfp8)
        print("block_m_dq_bwd:", block_m_dq_bwd)
        print("block_n_dq_bwd:", block_n_dq_bwd)
        print("block_m_dkv_bwd:", block_m_dkv_bwd)
        print("block_n_dkv_bwd:", block_n_dkv_bwd)
        print("quant_block_size:", quant_block_size)

    # make contigious
    assert is_cdna4(), "mxfp8 is only supported by gfx950 and newer version"
    if not do.is_contiguous():
        do = do.contiguous()
    assert q.is_contiguous()
    assert k.is_contiguous()
    assert v.is_contiguous()
    assert o.is_contiguous()
    assert softmax_lse.is_contiguous()
    assert q_scale.is_contiguous()
    assert k_scale.is_contiguous()

    # get strides and shape
    batch, nheads_q, nheads_k, head_size_qk, head_size_v, max_seqlen_q, max_seqlen_k = get_shape_from_layout(
        q, k, v, layout, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k
    )

    assert quant_block_size % quant_size == 0, "quant block must be divided by quant size"
    assert block_m_dq_bwd % quant_block_size == 0, "block M in dq bwd must be divided by quant size"
    assert block_m_dkv_bwd % quant_block_size == 0, "block M in dkv bwd must be divided by quant size"
    assert block_n_dq_bwd % quant_block_size == 0, "block N in dq bwd must be divided by quant size"
    assert block_n_dkv_bwd % quant_block_size == 0, "block N in dkv bwd must be divided by quant size"

    q_strides = get_strides_from_layout(q, layout)
    k_strides = get_strides_from_layout(k, layout)
    v_strides = get_strides_from_layout(v, layout)
    o_strides = get_strides_from_layout(o, layout)
    do_strides = get_strides_from_layout(do, layout)
    stride_qz, stride_qh, stride_qm, stride_qk = q_strides
    stride_kz, stride_kh, stride_kn, stride_kk = k_strides
    stride_vz, stride_vh, stride_vn, stride_vk = v_strides
    stride_oz, stride_oh, stride_om, stride_ok = o_strides
    stride_doz, stride_doh, stride_dom, stride_dok = do_strides
    batch_headsize_q = batch * nheads_q
    batch_headsize_k = batch * nheads_k
    is_varlen = layout == "thd"

    assert head_size_qk >= 32 and head_size_v >= 32
    assert head_size_qk % 2 == 0 and head_size_v % 2 == 0
    padded_d_model_qk = get_padded_head_dim(head_size_qk)
    padded_d_model_v = get_padded_head_dim(head_size_v)
    assert padded_d_model_qk % quant_block_size == 0, "padded_d_model_qk must be divided by quant size"
    assert padded_d_model_v % quant_block_size == 0, "padded_d_model_v must be divided by quant size"

    # NOTE: we might need to copy the output tensor if they are not continuous or have other issues
    copy_back = {"dq": False, "dk": False, "dv": False}

    # deal with dq
    if dq is None:
        dq = torch.zeros_like(q, dtype=bwd_torch_dtype)
    else:
        dq_og = dq
        if not dq.is_contiguous():
            dq = dq.contiguous()
            copy_back["dq"] = True

        dq.zero_()
    dq.stride()[0]

    # deal with dk, dv
    if (dk is None) or (dv is None):
        dk = torch.zeros_like(k, dtype=bwd_torch_dtype)
        dv = torch.zeros_like(v, dtype=bwd_torch_dtype)
    else:
        if not dk.is_contiguous():
            dk_og = dk
            dk = dk.contiguous()
            copy_back["dk"] = True

        if not dv.is_contiguous():
            dv_og = dv
            dv = dv.contiguous()
            copy_back["dv"] = True

    if DEBUG:
        print("copy_back:", copy_back)

    # init delta
    delta = torch.empty_like(softmax_lse)
    if is_varlen:
        stride_lse_delta_m, stride_lse_delta_h = softmax_lse.stride()
        stride_lse_delta_z = 0
    else:
        stride_lse_delta_z, stride_lse_delta_h, stride_lse_delta_m = softmax_lse.stride()

    if use_mxfp8:
        # _shape = (batch, nheads_q, triton.cdiv(max_seqlen_q, quant_block_size), triton.cdiv(head_size_v, quant_block_size))
        m_blocks_q = triton.cdiv(max_seqlen_q, quant_block_size)
        dv_blocks = triton.cdiv(head_size_v, quant_block_size)
        if layout == "bhsd":
            _shape = (batch, nheads_q, m_blocks_q, dv_blocks)
        elif layout == "bshd":
            _shape = (batch, m_blocks_q, nheads_q, dv_blocks)
        elif layout == "thd":
            _shape = (q_scale.shape[0], q_scale.shape[1], dv_blocks)
        else:
            raise AssertionError(f"Got unsupported layout for do_scale: {layout}")
        do_fp8 = torch.empty_like(do, dtype=get_f8_bwd_dtype())
        do_scale = torch.empty(_shape, dtype=torch.uint8, device=q.device)
        stride_dodescalez, stride_dodescaleh, stride_dodescalem, stride_dodescaled = get_strides_from_layout(
            do_scale, layout
        )
        stride_qdescalez, stride_qdescaleh, stride_qdescalem, stride_qdescaled = get_strides_from_layout(
            q_scale, layout
        )
        stride_kdescalez, stride_kdescaleh, stride_kdescalem, stride_kdescaled = get_strides_from_layout(
            k_scale, layout
        )
        stride_vdescalez, stride_vdescaleh, stride_vdescalem, stride_vdescaled = get_strides_from_layout(
            v_scale, layout
        )

    else:
        do_fp8 = None
        do_scale = None
        stride_dodescalez, stride_dodescaleh, stride_dodescalem, stride_dodescaled = None, None, None, None
        stride_qdescalez, stride_qdescaleh, stride_qdescalem, stride_qdescaled = None, None, None, None
        stride_kdescalez, stride_kdescaleh, stride_kdescalem, stride_kdescaled = None, None, None, None
        stride_vdescalez, stride_vdescaleh, stride_vdescalem, stride_vdescaled = None, None, None, None

    preprocess_o_block = 64 if max_seqlen_q > 64 else max_seqlen_q
    preprocess_o_block = quant_block_size if quant_block_size > preprocess_o_block else preprocess_o_block
    grid_prebwd = (triton.cdiv(max_seqlen_q, preprocess_o_block), batch_headsize_q)
    wrap_triton(_bwd_preprocess_use_o_mxfp8)[grid_prebwd](
        o,
        do,
        do_fp8,
        do_scale,
        delta,
        use_mxfp8,
        stride_oz,
        stride_oh,
        stride_om,
        stride_ok,
        stride_doz,
        stride_doh,
        stride_dom,
        stride_dok,
        stride_lse_delta_z,
        stride_lse_delta_h,
        stride_lse_delta_m,
        stride_dodescalez,
        stride_dodescaleh,
        stride_dodescalem,
        stride_dodescaled,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        BLOCK_M=preprocess_o_block,
        BLOCK_DMODEL_V=padded_d_model_v,
        ACTUAL_BLOCK_DMODEL_V=head_size_v,
        N_CTX_Q=max_seqlen_q,
        Z=batch,
        HQ=nheads_q,
        IS_VARLEN=is_varlen,
        F8_BWD_DTYPE=get_tl_f8_bwd_dtype(),
        QUANT_BLOCK_SIZE=quant_block_size,
    )

    if DEBUG and (not dist.is_initialized() or dist.get_rank()==0):
        print("####################################################")
        print("_bwd_kernel inputs")
        print("do:", do, do.shape)
        print("q:", q, q.shape)
        print("k:", k, k.shape)
        print("v:", v, v.shape)
        print("sm_scale", sm_scale)
        print("o:", o, o.shape)
        print("dq:", dq, dq.shape)
        print("dk:", dk, dk.shape)
        print("dv:", dv, dv.shape)
        print("L:", softmax_lse, softmax_lse.shape)
        # print("delta:", delta, delta.shape)
        print("stride_qz, stride_qh, stride_qm, stride_qk:", stride_qz, stride_qh, stride_qm, stride_qk)
        print("stride_kz, stride_kh, stride_kn, stride_kk:", stride_kz, stride_kh, stride_kn, stride_kk)
        print("stride_vz, stride_vh, stride_vn, stride_vk:", stride_vz, stride_vh, stride_vn, stride_vk)
        print("batch_q:", batch)
        print("heads_q:", nheads_q)
        print("max_seqlen_q:", max_seqlen_q)
        print("max_seqlen_k:", max_seqlen_k)
        print("BLOCK_DMODEL_QK:", padded_d_model_qk)
        print("BLOCK_DMODEL_V:", padded_d_model_v)
        print("CAUSAL:", causal)
        print("USE_EXP2:", use_exp2)

    num_block_m = triton.cdiv(max_seqlen_q, block_m_dq_bwd)
    grid_bwd = (
        batch_headsize_q,
        num_block_m,
    )
    kernel_kwargs = {}
    # use mfma_16x16x128 when these K can be divided by 128
    if block_n_dq_bwd % 128 == 0 and padded_d_model_qk % 128 == 0 and padded_d_model_v % 128 == 0:
        kernel_kwargs["matrix_instr_nonkdim"] = 16
    else:
        pass

    p_scale_t = math.pow(2.0, int(p_scale - 127))
    log_p_scale = math.log(p_scale_t)

    wrap_triton(_bwd_kernel_dq_mxfp8)[grid_bwd](
        q,
        k,
        v,
        sm_scale,
        p_scale,
        log_p_scale,
        q_scale,
        k_scale,
        v_scale,
        do_scale,
        o,
        do_fp8 if use_mxfp8 else do,
        dq,
        dk,
        dv,
        softmax_lse,
        delta,
        stride_qz,
        stride_qh,
        stride_qm,
        stride_qk,
        stride_kz,
        stride_kh,
        stride_kn,
        stride_kk,
        stride_vz,
        stride_vh,
        stride_vn,
        stride_vk,
        stride_doz,
        stride_doh,
        stride_dom,
        stride_dok,
        stride_lse_delta_z,
        stride_lse_delta_h,
        stride_lse_delta_m,
        stride_dodescalez,
        stride_dodescaleh,
        stride_dodescalem,
        stride_dodescaled,
        stride_qdescalez,
        stride_qdescaleh,
        stride_qdescalem,
        stride_qdescaled,
        stride_kdescalez,
        stride_kdescaleh,
        stride_kdescalem,
        stride_kdescaled,
        stride_vdescalez,
        stride_vdescaleh,
        stride_vdescalem,
        stride_vdescaled,
        batch,
        nheads_q,
        nheads_k,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        num_block_m=num_block_m,
        BLOCK_M=block_m_dq_bwd,
        BLOCK_N=block_n_dq_bwd,
        BLOCK_DMODEL_QK=padded_d_model_qk,
        BLOCK_DMODEL_V=padded_d_model_v,
        ACTUAL_BLOCK_DMODEL_QK=head_size_qk,
        ACTUAL_BLOCK_DMODEL_V=head_size_v,
        CAUSAL=causal,
        USE_EXP2=use_exp2,
        IS_VARLEN=is_varlen,
        use_mxfp8=use_mxfp8,
        F8_BWD_DTYPE=get_tl_f8_bwd_dtype(),
        QUANT_BLOCK_SIZE=quant_block_size,
        QUANT_SIZE=quant_size,
        **kernel_kwargs,
    )

    # use mfma_16x16x128 when these K can be divided by 128
    if block_m_dkv_bwd % 128 == 0 and padded_d_model_qk % 128 == 0 and padded_d_model_v % 128 == 0:
        kernel_kwargs["matrix_instr_nonkdim"] = 16
    else:
        kernel_kwargs = {}

    grid_bwd_dkdv = (
        batch_headsize_k,
        triton.cdiv(max_seqlen_k, block_n_dkv_bwd),
    )
    wrap_triton(_bwd_kernel_dkdv_mxfp8)[grid_bwd_dkdv](
        q,
        k,
        v,
        p_scale,
        log_p_scale,
        sm_scale,
        q_scale,
        k_scale,
        v_scale,
        do_scale,
        o,
        do_fp8 if use_mxfp8 else do,
        dq,
        dk,
        dv,
        softmax_lse,
        delta,
        stride_qz,
        stride_qh,
        stride_qm,
        stride_qk,
        stride_kz,
        stride_kh,
        stride_kn,
        stride_kk,
        stride_vz,
        stride_vh,
        stride_vn,
        stride_vk,
        stride_doz,
        stride_doh,
        stride_dom,
        stride_dok,
        stride_lse_delta_z,
        stride_lse_delta_h,
        stride_lse_delta_m,
        stride_dodescalez,
        stride_dodescaleh,
        stride_dodescalem,
        stride_dodescaled,
        stride_qdescalez,
        stride_qdescaleh,
        stride_qdescalem,
        stride_qdescaled,
        stride_kdescalez,
        stride_kdescaleh,
        stride_kdescalem,
        stride_kdescaled,
        stride_vdescalez,
        stride_vdescaleh,
        stride_vdescalem,
        stride_vdescaled,
        batch,
        nheads_q,
        nheads_k,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        num_block_m=num_block_m,
        BLOCK_M=block_m_dkv_bwd,
        BLOCK_N=block_n_dkv_bwd,
        BLOCK_DMODEL_QK=padded_d_model_qk,
        BLOCK_DMODEL_V=padded_d_model_v,
        ACTUAL_BLOCK_DMODEL_QK=head_size_qk,
        ACTUAL_BLOCK_DMODEL_V=head_size_v,
        CAUSAL=causal,
        USE_EXP2=use_exp2,
        IS_VARLEN=is_varlen,
        use_mxfp8=use_mxfp8,
        F8_BWD_DTYPE=get_tl_f8_bwd_dtype(),
        QUANT_BLOCK_SIZE=quant_block_size,
        QUANT_SIZE=quant_size,
        **kernel_kwargs,
    )

    if DEBUG and (not dist.is_initialized() or dist.get_rank()==0):
        print("####################################################")
        print("_bwd_kernel scales")
        print("q_scale:", q_scale, q_scale.shape, q_scale.dtype)
        print("k_scale:", k_scale, k_scale.shape, k_scale.dtype)
        print("v_scale:", v_scale, v_scale.shape, v_scale.dtype)

        print("####################################################")
        print("attention_prefill_backward_triton_new_impl outputs")
        print("dq:", dq, dq.shape, dq.dtype)
        print("dk:", dk, dk.shape, dk.dtype)
        print("dv:", dv, dv.shape, dv.dtype)
        print("copy_back:", copy_back)
        # print("delta:", delta, delta.shape)

    if copy_back["dq"]:
        dq_og.copy_(dq)
        dq = dq_og
    if copy_back["dk"]:
        dk_og.copy_(dk)
        dk = dk_og
    if copy_back["dv"]:
        dv_og.copy_(dv)
        dv = dv_og

    return dq, dk, dv


@attention_triton_mxfp8_backward_triton_impl.register_fake
def fake_attention_triton_mxfp8_backward_triton_impl(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
    q_scale: torch.Tensor,  # p_scale = 127 if not quantize p
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    sm_scale: float,
    p_scale: int,
    alibi_slopes: Optional[torch.Tensor],
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    cu_seqlens_q: Optional[int],
    cu_seqlens_k: Optional[int],
    max_seqlen_q: Optional[int],
    max_seqlen_k: Optional[int],
    use_mxfp8: bool,
    block_m_dq_bwd: int = 64,  # block of dq seq len in bwd
    block_n_dq_bwd: int = 64,  # block of dq seq len in bwd
    block_m_dkv_bwd: int = 64,  # block of dkv seq len in bwd
    block_n_dkv_bwd: int = 64,  # block of dkv seq len in bwd
    quant_block_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.empty_like(q, dtype=fwd_torch_dtype),
        torch.empty_like(k, dtype=fwd_torch_dtype),
        torch.empty_like(v, dtype=fwd_torch_dtype),
    )
