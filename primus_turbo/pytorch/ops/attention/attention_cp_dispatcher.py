###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Literal, Optional
from .attention_with_cp_a2a import (
    AttentionCKFunctionCPA2A,
    AttentionTritonFunctionCPA2A,
    AttentionTritonMXFP8FunctionCPA2A,
)

def dispatch_attention_cp_functions(
    q,
    k,
    v,
    dropout_p,
    softmax_scale,
    causal,
    window_size,
    bias,
    alibi_slopes,
    deterministic,
    return_lse,
    return_attn_probs,
    is_grad_enabled,
    backend_type,    
    quant_type : Optional[Literal["fp8_blockwise", "mxfp8"]], # "fp8", "mxfp8", "None"
    cp_group,
    cp_comm_type,    
    block_m_fwd: int = 64,  # block of query seq len in fwd
    block_n_fwd: int = 64,  # block of key/value seq len in fwd
    block_m_dq_bwd: int = 64,  # block of dq seq len in bwd
    block_n_dq_bwd: int = 64,  # block of dq seq len in bwd
    block_m_dkv_bwd: int = 64,  # block of dkv seq len in bwd
    block_n_dkv_bwd: int = 64,  # block of dkv seq len in bwd
    quant_block_size: int = 32
):
    if backend_type == "triton":
        if cp_comm_type == "a2a":
            if quant_type == "fp8_blockwise" or quant_type is None:
                return AttentionTritonFunctionCPA2A.apply(
                    q,
                    k,
                    v,
                    dropout_p,
                    softmax_scale,
                    causal,
                    window_size,
                    bias,
                    alibi_slopes,
                    return_lse,
                    return_attn_probs,
                    is_grad_enabled,
                    True if quant_type == "fp8" else False,
                    cp_group,
                )
            elif quant_type == "mxfp8":
                return AttentionTritonMXFP8FunctionCPA2A.apply(
                    q,
                    k,
                    v,
                    dropout_p,
                    softmax_scale,
                    causal,
                    window_size,
                    bias,
                    alibi_slopes,
                    return_lse,
                    return_attn_probs,
                    is_grad_enabled,
                    True,
                    cp_group,
                    block_m_fwd,
                    block_n_fwd,
                    block_m_dq_bwd,
                    block_n_dq_bwd,
                    block_m_dkv_bwd,
                    block_n_dkv_bwd,
                    quant_block_size
                )
            else:
                raise NotImplementedError(
                    f"not supported quant_type {quant_type} backend_type {backend_type} cp_comm_type {cp_comm_type} yet"
                )
        else:
            raise NotImplementedError(
                f"not supported backend_type {backend_type} cp_comm_type {cp_comm_type} yet"
            )
    elif backend_type == "ck":
        if cp_comm_type == "a2a":
            return AttentionCKFunctionCPA2A.apply(
                q,
                k,
                v,
                dropout_p,
                softmax_scale,
                causal,
                window_size,
                bias,
                alibi_slopes,
                deterministic,
                return_lse,
                return_attn_probs,
                is_grad_enabled,
                cp_group,
            )
        else:
            raise NotImplementedError(
                f"not supported backend_type {backend_type} cp_comm_type {cp_comm_type} yet"
            )
    else:
        raise NotImplementedError(
            f"not supported backend_type {backend_type} cp_comm_type {cp_comm_type} yet"
        )
