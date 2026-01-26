###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Literal, Optional

import torch
import torch.utils.benchmark as benchmark
from flash_attn import flash_attn_func

import primus_turbo.pytorch as pt
from tests.pytorch.ref.attention_ref import (
    AttnConfig,
    attention_vanilla_forward_pytorch_ref_impl,
)
from tests.test_utils import compute_snr

test_cases_turbo = [
    AttnConfig(seqlen_q=4096, seqlen_kv=4096, num_head_q=32, num_head_kv=32, head_dim_qk=128, head_dim_v=128),
    AttnConfig(seqlen_q=4096, seqlen_kv=4096, num_head_q=64, num_head_kv=8, head_dim_qk=128, head_dim_v=128),
    AttnConfig(seqlen_q=4096, seqlen_kv=4096, num_head_q=32, num_head_kv=8, head_dim_qk=128, head_dim_v=128),
    AttnConfig(seqlen_q=4096, seqlen_kv=4096, num_head_q=64, num_head_kv=8, head_dim_qk=128, head_dim_v=128),
    AttnConfig(seqlen_q=4096, seqlen_kv=4096, num_head_q=16, num_head_kv=16, head_dim_qk=192, head_dim_v=128),
    AttnConfig(
        seqlen_q=4096, seqlen_kv=4096, num_head_q=128, num_head_kv=128, head_dim_qk=192, head_dim_v=128
    ),
    AttnConfig(seqlen_q=4096, seqlen_kv=4096, num_head_q=32, num_head_kv=8, head_dim_qk=128, head_dim_v=128),
    AttnConfig(seqlen_q=4096, seqlen_kv=4096, num_head_q=48, num_head_kv=8, head_dim_qk=128, head_dim_v=128),
]

test_cases_flash_attn = [
    AttnConfig(seqlen_q=4096, seqlen_kv=4096, num_head_q=32, num_head_kv=32, head_dim_qk=128, head_dim_v=128),
    AttnConfig(seqlen_q=4096, seqlen_kv=4096, num_head_q=64, num_head_kv=8, head_dim_qk=128, head_dim_v=128),
    AttnConfig(seqlen_q=4096, seqlen_kv=4096, num_head_q=32, num_head_kv=8, head_dim_qk=128, head_dim_v=128),
    AttnConfig(seqlen_q=4096, seqlen_kv=4096, num_head_q=64, num_head_kv=8, head_dim_qk=128, head_dim_v=128),
    AttnConfig(seqlen_q=4096, seqlen_kv=4096, num_head_q=32, num_head_kv=8, head_dim_qk=128, head_dim_v=128),
    AttnConfig(seqlen_q=4096, seqlen_kv=4096, num_head_q=48, num_head_kv=8, head_dim_qk=128, head_dim_v=128),
]


def bench_turbo_attention(
    batch,
    config,
    causal: bool,
    backend_type: str,
    quant_type: Optional[Literal["fp8_blockwise", "mxfp8"]],
    test_backward: bool,
    block_m: int = 64,
    block_n: int = 64,
    quant_block_size: int = 32,
):
    device = "cuda"
    dtype = torch.bfloat16
    seqlen_q, seqlen_kv, num_head_q, num_head_kv, head_dim_qk, head_dim_v = (
        config.seqlen_q,
        config.seqlen_kv,
        config.num_head_q,
        config.num_head_kv,
        config.head_dim_qk,
        config.head_dim_v,
    )
    q_layout = (batch, seqlen_q, num_head_q, head_dim_qk)
    k_layout = (batch, seqlen_kv, num_head_kv, head_dim_qk)
    v_layout = (batch, seqlen_kv, num_head_kv, head_dim_v)

    query = torch.randn(q_layout, device=device, dtype=dtype, requires_grad=True)
    key = torch.randn(k_layout, device=device, dtype=dtype, requires_grad=True)
    value = torch.randn(v_layout, device=device, dtype=dtype, requires_grad=True)
    query_ref = query.clone().detach().requires_grad_()
    key_ref = key.clone().detach().requires_grad_()
    value_ref = value.clone().detach().requires_grad_()

    sm_scale = query.shape[-1] ** (-0.5)

    o_ref = attention_vanilla_forward_pytorch_ref_impl(query_ref, key_ref, value_ref, sm_scale, causal)
    if quant_type is None:
        fn_forward = lambda: pt.ops.attention(
            query,
            key,
            value,
            dropout_p=0.0,
            softmax_scale=sm_scale,
            causal=causal,
            window_size=(-1, -1),
            bias=None,
            alibi_slopes=None,
            deterministic=False,
            return_lse=False,
            return_attn_probs=False,
            backend_type=backend_type,
        )
    else:
        fn_forward = lambda: pt.ops.attention_fp8_quant(
            query,
            key,
            value,
            dropout_p=0.0,
            softmax_scale=sm_scale,
            causal=causal,
            window_size=(-1, -1),
            bias=None,
            alibi_slopes=None,
            deterministic=False,
            return_lse=False,
            return_attn_probs=False,
            backend_type=backend_type,
            quant_type=quant_type,
            block_m_fwd=block_m,
            block_n_fwd=block_n,
            block_m_dq_bwd=block_m,
            block_n_dq_bwd=block_n,
            block_m_dkv_bwd=block_m,
            block_n_dkv_bwd=block_n,
            quant_block_size=quant_block_size,
        )

    # Forward pass
    o = fn_forward()
    grad_output = torch.randn_like(o)

    # Backward pass for reference
    o_ref.backward(grad_output, retain_graph=True)
    # Backward pass for implementation
    o.backward(grad_output, retain_graph=True)

    # Compute SNRs
    out_snr = compute_snr(o_ref, o)
    query_grad_snr = compute_snr(query_ref.grad, query.grad)
    key_grad_snr = compute_snr(key_ref.grad, key.grad)
    value_grad_snr = compute_snr(value_ref.grad, value.grad)

    # Verify SNRs meet requirements
    assert out_snr > 20, "out_snr too low"
    assert query_grad_snr > 15, "query_grad_snr too low"
    assert key_grad_snr > 15, "key_grad_snr too low"
    assert value_grad_snr > 15, "value_grad_snr too low"

    # Calculate FLOPs
    total_flops = (
        2 * batch * seqlen_q * seqlen_kv * num_head_q * head_dim_qk
        + 2 * batch * seqlen_q * seqlen_kv * num_head_q * head_dim_v
    )
    if causal:
        total_flops //= 2

    if test_backward:
        # Re-run forward pass to get fresh output
        o = fn_forward()
        do = torch.randn_like(o)
        fn = lambda: o.backward(do, retain_graph=True)
        total_flops *= 2.5  # Approximate factor for backward pass
    else:
        fn = fn_forward

    # Warmup
    warmup = 5
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()

    # Benchmark
    timer = benchmark.Timer(
        stmt="fn()",
        globals={"fn": fn},
    )
    measurement = timer.timeit(100)
    mean_time_ms = measurement.mean * 1e3
    flops_per_sec = total_flops / (mean_time_ms * 1e-3) / 1e12  # TFLOPs/s

    print(f"Mean time: {mean_time_ms:.3f} ms | TFLOPS: {flops_per_sec:.2f}")
    return mean_time_ms, flops_per_sec, total_flops


def bench_flash_attention(
    batch,
    config,
    causal: bool,
    backend_type: str,
    quant_type: Optional[Literal["fp8_blockwise", "mxfp8"]],
    test_backward: bool,
    block_m: int = 64,
    block_n: int = 64,
    quant_block_size: int = 32,
):
    device = "cuda"
    dtype = torch.bfloat16
    seqlen_q, seqlen_kv, num_head_q, num_head_kv, head_dim_qk, head_dim_v = (
        config.seqlen_q,
        config.seqlen_kv,
        config.num_head_q,
        config.num_head_kv,
        config.head_dim_qk,
        config.head_dim_v,
    )
    q_layout = (batch, seqlen_q, num_head_q, head_dim_qk)
    k_layout = (batch, seqlen_kv, num_head_kv, head_dim_qk)
    v_layout = (batch, seqlen_kv, num_head_kv, head_dim_v)

    query = torch.randn(q_layout, device=device, dtype=dtype, requires_grad=True)
    key = torch.randn(k_layout, device=device, dtype=dtype, requires_grad=True)
    value = torch.randn(v_layout, device=device, dtype=dtype, requires_grad=True)
    sm_scale = query.shape[-1] ** (-0.5)

    fn_forward = lambda: flash_attn_func(
        query,
        key,
        value,
        dropout_p=0.0,
        softmax_scale=sm_scale,
        causal=causal,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
    )

    # Forward pass
    o = fn_forward()
    grad_output = torch.randn_like(o)

    # Backward pass for implementation
    o.backward(grad_output, retain_graph=True)

    # Calculate FLOPs
    total_flops = (
        2 * batch * seqlen_q * seqlen_kv * num_head_q * head_dim_qk
        + 2 * batch * seqlen_q * seqlen_kv * num_head_q * head_dim_v
    )
    if causal:
        total_flops //= 2

    if test_backward:
        # Re-run forward pass to get fresh output
        o = fn_forward()
        do = torch.randn_like(o)
        fn = lambda: o.backward(do, retain_graph=True)
        total_flops *= 2.5  # Approximate factor for backward pass
    else:
        fn = fn_forward

    # Warmup
    warmup = 5
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()

    # Benchmark
    timer = benchmark.Timer(
        stmt="fn()",
        globals={"fn": fn},
    )
    measurement = timer.timeit(100)
    mean_time_ms = measurement.mean * 1e3
    flops_per_sec = total_flops / (mean_time_ms * 1e-3) / 1e12  # TFLOPs/s

    print(f"Mean time: {mean_time_ms:.3f} ms | TFLOPS: {flops_per_sec:.2f}")
    return mean_time_ms, flops_per_sec, total_flops


if __name__ == "__main__":
    import pandas as pd
    from tabulate import tabulate

    def run_benchmark(bench_func, test_cases, test_configs):
        # DataFrame to store results
        results = pd.DataFrame(
            columns=[
                "Test id",
                "Causal",
                "Backend",
                "Quant_type",
                "Block_M",
                "Block_N",
                "quant_block_size",
                "Test Backward",
                "num_head_q",
                "num_head_kv",
                "head_dim_qk",
                "head_dim_v",
                "Time (ms)",
                "TFLOPS",
            ]
        )
        test_id = 0
        for test_case in test_cases:
            print(f"\n{'='*50}")
            print(f"Testing config: {test_case}")
            print(f"{'='*50}")
            for config in test_configs:
                test_id += 1
                try:
                    # Run benchmark
                    time_ms, tflops, _ = bench_func(
                        batch=4,
                        config=test_case,
                        causal=config["causal"],
                        backend_type=config["backend"],
                        quant_type=config["quant_type"],
                        test_backward=config["test_backward"],
                        block_m=config["block_m"],
                        block_n=config["block_n"],
                        quant_block_size=config["quant_block_size"],
                    )

                    # Add to results table
                    new_row = {
                        "Test id": test_id,
                        "Causal": config["causal"],
                        "Backend": config["backend"],
                        "Quant_type": config["quant_type"],
                        "Block_M": config["block_m"],
                        "Block_N": config["block_n"],
                        "quant_block_size": config["quant_block_size"],
                        "Test Backward": config["test_backward"],
                        "Time (ms)": f"{time_ms:.2f}",
                        "TFLOPS": f"{tflops:.2f}",
                        "num_head_q": test_case.num_head_q,
                        "num_head_kv": test_case.num_head_kv,
                        "head_dim_qk": test_case.head_dim_qk,
                        "head_dim_v": test_case.head_dim_v,
                    }
                    results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)

                except Exception as e:
                    print(f"Failed to run {config}: {str(e)}")
                    new_row = {
                        "Test id": test_id,
                        "Causal": config["causal"],
                        "Backend": config["backend"],
                        "Quant_type": config["quant_type"],
                        "Block_M": config["block_m"],
                        "Block_N": config["block_n"],
                        "quant_block_size": config["quant_block_size"],
                        "Test Backward": config["test_backward"],
                        "Time (ms)": "Failed",
                        "TFLOPS": "N/A",
                        "num_head_q": test_case.num_head_q,
                        "num_head_kv": test_case.num_head_kv,
                        "head_dim_qk": test_case.head_dim_qk,
                        "head_dim_v": test_case.head_dim_v,
                    }
                    results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)

        return results

    # Define test configurations
    test_configs_turbo = [
        {
            "causal": False,
            "backend": "ck",
            "quant_type": None,
            "test_backward": False,
            "block_m": 32,
            "block_n": 32,
            "quant_block_size": 32,
        },
        {
            "causal": True,
            "backend": "ck",
            "quant_type": None,
            "test_backward": False,
            "block_m": 32,
            "block_n": 32,
            "quant_block_size": 32,
        },
        {
            "causal": False,
            "backend": "ck",
            "quant_type": None,
            "test_backward": True,
            "block_m": 32,
            "block_n": 32,
            "quant_block_size": 32,
        },
        {
            "causal": True,
            "backend": "ck",
            "quant_type": None,
            "test_backward": True,
            "block_m": 32,
            "block_n": 32,
            "quant_block_size": 32,
        },
        {
            "causal": False,
            "backend": "triton",
            "quant_type": "fp8_blockwise",
            "test_backward": False,
            "block_m": 32,
            "block_n": 32,
            "quant_block_size": 32,
        },
        {
            "causal": True,
            "backend": "triton",
            "quant_type": "fp8_blockwise",
            "test_backward": False,
            "block_m": 32,
            "block_n": 32,
            "quant_block_size": 32,
        },
        {
            "causal": False,
            "backend": "triton",
            "quant_type": "fp8_blockwise",
            "test_backward": True,
            "block_m": 32,
            "block_n": 32,
            "quant_block_size": 32,
        },
        {
            "causal": True,
            "backend": "triton",
            "quant_type": "fp8_blockwise",
            "test_backward": True,
            "block_m": 32,
            "block_n": 32,
            "quant_block_size": 32,
        },
        {
            "causal": False,
            "backend": "triton",
            "quant_type": "mxfp8",
            "test_backward": False,
            "block_m": 32,
            "block_n": 32,
            "quant_block_size": 32,
        },
        {
            "causal": True,
            "backend": "triton",
            "quant_type": "mxfp8",
            "test_backward": False,
            "block_m": 32,
            "block_n": 32,
            "quant_block_size": 32,
        },
        {
            "causal": False,
            "backend": "triton",
            "quant_type": "mxfp8",
            "test_backward": True,
            "block_m": 32,
            "block_n": 32,
            "quant_block_size": 32,
        },
        {
            "causal": True,
            "backend": "triton",
            "quant_type": "mxfp8",
            "test_backward": True,
            "block_m": 32,
            "block_n": 32,
            "quant_block_size": 32,
        },
        {
            "causal": False,
            "backend": "triton",
            "quant_type": "mxfp8",
            "test_backward": False,
            "block_m": 64,
            "block_n": 64,
            "quant_block_size": 32,
        },
        {
            "causal": True,
            "backend": "triton",
            "quant_type": "mxfp8",
            "test_backward": False,
            "block_m": 64,
            "block_n": 64,
            "quant_block_size": 32,
        },
        {
            "causal": False,
            "backend": "triton",
            "quant_type": "mxfp8",
            "test_backward": True,
            "block_m": 64,
            "block_n": 64,
            "quant_block_size": 32,
        },
        {
            "causal": True,
            "backend": "triton",
            "quant_type": "mxfp8",
            "test_backward": True,
            "block_m": 64,
            "block_n": 64,
            "quant_block_size": 32,
        },
        {
            "causal": False,
            "backend": "triton",
            "quant_type": "mxfp8",
            "test_backward": False,
            "block_m": 64,
            "block_n": 64,
            "quant_block_size": 64,
        },
        {
            "causal": True,
            "backend": "triton",
            "quant_type": "mxfp8",
            "test_backward": False,
            "block_m": 64,
            "block_n": 64,
            "quant_block_size": 64,
        },
        {
            "causal": False,
            "backend": "triton",
            "quant_type": "mxfp8",
            "test_backward": True,
            "block_m": 64,
            "block_n": 64,
            "quant_block_size": 64,
        },
        {
            "causal": True,
            "backend": "triton",
            "quant_type": "mxfp8",
            "test_backward": True,
            "block_m": 64,
            "block_n": 64,
            "quant_block_size": 64,
        },
        {
            "causal": False,
            "backend": "triton",
            "quant_type": "mxfp8",
            "test_backward": False,
            "block_m": 128,
            "block_n": 128,
            "quant_block_size": 64,
        },
        {
            "causal": True,
            "backend": "triton",
            "quant_type": "mxfp8",
            "test_backward": False,
            "block_m": 128,
            "block_n": 128,
            "quant_block_size": 64,
        },
        {
            "causal": False,
            "backend": "triton",
            "quant_type": "mxfp8",
            "test_backward": True,
            "block_m": 128,
            "block_n": 128,
            "quant_block_size": 64,
        },
        {
            "causal": True,
            "backend": "triton",
            "quant_type": "mxfp8",
            "test_backward": True,
            "block_m": 128,
            "block_n": 128,
            "quant_block_size": 64,
        },
    ]
    # Run benchmarks with bench_turbo_attention
    aiter_results = run_benchmark(bench_turbo_attention, test_cases_turbo, test_configs_turbo)
    print("\nFinal AIter Results:")
    print(tabulate(aiter_results, headers="keys", tablefmt="grid", showindex=False))
    aiter_results.to_csv("aiter_attention_benchmark_results.csv", index=False)
    print("AITer results saved to aiter_attention_benchmark_results.csv")

    test_configs_flash_attn = [
        {
            "causal": False,
            "backend": "ck",
            "quant_type": None,
            "test_backward": False,
            "block_m": 32,
            "block_n": 32,
            "quant_block_size": 32,
        },
        {
            "causal": True,
            "backend": "ck",
            "quant_type": None,
            "test_backward": False,
            "block_m": 32,
            "block_n": 32,
            "quant_block_size": 32,
        },
        {
            "causal": False,
            "backend": "ck",
            "quant_type": None,
            "test_backward": True,
            "block_m": 32,
            "block_n": 32,
            "quant_block_size": 32,
        },
        {
            "causal": True,
            "backend": "ck",
            "quant_type": None,
            "test_backward": True,
            "block_m": 32,
            "block_n": 32,
            "quant_block_size": 32,
        },
    ]
    # Run benchmarks with bench_flash_attention
    flash_results = run_benchmark(bench_flash_attention, test_cases_flash_attn, test_configs_flash_attn)
    print("\nFinal Flash Attention Results:")
    print(tabulate(flash_results, headers="keys", tablefmt="grid", showindex=False))
    flash_results.to_csv("flash_attention_benchmark_results.csv", index=False)
    print("Flash Attention results saved to flash_attention_benchmark_results.csv")
