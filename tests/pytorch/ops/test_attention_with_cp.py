###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)

import primus_turbo.pytorch as pt
from primus_turbo.pytorch.ops.attention.attention_utils import All2AllAttentionSharder
from tests.pytorch.ref.attention_ref import (
    AttnConfig,
    attention_vanilla_forward_pytorch_ref_impl,
)
from tests.test_utils import compute_snr

test_cases = [
    # AttnConfig(seqlen_q=1024, seqlen_kv=1024, num_head_q=32, num_head_kv=32, head_dim_qk=128, head_dim_v=128),
    AttnConfig(seqlen_q=1024, seqlen_kv=1024, num_head_q=64, num_head_kv=8, head_dim_qk=128, head_dim_v=128),
    # AttnConfig(seqlen_q=1024, seqlen_kv=1024, num_head_q=32, num_head_kv=8, head_dim_qk=128, head_dim_v=128),
    AttnConfig(seqlen_q=1024, seqlen_kv=1024, num_head_q=64, num_head_kv=8, head_dim_qk=128, head_dim_v=128),
    AttnConfig(seqlen_q=1024, seqlen_kv=1024, num_head_q=16, num_head_kv=16, head_dim_qk=192, head_dim_v=128),
    AttnConfig(
        seqlen_q=1024, seqlen_kv=1024, num_head_q=128, num_head_kv=128, head_dim_qk=192, head_dim_v=128
    ),
    # AttnConfig(seqlen_q=1024, seqlen_kv=1024, num_head_q=48, num_head_kv=8, head_dim_qk=128, head_dim_v=128),
]


@instantiate_parametrized_tests
class AttentionWithCPTestCase(MultiProcessTestCase):
    """AttentionWithCPTestCase"""

    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    @property
    def world_size(self) -> int:
        return torch.cuda.device_count()

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.rank)

    def _init_process(self):
        torch.cuda.set_device(self.device)
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        torch.manual_seed(42)

    @skip_if_lt_x_gpu(2)
    @parametrize("batch", [4])
    @parametrize("config", test_cases)
    @parametrize("causal", [True])
    @parametrize("backend_type", ["ck", "triton"])
    @parametrize("cp_comm_type", ["a2a"])
    @parametrize("quant_type", [None, "fp8_blockwise", "mxfp8"])
    @parametrize("block_m", [32, 64, 128])
    @parametrize("block_n", [32, 64, 128])
    @parametrize("quant_block_size", [32, 64, 128])
    def test_attention_with_cp(self, batch, config, causal, backend_type, cp_comm_type, quant_type,
                               block_m, block_n, quant_block_size):

        if quant_type == "mxfp8":
            if (config.seqlen_q % block_m != 0 or  
            config.seqlen_kv % block_n != 0 or
            block_m % quant_block_size != 0 or 
            block_n % quant_block_size !=0 or 
            config.head_dim_qk % quant_block_size !=0 or
            config.head_dim_v % quant_block_size != 0):
                return
            
        self._init_process()
        cp_group = dist.group.WORLD

        # ck backend do not support fp8 yet
        if backend_type == "ck" and quant_type is not None:
            return

        # not test all block size for none-mxfp8 quantization 
        if quant_type != "mxfp8":
            if block_m != 32 or block_n != 32 or quant_block_size != 32:
                return

        device = "cuda"
        dtype = torch.bfloat16
        if cp_comm_type == "a2a":
            input_sharder = All2AllAttentionSharder()
        else:
            raise NotImplementedError()

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

        grad_ref = torch.randn(*o_ref.shape, device=device, dtype=dtype)
        o_ref.backward(grad_ref)
        o_ref, dq_ref, dk_ref, dv_ref = input_sharder.shard_cp_input(
            [o_ref, query_ref.grad, key_ref.grad, value_ref.grad], cp_group
        )

        # attention with CP
        cp_stream = torch.cuda.Stream()
        cp_param_bundle = {"cp_group": cp_group, "cp_stream": cp_stream, "cp_comm_type": cp_comm_type}

        query_local_token, key_local_token, value_local_token = input_sharder.shard_cp_input(
            [query, key, value], cp_group
        )

        if quant_type is not None:
            o = pt.ops.attention_fp8_quant(
                query_local_token,
                key_local_token,
                value_local_token,
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
                cp_param_bundle=cp_param_bundle,
                quant_type = quant_type, # "fp8", "mxfp8"
                block_m_fwd=block_m,
                block_n_fwd=block_n,
                block_m_dq_bwd=block_m,
                block_n_dq_bwd=block_n,
                block_m_dkv_bwd=block_m,
                block_n_dkv_bwd=block_n,
                quant_block_size=quant_block_size,
            )
        else:
            o = pt.ops.attention(
                query_local_token,
                key_local_token,
                value_local_token,
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
                cp_param_bundle=cp_param_bundle,
            )

        grad = input_sharder.shard_cp_input([grad_ref], cp_group)[0]
        o.backward(grad)

        dq, dk, dv = input_sharder.shard_cp_input([query.grad, key.grad, value.grad], cp_group)

        out_snr = compute_snr(o_ref, o)
        query_grad_snr = compute_snr(dq_ref, dq)
        key_grad_snr = compute_snr(dk_ref, dk)
        value_grad_snr = compute_snr(dv_ref, dv)

        assert out_snr > 20, "out_snr too low"
        assert query_grad_snr > 15, "query_grad_snr too low"
        assert key_grad_snr > 15, "key_grad_snr too low"
        assert value_grad_snr > 15, "value_grad_snr too low"
