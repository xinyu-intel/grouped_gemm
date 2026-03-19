# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch
import triton
import triton.language as tl


@triton.jit
def moe_mmk(
    a_desc,
    b_desc,
    K,
    expert_id,
    a_scale_ptr,
    b_scale_ptr,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_ak: tl.int64,
    stride_bk: tl.int64,
    stride_ase: tl.int64,
    stride_asm: tl.int64,
    stride_ask: tl.int64,
    stride_bse: tl.int64,
    stride_bsk: tl.int64,
    stride_bsn: tl.int64,
    # Offsets and masks
    offs_m,
    offs_n,
    offs_bn,
    mask_m,
    # Block size for block-wise quantization
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    pid_m,
    pid_n,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    compute_type: tl.constexpr,
    use_w8a8: tl.constexpr,
    use_w8a16: tl.constexpr,
    per_act_token_quant: tl.constexpr,
):
    if use_w8a16:
        b_scale_ptrs = (
            b_scale_ptr + expert_id * stride_bse + offs_n[None, :] * stride_bsn
        )
        b_scale = tl.load(b_scale_ptrs)

    if use_w8a8:
        # block-wise
        if group_k > 0 and group_n > 0:
            a_scale_ptrs = a_scale_ptr + offs_m * stride_asm
            offs_bsn = offs_bn // group_n
            b_scale_ptrs = b_scale_ptr + offs_bsn * stride_bsn

        # per act token
        elif per_act_token_quant:
            # Load per-token scale for activations
            a_scale_ptrs = a_scale_ptr + offs_m * stride_asm
            a_scale = tl.load(a_scale_ptrs, mask=mask_m, other=0.0)[:, None]

            b_scale_ptrs = b_scale_ptr + offs_bn[None, :] * stride_bsn
            b_scale = tl.load(b_scale_ptrs)

        # tensor-wise
        else:
            a_scale = tl.load(a_scale_ptr)
            b_scale = tl.load(b_scale_ptr)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load the next block of A and B using tensor descriptors
        a = a_desc.load([pid_m * BLOCK_M, k * BLOCK_K])
        b = b_desc.load([pid_n * BLOCK_N, k * BLOCK_K]).T
        # We accumulate along the K dimension.
        if use_w8a16:
            accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
        elif use_w8a8:
            if group_k > 0 and group_n > 0:
                k_start = k * BLOCK_K
                offs_ks = k_start // group_k
                a_scale = tl.load(
                    a_scale_ptrs + offs_ks * stride_ask, mask=mask_m, other=0.0
                )
                b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)

                accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
            else:
                # acc used to enable fp8_fast_accum
                accumulator = tl.dot(a, b, acc=accumulator)
        else:
            accumulator += tl.dot(a, b)

    if use_w8a16:
        accumulator = (accumulator * b_scale).to(compute_type)
    elif use_w8a8:
        if group_k > 0 and group_n > 0:
            accumulator = accumulator.to(compute_type)
        else:
            accumulator = (accumulator * a_scale * b_scale).to(compute_type)
    else:
        accumulator = accumulator.to(compute_type)

    return accumulator


@triton.jit
def expert_triton_kernel(
    a_desc,  # [max_tokens, K]
    b_desc,  # [K, N]
    c_desc,  # [max_tokens, N]
    expert_id,
    compute_type: tl.constexpr,
    # Dimensions
    M,
    N,
    K,
    # Quantization data
    a_scale_ptr,
    b_scale_ptr,
    b_zp_ptr,
    # strides
    stride_am: tl.int64,
    stride_ak: tl.int64,
    stride_bk: tl.int64,
    stride_bn: tl.int64,
    stride_cm: tl.int64,
    stride_cn: tl.int64,
    stride_ase: tl.int64,
    stride_asm: tl.int64,
    stride_ask: tl.int64,
    stride_bse: tl.int64,
    stride_bsk: tl.int64,
    stride_bsn: tl.int64,
    # offsets
    offs_bn,
    # Blockwise quantization data
    group_n,
    group_k,
    pid_m,
    pid_n,
    # Quantization schemes
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    per_act_token_quant: tl.constexpr,
    # Kernel config
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N) % N
    mask_m = offs_m < M

    accumulator = moe_mmk(
        a_desc,
        b_desc,
        K,
        expert_id,
        a_scale_ptr,
        b_scale_ptr,
        # The stride variables represent how much to increase the ptr by when
        # moving by 1 element in a particular dimension. E.g. `stride_am` is
        # how much to increase `a_ptr` by to get the element one row down
        # (A has M rows).
        stride_ak,
        stride_bk,
        stride_ase,
        stride_asm,
        stride_ask,
        stride_bse,
        stride_bsk,
        stride_bsn,
        # Offsets and masks
        offs_m,
        offs_n,
        offs_bn,
        mask_m,
        # Block size for block-wise quantization
        group_n,
        group_k,
        pid_m,
        pid_n,
        # Meta-parameters
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        compute_type,
        use_fp8_w8a8,
        use_int8_w8a16,
        per_act_token_quant,
    )

    # store in C
    c_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], accumulator)


def get_batched_moe_configs():
    return (
        [
            triton.Config(
                {"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 32, "grf_mode": "256"},
                num_stages=s,
                num_warps=32,
            )
            for s in [2, 3]
        ]
        + [
            triton.Config(
                {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "grf_mode": m},
                num_stages=s,
                num_warps=w,
            )
            for s in [2]
            for (m, w) in ([("256", 32), ("128", 64)])
        ]
        + [
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "grf_mode": "256"},
                num_stages=s,
                num_warps=32,
            )
            for s in [2]
        ]
        + [
            triton.Config(
                {"BLOCK_M": 8, "BLOCK_N": 512, "BLOCK_K": 64, "grf_mode": "256"},
                num_stages=s,
                num_warps=32,
            )
            for s in [2]
        ]
        + [
            triton.Config(
                {"BLOCK_M": 8, "BLOCK_N": 128, "BLOCK_K": 64, "grf_mode": "256"},
                num_stages=s,
                num_warps=4,
            )
            for s in [2]
        ]
    )


@triton.autotune(
    configs=get_batched_moe_configs(),
    key=["max_num_tokens", "K", "N"],
)
@triton.jit
def batched_triton_kernel(
    a_ptr,  # [E, max_num_tokens, K]
    b_ptr,  # [E, K, N]
    c_ptr,  # [E, max_num_tokens, N]
    expert_num_tokens,  # [E]
    compute_type: tl.constexpr,
    # Dimensions
    max_num_tokens: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    # Quantization data
    a_scale_ptr,
    b_scale_ptr,
    b_zp_ptr,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_ae: tl.int64,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_be: tl.int64,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_ce: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    stride_ase: tl.constexpr,
    stride_asm: tl.constexpr,
    stride_ask: tl.constexpr,
    stride_bse: tl.constexpr,
    stride_bsk: tl.constexpr,
    stride_bsn: tl.constexpr,
    # Blockwise quantization data
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    # Quantization schemes
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    per_act_token_quant: tl.constexpr,
    # Kernel config
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    expert_id = tl.program_id(axis=1)
    e_num_tokens = tl.load(expert_num_tokens + expert_id)
    if e_num_tokens == 0:
        # Early exit
        return

    # axis 0 is M_blocks * N_blocks
    pid_mn = tl.program_id(axis=0)
    # num_pid_m = tl.cdiv(max_num_tokens, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid_mn // num_pid_n
    pid_n = pid_mn % num_pid_n

    cta_m_start = pid_m * BLOCK_M
    cta_n_start = pid_n * BLOCK_N
    if cta_m_start >= e_num_tokens:
        # Early exit
        return

    cta_m_size = min(BLOCK_M, e_num_tokens - cta_m_start)
    cta_n_size = min(BLOCK_N, N - cta_n_start)

    a_desc = tl.make_tensor_descriptor(
        base=a_ptr + expert_id * stride_ae,
        shape=(e_num_tokens, K),
        strides=(stride_am, stride_ak),
        block_shape=(BLOCK_M, BLOCK_K),
    )
    b_desc = tl.make_tensor_descriptor(
        base=b_ptr + expert_id * stride_be,
        shape=(N, K),
        strides=(stride_bn, stride_bk),
        block_shape=(BLOCK_N, BLOCK_K),
    )
    c_desc = tl.make_tensor_descriptor(
        base=c_ptr + expert_id * stride_ce,
        shape=(e_num_tokens, N),
        strides=(stride_cm, stride_cn),
        block_shape=(BLOCK_M, BLOCK_N),
    )

    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)) % N

    if use_fp8_w8a8:
        a_scale_ptr = a_scale_ptr + expert_id * stride_ase
        b_scale_ptr = b_scale_ptr + expert_id * stride_bse

        # block-wise
        if group_k > 0 and group_n > 0 or per_act_token_quant:
            a_scale_ptr = a_scale_ptr + cta_m_start * stride_asm

    expert_triton_kernel(
        a_desc,
        b_desc,
        c_desc,
        expert_id,
        compute_type,
        cta_m_size,  # M
        cta_n_size,  # N
        K,  # K
        a_scale_ptr,
        b_scale_ptr,
        b_zp_ptr,
        # Strides
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        stride_ase,
        stride_asm,
        stride_ask,
        stride_bse,
        stride_bsk,
        stride_bsn,
        # offsets
        offs_bn,
        # Blockwise quantization data
        group_n,
        group_k,
        pid_m,
        pid_n,
        # Quantization schemes
        use_fp8_w8a8,
        use_int8_w8a16,
        per_act_token_quant,
        # Kernel config
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
    )


def normalize_batched_scales_shape(
    scales: torch.Tensor | None,
    num_experts: int,
) -> torch.Tensor | None:
    if scales is not None and scales.ndim < 3:
        if scales.numel() == 1:
            scales = scales.view(1)
            scales = torch.repeat_interleave(scales, num_experts, dim=0).view(
                num_experts, 1, 1
            )
        else:
            scales = scales.view(num_experts, -1, scales.size(-1))

    return scales


def invoke_moe_batched_triton_kernel(
    A: torch.Tensor,  # [E, max_tokens, K]
    B: torch.Tensor,  # [E, N, K]
    C: torch.Tensor,  # [E, max_tokens, N]
    expert_num_tokens: torch.Tensor,  # [E]
    compute_type: tl.dtype,
    # Quantization data
    A_scale: torch.Tensor | None,
    B_scale: torch.Tensor | None,
    B_zp: torch.Tensor,
    # Quantization schemes
    use_fp8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    config: dict[str, int],
    per_act_token_quant: bool,
    block_shape: list[int] | None = None,
):
    assert not use_int4_w4a16
    max_num_tokens = A.size(1)
    K = A.size(2)
    N = C.size(2)

    BLOCK_M = config["BLOCK_SIZE_M"]
    BLOCK_N = config["BLOCK_SIZE_N"]

    grid = (
        triton.cdiv(max_num_tokens, BLOCK_M) * triton.cdiv(B.size(1), BLOCK_N),
        expert_num_tokens.size(0),
    )

    A_scale = normalize_batched_scales_shape(A_scale, expert_num_tokens.shape[0])

    if B_scale is not None and B_scale.ndim == 1:
        assert B_scale.numel() == expert_num_tokens.shape[0]
        B_scale = B_scale.view(-1, 1, 1)

    assert A_scale is None or A_scale.ndim == 3, (
        f"{0 if A_scale is None else A_scale.shape}"
    )
    assert B_scale is None or B_scale.ndim == 1 or B_scale.ndim == 3, (
        f"{0 if B_scale is None else B_scale.shape}"
    )

    if B_scale is not None:
        if B_scale.ndim == 1:
            stride_bse = 1
            stride_bsk = 0
            stride_bsn = 0
        else:
            stride_bse = B_scale.stride(0)
            stride_bsk = B_scale.stride(2)
            stride_bsn = B_scale.stride(1)

    else:
        stride_bse = 0
        stride_bsk = 0
        stride_bsn = 0

    if A_scale is not None:
        stride_ase = A_scale.stride(0)
        stride_asm = A_scale.stride(1)
        stride_ask = A_scale.stride(2)
    else:
        stride_ase = 0
        stride_asm = 0
        stride_ask = 0

    batched_triton_kernel[grid](
        A,
        B,
        C,
        expert_num_tokens,
        compute_type,
        # Dimensions
        max_num_tokens,
        K,
        N,
        # Quantization data
        A_scale,
        B_scale,
        B_zp,
        # Strides
        A.stride(0),
        A.stride(1),
        A.stride(2),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        C.stride(2),
        stride_ase,
        stride_asm,
        stride_ask,
        stride_bse,
        stride_bsk,
        stride_bsn,
        # Blockwise quantization data
        0 if block_shape is None else block_shape[0],
        0 if block_shape is None else block_shape[1],
        # Quantization schemes
        use_fp8_w8a8,
        use_int8_w8a16,
        per_act_token_quant,
    )
