#!/usr/bin/env python3
"""TileLang MUSA smoke test — run from this directory (same as self_cognition.py).

Twinkle 训练 demo 本身不依赖 tilelang；本脚本用于在同一环境中验证
tilelang_musa 是否能正常 JIT 编译并在 musa 设备上执行 GEMM。

Usage:
    cd /data/wangmx/twinkle_musa/cookbook/client/twinkle/self_host
    python test_tilelang_musa.py
"""

import sys

import torch
import tilelang
import tilelang.language as T


@tilelang.jit(target="musa")
def matmul(A, B, block_M, block_N, block_K, dtype="float16", accum_dtype="float32"):
    M, N, K = T.const("M N K")
    A: T.Tensor[[M, K], dtype]
    B: T.Tensor[[K, N], dtype]
    C = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        a_shared = T.alloc_shared((block_M, block_K), dtype)
        b_shared = T.alloc_shared((block_K, block_N), dtype)
        c_local = T.alloc_fragment((block_M, block_N), accum_dtype)
        T.clear(c_local)
        for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
            T.copy(A[by * block_M, ko * block_K], a_shared)
            T.copy(B[ko * block_K, bx * block_N], b_shared)
            T.gemm(a_shared, b_shared, c_local)
        T.copy(c_local, C[by * block_M, bx * block_N])
    return C


def main() -> int:
    print("=== TileLang MUSA smoke test ===")
    print(f"tilelang: {tilelang.__version__}")
    print(f"tilelang path: {tilelang.__file__}")
    print(f"torch: {torch.__version__}")
    print(f"musa available: {torch.musa.is_available()}, count: {torch.musa.device_count()}")

    if not torch.musa.is_available():
        print("FAIL: torch.musa not available")
        return 1

    M = N = K = 256
    block = 64
    a = torch.randn(M, K, device="musa", dtype=torch.float16)
    b = torch.randn(K, N, device="musa", dtype=torch.float16)

    kernel = matmul.compile(
        M=M, N=N, K=K,
        block_M=block, block_N=block, block_K=block,
        dtype="float16", accum_dtype="float32",
    )
    c = kernel(a, b)
    ref = a @ b
    torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)

    print(f"GEMM ok: shape={tuple(c.shape)}, max_err={(c - ref).abs().max().item():.6f}")
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
