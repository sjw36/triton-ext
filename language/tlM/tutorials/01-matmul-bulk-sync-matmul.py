"""Chained matmul with a grid-wide barrier: ``D = (A @ B) @ C``.

This tutorial demonstrates the ``tlM.bulk_sync`` grid-wide barrier by fusing two
dependent GEMMs into a *single* persistent kernel launch:

    P = A @ B            # GEMM 1   (A: MxK, B: KxN  -> P: MxN)
    <grid-wide barrier>
    D = P @ C            # GEMM 2   (P: MxN, C: NxL  -> D: MxL)

Why a grid-wide barrier?
------------------------
Each program (CTA) in GEMM 2 reads a *row-block* of ``P`` that spans the full
``N`` dimension -- columns that were produced by *many different* programs in
GEMM 1. A normal kernel cannot read another program's output safely, because
there is no ordering guarantee across CTAs. ``tlM.bulk_sync`` provides exactly
that: every program publishes its GEMM-1 stores, then waits until *all*
cooperating programs have arrived, so the entire ``P`` matrix is globally
visible before any GEMM-2 read begins.

Persistent / cooperative launch (important!)
--------------------------------------------
A grid-wide spin barrier only terminates if *all* participating programs are
resident on the GPU at the same time. We therefore launch a **persistent**
kernel with ``grid = (NUM_SMS,)`` (one program per SM) and have each program
loop over output tiles. If you launch more programs than can be co-resident,
the barrier can deadlock.

How the barrier is lowered
--------------------------
``tlM.bulk_sync`` emits a ``mega.bulk_sync`` op into TTIR. That op is lowered to
plain Triton atomics + ``ttg.barrier`` + ``scf`` control flow by the
out-of-tree ``mega-bulk-sync-lowering`` pass. The standard Triton pipeline does
not know about it, so this tutorial injects the pass right after the TTIR stage
via ``knobs.runtime.add_stages_inspection_hook`` (the same mechanism other
out-of-tree extensions use).

Requirements
------------
Both plugin libraries must be loadable (this file discovers them under
``<repo>/build*/lib``):
  * ``libmega.so``            -- the ``mega`` dialect + its builder method
  * ``libmega_bulk_sync.so``  -- the ``mega-bulk-sync-lowering`` pass

Three things have to be registered before the kernel compiles: the `mega`
dialect, the ``create_mega_bulk_sync`` builder method behind
``tlM.bulk_sync``, and the ``add_mega_bulk_sync`` pass wrapper used by the
stages hook below. How that happens depends on the Triton build, so step 0
does both: it sets ``TRITON_PLUGIN_PATHS`` *before* importing triton (older
builds, including the revision pinned in ``ci/triton-hash.txt``, enumerate it
while ``libtriton`` is imported) and then calls the explicit
``extend_with``-style APIs when the bindings provide them (newer builds load
nothing on their own).

Triton itself must be built with ``TRITON_EXT_ENABLED=1``; otherwise
``libtriton.so`` hides its MLIR symbols, the plugins bind to a second copy of
MLIR, and loading them crashes (or is refused outright by ``triton-opt``).
The plugins must also be built against the same LLVM that Triton uses --
mixing two LLVM revisions corrupts the heap rather than failing cleanly.

Run it:

    python language/tlM/tutorials/01-matmul-bulk-sync-matmul.py
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import sys

# --------------------------------------------------------------------------- #
# 0. Locate the plugin libraries and register them, BEFORE importing triton:
#    older builds enumerate TRITON_PLUGIN_PATHS as `libtriton` is imported.
# --------------------------------------------------------------------------- #
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_TLM_PYTHON_DIR = pathlib.Path(__file__).resolve().parents[1] / "python"


def _find_lib(lib_name: str, env_var: str) -> pathlib.Path:
    if env := os.environ.get(env_var):
        p = pathlib.Path(env)
        if p.exists():
            return p
    for build_dir in ("build", "build-release", "build-debug"):
        cand = _REPO_ROOT / build_dir / "lib" / lib_name
        if cand.exists():
            return cand
    raise RuntimeError(
        f"Could not find {lib_name}. Build the extensions first (see README) "
        f"so they exist under <repo>/build/lib/, or point {env_var} at it.")


_MEGA_LIB = _find_lib("libmega.so", "MEGA_PLUGIN")
_BULK_SYNC_LIB = _find_lib("libmega_bulk_sync.so", "MEGA_BULK_SYNC_PLUGIN")
_libs = [str(_MEGA_LIB), str(_BULK_SYNC_LIB)]
_libs += [p for p in os.environ.get("TRITON_PLUGIN_PATHS", "").split(os.pathsep)
          if p]
os.environ["TRITON_PLUGIN_PATHS"] = os.pathsep.join(dict.fromkeys(_libs))

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from triton import knobs  # noqa: E402
from triton._C.libtriton import ir, passes  # noqa: E402

# Register the `tlM` language extension (`triton.language.extra.tlM`).
if str(_TLM_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_TLM_PYTHON_DIR))
import tlM  # noqa: E402,F401
import triton.language.extra.tlM as tlM  # noqa: E402

# `mega` dialect + the `create_mega_bulk_sync` TritonOpBuilder method. On a
# Triton that auto-loaded the plugin above, this only validates the setup.
tlM.register_plugin(str(_MEGA_LIB))
# `mega-bulk-sync-lowering` -> `passes.plugin.add_mega_bulk_sync`; needed only
# on builds that expose the explicit loader.
if hasattr(passes.plugin, "extend_with"):
    passes.plugin.extend_with(str(_BULK_SYNC_LIB))


# --------------------------------------------------------------------------- #
# 1. Inject the `mega-bulk-sync-lowering` pass right after the TTIR stage so
#    `mega.bulk_sync` becomes runnable atomics + barrier + control flow.
# --------------------------------------------------------------------------- #
def _bulk_sync_stages_hook(self=None,
                           stages=None,
                           options=None,
                           language=None,
                           capability=None):
    # Probe protocol: when called with everything None, return a cache (key, hash).
    key = pathlib.Path(__file__).read_text()
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    if all(a is None for a in (stages, options, language, capability)):
        return key, digest

    base_make_ttir = self.make_ttir

    def make_ttir_with_lowering(mod, metadata, opt, cap):
        mod = base_make_ttir(mod, metadata, opt, cap)
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.plugin.add_mega_bulk_sync(pm)
        pm.run(mod, "mega_bulk_sync_lowering")
        return mod

    stages["ttir"] = lambda src, metadata: make_ttir_with_lowering(
        src, metadata, options, capability)
    return key, digest


knobs.runtime.add_stages_inspection_hook = _bulk_sync_stages_hook


# --------------------------------------------------------------------------- #
# 2. A reusable persistent GEMM, plus the chained kernel that calls it twice.
# --------------------------------------------------------------------------- #
@triton.jit
def persistent_gemm(
    lhs_ptr, rhs_ptr, out_ptr,  # out[Mo, No] = lhs[Mo, Ko] @ rhs[Ko, No]
    Mo, No, Ko,  #
    stride_lm, stride_lk,  # lhs strides (row, col)
    stride_rk, stride_rn,  # rhs strides (row, col)
    stride_om, stride_on,  # out strides (row, col)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Persistent matmul: each program strides over the `Mo x No` output tiles.

    Called as a device function from `chained_matmul_kernel` (Triton inlines
    `@triton.jit` callees). The grid is sized to the SM count by the host, so
    every program is co-resident -- a precondition for the grid-wide barrier
    that runs between the two GEMMs.
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    num_pid_m = tl.cdiv(Mo, BLOCK_M)
    num_pid_n = tl.cdiv(No, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n

    for tile_id in range(pid, num_tiles, num_programs):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        lhs_ptrs = lhs_ptr + (offs_m[:, None] * stride_lm +
                              offs_k[None, :] * stride_lk)
        rhs_ptrs = rhs_ptr + (offs_k[:, None] * stride_rk +
                              offs_n[None, :] * stride_rn)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for _ in range(0, Ko, BLOCK_K):
            lhs = tl.load(lhs_ptrs)
            rhs = tl.load(rhs_ptrs)
            acc += tl.dot(lhs, rhs)
            lhs_ptrs += BLOCK_K * stride_lk
            rhs_ptrs += BLOCK_K * stride_rk
        out_ptrs = out_ptr + (offs_m[:, None] * stride_om +
                              offs_n[None, :] * stride_on)
        tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty))


@triton.jit
def chained_matmul_kernel(
    a_ptr, b_ptr, c_ptr, p_ptr, d_ptr,  # data
    arrival_ptr, release_ptr,  # grid-barrier scratch (int32, zero-initialized)
    M, N, K, L,  # A:MxK  B:KxN  ->  P:MxN ;  C:NxL  ->  D:MxL
    stride_am, stride_ak,  #
    stride_bk, stride_bn,  #
    stride_cn, stride_cl,  #
    stride_pm, stride_pn,  #
    stride_dm, stride_dl,  #
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    # ----- GEMM 1: P = A @ B  (reduce over K) -----
    persistent_gemm(
        a_ptr, b_ptr, p_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_pm, stride_pn,
        BLOCK_M, BLOCK_N, BLOCK_K,
    )

    # ----- grid-wide barrier: make all of P globally visible -----
    # `sense=1`: the release flag starts at 0; the last arriver flips it to 1
    # and resets the arrival counter, while everyone else spins until they see 1.
    tlM.bulk_sync(arrival_ptr, release_ptr, tl.num_programs(0), 1)

    # ----- GEMM 2: D = P @ C  (reduce over N) -----
    # Reuse persistent_gemm with lhs=P, rhs=C, out=D: the output is M x L and the
    # contraction dim is N, so BLOCK_N here plays the "BLOCK_K" role.
    persistent_gemm(
        p_ptr, c_ptr, d_ptr,
        M, L, N,
        stride_pm, stride_pn,
        stride_cn, stride_cl,
        stride_dm, stride_dl,
        BLOCK_M, BLOCK_L, BLOCK_N,
    )


# --------------------------------------------------------------------------- #
# 3. Host wrapper.
# --------------------------------------------------------------------------- #
def chained_matmul(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
                   num_sms: int | None = None) -> torch.Tensor:
    """Return ``(a @ b) @ c`` computed in a single grid-synchronized kernel."""
    M, K = a.shape
    K2, N = b.shape
    N2, L = c.shape
    assert K == K2 and N == N2, "incompatible matmul shapes"

    BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_L = 64, 64, 64, 64
    for dim, blk, name in ((M, BLOCK_M, "M"), (N, BLOCK_N, "N"),
                           (K, BLOCK_K, "K"), (L, BLOCK_L, "L")):
        assert dim % blk == 0, (
            f"this tutorial requires {name} ({dim}) to be a multiple of its "
            f"block size ({blk})")

    p = torch.empty((M, N), device=a.device, dtype=a.dtype)
    d = torch.empty((M, L), device=a.device, dtype=a.dtype)
    # Grid-barrier scratch: an arrival counter and a release flag, both zeroed.
    arrival = torch.zeros(1, device=a.device, dtype=torch.int32)
    release = torch.zeros(1, device=a.device, dtype=torch.int32)

    if num_sms is None:
        num_sms = torch.cuda.get_device_properties(
            a.device).multi_processor_count

    grid = (num_sms, )
    chained_matmul_kernel[grid](
        a, b, c, p, d,
        arrival, release,
        M, N, K, L,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        p.stride(0), p.stride(1),
        d.stride(0), d.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, BLOCK_L=BLOCK_L,
    )
    return d


# --------------------------------------------------------------------------- #
# 4. Driver: validate against torch.
# --------------------------------------------------------------------------- #
def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA device not available; skipping (this kernel needs a GPU).")
        return

    device = "cuda"
    torch.manual_seed(0)
    M, K, N, L = 256, 128, 192, 320
    a = torch.randn((M, K), device=device, dtype=torch.float16)
    b = torch.randn((K, N), device=device, dtype=torch.float16)
    c = torch.randn((N, L), device=device, dtype=torch.float16)

    out = chained_matmul(a, b, c)
    ref = (a @ b) @ c

    torch.testing.assert_close(out, ref, atol=1e-1, rtol=1e-1)
    print(f"OK: chained_matmul matches torch reference "
          f"(M={M}, K={K}, N={N}, L={L}, dtype={a.dtype}).")


if __name__ == "__main__":
    main()
