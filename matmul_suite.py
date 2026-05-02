"""Loopy GEMM performance suite using loopy_perf_artifacts.py.

This rewrites the original ``compute-examples/matmul.py`` benchmark driver so
that matrix-shape sweeps, timing, compiler-artifact dumps, and post-run analysis
flow through the generic helpers:

* ``loopy_perf_artifacts.py``: per-case OpenCL/CUDA timing + artifact dumps
* ``loopy_perf_suite.py``: runs many ``LoopyPerfCase`` objects under one suite
* ``analyze_compiler_artifacts.py``: suite CSV summaries and plots

Typical H200-style run::

    python matmul_suite.py \
        --variant register \
        --shape 2048x2048x2048 --shape 4096x4096x4096 \
        --bm 64 --bn 64 --bk 32 --tm 4 --tn 4 \
        --use-cuda --dump-artifacts --cuda-arch sm_90 \
        --opencl-build-option=-cl-nv-verbose \
        --analyze --plot-metric gflops,ptxas_registers_max

Cartesian-product sweep::

    python matmul_suite.py \
        --m 1024,2048,4096 --n 1024,2048 --k 1024,2048 \
        --variant register --use-cuda --analyze \
        --plot-metric gflops --plot-x metadata.m
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Iterator, Mapping, Sequence

import namedisl as nisl
import numpy as np
import numpy.linalg as la
import loopy as lp
from loopy.transform.compute import compute
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2  # noqa: F401

from loopy_perf_artifacts import CudaLaunchSpec, LoopyPerfCase, make_pyopencl_target
from loopy_perf_suite import SuiteResult, iter_param_grid, run_loopy_perf_suite


# {{{ kernel transformations, mostly preserved from compute-examples/matmul.py


def naive_matmul(knl: lp.TranslationUnit, bm: int, bn: int, bk: int) -> lp.TranslationUnit:
    knl = lp.split_iname(knl, "i", bm, inner_iname="ii", outer_iname="io")
    knl = lp.split_iname(knl, "j", bn, inner_iname="ji", outer_iname="jo")
    knl = lp.split_iname(knl, "k", bk, inner_iname="ki", outer_iname="ko")
    iname_tags = {
        "io": "g.1",
        "jo": "g.0",
        "ii": "l.1",
        "ji": "l.0",
    }
    return lp.tag_inames(knl, iname_tags)


def shared_memory_tiled_matmul(
    knl: lp.TranslationUnit,
    bm: int,
    bn: int,
    bk: int,
) -> lp.TranslationUnit:
    knl = lp.split_iname(knl, "i", bm, inner_iname="ii", outer_iname="io")
    knl = lp.split_iname(knl, "j", bn, inner_iname="ji", outer_iname="jo")
    knl = lp.split_iname(knl, "k", bk, inner_iname="ki", outer_iname="ko")

    compute_map_a = nisl.make_map(
        f"""{{ [is, ks] -> [a_ii, io, a_ki, ko] :
            is = io * {bm} + a_ii and ks = ko * {bk} + a_ki }}"""
    )
    compute_map_b = nisl.make_map(
        f"""{{ [ks, js] -> [b_ki, ko, b_ji, jo] :
            js = jo * {bn} + b_ji and ks = ko * {bk} + b_ki }}"""
    )

    knl = compute(
        knl,
        "a_",
        compute_map=compute_map_a,
        storage_indices=["a_ii", "a_ki"],
        extra_context_inames=["jo"],
        temporary_name="a_tile",
        temporary_address_space=lp.AddressSpace.LOCAL,
        compute_insn_id="a_load",
    )
    knl = compute(
        knl,
        "b_",
        compute_map=compute_map_b,
        storage_indices=["b_ki", "b_ji"],
        extra_context_inames=["io"],
        temporary_name="b_tile",
        temporary_address_space=lp.AddressSpace.LOCAL,
        compute_insn_id="b_load",
    )

    iname_tags = {
        "io": "g.1",
        "ii": "l.1",
        "jo": "g.0",
        "ji": "l.0",
        "a_ii": "l.1",
        "a_ki": "l.0",
        "b_ki": "l.1",
        "b_ji": "l.0",
    }
    return lp.tag_inames(knl, iname_tags)


def register_tiled_matmul(
    knl: lp.TranslationUnit,
    bm: int,
    bn: int,
    bk: int,
    tm: int,
    tn: int,
) -> lp.TranslationUnit:
    # Shared-memory-level split / compute.
    knl = lp.split_iname(knl, "i", bm, inner_iname="ii", outer_iname="io")
    knl = lp.split_iname(knl, "j", bn, inner_iname="ji", outer_iname="jo")
    knl = lp.split_iname(knl, "k", bk, inner_iname="ki", outer_iname="ko")

    compute_map_a = nisl.make_map(
        f"""{{ [is, ks] -> [a_ii, io, a_ki, ko] :
            is = io * {bm} + a_ii and ks = ko * {bk} + a_ki }}"""
    )
    compute_map_b = nisl.make_map(
        f"""{{ [ks, js] -> [b_ki, ko, b_ji, jo] :
            js = jo * {bn} + b_ji and ks = ko * {bk} + b_ki }}"""
    )

    knl = compute(
        knl,
        "a_",
        compute_map=compute_map_a,
        storage_indices=["a_ii", "a_ki"],
        extra_context_inames=["jo"],
        temporary_name="a_smem",
        temporary_address_space=lp.AddressSpace.LOCAL,
        compute_insn_id="a_load",
    )
    knl = compute(
        knl,
        "b_",
        compute_map=compute_map_b,
        storage_indices=["b_ki", "b_ji"],
        extra_context_inames=["io"],
        temporary_name="b_smem",
        temporary_address_space=lp.AddressSpace.LOCAL,
        compute_insn_id="b_load",
    )

    wg_size_i = bm // tm
    wg_size_j = bn // tn
    knl = lp.split_iname(knl, "a_ii", wg_size_i, inner_iname="a_local", outer_iname="a_tile")
    knl = lp.split_iname(knl, "b_ji", wg_size_j, inner_iname="b_local", outer_iname="b_tile")

    # Register-level split / compute.
    knl = lp.extract_subst(knl, "a_smem_", "a_smem[is, ks]", parameters="is, ks")
    knl = lp.extract_subst(knl, "b_smem_", "b_smem[ks, js]", parameters="ks, js")
    knl = lp.split_iname(knl, "ii", tm, inner_iname="ii_reg", outer_iname="ii_thr")
    knl = lp.split_iname(knl, "ji", tn, inner_iname="ji_reg", outer_iname="ji_thr")
    knl = lp.split_iname(knl, "ki", 8, inner_iname="dot", outer_iname="ki_outer")

    a_reg_tile = nisl.make_map(
        f"""{{ [is, ks] -> [a_reg_i, ii_thr, ki_outer, dot] :
            is = ii_thr * {tm} + a_reg_i and ks = ki_outer * 8 + dot }}"""
    )
    b_reg_tile = nisl.make_map(
        f"""{{ [ks, js] -> [b_reg_j, ki_outer, dot, ji_thr] :
            ks = ki_outer * 8 + dot and js = ji_thr * {tn} + b_reg_j }}"""
    )

    knl = compute(
        knl,
        "a_smem_",
        compute_map=a_reg_tile,
        storage_indices=["a_reg_i"],
        extra_context_inames=["ji_thr", "io", "jo", "ko"],
        temporary_name="a_reg",
        temporary_address_space=lp.AddressSpace.PRIVATE,
        compute_insn_id="a_reg_load",
    )
    knl = compute(
        knl,
        "b_smem_",
        compute_map=b_reg_tile,
        storage_indices=["b_reg_j"],
        extra_context_inames=["ii_thr", "io", "jo", "ko"],
        temporary_name="b_reg",
        temporary_address_space=lp.AddressSpace.PRIVATE,
        compute_insn_id="b_reg_load",
    )

    iname_tags = {
        # global tiles
        "io": "g.1",
        "jo": "g.0",
        # a local storage axes
        "a_local": "l.1",
        "a_ki": "l.0",
        # b local storage axes
        "b_local": "l.0",
        "b_ki": "l.1",
        # register tiles
        "ii_thr": "l.1",
        "ji_thr": "l.0",
        # register storage axes
        "a_reg_i": "ilp",
        "b_reg_j": "ilp",
        # compute axes
        "ii_reg": "ilp",
        "ji_reg": "ilp",
    }
    return lp.tag_inames(knl, iname_tags)


# }}}


@dataclass(frozen=True)
class MatmulConfig:
    m: int
    n: int
    k: int
    bm: int
    bn: int
    bk: int
    tm: int
    tn: int
    variant: str


def make_base_matmul_kernel(
    *,
    m: int,
    n: int,
    k: int,
    dtype: np.dtype[Any] | type[np.generic],
) -> lp.TranslationUnit:
    dtype = np.dtype(dtype).type
    knl = lp.make_kernel(
        "{ [i, j, k] : 0 <= i < M and 0 <= j < N and 0 <= k < K }",
        """
        a_(is, ks) := a[is, ks]
        b_(ks, js) := b[ks, js]
        c[i, j] = sum([k], a_(i, k) * b_(k, j))
        """,
        [
            lp.GlobalArg("a", shape=(m, k), dtype=dtype),
            lp.GlobalArg("b", shape=(k, n), dtype=dtype),
            lp.GlobalArg("c", shape=(m, n), dtype=dtype, is_output=True),
        ],
        target=make_pyopencl_target(),
    )
    return lp.fix_parameters(knl, M=m, N=n, K=k)


def check_config(cfg: MatmulConfig, *, backends: Sequence[str]) -> None:
    if cfg.variant not in {"naive", "shared", "register"}:
        raise ValueError(f"unknown matmul variant: {cfg.variant}")
    if cfg.m <= 0 or cfg.n <= 0 or cfg.k <= 0:
        raise ValueError(f"matrix dimensions must be positive: {cfg}")
    if cfg.bm <= 0 or cfg.bn <= 0 or cfg.bk <= 0 or cfg.tm <= 0 or cfg.tn <= 0:
        raise ValueError(f"tile dimensions must be positive: {cfg}")
    if cfg.m % cfg.bm or cfg.n % cfg.bn or cfg.k % cfg.bk:
        raise ValueError(
            "this suite driver requires exact tiling for CUDA launch geometry: "
            f"got m,n,k=({cfg.m},{cfg.n},{cfg.k}) and bm,bn,bk=({cfg.bm},{cfg.bn},{cfg.bk})"
        )
    if cfg.variant == "register":
        if cfg.bm % cfg.tm or cfg.bn % cfg.tn:
            raise ValueError(f"register tiling requires bm % tm == 0 and bn % tn == 0: {cfg}")
        if cfg.bk % 8:
            raise ValueError(f"register_tiled_matmul currently splits ki by 8, so bk must be divisible by 8: {cfg}")

    if "cuda" in backends:
        block = cuda_block_dim(cfg)
        nthreads = block[0] * block[1] * block[2]
        if nthreads > 1024:
            raise ValueError(
                f"CUDA block has {nthreads} threads, which exceeds the usual 1024-thread limit. "
                f"Use smaller bm/bn or larger tm/tn. config={cfg}"
            )


def make_matmul_kernel(cfg: MatmulConfig, *, dtype: np.dtype[Any] | type[np.generic]) -> lp.TranslationUnit:
    knl = make_base_matmul_kernel(m=cfg.m, n=cfg.n, k=cfg.k, dtype=dtype)
    if cfg.variant == "naive":
        return naive_matmul(knl, cfg.bm, cfg.bn, cfg.bk)
    if cfg.variant == "shared":
        return shared_memory_tiled_matmul(knl, cfg.bm, cfg.bn, cfg.bk)
    if cfg.variant == "register":
        return register_tiled_matmul(knl, cfg.bm, cfg.bn, cfg.bk, cfg.tm, cfg.tn)
    raise AssertionError(f"unhandled variant: {cfg.variant}")


def cuda_block_dim(cfg: MatmulConfig) -> tuple[int, int, int]:
    if cfg.variant == "register":
        return (cfg.bm // cfg.tm, cfg.bn // cfg.tn, 1)
    return (cfg.bm, cfg.bn, 1)


def cuda_grid_dim(cfg: MatmulConfig) -> tuple[int, int, int]:
    return (cfg.m // cfg.bm, cfg.n // cfg.bn, 1)


def case_name(cfg: MatmulConfig) -> str:
    return (
        f"matmul_{cfg.variant}"
        f"_m{cfg.m}_n{cfg.n}_k{cfg.k}"
        f"_bm{cfg.bm}_bn{cfg.bn}_bk{cfg.bk}"
        f"_tm{cfg.tm}_tn{cfg.tn}"
    )


def make_validation_fn(
    *,
    a: np.ndarray,
    b: np.ndarray,
    tolerance: float,
):
    c_ref_cache: dict[str, np.ndarray] = {}

    def validate(backend: str, dev_args: Mapping[str, Any], _last_outputs: Any) -> dict[str, Any]:
        if "c_ref" not in c_ref_cache:
            c_ref_cache["c_ref"] = a @ b
        c_ref = c_ref_cache["c_ref"]

        c_dev = dev_args["c"]
        if backend == "opencl":
            c = c_dev.get()
        elif backend == "cuda":
            import cupy as cp  # type: ignore

            c = cp.asnumpy(c_dev)
        else:
            raise ValueError(f"unknown backend for validation: {backend}")

        denom = la.norm(c_ref)
        rel_l2 = la.norm(c - c_ref) / (denom if denom else 1.0)
        return {
            "relative_l2_error": float(rel_l2),
            "tolerance": float(tolerance),
            "ok": bool(rel_l2 <= tolerance),
        }

    return validate


def make_case_factory(
    cfg: MatmulConfig,
    *,
    dtype: np.dtype[Any],
    seed: int,
    validate: bool,
    validation_tolerance: float,
    print_kernel: bool,
    print_device_code: bool,
    backends: Sequence[str],
):
    def factory() -> LoopyPerfCase:
        check_config(cfg, backends=backends)
        knl = make_matmul_kernel(cfg, dtype=dtype)

        rng = np.random.default_rng(seed)
        a = rng.standard_normal((cfg.m, cfg.k)).astype(dtype, copy=False)
        b = rng.standard_normal((cfg.k, cfg.n)).astype(dtype, copy=False)
        c = np.empty((cfg.m, cfg.n), dtype=dtype)

        if print_kernel:
            print(f"\n===== {case_name(cfg)} Loopy kernel =====")
            print(knl)
        if print_device_code:
            print(f"\n===== {case_name(cfg)} OpenCL device code =====")
            print(lp.generate_code_v2(knl).device_code())
            if "cuda" in backends:
                print(f"\n===== {case_name(cfg)} CUDA device code =====")
                print(lp.generate_code_v2(knl.copy(target=lp.CudaTarget())).device_code())

        metadata = {
            "suite": "matmul",
            "variant": cfg.variant,
            "m": cfg.m,
            "n": cfg.n,
            "k": cfg.k,
            "shape": f"{cfg.m}x{cfg.n}x{cfg.k}",
            "bm": cfg.bm,
            "bn": cfg.bn,
            "bk": cfg.bk,
            "tm": cfg.tm,
            "tn": cfg.tn,
            "dtype": str(dtype),
            "output_elements": cfg.m * cfg.n,
            "flop_count_formula": "2*m*n*k",
            "cuda_grid": cuda_grid_dim(cfg),
            "cuda_block": cuda_block_dim(cfg),
        }

        validate_fn = None
        if validate:
            validate_fn = make_validation_fn(a=a, b=b, tolerance=validation_tolerance)

        return LoopyPerfCase(
            name=case_name(cfg),
            knl=knl,
            args={"a": a, "b": b, "c": c},
            flop_count=2 * cfg.m * cfg.n * cfg.k,
            cuda_launch=CudaLaunchSpec(
                grid=cuda_grid_dim(cfg),
                block=cuda_block_dim(cfg),
                arg_order=("a", "b", "c"),
            ),
            validate=validate_fn,
            metadata=metadata,
        )

    return factory


# {{{ CLI parsing / suite construction


def parse_csv_ints(text: str) -> list[int]:
    vals: list[int] = []
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(int(part))
    if not vals:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return vals


def parse_variants(items: Sequence[str] | None) -> list[str]:
    if not items:
        return ["register"]
    variants: list[str] = []
    aliases = {"shared_memory_tiled": "shared", "shared-memory-tiled": "shared", "register_tiled": "register", "register-tiled": "register"}
    for item in items:
        for raw in item.split(","):
            v = aliases.get(raw.strip(), raw.strip())
            if v:
                variants.append(v)
    if "all" in variants:
        return ["naive", "shared", "register"]
    bad = sorted(set(variants) - {"naive", "shared", "register"})
    if bad:
        raise argparse.ArgumentTypeError(f"unknown variant(s): {bad}")
    return variants or ["register"]


def parse_backends(args: argparse.Namespace) -> tuple[str, ...]:
    if args.backend:
        out: list[str] = []
        for item in args.backend:
            out.extend(part.strip().lower() for part in item.split(",") if part.strip())
    else:
        out = []
        if not args.no_cl:
            out.append("opencl")
        if args.use_cuda:
            out.append("cuda")
    bad = sorted(set(out) - {"opencl", "cuda"})
    if bad:
        raise ValueError(f"unknown backend(s): {bad}")
    if not out:
        raise ValueError("no backends requested; use OpenCL, --use-cuda, or --backend")
    # Preserve user order while removing duplicates.
    deduped: list[str] = []
    for b in out:
        if b not in deduped:
            deduped.append(b)
    return tuple(deduped)


def parse_shape(text: str) -> tuple[int, int, int]:
    normalized = text.lower().replace("*", "x").replace(",", "x")
    parts = [p.strip() for p in normalized.split("x") if p.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"expected shape MxNxK or M,N,K, got {text!r}")
    return (int(parts[0]), int(parts[1]), int(parts[2]))


def iter_problem_shapes(args: argparse.Namespace) -> Iterator[tuple[int, int, int]]:
    if args.shape:
        seen: set[tuple[int, int, int]] = set()
        for raw in args.shape:
            shape = parse_shape(raw)
            if shape not in seen:
                seen.add(shape)
                yield shape
        return

    for p in iter_param_grid({"m": args.m, "n": args.n, "k": args.k}):
        yield (int(p["m"]), int(p["n"]), int(p["k"]))


def iter_configs(args: argparse.Namespace) -> Iterator[MatmulConfig]:
    variants = parse_variants(args.variant)
    if args.shared_memory_tiled:
        variants = ["shared"]
    if args.register_tiled:
        variants = ["register"]

    shape_list = list(iter_problem_shapes(args))
    tile_grid = {
        "bm": args.bm,
        "bn": args.bn,
        "bk": args.bk,
        "tm": args.tm,
        "tn": args.tn,
        "variant": variants,
    }

    seen: set[MatmulConfig] = set()
    for m, n, k in shape_list:
        for p in iter_param_grid(tile_grid):
            cfg = MatmulConfig(
                m=m,
                n=n,
                k=k,
                bm=int(p["bm"]),
                bn=int(p["bn"]),
                bk=int(p["bk"]),
                tm=int(p["tm"]),
                tn=int(p["tn"]),
                variant=str(p["variant"]),
            )
            if cfg not in seen:
                seen.add(cfg)
                yield cfg


def dtype_from_name(name: str) -> np.dtype[Any]:
    aliases = {
        "fp32": "float32",
        "single": "float32",
        "f32": "float32",
        "fp64": "float64",
        "double": "float64",
        "f64": "float64",
    }
    try:
        dtype = np.dtype(aliases.get(name.lower(), name))
    except Exception as exc:
        raise argparse.ArgumentTypeError(f"unknown dtype {name!r}") from exc
    if dtype not in {np.dtype("float32"), np.dtype("float64")}:
        raise argparse.ArgumentTypeError("this matmul benchmark currently expects float32 or float64")
    return dtype


def maybe_run_analyzer(
    suite: SuiteResult,
    *,
    analyze: bool,
    plot_metrics: Sequence[str],
    plot_x: str,
    plot_series: str,
    metrics: Sequence[str],
) -> None:
    if not analyze and not plot_metrics:
        return

    analyzer = Path(__file__).with_name("analyze_compiler_artifacts.py")
    if not analyzer.exists():
        print(
            "analyze_compiler_artifacts.py was not found next to this script. "
            f"Run manually: python analyze_compiler_artifacts.py --suite-root {suite.suite_dir}",
            file=sys.stderr,
        )
        return

    cmd = [
        sys.executable,
        str(analyzer),
        "--suite-root",
        str(suite.suite_dir),
        "--csv",
        str(suite.suite_dir / "analyzed_suite.csv"),
    ]
    for metric in metrics:
        cmd.extend(["--metric", metric])
    for metric in plot_metrics:
        cmd.extend(["--plot-metric", metric])
    if plot_metrics:
        cmd.extend([
            "--plot-x",
            plot_x,
            "--plot-series",
            plot_series,
            "--plot-dir",
            str(suite.suite_dir / "plots"),
        ])

    print("\nRunning analyzer:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=False)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a Loopy matmul performance/artifact suite.")

    shape_group = parser.add_argument_group("problem shapes")
    shape_group.add_argument("--shape", action="append", default=[], help="Exact shape MxNxK or M,N,K. May be repeated. Overrides --m/--n/--k Cartesian product.")
    shape_group.add_argument("--m", type=parse_csv_ints, default=[1024], help="Comma-separated M values. Default: 1024")
    shape_group.add_argument("--n", type=parse_csv_ints, default=[1024], help="Comma-separated N values. Default: 1024")
    shape_group.add_argument("--k", type=parse_csv_ints, default=[1024], help="Comma-separated K values. Default: 1024")

    tile_group = parser.add_argument_group("tile parameters")
    tile_group.add_argument("--bm", type=parse_csv_ints, default=[64])
    tile_group.add_argument("--bn", type=parse_csv_ints, default=[64])
    tile_group.add_argument("--bk", type=parse_csv_ints, default=[32])
    tile_group.add_argument("--tm", type=parse_csv_ints, default=[4])
    tile_group.add_argument("--tn", type=parse_csv_ints, default=[4])

    variant_group = parser.add_argument_group("kernel variants")
    variant_group.add_argument("--variant", action="append", default=None, help="naive, shared, register, all. May be repeated or comma-separated. Default: register")
    variant_group.add_argument("--shared-memory-tiled", action="store_true", help="Compatibility alias for --variant shared")
    variant_group.add_argument("--register-tiled", action="store_true", help="Compatibility alias for --variant register")

    run_group = parser.add_argument_group("benchmark execution")
    run_group.add_argument("--backend", action="append", default=None, help="Explicit backend list: opencl,cuda. Overrides --use-cuda/--no-cl.")
    run_group.add_argument("--use-cuda", action="store_true", help="Also run CUDA through CuPy RawModule.")
    run_group.add_argument("--no-cl", action="store_true", help="Do not run OpenCL.")
    run_group.add_argument("--dtype", type=dtype_from_name, default=np.dtype("float32"))
    run_group.add_argument("--seed", type=int, default=0)
    run_group.add_argument("--nwarmup", type=int, default=5)
    run_group.add_argument("--niterations", type=int, default=100)
    run_group.add_argument("--validate", action="store_true", help="Check each backend result against NumPy a @ b. Off by default because it is expensive for large sweeps.")
    run_group.add_argument("--validation-tolerance", type=float, default=1.0e-4)
    run_group.add_argument("--fail-fast", action="store_true")
    run_group.add_argument("--quiet", action="store_true", help="Suppress per-case benchmark printing from the suite runner.")

    artifact_group = parser.add_argument_group("artifact collection")
    artifact_group.add_argument("--artifact-dir", type=Path, default=Path("compiler_artifacts"))
    artifact_group.add_argument("--suite-name", default="matmul_suite")
    artifact_group.add_argument("--dump-artifacts", dest="dump_artifacts", action="store_true", default=True)
    artifact_group.add_argument("--no-dump-artifacts", dest="dump_artifacts", action="store_false")
    artifact_group.add_argument("--cuda-arch", default="sm_90")
    artifact_group.add_argument("--cuda-nvcc-option", action="append", default=[])
    artifact_group.add_argument("--opencl-build-option", action="append", default=[])
    artifact_group.add_argument("--skip-external-tools", action="store_true", help="Skip nvcc/ptxas/cuobjdump/nvdisasm artifact passes.")

    analyze_group = parser.add_argument_group("post-run analysis and plots")
    analyze_group.add_argument("--analyze", action="store_true", help="Run analyze_compiler_artifacts.py on the generated suite directory.")
    analyze_group.add_argument("--metric", action="append", default=["run_status", "gflops", "s_per_iter", "ptxas_registers_max", "sass_instruction_lines", "backend_error_repr"], help="Analyzer metric to show. May be repeated or comma-separated.")
    analyze_group.add_argument("--plot-metric", action="append", default=[], help="Analyzer metric to plot, e.g. gflops,ptxas_registers_max. May be repeated or comma-separated.")
    analyze_group.add_argument("--plot-x", default="metadata.m")
    analyze_group.add_argument("--plot-series", default="backend")

    debug_group = parser.add_argument_group("debug")
    debug_group.add_argument("--print-kernel", action="store_true")
    debug_group.add_argument("--print-device-code", action="store_true")
    debug_group.add_argument("--dry-run", action="store_true", help="Print the expanded case list without running kernels.")

    return parser


def split_csv_items(items: Sequence[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        out.extend(part.strip() for part in item.split(",") if part.strip())
    return out


def main(argv: Sequence[str] | None = None) -> SuiteResult | None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    backends = parse_backends(args)
    configs = list(iter_configs(args))
    if not configs:
        parser.error("no matmul configurations were generated")

    for cfg in configs:
        check_config(cfg, backends=backends)

    print(f"Generated {len(configs)} matmul case(s):")
    for cfg in configs:
        print(f"  - {case_name(cfg)} grid={cuda_grid_dim(cfg)} block={cuda_block_dim(cfg)}")

    if args.dry_run:
        return None

    factories = [
        make_case_factory(
            cfg,
            dtype=args.dtype,
            seed=args.seed + i,
            validate=args.validate,
            validation_tolerance=args.validation_tolerance,
            print_kernel=args.print_kernel,
            print_device_code=args.print_device_code,
            backends=backends,
        )
        for i, cfg in enumerate(configs)
    ]

    suite = run_loopy_perf_suite(
        factories,
        suite_name=args.suite_name,
        artifact_dir=args.artifact_dir,
        backends=backends,  # type: ignore[arg-type]
        dump_artifacts=args.dump_artifacts,
        cuda_arch=args.cuda_arch,
        cuda_nvcc_options=args.cuda_nvcc_option,
        opencl_build_options=args.opencl_build_option,
        skip_external_tools=args.skip_external_tools,
        nwarmup=args.nwarmup,
        niterations=args.niterations,
        fail_fast=args.fail_fast,
        print_results=not args.quiet,
    )

    print("\nSuite complete")
    print(f"  suite_dir: {suite.suite_dir}")
    print(f"  suite_manifest: {suite.manifest_path}")
    print(f"  suite_results_csv: {suite.csv_path}")
    print(f"  completed: {suite.completed}")
    print(f"  failed: {suite.failed}")

    maybe_run_analyzer(
        suite,
        analyze=args.analyze,
        plot_metrics=split_csv_items(args.plot_metric),
        plot_x=args.plot_x,
        plot_series=args.plot_series,
        metrics=split_csv_items(args.metric),
    )

    return suite


if __name__ == "__main__":
    main()
