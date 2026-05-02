"""Loopy 2.5D finite-difference performance suite.

This rewrites ``compute-examples/finite-difference-2-5D.py`` as a suite driver
for the generic helpers in this directory:

* ``loopy_perf_artifacts.py``: per-case OpenCL/CUDA timing + artifact dumps
* ``loopy_perf_suite.py``: runs many ``LoopyPerfCase`` objects under one suite
* ``analyze_compiler_artifacts.py``: suite CSV summaries and plots

Typical CUDA + OpenCL comparison::

    python finite_difference_2_5d_suite.py \
        --variant all \
        --npoints 64,128,256 \
        --stencil-width 5,9 \
        --use-cuda \
        --dump-artifacts \
        --cuda-arch sm_90 \
        --analyze \
        --plot-metric gflops,ptxas_registers_max \
        --plot-x metadata.npts

CUDA-only sanity run::

    python finite_difference_2_5d_suite.py \
        --variant compute \
        --npoints 128 \
        --stencil-width 5 \
        --use-cuda --no-cl \
        --analyze --plot-metric gflops
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterator, Mapping, Sequence

import namedisl as nisl
import numpy as np
import numpy.linalg as la
import loopy as lp
from loopy.transform.compute import compute
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2

from loopy_perf_artifacts import CudaLaunchSpec, LoopyPerfCase, make_pyopencl_target
from loopy_perf_suite import SuiteResult, iter_param_grid, run_loopy_perf_suite


# {{{ numerical setup


def centered_second_derivative_coefficients(radius: int, dtype: np.dtype[Any]) -> np.ndarray:
    """Return centered finite-difference coefficients for d^2/dx^2.

    The coefficient vector is ordered for offsets ``[-radius, ..., radius]``.
    """
    scalar_dtype = np.dtype(dtype).type
    offsets = np.arange(-radius, radius + 1, dtype=scalar_dtype)
    powers = np.arange(2 * radius + 1)
    vandermonde = offsets[np.newaxis, :] ** powers[:, np.newaxis]
    rhs = np.zeros(2 * radius + 1, dtype=scalar_dtype)
    rhs[2] = 2
    return np.linalg.solve(vandermonde, rhs).astype(scalar_dtype)


def f(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    return x**2 + y**2 + z**2


def laplacian_f(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    return 6 * np.ones_like(x)


def laplacian_flop_count(npts: int, stencil_width: int) -> int:
    """Match the FLOP convention used by the original example.

    For each stencil offset, the expression has approximately:

    * two additions for ``u_x + u_y + u_z``;
    * one multiply by ``c[l+r]``;
    * one accumulation into the reduction.
    """
    radius = stencil_width // 2
    output_points = (npts - 2 * radius) ** 3
    return int(4 * stencil_width * output_points)


# }}}


# {{{ kernel construction


@dataclass(frozen=True)
class FD25DConfig:
    npts: int
    stencil_width: int
    variant: str
    bm: int
    bn: int
    bk: int
    plane_tile: int

    @property
    def radius(self) -> int:
        return self.stencil_width // 2


def make_base_fd25d_kernel(
    cfg: FD25DConfig,
    *,
    dtype: np.dtype[Any] | type[np.generic],
) -> lp.TranslationUnit:
    scalar_dtype = np.dtype(dtype).type
    npts = int(cfg.npts)
    m = int(cfg.stencil_width)

    knl = lp.make_kernel(
        "{ [i, j, k, l] : r <= i, j, k < npts - r and -r <= l < r + 1 }",
        """
        u_(is, js, ks) := u[is, js, ks]
        lap_u[i, j, k] = sum(
            [l],
            c[l+r] * (u_(i-l, j, k) + u_(i, j-l, k) + u_(i, j, k-l))
        )
        """,
        [
            lp.GlobalArg("u", dtype=scalar_dtype, shape=(npts, npts, npts)),
            lp.GlobalArg("lap_u", dtype=scalar_dtype, shape=(npts, npts, npts), is_output=True),
            lp.GlobalArg("c", dtype=scalar_dtype, shape=(m,)),
        ],
        target=make_pyopencl_target(),
        lang_version=LOOPY_USE_LANGUAGE_VERSION_2018_2,
    )
    return lp.fix_parameters(knl, npts=npts, r=cfg.radius)


def apply_fd25d_transforms(cfg: FD25DConfig, knl: lp.TranslationUnit, *, dtype: np.dtype[Any]) -> lp.TranslationUnit:
    bm, bn, bk = int(cfg.bm), int(cfg.bn), int(cfg.bk)
    r = int(cfg.radius)

    knl = lp.split_iname(knl, "i", bm, inner_iname="ii", outer_iname="io")
    knl = lp.split_iname(knl, "j", bn, inner_iname="ji", outer_iname="jo")
    knl = lp.split_iname(knl, "k", bk, inner_iname="ki", outer_iname="ko")

    if cfg.variant == "compute":
        scalar_dtype = np.dtype(dtype).type
        plane_map = nisl.make_map(
            f"""{{ [is, js, ks] -> [io, ii_s, jo, ji_s, ko, ki] :
                is = io * {bm} + ii_s - {r}
                and js = jo * {bn} + ji_s - {r}
                and ks = ko * {bk} + ki }}"""
        )
        knl = compute(
            knl,
            "u_",
            compute_map=plane_map,
            storage_indices=["ii_s", "ji_s"],
            temporary_name="u_ij_plane",
            temporary_address_space=lp.AddressSpace.LOCAL,
            temporary_dtype=scalar_dtype,
            compute_insn_id="u_plane_compute",
        )

        ring_buffer_map = nisl.make_map(
            f"""{{ [is, js, ks] -> [io, ii, jo, ji, ko, ki, kb] :
                is = io * {bm} + ii
                and js = jo * {bn} + ji
                and kb = ks - (ko * {bk} + ki) + {r} }}"""
        )
        knl = compute(
            knl,
            "u_",
            compute_map=ring_buffer_map,
            storage_indices=["kb"],
            temporary_name="u_k_buf",
            temporary_address_space=lp.AddressSpace.PRIVATE,
            temporary_dtype=scalar_dtype,
            compute_insn_id="u_ring_buf_compute",
            inames_to_advance=["ki"],
        )

        nt = int(cfg.plane_tile)
        knl = lp.split_iname(knl, "ii_s", nt, outer_iname="ii_s_tile", inner_iname="ii_s_local")
        knl = lp.split_iname(knl, "ji_s", nt, outer_iname="ji_s_tile", inner_iname="ji_s_local")
        knl = lp.tag_inames(
            knl,
            {
                # 2D plane-compute storage loops.
                "ii_s_local": "l.1",
                "ji_s_local": "l.0",
                # Force the ring buffer to registers.
                "kb": "unr",
            },
        )

    if cfg.variant not in {"baseline", "compute"}:
        raise ValueError(f"unknown finite-difference variant: {cfg.variant}")

    return lp.tag_inames(
        knl,
        {
            # CUDA/OpenCL group axes. Loopy's g.0/g.1/g.2 map to x/y/z.
            "ko": "g.0",
            "jo": "g.1",
            "io": "g.2",
            # Local axes. Loopy's l.0/l.1 map to x/y.
            "ji": "l.0",
            "ii": "l.1",
        },
    )


def make_fd25d_kernel(cfg: FD25DConfig, *, dtype: np.dtype[Any] | type[np.generic]) -> lp.TranslationUnit:
    dtype = np.dtype(dtype)
    return apply_fd25d_transforms(cfg, make_base_fd25d_kernel(cfg, dtype=dtype), dtype=dtype)


# }}}


# {{{ config/case helpers


def ceil_div(a: int, b: int) -> int:
    return -(-int(a) // int(b))


def fd25d_cuda_grid_dim(cfg: FD25DConfig) -> tuple[int, int, int]:
    # The split inames are not offset by ``r``. For a domain
    # ``r <= axis < npts-r`` represented as ``axis_outer*tile + axis_inner``,
    # outer values are needed through ``floor((npts-r-1)/tile)``.
    n_minus_r = cfg.npts - cfg.radius
    return (
        ceil_div(n_minus_r, cfg.bk),
        ceil_div(n_minus_r, cfg.bn),
        ceil_div(n_minus_r, cfg.bm),
    )


def fd25d_cuda_block_dim(cfg: FD25DConfig) -> tuple[int, int, int]:
    # l.0 -> x, l.1 -> y. ``ki`` is sequential inside each work-item.
    return (int(cfg.bn), int(cfg.bm), 1)


def case_name(cfg: FD25DConfig) -> str:
    return (
        f"fd25d_{cfg.variant}"
        f"_n{cfg.npts}_sw{cfg.stencil_width}"
        f"_bm{cfg.bm}_bn{cfg.bn}_bk{cfg.bk}"
        f"_pt{cfg.plane_tile}"
    )


def check_config(cfg: FD25DConfig, *, backends: Sequence[str]) -> None:
    if cfg.variant not in {"baseline", "compute"}:
        raise ValueError(f"unknown finite-difference variant: {cfg.variant}")
    if cfg.npts <= 0:
        raise ValueError(f"npts must be positive: {cfg.npts}")
    if cfg.stencil_width <= 0 or cfg.stencil_width % 2 == 0:
        raise ValueError(f"stencil_width must be a positive odd integer: {cfg.stencil_width}")
    if cfg.npts <= 2 * cfg.radius:
        raise ValueError(
            f"npts must exceed 2*radius; got npts={cfg.npts}, radius={cfg.radius}"
        )
    if cfg.bm <= 0 or cfg.bn <= 0 or cfg.bk <= 0 or cfg.plane_tile <= 0:
        raise ValueError(f"tile dimensions must be positive: {cfg}")
    if "cuda" in backends:
        block = fd25d_cuda_block_dim(cfg)
        nthreads = block[0] * block[1] * block[2]
        if nthreads > 1024:
            raise ValueError(
                f"CUDA block has {nthreads} threads, exceeding the usual 1024-thread limit. "
                f"Use smaller --bm/--bn. config={cfg}"
            )


def make_problem_arrays(cfg: FD25DConfig, *, dtype: np.dtype[Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scalar_dtype = np.dtype(dtype).type
    pts = np.linspace(-1, 1, num=cfg.npts, endpoint=True, dtype=scalar_dtype)
    h = pts[1] - pts[0]
    x, y, z = np.meshgrid(*(pts,) * 3)
    x = x.reshape(*(cfg.npts,) * 3).astype(scalar_dtype, copy=False)
    y = y.reshape(*(cfg.npts,) * 3).astype(scalar_dtype, copy=False)
    z = z.reshape(*(cfg.npts,) * 3).astype(scalar_dtype, copy=False)

    u = f(x, y, z).astype(scalar_dtype, copy=False)
    lap_u = np.empty_like(u)
    c = (centered_second_derivative_coefficients(cfg.radius, dtype) / h**2).astype(scalar_dtype)
    true_lap = laplacian_f(x, y, z).astype(scalar_dtype, copy=False)
    return u, lap_u, c, true_lap


def make_validation_fn(
    *,
    cfg: FD25DConfig,
    true_lap: np.ndarray,
    tolerance: float,
):
    sl = (slice(cfg.radius, cfg.npts - cfg.radius),) * 3
    denom = la.norm(true_lap[sl])
    denom = float(denom if denom else 1.0)

    def validate(backend: str, dev_args: Mapping[str, Any], last_outputs: Any) -> dict[str, Any]:
        out_dev: Any | None = None
        if last_outputs is not None:
            if isinstance(last_outputs, (tuple, list)) and len(last_outputs) > 0:
                out_dev = last_outputs[0]
            else:
                out_dev = last_outputs
        if out_dev is None:
            out_dev = dev_args["lap_u"]

        if backend == "opencl":
            lap_u = out_dev.get()
        elif backend == "cuda":
            import cupy as cp  # type: ignore

            lap_u = cp.asnumpy(out_dev)
        else:
            raise ValueError(f"unknown backend for validation: {backend}")

        rel_l2 = la.norm(true_lap[sl] - lap_u[sl]) / denom
        return {
            "relative_l2_error": float(rel_l2),
            "tolerance": float(tolerance),
            "ok": bool(rel_l2 <= tolerance),
        }

    return validate


def make_case_factory(
    cfg: FD25DConfig,
    *,
    dtype: np.dtype[Any],
    validate: bool,
    validation_tolerance: float,
    print_kernel: bool,
    print_device_code: bool,
    backends: Sequence[str],
):
    def factory() -> LoopyPerfCase:
        check_config(cfg, backends=backends)
        knl = make_fd25d_kernel(cfg, dtype=dtype)
        u, lap_u, c, true_lap = make_problem_arrays(cfg, dtype=dtype)

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
            "suite": "finite_difference_2_5d",
            "variant": cfg.variant,
            "npts": cfg.npts,
            "stencil_width": cfg.stencil_width,
            "radius": cfg.radius,
            "bm": cfg.bm,
            "bn": cfg.bn,
            "bk": cfg.bk,
            "plane_tile": cfg.plane_tile,
            "dtype": str(dtype),
            "output_points": (cfg.npts - 2 * cfg.radius) ** 3,
            "flop_count_formula": "4*stencil_width*(npts-2*radius)^3",
            "cuda_grid": fd25d_cuda_grid_dim(cfg),
            "cuda_block": fd25d_cuda_block_dim(cfg),
        }

        validate_fn = None
        if validate:
            validate_fn = make_validation_fn(
                cfg=cfg,
                true_lap=true_lap,
                tolerance=validation_tolerance,
            )

        return LoopyPerfCase(
            name=case_name(cfg),
            knl=knl,
            args={"u": u, "lap_u": lap_u, "c": c},
            flop_count=laplacian_flop_count(cfg.npts, cfg.stencil_width),
            cuda_launch=CudaLaunchSpec(
                grid=fd25d_cuda_grid_dim(cfg),
                block=fd25d_cuda_block_dim(cfg),
                arg_order=("u", "lap_u", "c"),
            ),
            validate=validate_fn,
            metadata=metadata,
        )

    return factory


# }}}


# {{{ CLI


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


def split_csv_items(items: Sequence[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        out.extend(part.strip() for part in item.split(",") if part.strip())
    return out


def parse_variants(items: Sequence[str] | None) -> list[str]:
    if not items:
        return ["compute"]
    aliases = {
        "2.5d": "compute",
        "2-5d": "compute",
        "no_compute": "baseline",
        "no-compute": "baseline",
        "base": "baseline",
    }
    variants: list[str] = []
    for item in items:
        for raw in item.split(","):
            v = aliases.get(raw.strip().lower(), raw.strip().lower())
            if v:
                variants.append(v)
    if "all" in variants:
        return ["baseline", "compute"]
    bad = sorted(set(variants) - {"baseline", "compute"})
    if bad:
        raise argparse.ArgumentTypeError(f"unknown variant(s): {bad}")
    return variants or ["compute"]


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
    deduped: list[str] = []
    for b in out:
        if b not in deduped:
            deduped.append(b)
    return tuple(deduped)


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
        raise argparse.ArgumentTypeError("this benchmark expects float32 or float64")
    return dtype


def iter_configs(args: argparse.Namespace) -> Iterator[FD25DConfig]:
    variants = parse_variants(args.variant)
    if args.compare:
        variants = ["baseline", "compute"]
    if args.compute:
        variants = ["compute"]

    grid = {
        "npts": args.npoints,
        "stencil_width": args.stencil_width,
        "bm": args.bm,
        "bn": args.bn,
        "bk": args.bk,
        "plane_tile": args.plane_tile,
        "variant": variants,
    }
    seen: set[FD25DConfig] = set()
    for p in iter_param_grid(grid):
        cfg = FD25DConfig(
            npts=int(p["npts"]),
            stencil_width=int(p["stencil_width"]),
            bm=int(p["bm"]),
            bn=int(p["bn"]),
            bk=int(p["bk"]),
            plane_tile=int(p["plane_tile"]),
            variant=str(p["variant"]),
        )
        if cfg not in seen:
            seen.add(cfg)
            yield cfg


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
    parser = argparse.ArgumentParser(description="Run a Loopy 2.5D finite-difference performance/artifact suite.")

    problem_group = parser.add_argument_group("problem sizes")
    problem_group.add_argument("--npoints", "--npts", type=parse_csv_ints, default=[64], help="Comma-separated grid sizes. Default: 64")
    problem_group.add_argument("--stencil-width", type=parse_csv_ints, default=[5], help="Comma-separated odd stencil widths. Default: 5")

    tile_group = parser.add_argument_group("tile parameters")
    tile_group.add_argument("--bm", type=parse_csv_ints, default=[16])
    tile_group.add_argument("--bn", type=parse_csv_ints, default=[16])
    tile_group.add_argument("--bk", type=parse_csv_ints, default=[32])
    tile_group.add_argument("--plane-tile", type=parse_csv_ints, default=[16], help="Local split factor for compute-variant plane loads. Default: 16")

    variant_group = parser.add_argument_group("kernel variants")
    variant_group.add_argument("--variant", action="append", default=None, help="baseline, compute, all. May be repeated or comma-separated. Default: compute")
    variant_group.add_argument("--compute", action="store_true", help="Compatibility alias for --variant compute")
    variant_group.add_argument("--compare", action="store_true", help="Run both baseline and compute variants")

    run_group = parser.add_argument_group("benchmark execution")
    run_group.add_argument("--backend", action="append", default=None, help="Explicit backend list: opencl,cuda. Overrides --use-cuda/--no-cl.")
    run_group.add_argument("--use-cuda", action="store_true", help="Also run CUDA through CuPy RawModule.")
    run_group.add_argument("--no-cl", action="store_true", help="Do not run OpenCL.")
    run_group.add_argument("--dtype", type=dtype_from_name, default=np.dtype("float64"))
    run_group.add_argument("--nwarmup", type=int, default=3)
    run_group.add_argument("--niterations", type=int, default=10)
    run_group.add_argument("--validate", action="store_true", help="Check numerical error against analytic Laplacian on interior points.")
    run_group.add_argument("--validation-tolerance", type=float, default=None, help="Default: 1e-4 for float32, 1e-9 for float64")
    run_group.add_argument("--fail-fast", action="store_true")
    run_group.add_argument("--quiet", action="store_true", help="Suppress per-case benchmark printing from the suite runner.")

    artifact_group = parser.add_argument_group("artifact collection")
    artifact_group.add_argument("--artifact-dir", type=Path, default=Path("compiler_artifacts"))
    artifact_group.add_argument("--suite-name", default="finite_difference_2_5d_suite")
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
    analyze_group.add_argument("--plot-x", default="metadata.npts")
    analyze_group.add_argument("--plot-series", default="backend")

    debug_group = parser.add_argument_group("debug")
    debug_group.add_argument("--print-kernel", action="store_true")
    debug_group.add_argument("--print-device-code", action="store_true")
    debug_group.add_argument("--dry-run", action="store_true", help="Print the expanded case list without running kernels.")

    return parser


def main(argv: Sequence[str] | None = None) -> SuiteResult | None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    backends = parse_backends(args)
    configs = list(iter_configs(args))
    if not configs:
        parser.error("no finite-difference configurations were generated")

    for cfg in configs:
        check_config(cfg, backends=backends)

    validation_tolerance = args.validation_tolerance
    if validation_tolerance is None:
        validation_tolerance = 1.0e-4 if args.dtype == np.dtype("float32") else 1.0e-9

    print(f"Generated {len(configs)} finite-difference case(s):")
    for cfg in configs:
        print(f"  - {case_name(cfg)} grid={fd25d_cuda_grid_dim(cfg)} block={fd25d_cuda_block_dim(cfg)}")

    if args.dry_run:
        return None

    factories = [
        make_case_factory(
            cfg,
            dtype=args.dtype,
            validate=args.validate,
            validation_tolerance=float(validation_tolerance),
            print_kernel=args.print_kernel,
            print_device_code=args.print_device_code,
            backends=backends,
        )
        for cfg in configs
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
