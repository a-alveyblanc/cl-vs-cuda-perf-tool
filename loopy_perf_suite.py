"""Small suite runner for loopy_perf_artifacts.py.

This module deliberately keeps suite construction explicit: each example still
owns its Loopy kernel construction, argument allocation, FLOP count, and CUDA
launch geometry. The suite runner just executes many LoopyPerfCase objects,
places their artifacts under a shared suite directory, and writes a compact
suite-level CSV/JSON manifest.

Typical use from a matmul example::

    from loopy_perf_artifacts import LoopyPerfCase, CudaLaunchSpec
    from loopy_perf_suite import iter_param_grid, run_loopy_perf_suite

    def make_case(p):
        m, n, k = p["m"], p["n"], p["k"]
        knl = make_matmul_kernel(...)
        a = rng.standard_normal((m, k), dtype=np.float32)
        b = rng.standard_normal((k, n), dtype=np.float32)
        c = np.empty((m, n), dtype=np.float32)
        return LoopyPerfCase(
            name=f"matmul_m{m}_n{n}_k{k}",
            knl=knl,
            args={"a": a, "b": b, "c": c},
            flop_count=2*m*n*k,
            cuda_launch=CudaLaunchSpec(
                grid=(m // bm, n // bn, 1),
                block=(bm // tm, bn // tn, 1),
                arg_order=("a", "b", "c"),
            ),
            metadata={**p, "bm": bm, "bn": bn, "bk": bk, "tm": tm, "tn": tn},
        )

    params = iter_param_grid({"m": [1024, 2048, 4096], "n": [1024], "k": [1024, 2048]})
    run_loopy_perf_suite((make_case(p) for p in params), backends=("opencl", "cuda"))

Then analyze/plot with::

    python analyze_compiler_artifacts.py --suite-root compiler_artifacts/my_suite_* \
        --csv suite.csv --plot-metric gflops,ptxas_registers_max --plot-x metadata.m
"""

from __future__ import annotations

import csv
import datetime as _dt
from dataclasses import dataclass, field
import itertools
import json
from pathlib import Path
import re
import traceback
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from loopy_perf_artifacts import BackendName, LoopyPerfCase, run_loopy_perf_case

CaseFactory = Callable[[], LoopyPerfCase]
CaseLike = LoopyPerfCase | CaseFactory


def _mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_default(obj: Any) -> Any:
    try:
        import numpy as np  # type: ignore
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return {"kind": "numpy.ndarray", "shape": tuple(obj.shape), "dtype": str(obj.dtype), "nbytes": int(obj.nbytes)}
    except Exception:
        pass
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _mkdir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "loopy_suite"


def _exception_record(exc: BaseException) -> dict[str, Any]:
    return {
        "type": type(exc).__name__,
        "repr": repr(exc),
        "message": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    }


def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            _flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
    else:
        out[prefix] = value


def iter_param_grid(grid: Mapping[str, Sequence[Any]]) -> Iterator[dict[str, Any]]:
    """Yield dictionaries for the Cartesian product of a parameter grid."""
    keys = list(grid.keys())
    for values in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, values, strict=True))


@dataclass(frozen=True)
class SuiteResult:
    suite_dir: Path
    manifest_path: Path
    csv_path: Path
    completed: int
    failed: int
    run_dirs: tuple[Path, ...] = field(default_factory=tuple)


def _benchmark_rows_from_manifest(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    base: dict[str, Any] = {
        "case": manifest.get("case", ""),
        "run_dir": manifest.get("run_dir", ""),
        "created_at": manifest.get("created_at", ""),
    }
    _flatten("metadata", manifest.get("metadata") or {}, base)
    _flatten("parameters", manifest.get("parameters") or {}, base)

    rows: list[dict[str, Any]] = []
    for rec in manifest.get("benchmarks", []) or []:
        if not isinstance(rec, dict):
            continue
        row = dict(base)
        for k, v in rec.items():
            if isinstance(v, (dict, list, tuple)):
                row[k] = json.dumps(v, sort_keys=True, default=_json_default)
            else:
                row[k] = v
        rows.append(row)
    return rows


def _write_suite_csv(path: Path, manifests: Sequence[Mapping[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for manifest in manifests:
        rows.extend(_benchmark_rows_from_manifest(manifest))
    _mkdir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as outf:
        writer = csv.DictWriter(outf, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def run_loopy_perf_suite(
    cases: Iterable[CaseLike],
    *,
    suite_name: str = "loopy_suite",
    artifact_dir: str | Path = "compiler_artifacts",
    backends: Sequence[BackendName] = ("opencl",),
    dump_artifacts: bool = True,
    cuda_arch: str = "sm_90",
    cuda_nvcc_options: Iterable[str] = (),
    opencl_build_options: Iterable[str] = (),
    skip_external_tools: bool = False,
    nwarmup: int = 5,
    niterations: int = 100,
    fail_fast: bool = False,
    print_results: bool = True,
) -> SuiteResult:
    """Run many LoopyPerfCase objects under one suite artifact directory.

    Every case gets its own child run directory inside
    ``artifact_dir / f"{suite_name}_{timestamp}"``. Failures are recorded in
    ``suite_manifest.json`` and, unless ``fail_fast=True``, do not stop the
    remaining cases.
    """
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_dir = _mkdir(Path(artifact_dir) / f"{_safe_name(suite_name)}_{stamp}")
    completed_manifests: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    run_dirs: list[Path] = []

    for i, item in enumerate(cases):
        try:
            case = item() if callable(item) and not isinstance(item, LoopyPerfCase) else item
            if not isinstance(case, LoopyPerfCase):
                raise TypeError(f"suite item {i} did not produce a LoopyPerfCase")
            manifest = run_loopy_perf_case(
                case,
                artifact_dir=suite_dir,
                artifact_prefix=f"{i:04d}_{_safe_name(case.name)}",
                backends=backends,
                dump_artifacts=dump_artifacts,
                cuda_arch=cuda_arch,
                cuda_nvcc_options=cuda_nvcc_options,
                opencl_build_options=opencl_build_options,
                skip_external_tools=skip_external_tools,
                nwarmup=nwarmup,
                niterations=niterations,
                print_results=print_results,
            )
            if manifest.get("run_dir"):
                run_dirs.append(Path(str(manifest["run_dir"])))

            status = str(manifest.get("status", "ok"))
            benchmark_count = len(manifest.get("benchmarks") or [])
            completed_manifests.append(manifest)

            if status not in {"ok", "partial"} or benchmark_count == 0:
                err = {
                    "index": i,
                    "case": case.name,
                    "status": status,
                    "run_dir": manifest.get("run_dir"),
                    "benchmark_count": benchmark_count,
                    "backend_errors": manifest.get("backend_errors", {}),
                    "artifact_errors": manifest.get("artifact_errors", {}),
                }
                errors.append(err)
                if print_results:
                    print(f"[suite] case {i} failed: status={status!r}, benchmarks={benchmark_count}")
                    for backend, backend_err in (manifest.get("backend_errors") or {}).items():
                        print(f"[suite]   {backend}: {backend_err.get('repr', backend_err)}")
                if fail_fast:
                    break
        except Exception as exc:
            err = {"index": i, **_exception_record(exc)}
            errors.append(err)
            if print_results:
                print(f"[suite] case {i} failed: {exc!r}")
            if fail_fast:
                break

        successful_count = sum(
            1
            for m in completed_manifests
            if str(m.get("status", "ok")) in {"ok", "partial"}
            and len(m.get("benchmarks") or []) > 0
        )
        suite_manifest = {
            "suite_name": suite_name,
            "suite_dir": str(suite_dir),
            "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "completed": successful_count,
            "attempted": len(completed_manifests) + len([e for e in errors if "case" not in e]),
            "failed": len(errors),
            "run_dirs": [str(p) for p in run_dirs],
            "errors": errors,
            "parameters": {
                "backends": list(backends),
                "dump_artifacts": bool(dump_artifacts),
                "cuda_arch": cuda_arch,
                "cuda_nvcc_options": list(cuda_nvcc_options),
                "opencl_build_options": list(opencl_build_options),
                "skip_external_tools": bool(skip_external_tools),
                "nwarmup": int(nwarmup),
                "niterations": int(niterations),
            },
        }
        _write_json(suite_dir / "suite_manifest.json", suite_manifest)
        _write_suite_csv(suite_dir / "suite_results.csv", completed_manifests)

    return SuiteResult(
        suite_dir=suite_dir,
        manifest_path=suite_dir / "suite_manifest.json",
        csv_path=suite_dir / "suite_results.csv",
        completed=sum(
            1
            for m in completed_manifests
            if str(m.get("status", "ok")) in {"ok", "partial"}
            and len(m.get("benchmarks") or []) > 0
        ),
        failed=len(errors),
        run_dirs=tuple(run_dirs),
    )


__all__ = [
    "CaseFactory",
    "CaseLike",
    "SuiteResult",
    "iter_param_grid",
    "run_loopy_perf_suite",
]
