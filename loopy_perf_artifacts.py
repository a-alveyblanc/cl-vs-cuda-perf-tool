"""Generic Loopy performance and compiler-artifact harness.

This module is intended to replace the matmul-specific benchmarking layer in
``matmul_dump_artifacts.py`` while keeping the same artifact style:

* OpenCL source/build logs/binaries and best-effort POCL cache PTX/SASS capture.
* CUDA source/PTX/cubin/ptxas/SASS/resource-usage dumps.
* Per-backend timing and FLOP/s reporting in JSON and CSV.

The per-example code should still provide the two pieces that are not safe to
infer generically:

* FLOP count, as a number or a function of the host argument mapping.
* CUDA launch geometry and argument order.

Typical use::

    from loopy_perf_artifacts import CudaLaunchSpec, run_loopy_perf

    manifest = run_loopy_perf(
        knl,
        {"a": a, "b": b, "c": c},
        name="matmul_register_tiled_m4096_n4096_k4096",
        flop_count=2*m*n*k,
        backends=("opencl", "cuda"),
        cuda_launch=CudaLaunchSpec(
            grid=(m // bm, n // bn, 1),
            block=(bm // tm, bn // tn, 1),
            arg_order=("a", "b", "c"),
        ),
        dump_artifacts=True,
        artifact_dir="compiler_artifacts",
        cuda_arch="sm_90",
    )
"""

from __future__ import annotations

import csv
import datetime as _dt
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import traceback
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import loopy as lp

BackendName = Literal["opencl", "cuda"]
HostArgs = Mapping[str, Any]
FlopCount = int | float | Callable[[HostArgs], int | float]
ValidateFn = Callable[[BackendName, Mapping[str, Any], Any], Mapping[str, Any] | None]


def make_pyopencl_target() -> Any:
    """Return Loopy's PyOpenCL execution target.

    ``lp.OpenCLTarget`` can generate OpenCL C, but it intentionally does not
    implement ``get_kernel_executor``. The PyOpenCL executor path requires
    ``loopy.target.pyopencl.PyOpenCLTarget``. Keeping this behind a tiny helper
    makes the rest of this module robust across Loopy versions that may or may
    not re-export the class at top level.
    """
    try:
        return lp.PyOpenCLTarget()
    except AttributeError:
        from loopy.target.pyopencl import PyOpenCLTarget

        return PyOpenCLTarget()


def ensure_pyopencl_target(knl: lp.TranslationUnit) -> lp.TranslationUnit:
    """Return *knl* retargeted for PyOpenCL execution if necessary."""
    try:
        from loopy.target.pyopencl import PyOpenCLTarget

        if isinstance(getattr(knl, "target", None), PyOpenCLTarget):
            return knl
    except Exception:
        # Fall through to an unconditional retarget.
        pass

    return knl.copy(target=make_pyopencl_target())


@dataclass(frozen=True)
class CudaLaunchSpec:
    """CUDA launch metadata for a Loopy-generated raw CUDA kernel.

    ``grid`` and ``block`` may be fixed ``(x, y, z)`` tuples or callables that
    receive the host argument mapping and return such tuples. ``arg_order`` must
    match the generated CUDA function parameter order. If omitted, mapping
    insertion order is used, which is convenient but less robust.
    """

    grid: tuple[int, int, int] | Callable[[HostArgs], tuple[int, int, int]]
    block: tuple[int, int, int] | Callable[[HostArgs], tuple[int, int, int]]
    arg_order: Sequence[str] | None = None
    kernel_name: str | None = None
    shared_mem: int = 0
    raw_module_options: Sequence[str] = ()
    raw_module_backend: str | None = None


@dataclass(frozen=True)
class LoopyPerfCase:
    """A benchmarkable Loopy kernel plus concrete runtime metadata."""

    name: str
    knl: lp.TranslationUnit
    args: HostArgs
    flop_count: FlopCount | None = None
    cuda_launch: CudaLaunchSpec | None = None
    validate: ValidateFn | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


# {{{ small artifact utilities


def _mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_text(path: Path, text: str) -> Path:
    _mkdir(path.parent)
    path.write_text(text, encoding="utf-8", errors="replace")
    return path


def _write_bytes(path: Path, data: bytes) -> Path:
    _mkdir(path.parent)
    path.write_bytes(data)
    return path


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return {
            "kind": "numpy.ndarray",
            "shape": tuple(int(i) for i in obj.shape),
            "dtype": str(obj.dtype),
            "nbytes": int(obj.nbytes),
        }
    try:
        import cupy as cp  # type: ignore

        if isinstance(obj, cp.ndarray):
            return {
                "kind": "cupy.ndarray",
                "shape": tuple(int(i) for i in obj.shape),
                "dtype": str(obj.dtype),
                "nbytes": int(obj.nbytes),
            }
    except Exception:
        pass
    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        return {
            "kind": type(obj).__name__,
            "shape": tuple(int(i) for i in obj.shape),
            "dtype": str(obj.dtype),
        }
    return str(obj)


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    _mkdir(path.parent)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    return path


def _exception_record(exc: BaseException) -> dict[str, Any]:
    return {
        "type": type(exc).__name__,
        "repr": repr(exc),
        "message": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    }


def _run_capture(
    argv: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run a command and capture stdout/stderr without throwing on failure."""
    exe = shutil.which(argv[0])
    if exe is None:
        return {
            "cmd": argv,
            "returncode": None,
            "stdout": "",
            "stderr": f"tool not found in PATH: {argv[0]}",
            "skipped": True,
        }

    proc = subprocess.run(
        [exe, *argv[1:]],
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        check=False,
    )
    return {
        "cmd": [exe, *argv[1:]],
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "skipped": False,
    }


def _save_command_result(out_dir: Path, stem: str, result: Mapping[str, Any]) -> None:
    _write_json(out_dir / f"{stem}.cmd.json", result)
    _write_text(out_dir / f"{stem}.stdout.txt", str(result.get("stdout", "")))
    _write_text(out_dir / f"{stem}.stderr.txt", str(result.get("stderr", "")))


def _looks_like_text(data: bytes) -> bool:
    if not data:
        return True
    sample = data[:4096]
    if sample.startswith(b"\x7fELF"):
        return False
    printable = sum((32 <= b <= 126) or b in b"\r\n\t\f\b" for b in sample)
    return printable / len(sample) > 0.92


def _strings_from_binary(data: bytes, min_len: int = 4) -> str:
    chunks: list[str] = []
    cur: list[int] = []
    for b in data:
        if 32 <= b <= 126:
            cur.append(b)
        else:
            if len(cur) >= min_len:
                chunks.append(bytes(cur).decode("ascii", errors="replace"))
            cur = []
    if len(cur) >= min_len:
        chunks.append(bytes(cur).decode("ascii", errors="replace"))
    return "\n".join(chunks) + ("\n" if chunks else "")


def _normalize_sm(cuda_arch: str) -> str:
    cuda_arch = str(cuda_arch).strip()
    if cuda_arch.startswith("sm_"):
        return cuda_arch
    if cuda_arch.startswith("compute_"):
        return "sm_" + cuda_arch.split("_", 1)[1]
    if cuda_arch.isdigit():
        return f"sm_{cuda_arch}"
    return cuda_arch


# }}}

# {{{ generated-code and artifact dumping


def dump_loopy_device_code(
    knl: lp.TranslationUnit,
    out_dir: str | Path,
    backend: BackendName,
) -> str:
    """Dump Loopy IR and generated source for one backend."""
    out_dir = _mkdir(Path(out_dir))

    if backend == "cuda":
        backend_knl = knl.copy(target=lp.CudaTarget())
        ext = "cu"
    elif backend == "opencl":
        backend_knl = ensure_pyopencl_target(knl)
        ext = "cl"
    else:
        raise ValueError(f"unknown backend: {backend}")

    _write_text(out_dir / "loopy_translation_unit.txt", str(backend_knl))
    codegen = lp.generate_code_v2(backend_knl)
    device_code = codegen.device_code()
    _write_text(out_dir / f"loopy_kernel.{ext}", device_code)
    return device_code


_ENVREG_RE = re.compile(
    r"^[\t ]*mov\.(?:b32|u32)\s+(%r[0-9]+),\s*%envreg[0-9]+\s*;",
    flags=re.MULTILINE,
)


def _sanitize_opencl_ptx_for_ptxas(text: str) -> str:
    return _ENVREG_RE.sub(r"\tmov.u32 \1, 0;", text)


def _ptx_envreg_count(text: str) -> int:
    return len(_ENVREG_RE.findall(text))


def _is_probably_ptx_bytes(data: bytes) -> bool:
    head = data[:8192]
    return b".version" in head and (b".entry" in data or b".visible .entry" in data)


def _extract_ptx_from_blob(data: bytes) -> tuple[bytes | None, dict[str, Any]]:
    info: dict[str, Any] = {"method": "none", "offset": -1, "nul": -1, "bytes": 0}
    start = data.find(b".version")
    if start < 0:
        return None, info
    nul = data.find(b"\x00", start)
    end = nul if nul >= 0 else len(data)
    ptx = data[start:end].lstrip()
    if b".entry" not in ptx and b".visible .entry" not in ptx:
        return None, info
    info.update(
        {
            "method": "embedded_or_nul_terminated_ptx",
            "offset": start,
            "nul": nul,
            "bytes": len(ptx),
        }
    )
    return ptx, info


def _assemble_ptx_and_dump_tools(
    out_dir: Path,
    stem: str,
    ptx_path: Path,
    ptxas_arch: str,
    run_external_tools: bool = True,
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    if not run_external_tools:
        return outputs

    text = ptx_path.read_text(encoding="utf-8", errors="replace")
    tool_ptx = ptx_path
    envregs = _ptx_envreg_count(text)
    if envregs:
        sanitized = out_dir / f"{stem}.sanitized_for_ptxas.ptx"
        _write_text(sanitized, _sanitize_opencl_ptx_for_ptxas(text))
        tool_ptx = sanitized
        outputs.append(
            {
                "kind": "sanitized_ptx",
                "path": str(sanitized),
                "envreg_moves_replaced": envregs,
            }
        )

    arch = _normalize_sm(ptxas_arch)
    cubin_path = out_dir / f"{stem}.cubin"
    res = _run_capture(["ptxas", "-v", f"-arch={arch}", "-o", str(cubin_path), str(tool_ptx)])
    _save_command_result(out_dir, f"{stem}.ptxas_from_ptx", res)
    outputs.append({"kind": "ptxas_from_ptx", "path": str(tool_ptx), "returncode": res.get("returncode")})
    if res.get("returncode") != 0:
        return outputs

    sass = _run_capture(["cuobjdump", "--dump-sass", str(cubin_path)])
    _save_command_result(out_dir, f"{stem}.cuobjdump_sass", sass)
    _write_text(out_dir / f"{stem}.sass", sass.get("stdout", ""))
    outputs.append({"kind": "sass", "path": str(out_dir / f"{stem}.sass"), "returncode": sass.get("returncode")})

    rsrc = _run_capture(["cuobjdump", "--dump-resource-usage", str(cubin_path)])
    _save_command_result(out_dir, f"{stem}.cuobjdump_resource_usage", rsrc)
    _write_text(out_dir / f"{stem}.resource_usage.txt", rsrc.get("stdout", ""))
    outputs.append(
        {
            "kind": "resource_usage",
            "path": str(out_dir / f"{stem}.resource_usage.txt"),
            "returncode": rsrc.get("returncode"),
        }
    )

    dis = _run_capture(["nvdisasm", "--print-code", "--print-line-info", str(cubin_path)])
    _save_command_result(out_dir, f"{stem}.nvdisasm", dis)
    _write_text(out_dir / f"{stem}.nvdisasm.sass", dis.get("stdout", ""))
    outputs.append(
        {
            "kind": "nvdisasm_sass",
            "path": str(out_dir / f"{stem}.nvdisasm.sass"),
            "returncode": dis.get("returncode"),
        }
    )

    return outputs


def _is_pocl_context(ctx: cl.Context) -> bool:
    try:
        for dev in ctx.devices:
            platform = dev.platform
            text = f"{platform.name} {platform.vendor} {platform.version}".lower()
            if "pocl" in text or "portable computing language" in text:
                return True
    except Exception:
        pass
    return False


def setup_pocl_capture_env(run_dir: str | Path, enabled: bool = True) -> dict[str, Any]:
    """Force POCL to use a run-local cache.

    Call this before creating the OpenCL context or constructing a Loopy executor.
    """
    run_dir = Path(run_dir)
    info: dict[str, Any] = {
        "enabled": enabled,
        "cache_dir": None,
        "changes": {},
        "warnings": [],
    }
    if not enabled:
        return info

    cache_dir = run_dir / "opencl" / "pocl-cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)
    _mkdir(cache_dir)

    updates = {
        "POCL_CACHE_DIR": str(cache_dir.resolve()),
        "POCL_LEAVE_KERNEL_COMPILER_TEMP_FILES": "1",
        "POCL_KERNEL_CACHE": "1",
        "PYOPENCL_NO_CACHE": "1",
        "LOOPY_NO_CACHE": "1",
    }
    for key, val in updates.items():
        old = os.environ.get(key)
        os.environ[key] = val
        info["changes"][key] = {"old": old, "new": val}

    info["cache_dir"] = updates["POCL_CACHE_DIR"]
    if os.environ.get("POCL_DEVICES") is None:
        info["warnings"].append("POCL_DEVICES is not set; set POCL_DEVICES=CUDA externally if needed.")
    return info


def collect_pocl_cache_artifacts(
    out_dir: str | Path,
    cache_dir: str | Path | None,
    ptxas_arch: str,
    run_external_tools: bool = True,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    dst_root = _mkdir(out_dir / "pocl-cache-extracted")
    result: dict[str, Any] = {
        "kind": "run_local_pocl_cache",
        "env": {
            k: os.environ.get(k)
            for k in [
                "POCL_CACHE_DIR",
                "POCL_KERNEL_CACHE",
                "POCL_LEAVE_KERNEL_COMPILER_TEMP_FILES",
                "POCL_DEVICES",
                "POCL_CUDA_GPU_ARCH",
                "POCL_CUDA_DUMP_NVVM",
                "POCL_DEBUG",
                "PYOPENCL_NO_CACHE",
                "LOOPY_NO_CACHE",
            ]
        },
        "cache_dir": str(cache_dir) if cache_dir is not None else os.environ.get("POCL_CACHE_DIR"),
        "searched_dirs": [],
        "copied_files": [],
        "ptx_files": [],
        "primary_ptx": None,
        "tool_outputs": [],
        "warnings": [],
    }

    if cache_dir is None:
        env_cache = os.environ.get("POCL_CACHE_DIR")
        cache_dir_path = Path(env_cache) if env_cache else None
    else:
        cache_dir_path = Path(cache_dir)

    if cache_dir_path is None:
        result["warnings"].append("No run-local POCL_CACHE_DIR was configured.")
        _write_json(out_dir / "pocl_cache_snapshot_manifest.json", result)
        return result

    cache_dir_path = cache_dir_path.resolve()
    result["cache_dir"] = str(cache_dir_path)

    interesting_exts = {".ptx", ".ll", ".bc", ".cubin", ".fatbin", ".s", ".asm", ".o", ".so", ".cl"}
    seen_copy: set[str] = set()
    ptx_candidates: list[Path] = []

    result["searched_dirs"].append(str(cache_dir_path))
    if cache_dir_path.exists():
        for path in cache_dir_path.rglob("*"):
            if not path.is_file():
                continue
            try:
                data = path.read_bytes()
            except Exception:
                continue

            suffix = path.suffix.lower()
            ptx_data, embedded_info = _extract_ptx_from_blob(data)
            is_ptx = suffix == ".ptx" or ptx_data is not None or _is_probably_ptx_bytes(data)
            if suffix not in interesting_exts and not is_ptx:
                continue

            digest = _sha256_bytes(data)
            if digest in seen_copy:
                continue
            seen_copy.add(digest)

            rel = path.relative_to(cache_dir_path)
            out_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(rel))[:180]
            dst = dst_root / out_name
            _write_bytes(dst, data)

            rec = {
                "source": str(path),
                "path": str(dst),
                "bytes": len(data),
                "suffix": suffix,
                "sha256": digest,
                "is_ptx": is_ptx,
            }
            result["copied_files"].append(rec)

            if is_ptx:
                ptx_path = dst
                if suffix != ".ptx" and ptx_data is not None:
                    ptx_path = dst_root / (out_name + ".extracted.ptx")
                    _write_bytes(ptx_path, ptx_data)
                    rec["ptx_extraction"] = embedded_info
                if ptx_path.suffix != ".ptx":
                    normalized = dst_root / (out_name + ".ptx")
                    _write_bytes(normalized, ptx_path.read_bytes())
                    ptx_path = normalized

                text = ptx_path.read_text(encoding="utf-8", errors="replace")
                ptx_rec = {
                    "path": str(ptx_path),
                    "source": str(path),
                    "bytes": ptx_path.stat().st_size,
                    "envreg_moves": _ptx_envreg_count(text),
                }
                result["ptx_files"].append(ptx_rec)
                ptx_candidates.append(ptx_path)
    else:
        result["warnings"].append(f"POCL cache directory does not exist: {cache_dir_path}")

    if ptx_candidates:
        primary = max(ptx_candidates, key=lambda p: p.stat().st_size)
        result["primary_ptx"] = str(primary)
        primary_copy = out_dir / "pocl_primary.ptx"
        _write_text(primary_copy, primary.read_text(encoding="utf-8", errors="replace"))
        result["tool_outputs"].extend(
            _assemble_ptx_and_dump_tools(out_dir, "pocl_primary", primary_copy, ptxas_arch, run_external_tools)
        )
    else:
        result["warnings"].append(
            "No PTX found in this run-local POCL cache. Make sure collection happens after a kernel launch and POCL_KERNEL_CACHE is not 0."
        )

    _write_json(out_dir / "pocl_cache_snapshot_manifest.json", result)
    return result


def dump_opencl_artifacts(
    knl: lp.TranslationUnit,
    ctx: cl.Context,
    out_dir: str | Path,
    build_options: Iterable[str] = (),
    run_external_tools: bool = True,
    ptxas_arch: str = "sm_90",
    collect_pocl_cache: bool = True,
) -> dict[str, Any]:
    """Dump OpenCL source, build logs, program binaries, and best-effort SASS/PTX."""
    out_dir = _mkdir(Path(out_dir))
    src = dump_loopy_device_code(knl, out_dir, "opencl")
    build_options = list(build_options)
    result: dict[str, Any] = {
        "backend": "opencl",
        "source": str(out_dir / "loopy_kernel.cl"),
        "build_options": build_options,
        "devices": [],
        "binaries": [],
        "errors": [],
        "is_pocl_context": _is_pocl_context(ctx),
    }

    try:
        program = cl.Program(ctx, src).build(options=build_options)
    except Exception as exc:
        result["errors"].append(f"OpenCL manual artifact build failed: {exc!r}")
        _write_text(out_dir / "opencl_build_exception.txt", repr(exc))
        _write_json(out_dir / "artifact_manifest.json", result)
        return result

    for dev in ctx.devices:
        dev_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", dev.name)
        try:
            log = program.get_build_info(dev, cl.program_build_info.LOG)
        except Exception:
            log = ""
        log_path = out_dir / f"opencl_build_log.{dev_name}.txt"
        _write_text(log_path, log)
        result["devices"].append(
            {
                "name": dev.name,
                "vendor": dev.vendor,
                "version": dev.version,
                "driver_version": dev.driver_version,
                "build_log": str(log_path),
            }
        )

    try:
        binaries = list(program.binaries)
    except Exception as exc:
        binaries = []
        result["errors"].append(f"failed to retrieve OpenCL program binaries: {exc!r}")

    for idx, binary in enumerate(binaries):
        data = bytes(binary)
        binary_kind = "elf_cubin" if data.startswith(b"\x7fELF") else "text_ptx" if _looks_like_text(data) else "unknown_bin"
        suffix = "cubin" if binary_kind == "elf_cubin" else "ptx" if binary_kind == "text_ptx" else "bin"
        bin_path = out_dir / f"opencl_device_{idx}.{suffix}"
        _write_bytes(bin_path, data)
        _write_text(out_dir / f"opencl_device_{idx}.strings.txt", _strings_from_binary(data))
        bin_record = {
            "device_index": idx,
            "path": str(bin_path),
            "kind": binary_kind,
            "bytes": len(data),
            "sha256": _sha256_bytes(data),
        }
        result["binaries"].append(bin_record)

        if run_external_tools and binary_kind == "elf_cubin":
            sass = _run_capture(["cuobjdump", "--dump-sass", str(bin_path)])
            _save_command_result(out_dir, f"opencl_device_{idx}.cuobjdump_sass", sass)
            _write_text(out_dir / f"opencl_device_{idx}.sass", sass.get("stdout", ""))

            rsrc = _run_capture(["cuobjdump", "--dump-resource-usage", str(bin_path)])
            _save_command_result(out_dir, f"opencl_device_{idx}.cuobjdump_resource_usage", rsrc)
            _write_text(out_dir / f"opencl_device_{idx}.resource_usage.txt", rsrc.get("stdout", ""))

            dis = _run_capture(["nvdisasm", "--print-code", "--print-line-info", str(bin_path)])
            _save_command_result(out_dir, f"opencl_device_{idx}.nvdisasm", dis)
            _write_text(out_dir / f"opencl_device_{idx}.nvdisasm.sass", dis.get("stdout", ""))

    if collect_pocl_cache and result.get("is_pocl_context"):
        cache_dir_env = os.environ.get("POCL_CACHE_DIR")
        result["pocl_cache_snapshot"] = collect_pocl_cache_artifacts(
            out_dir,
            cache_dir=Path(cache_dir_env) if cache_dir_env else None,
            ptxas_arch=ptxas_arch,
            run_external_tools=run_external_tools,
        )

    _write_json(out_dir / "artifact_manifest.json", result)
    return result


def dump_cuda_artifacts(
    knl: lp.TranslationUnit,
    out_dir: str | Path,
    cuda_arch: str = "sm_90",
    nvcc_options: Iterable[str] = (),
    run_external_tools: bool = True,
) -> dict[str, Any]:
    """Dump CUDA source, PTX, cubin, ptxas logs, SASS, and resource usage."""
    out_dir = _mkdir(Path(out_dir))
    _ = dump_loopy_device_code(knl, out_dir, "cuda")
    cu_path = out_dir / "loopy_kernel.cu"
    cuda_arch = _normalize_sm(cuda_arch)
    nvcc_options = list(nvcc_options)
    result: dict[str, Any] = {
        "backend": "cuda",
        "source": str(cu_path),
        "cuda_arch": cuda_arch,
        "nvcc_options": nvcc_options,
        "outputs": [],
        "errors": [],
    }

    if not run_external_tools:
        _write_json(out_dir / "artifact_manifest.json", result)
        return result

    ptx_path = out_dir / "loopy_kernel.ptx"
    cubin_path = out_dir / "loopy_kernel.cubin"
    common = [f"-arch={cuda_arch}", "-lineinfo", "-Xptxas=-v", *nvcc_options]

    ptx_res = _run_capture(["nvcc", "--ptx", *common, "-o", str(ptx_path), str(cu_path)])
    _save_command_result(out_dir, "nvcc_ptx", ptx_res)
    result["outputs"].append({"kind": "nvcc_ptx", "cmd": ptx_res["cmd"], "returncode": ptx_res["returncode"]})

    cubin_res = _run_capture(["nvcc", "--cubin", *common, "-o", str(cubin_path), str(cu_path)])
    _save_command_result(out_dir, "nvcc_cubin", cubin_res)
    result["outputs"].append({"kind": "nvcc_cubin", "cmd": cubin_res["cmd"], "returncode": cubin_res["returncode"]})

    if ptx_path.exists():
        ptx_data = ptx_path.read_bytes()
        result["outputs"].append(
            {"kind": "ptx", "path": str(ptx_path), "bytes": len(ptx_data), "sha256": _sha256_bytes(ptx_data)}
        )

        ptxas_cubin = out_dir / "loopy_kernel.ptxas.cubin"
        ptxas_res = _run_capture(["ptxas", "-v", f"-arch={cuda_arch}", "-o", str(ptxas_cubin), str(ptx_path)])
        _save_command_result(out_dir, "ptxas_from_ptx", ptxas_res)
        result["outputs"].append({"kind": "ptxas", "cmd": ptxas_res["cmd"], "returncode": ptxas_res["returncode"]})

    if cubin_path.exists():
        cubin_data = cubin_path.read_bytes()
        result["outputs"].append(
            {
                "kind": "cubin",
                "path": str(cubin_path),
                "bytes": len(cubin_data),
                "sha256": _sha256_bytes(cubin_data),
            }
        )

        sass_res = _run_capture(["cuobjdump", "--dump-sass", str(cubin_path)])
        _save_command_result(out_dir, "cuobjdump_sass", sass_res)
        _write_text(out_dir / "loopy_kernel.sass", sass_res.get("stdout", ""))
        result["outputs"].append(
            {"kind": "sass", "path": str(out_dir / "loopy_kernel.sass"), "returncode": sass_res.get("returncode")}
        )

        resource_res = _run_capture(["cuobjdump", "--dump-resource-usage", str(cubin_path)])
        _save_command_result(out_dir, "cuobjdump_resource_usage", resource_res)
        _write_text(out_dir / "loopy_kernel.resource_usage.txt", resource_res.get("stdout", ""))
        result["outputs"].append(
            {
                "kind": "resource_usage",
                "path": str(out_dir / "loopy_kernel.resource_usage.txt"),
                "returncode": resource_res.get("returncode"),
            }
        )

        dis_res = _run_capture(["nvdisasm", "--print-code", "--print-line-info", str(cubin_path)])
        _save_command_result(out_dir, "nvdisasm", dis_res)
        _write_text(out_dir / "loopy_kernel.nvdisasm.sass", dis_res.get("stdout", ""))
        result["outputs"].append(
            {
                "kind": "nvdisasm_sass",
                "path": str(out_dir / "loopy_kernel.nvdisasm.sass"),
                "returncode": dis_res.get("returncode"),
            }
        )

    _write_json(out_dir / "artifact_manifest.json", result)
    return result


# }}}

# {{{ benchmarking helpers


def _summarize_args(args: HostArgs) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, value in args.items():
        if isinstance(value, np.ndarray):
            out[name] = {
                "kind": "numpy.ndarray",
                "shape": tuple(int(i) for i in value.shape),
                "dtype": str(value.dtype),
                "nbytes": int(value.nbytes),
            }
        elif isinstance(value, np.generic):
            out[name] = {"kind": type(value).__name__, "value": value.item()}
        elif np.isscalar(value):
            out[name] = {"kind": type(value).__name__, "value": value}
        else:
            out[name] = {"kind": type(value).__name__, "repr": repr(value)}
    return out


def _resolve_call_or_value(value: Any, args: HostArgs) -> Any:
    return value(args) if callable(value) else value


def _resolve_flop_count(flop_count: FlopCount | None, args: HostArgs) -> float | None:
    if flop_count is None:
        return None
    return float(flop_count(args) if callable(flop_count) else flop_count)


def _bench_record(
    *,
    backend: BackendName,
    total_elapsed_s: float,
    niterations: int,
    flop_count: float | None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    s_per_iter = total_elapsed_s / float(niterations)
    gflops = None if flop_count is None else flop_count / s_per_iter * 1e-9
    record: dict[str, Any] = {
        "backend": backend,
        "niterations": int(niterations),
        "total_elapsed_s": float(total_elapsed_s),
        "s_per_iter": float(s_per_iter),
        "flop_count": flop_count,
        "gflops": None if gflops is None else float(gflops),
    }
    if extra:
        record.update(extra)
    return record


def _as_opencl_arg(queue: cl.CommandQueue, value: Any) -> Any:
    if isinstance(value, cl_array.Array):
        return value
    if isinstance(value, np.ndarray):
        return cl_array.to_device(queue, value)
    return value


def make_opencl_device_args(queue: cl.CommandQueue, args: HostArgs) -> dict[str, Any]:
    """Copy NumPy array arguments to the OpenCL device, leaving scalars unchanged."""
    return {name: _as_opencl_arg(queue, value) for name, value in args.items()}


def _as_cuda_arg(value: Any) -> Any:
    import cupy as cp  # type: ignore

    if isinstance(value, cp.ndarray):
        return value
    if isinstance(value, np.ndarray):
        return cp.asarray(value)
    return value


def make_cuda_device_args(args: HostArgs) -> dict[str, Any]:
    """Copy NumPy array arguments to the CUDA device via CuPy, leaving scalars unchanged."""
    return {name: _as_cuda_arg(value) for name, value in args.items()}


def infer_cuda_kernel_name(knl: lp.TranslationUnit, device_code: str | None = None) -> str:
    """Best-effort inference of the generated CUDA function name."""
    for attr in ("default_entrypoint", "entrypoint"):
        try:
            entry = getattr(knl, attr)
            name = getattr(entry, "name", None)
            if isinstance(name, str) and name:
                return name
        except Exception:
            pass

    try:
        entrypoints = getattr(knl, "entrypoints")
        for entry in entrypoints:
            name = getattr(entry, "name", None)
            if isinstance(name, str) and name:
                return name
    except Exception:
        pass

    if device_code is not None:
        # Loopy's CUDA target normally emits extern "C" __global__ void name(...)
        patterns = [
            r'extern\s+"C"\s+__global__\s+void\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(',
            r'__global__\s+void\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(',
        ]
        for pattern in patterns:
            match = re.search(pattern, device_code)
            if match:
                return match.group(1)

    return "loopy_kernel"


def benchmark_opencl(
    knl: lp.TranslationUnit,
    args: HostArgs,
    *,
    ctx: cl.Context | None = None,
    queue: cl.CommandQueue | None = None,
    flop_count: FlopCount | None = None,
    nwarmup: int = 5,
    niterations: int = 100,
    validate: ValidateFn | None = None,
) -> tuple[dict[str, Any], dict[str, Any], Any, cl.Context, cl.CommandQueue]:
    """Benchmark a Loopy kernel through its PyOpenCL executor."""
    if niterations <= 0:
        raise ValueError("niterations must be positive")

    if queue is None:
        if ctx is None:
            ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    elif ctx is None:
        ctx = queue.context

    opencl_knl = ensure_pyopencl_target(knl)
    dev_args = make_opencl_device_args(queue, args)
    executor = opencl_knl.executor(queue)

    last_outputs: Any = None
    for _ in range(nwarmup):
        evt, last_outputs = executor(queue, **dev_args)
        evt.wait()
    queue.finish()

    start_evt = cl.enqueue_marker(queue)
    for _ in range(niterations):
        evt, last_outputs = executor(queue, **dev_args)
    end_evt = cl.enqueue_marker(queue)
    end_evt.wait()
    start_evt.wait()
    queue.finish()

    total_elapsed_s = (end_evt.profile.end - start_evt.profile.end) * 1e-9
    flops = _resolve_flop_count(flop_count, args)
    record = _bench_record(
        backend="opencl",
        total_elapsed_s=total_elapsed_s,
        niterations=niterations,
        flop_count=flops,
        extra={"nwarmup": int(nwarmup)},
    )

    if validate is not None:
        validation = validate("opencl", dev_args, last_outputs)
        if validation is not None:
            record["validation"] = dict(validation)

    return record, dev_args, last_outputs, ctx, queue


def benchmark_cuda(
    knl: lp.TranslationUnit,
    args: HostArgs,
    launch: CudaLaunchSpec,
    *,
    flop_count: FlopCount | None = None,
    nwarmup: int = 5,
    niterations: int = 100,
    validate: ValidateFn | None = None,
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    """Benchmark a Loopy kernel as a CuPy RawModule CUDA kernel."""
    if niterations <= 0:
        raise ValueError("niterations must be positive")

    import cupy as cp  # type: ignore

    # POCL's CUDA backend and CuPy both use the CUDA driver/runtime.  After an
    # OpenCL/POCL launch, the process' current CUDA context can be different
    # from CuPy's primary context.  Re-select the CuPy device before creating
    # modules, arrays, streams, or events.  This avoids many
    # CUDA_ERROR_INVALID_HANDLE failures when OpenCL and CUDA are benchmarked
    # in the same Python process.
    device = cp.cuda.Device()
    device.use()

    cuda_knl = knl.copy(target=lp.CudaTarget())
    device_code = lp.generate_code_v2(cuda_knl).device_code()
    kernel_name = launch.kernel_name or infer_cuda_kernel_name(cuda_knl, device_code)

    raw_module_kwargs: dict[str, Any] = {
        "code": device_code,
        "options": tuple(launch.raw_module_options),
    }
    if launch.raw_module_backend is not None:
        raw_module_kwargs["backend"] = launch.raw_module_backend

    module = cp.RawModule(**raw_module_kwargs)
    raw_kernel = module.get_function(kernel_name)

    dev_args = make_cuda_device_args(args)
    arg_order = tuple(launch.arg_order) if launch.arg_order is not None else tuple(args.keys())
    missing = [name for name in arg_order if name not in dev_args]
    if missing:
        raise KeyError(f"CUDA arg_order references missing args: {missing}")
    arg_tuple = tuple(dev_args[name] for name in arg_order)

    grid = tuple(int(i) for i in _resolve_call_or_value(launch.grid, args))
    block = tuple(int(i) for i in _resolve_call_or_value(launch.block, args))

    # Use an explicit CuPy stream rather than whatever stream/context another
    # CUDA-using library may have left current.  If CUDA event timing still
    # fails, fall back to host-side timing around synchronized kernel launches.
    stream = cp.cuda.Stream(non_blocking=False)

    def launch_once() -> None:
        raw_kernel(grid, block, arg_tuple, shared_mem=launch.shared_mem, stream=stream)

    with stream:
        for _ in range(nwarmup):
            launch_once()
    stream.synchronize()
    device.synchronize()

    timing_method = "cuda_event"
    timing_warning: str | None = None
    try:
        start_evt = cp.cuda.Event()
        end_evt = cp.cuda.Event()
        with stream:
            start_evt.record(stream)
            for _ in range(niterations):
                launch_once()
            end_evt.record(stream)
        end_evt.synchronize()
        total_elapsed_s = cp.cuda.get_elapsed_time(start_evt, end_evt) * 1e-3
    except Exception as event_exc:
        # This is primarily a guard for CUDA_ERROR_INVALID_HANDLE from event
        # handles after POCL/CUDA activity in the same process.  The fallback is
        # less precise because it includes Python launch overhead, but it keeps
        # the suite producing a usable GFLOP/s value and records the downgrade in
        # the benchmark row.
        timing_method = "host_synchronized_fallback"
        timing_warning = repr(event_exc)
        try:
            stream.synchronize()
            device.synchronize()
        except Exception:
            pass
        t0 = time.perf_counter()
        with stream:
            for _ in range(niterations):
                launch_once()
        stream.synchronize()
        device.synchronize()
        total_elapsed_s = time.perf_counter() - t0

    flops = _resolve_flop_count(flop_count, args)

    extra: dict[str, Any] = {
        "nwarmup": int(nwarmup),
        "kernel_name": kernel_name,
        "grid": grid,
        "block": block,
        "shared_mem": int(launch.shared_mem),
        "arg_order": arg_order,
        "timing_method": timing_method,
    }
    if timing_warning is not None:
        extra["timing_warning"] = timing_warning
    try:
        extra["raw_kernel_attributes"] = dict(raw_kernel.attributes)
    except Exception:
        pass

    record = _bench_record(
        backend="cuda",
        total_elapsed_s=total_elapsed_s,
        niterations=niterations,
        flop_count=flops,
        extra=extra,
    )

    if validate is not None:
        validation = validate("cuda", dev_args, None)
        if validation is not None:
            record["validation"] = dict(validation)

    return record, dev_args, None


# }}}

# {{{ top-level runners


def _print_benchmark_record(record: Mapping[str, Any], *, label: str | None = None) -> None:
    title = label or str(record.get("backend", "benchmark")).upper()
    print(f"================= {title} Results =================")
    print(f"Backend: {record.get('backend')}")
    if record.get("kernel_name"):
        print(f"Kernel: {record.get('kernel_name')}")
    if record.get("grid"):
        print(f"Grid: {record.get('grid')}")
    if record.get("block"):
        print(f"Block: {record.get('block')}")
    print(f"Warmups: {record.get('nwarmup')}")
    print(f"Iterations: {record.get('niterations')}")
    print(f"Total time (s): {float(record.get('total_elapsed_s', 0.0)):.6g}")
    print(f"Time per iter (s): {float(record.get('s_per_iter', 0.0)):.6g}")
    gflops = record.get("gflops")
    if gflops is not None:
        print(f"GFLOP/s: {float(gflops):.6g}")
    if "validation" in record:
        print(f"Validation: {record['validation']}")
    print("================================================")


def _write_benchmarks_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    _mkdir(path.parent)
    columns = [
        "backend",
        "kernel_name",
        "niterations",
        "nwarmup",
        "total_elapsed_s",
        "s_per_iter",
        "flop_count",
        "gflops",
        "grid",
        "block",
    ]
    with path.open("w", newline="", encoding="utf-8") as outf:
        writer = csv.DictWriter(outf, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            row = dict(rec)
            if isinstance(row.get("grid"), (tuple, list)):
                row["grid"] = "x".join(str(i) for i in row["grid"])
            if isinstance(row.get("block"), (tuple, list)):
                row["block"] = "x".join(str(i) for i in row["block"])
            writer.writerow(row)


def run_loopy_perf_case(
    case: LoopyPerfCase,
    *,
    artifact_dir: str | Path = "compiler_artifacts",
    artifact_prefix: str | None = None,
    backends: Sequence[BackendName] = ("opencl",),
    dump_artifacts: bool = True,
    cuda_arch: str = "sm_90",
    cuda_nvcc_options: Iterable[str] = (),
    opencl_build_options: Iterable[str] = (),
    skip_external_tools: bool = False,
    nwarmup: int = 5,
    niterations: int = 100,
    ctx: cl.Context | None = None,
    queue: cl.CommandQueue | None = None,
    print_results: bool = True,
) -> dict[str, Any]:
    """Run one Loopy performance case and optionally dump compiler artifacts.

    Returns the manifest that is also written to ``run_manifest.json`` when
    ``dump_artifacts`` is true. ``benchmarks.csv`` is always written in the run
    directory to make quick comparisons easy.
    """
    if not backends:
        raise ValueError("at least one backend must be requested")
    invalid_backends = sorted(set(backends) - {"opencl", "cuda"})
    if invalid_backends:
        raise ValueError(f"invalid backend(s): {invalid_backends}")

    run_stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_case_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", case.name).strip("_") or "loopy_case"
    prefix = artifact_prefix or f"{safe_case_name}_{run_stamp}"
    run_dir = _mkdir(Path(artifact_dir) / prefix)

    manifest: dict[str, Any] = {
        "case": case.name,
        "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "metadata": dict(case.metadata),
        "args": _summarize_args(case.args),
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
        "benchmarks": [],
        "artifacts": {},
        "warnings": [],
    }

    if "cuda" in backends and case.cuda_launch is None:
        raise ValueError("CUDA backend requested but case.cuda_launch is None")

    # Capture POCL cache only when OpenCL artifacts are requested; this must run
    # before context creation and before Loopy executor construction.
    if dump_artifacts and "opencl" in backends:
        manifest["pocl_capture_env"] = setup_pocl_capture_env(run_dir, enabled=True)

    manifest["backend_errors"] = {}
    manifest["artifact_errors"] = {}
    manifest["status"] = "started"
    _write_json(run_dir / "run_manifest.pre.json", manifest)

    def record_artifact_error(backend: BackendName, exc: BaseException) -> None:
        manifest.setdefault("artifact_errors", {})[backend] = _exception_record(exc)
        manifest.setdefault("warnings", []).append(f"{backend} artifact dump failed: {exc!r}")
        _write_text(run_dir / f"{backend}_artifact_exception.txt", manifest["artifact_errors"][backend]["traceback"])

    def record_backend_error(backend: BackendName, exc: BaseException) -> None:
        manifest.setdefault("backend_errors", {})[backend] = _exception_record(exc)
        manifest.setdefault("warnings", []).append(f"{backend} benchmark failed: {exc!r}")
        _write_text(run_dir / f"{backend}_benchmark_exception.txt", manifest["backend_errors"][backend]["traceback"])
        if print_results:
            print(f"[{case.name}] {backend} benchmark failed: {exc!r}")

    # Dump CUDA artifacts before benchmarking, matching the original matmul
    # driver. Artifact failures should not prevent benchmark attempts.
    if "cuda" in backends and dump_artifacts:
        try:
            manifest["artifacts"]["cuda"] = dump_cuda_artifacts(
                case.knl,
                run_dir / "cuda",
                cuda_arch=cuda_arch,
                nvcc_options=cuda_nvcc_options,
                run_external_tools=not skip_external_tools,
            )
        except Exception as exc:
            record_artifact_error("cuda", exc)
        finally:
            _write_json(run_dir / "run_manifest.pre.json", manifest)

    opencl_ctx: cl.Context | None = ctx
    opencl_queue: cl.CommandQueue | None = queue

    if "opencl" in backends:
        try:
            record, _dev_args, _outputs, opencl_ctx, opencl_queue = benchmark_opencl(
                case.knl,
                case.args,
                ctx=opencl_ctx,
                queue=opencl_queue,
                flop_count=case.flop_count,
                nwarmup=nwarmup,
                niterations=niterations,
                validate=case.validate,
            )
            manifest["benchmarks"].append(record)
            if print_results:
                _print_benchmark_record(record, label=f"{case.name} / OpenCL")
        except Exception as exc:
            record_backend_error("opencl", exc)

        # OpenCL/POCL artifacts are most useful after a kernel launch, but if
        # the benchmark failed we still try to dump source/build diagnostics.
        if dump_artifacts:
            try:
                if opencl_ctx is None:
                    opencl_ctx = cl.create_some_context()
                manifest["artifacts"]["opencl"] = dump_opencl_artifacts(
                    case.knl,
                    opencl_ctx,
                    run_dir / "opencl",
                    build_options=opencl_build_options,
                    run_external_tools=not skip_external_tools,
                    ptxas_arch=cuda_arch,
                    collect_pocl_cache=True,
                )
            except Exception as exc:
                record_artifact_error("opencl", exc)
            finally:
                _write_json(run_dir / "run_manifest.pre.json", manifest)

    if "cuda" in backends:
        assert case.cuda_launch is not None
        try:
            record, _dev_args, _outputs = benchmark_cuda(
                case.knl,
                case.args,
                case.cuda_launch,
                flop_count=case.flop_count,
                nwarmup=nwarmup,
                niterations=niterations,
                validate=case.validate,
            )
            manifest["benchmarks"].append(record)
            if print_results:
                _print_benchmark_record(record, label=f"{case.name} / CUDA")
        except Exception as exc:
            record_backend_error("cuda", exc)

    requested_backends = set(backends)
    successful_backends = {str(rec.get("backend")) for rec in manifest.get("benchmarks", []) if isinstance(rec, dict)}
    failed_backends = set((manifest.get("backend_errors") or {}).keys())
    if successful_backends == requested_backends:
        manifest["status"] = "ok"
    elif successful_backends:
        manifest["status"] = "partial"
    elif failed_backends:
        manifest["status"] = "failed"
    else:
        manifest["status"] = "artifact_only"

    _write_benchmarks_csv(run_dir / "benchmarks.csv", manifest["benchmarks"])
    _write_json(run_dir / "run_manifest.json", manifest)

    if dump_artifacts and print_results:
        print(f"Wrote compiler artifacts to: {run_dir}")

    return manifest


def run_loopy_perf(
    knl: lp.TranslationUnit,
    args: HostArgs,
    *,
    name: str = "loopy_case",
    flop_count: FlopCount | None = None,
    cuda_launch: CudaLaunchSpec | None = None,
    validate: ValidateFn | None = None,
    metadata: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Convenience wrapper around :func:`run_loopy_perf_case`.

    This is the generic routine you can call directly from each example.
    """
    case = LoopyPerfCase(
        name=name,
        knl=knl,
        args=args,
        flop_count=flop_count,
        cuda_launch=cuda_launch,
        validate=validate,
        metadata={} if metadata is None else metadata,
    )
    return run_loopy_perf_case(case, **kwargs)


# }}}

__all__ = [
    "BackendName",
    "CudaLaunchSpec",
    "FlopCount",
    "HostArgs",
    "LoopyPerfCase",
    "ValidateFn",
    "benchmark_cuda",
    "benchmark_opencl",
    "collect_pocl_cache_artifacts",
    "dump_cuda_artifacts",
    "dump_loopy_device_code",
    "dump_opencl_artifacts",
    "infer_cuda_kernel_name",
    "make_cuda_device_args",
    "make_opencl_device_args",
    "run_loopy_perf",
    "run_loopy_perf_case",
    "setup_pocl_capture_env",
]
