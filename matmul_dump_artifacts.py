"""
Loopy GEMM benchmark driver with compiler-artifact dumping for OpenCL and CUDA.

Typical H200 run:

  python matmul_dump_artifacts.py \
      --m 4096 --n 4096 --k 4096 \
      --bm 64 --bn 64 --bk 32 --tm 4 --tn 4 \
      --register-tiled --use-cuda \
      --dump-artifacts --artifact-dir artifacts --cuda-arch sm_90 \
      --opencl-build-option=-cl-nv-verbose

This emits per-backend source, binary, PTX/SASS/resource logs when the relevant
NVIDIA tools are available in PATH: nvcc, ptxas, cuobjdump, nvdisasm.

When dumping artifacts for a POCL OpenCL context, the script forces POCL to use
a per-run cache under:

  <artifact-run>/opencl/pocl-cache

The dumper/analyzer treat that run-local cache as the only POCL cache source.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
from typing import Any, Iterable

import namedisl as nisl
import numpy as np
import numpy.linalg as la

import pyopencl as cl
import pyopencl.array as cl_array

import loopy as lp
from loopy.transform.compute import compute
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2  # noqa: F401


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
    return str(obj)


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    _mkdir(path.parent)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    return path


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


def _save_command_result(out_dir: Path, stem: str, result: dict[str, Any]) -> None:
    _write_json(out_dir / f"{stem}.cmd.json", result)
    _write_text(out_dir / f"{stem}.stdout.txt", result.get("stdout", ""))
    _write_text(out_dir / f"{stem}.stderr.txt", result.get("stderr", ""))


def _looks_like_text(data: bytes) -> bool:
    if not data:
        return True
    sample = data[:4096]
    # PTX is ASCII-like. Cubin is ELF and should not pass this check.
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
    cuda_arch = cuda_arch.strip()
    if cuda_arch.startswith("sm_"):
        return cuda_arch
    if cuda_arch.startswith("compute_"):
        return "sm_" + cuda_arch.split("_", 1)[1]
    if cuda_arch.isdigit():
        return f"sm_{cuda_arch}"
    return cuda_arch


# }}}


# {{{ artifact dumping


def dump_loopy_device_code(
    knl: lp.TranslationUnit,
    out_dir: Path,
    backend: str,
) -> str:
    """Dump Loopy IR and generated source for one backend."""
    _mkdir(out_dir)
    if backend == "cuda":
        backend_knl = knl.copy(target=lp.CudaTarget())
        ext = "cu"
    elif backend == "opencl":
        backend_knl = knl
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
    info.update({"method": "embedded_or_nul_terminated_ptx", "offset": start, "nul": nul, "bytes": len(ptx)})
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
        outputs.append({"kind": "sanitized_ptx", "path": str(sanitized), "envreg_moves_replaced": envregs})

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
    outputs.append({"kind": "resource_usage", "path": str(out_dir / f"{stem}.resource_usage.txt"), "returncode": rsrc.get("returncode")})

    dis = _run_capture(["nvdisasm", "--print-code", "--print-line-info", str(cubin_path)])
    _save_command_result(out_dir, f"{stem}.nvdisasm", dis)
    _write_text(out_dir / f"{stem}.nvdisasm.sass", dis.get("stdout", ""))
    outputs.append({"kind": "nvdisasm_sass", "path": str(out_dir / f"{stem}.nvdisasm.sass"), "returncode": dis.get("returncode")})
    return outputs

def _is_pocl_context(ctx: cl.Context) -> bool:
    try:
        for dev in ctx.devices:
            p = dev.platform
            text = f"{p.name} {p.vendor} {p.version}".lower()
            if "pocl" in text or "portable computing language" in text:
                return True
    except Exception:
        pass
    return False

def setup_pocl_capture_env(run_dir: Path, enabled: bool = True) -> dict[str, Any]:
    """Force POCL to use a cache owned by this artifact run.

    This must run before cl.create_some_context()/Loopy executor construction so
    POCL never consults ~/.cache/pocl or /tmp/POCL_CACHE for this benchmark.
    """
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
        # Single source of truth for POCL intermediates for this run.
        "POCL_CACHE_DIR": str(cache_dir.resolve()),
        # Keep compiler intermediates and keep the kernel cache enabled.
        "POCL_LEAVE_KERNEL_COMPILER_TEMP_FILES": "1",
        "POCL_KERNEL_CACHE": "1",
        # Avoid reusing host-side PyOpenCL/Loopy program caches from another run.
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
    out_dir: Path,
    cache_dir: Path | None,
    ptxas_arch: str,
    run_external_tools: bool = True,
) -> dict[str, Any]:
    dst_root = _mkdir(out_dir / "pocl-cache-extracted")
    result: dict[str, Any] = {
        "kind": "run_local_pocl_cache",
        "env": {k: os.environ.get(k) for k in [
            "POCL_CACHE_DIR", "POCL_KERNEL_CACHE", "POCL_LEAVE_KERNEL_COMPILER_TEMP_FILES",
            "POCL_DEVICES", "POCL_CUDA_GPU_ARCH", "POCL_CUDA_DUMP_NVVM", "POCL_DEBUG",
            "PYOPENCL_NO_CACHE", "LOOPY_NO_CACHE",
        ]},
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
        cache_dir = Path(env_cache) if env_cache else None
    if cache_dir is None:
        result["warnings"].append("No run-local POCL_CACHE_DIR was configured.")
        _write_json(out_dir / "pocl_cache_snapshot_manifest.json", result)
        return result

    cache_dir = cache_dir.resolve()
    result["cache_dir"] = str(cache_dir)
    candidate_roots = [cache_dir]
    interesting_exts = {".ptx", ".ll", ".bc", ".cubin", ".fatbin", ".s", ".asm", ".o", ".so", ".cl"}
    seen_copy: set[str] = set()
    ptx_candidates: list[Path] = []
    for root in candidate_roots:
        result["searched_dirs"].append(str(root))
        if not root.exists():
            continue
        for path in root.rglob("*"):
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
            rel = path.relative_to(root)
            out_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(rel))[:180]
            dst = dst_root / out_name
            _write_bytes(dst, data)
            rec = {"source": str(path), "path": str(dst), "bytes": len(data), "suffix": suffix, "sha256": digest, "is_ptx": is_ptx}
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
                ptx_rec = {"path": str(ptx_path), "source": str(path), "bytes": ptx_path.stat().st_size, "envreg_moves": _ptx_envreg_count(text)}
                result["ptx_files"].append(ptx_rec)
                ptx_candidates.append(ptx_path)

    if ptx_candidates:
        # Use one primary PTX for tool output to avoid creating hundreds of files
        # and overcounting cache artifacts. Prefer largest, which is normally the
        # Loopy kernel over tiny helper kernels.
        primary = max(ptx_candidates, key=lambda p: p.stat().st_size)
        result["primary_ptx"] = str(primary)
        stem = "pocl_primary"
        _write_text(out_dir / "pocl_primary.ptx", primary.read_text(encoding="utf-8", errors="replace"))
        result["tool_outputs"].extend(_assemble_ptx_and_dump_tools(out_dir, stem, out_dir / "pocl_primary.ptx", ptxas_arch, run_external_tools))
    else:
        result["warnings"].append("No PTX found in this run-local POCL cache. Make sure collection happens after a kernel launch and POCL_KERNEL_CACHE is not 0.")
    _write_json(out_dir / "pocl_cache_snapshot_manifest.json", result)
    return result

def dump_opencl_artifacts(
    knl: lp.TranslationUnit,
    ctx: cl.Context,
    out_dir: Path,
    build_options: Iterable[str] = (),
    run_external_tools: bool = True,
    ptxas_arch: str = "sm_90",
    collect_pocl_cache: bool = True,
) -> dict[str, Any]:
    """Dump OpenCL source, build log, program binaries, and best-effort SASS/PTX."""
    out_dir = _mkdir(out_dir)
    src = dump_loopy_device_code(knl, out_dir, "opencl")

    result: dict[str, Any] = {
        "backend": "opencl",
        "source": str(out_dir / "loopy_kernel.cl"),
        "build_options": list(build_options),
        "devices": [],
        "binaries": [],
        "errors": [],
        "is_pocl_context": _is_pocl_context(ctx),
    }

    try:
        program = cl.Program(ctx, src).build(options=list(build_options))
    except Exception as exc:  # keep source artifacts even if manual build fails
        result["errors"].append(f"OpenCL manual artifact build failed: {exc!r}")
        _write_text(out_dir / "opencl_build_exception.txt", repr(exc))
        _write_json(out_dir / "artifact_manifest.json", result)
        return result

    # Build logs are per device.
    for dev in ctx.devices:
        dev_name = dev.name.replace("/", "_").replace(" ", "_")
        try:
            log = program.get_build_info(dev, cl.program_build_info.LOG)
        except Exception as exc:
            log = f"<failed to get build log: {exc!r}>"
        _write_text(out_dir / f"opencl_build_log.{dev_name}.txt", log)
        result["devices"].append(
            {
                "name": dev.name,
                "vendor": dev.vendor,
                "version": dev.version,
                "driver_version": dev.driver_version,
                "build_log": str(out_dir / f"opencl_build_log.{dev_name}.txt"),
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

            res = _run_capture(["cuobjdump", "--dump-resource-usage", str(bin_path)])
            _save_command_result(out_dir, f"opencl_device_{idx}.cuobjdump_resource_usage", res)
            _write_text(out_dir / f"opencl_device_{idx}.resource_usage.txt", res.get("stdout", ""))

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
    out_dir: Path,
    cuda_arch: str = "sm_90",
    nvcc_options: Iterable[str] = (),
    run_external_tools: bool = True,
) -> dict[str, Any]:
    """Dump CUDA source, PTX, cubin, ptxas logs, SASS, and resource usage."""
    out_dir = _mkdir(out_dir)
    src = dump_loopy_device_code(knl, out_dir, "cuda")
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

    ptx_cmd = ["nvcc", "--ptx", *common, "-o", str(ptx_path), str(cu_path)]
    ptx_res = _run_capture(ptx_cmd)
    _save_command_result(out_dir, "nvcc_ptx", ptx_res)
    result["outputs"].append({"kind": "nvcc_ptx", "cmd": ptx_res["cmd"], "returncode": ptx_res["returncode"]})

    cubin_cmd = ["nvcc", "--cubin", *common, "-o", str(cubin_path), str(cu_path)]
    cubin_res = _run_capture(cubin_cmd)
    _save_command_result(out_dir, "nvcc_cubin", cubin_res)
    result["outputs"].append({"kind": "nvcc_cubin", "cmd": cubin_res["cmd"], "returncode": cubin_res["returncode"]})

    if ptx_path.exists():
        ptx_data = ptx_path.read_bytes()
        result["outputs"].append(
            {"kind": "ptx", "path": str(ptx_path), "bytes": len(ptx_data), "sha256": _sha256_bytes(ptx_data)}
        )

        # Direct ptxas is useful because its stderr usually contains register/spill info.
        ptxas_cubin = out_dir / "loopy_kernel.ptxas.cubin"
        ptxas_cmd = ["ptxas", "-v", f"-arch={cuda_arch}", "-o", str(ptxas_cubin), str(ptx_path)]
        ptxas_res = _run_capture(ptxas_cmd)
        _save_command_result(out_dir, "ptxas_from_ptx", ptxas_res)
        result["outputs"].append({"kind": "ptxas", "cmd": ptxas_res["cmd"], "returncode": ptxas_res["returncode"]})

    if cubin_path.exists():
        cubin_data = cubin_path.read_bytes()
        result["outputs"].append(
            {"kind": "cubin", "path": str(cubin_path), "bytes": len(cubin_data), "sha256": _sha256_bytes(cubin_data)}
        )

        sass_res = _run_capture(["cuobjdump", "--dump-sass", str(cubin_path)])
        _save_command_result(out_dir, "cuobjdump_sass", sass_res)
        _write_text(out_dir / "loopy_kernel.sass", sass_res.get("stdout", ""))

        resource_res = _run_capture(["cuobjdump", "--dump-resource-usage", str(cubin_path)])
        _save_command_result(out_dir, "cuobjdump_resource_usage", resource_res)
        _write_text(out_dir / "loopy_kernel.resource_usage.txt", resource_res.get("stdout", ""))

        dis_res = _run_capture(["nvdisasm", "--print-code", "--print-line-info", str(cubin_path)])
        _save_command_result(out_dir, "nvdisasm", dis_res)
        _write_text(out_dir / "loopy_kernel.nvdisasm.sass", dis_res.get("stdout", ""))

    _write_json(out_dir / "artifact_manifest.json", result)
    return result


# }}}


# {{{ benchmarking


def benchmark_kernel_with_cl(
    knl: lp.TranslationUnit,
    kernel_version: str,
    queue: cl.CommandQueue,
    a: np.ndarray,
    b: np.ndarray,
    m: int,
    n: int,
    k: int,
    bm: int,
    bn: int,
    bk: int,
    tm: int = 1,
    tn: int = 1,
    nwarmup: int = 5,
    niterations: int = 100,
) -> dict[str, Any]:
    ex = knl.executor(queue)

    a_cl = cl_array.to_device(queue, a)
    b_cl = cl_array.to_device(queue, b)
    c_cl = cl_array.zeros(queue, (a.shape[0], b.shape[1]), dtype=a_cl.dtype)

    start = cl.enqueue_marker(queue)
    for _ in range(nwarmup):
        ex(queue, a=a_cl, b=b_cl, c=c_cl)
    end = cl.enqueue_marker(queue)
    end.wait()
    start.wait()

    start = cl.enqueue_marker(queue)
    for _ in range(niterations):
        ex(queue, a=a_cl, b=b_cl, c=c_cl)
    end = cl.enqueue_marker(queue)
    end.wait()
    start.wait()

    total_ns = end.profile.end - start.profile.end
    total_elapsed_s = total_ns * 1e-9
    s_per_iter = total_elapsed_s / niterations

    total_flops = 2 * a.shape[0] * a.shape[1] * b.shape[1]
    gflops = (total_flops / s_per_iter) * 1e-9

    c_ref = a @ b
    _, c_res = ex(queue, a=a_cl, b=b_cl, c=c_cl)

    error = la.norm(c_res[0].get() - c_ref) / la.norm(c_ref)

    m, k = a.shape
    _, n = b.shape
    print(f"================= OpenCL Results =================")
    print(f"Kernel version = {kernel_version}")
    print(f"Global problem (m, n, k) = ({m}, {n}, {k})")
    print(f"Block problem size (bm, bn, bk) = ({bm}, {bn}, {bk})")
    print(f"Per-thread output tile (tm, tn) = ({tm}, {tn})")
    print(f"           Error = {error:.4}")
    print(f"   Total time (s): {total_elapsed_s:.4}")
    print(f"Time per iter (s): {s_per_iter:.4}")
    print(f"          GFLOP/s: {gflops:.4f}")
    print(f"================================================")

    return {
        "backend": "opencl",
        "kernel_version": kernel_version,
        "m": m,
        "n": n,
        "k": k,
        "bm": bm,
        "bn": bn,
        "bk": bk,
        "tm": tm,
        "tn": tn,
        "nwarmup": nwarmup,
        "niterations": niterations,
        "total_elapsed_s": total_elapsed_s,
        "s_per_iter": s_per_iter,
        "gflops": gflops,
        "error": float(error),
    }


def benchmark_kernel_with_cuda(
    knl: lp.TranslationUnit,
    kernel_version: str,
    a: np.ndarray,
    b: np.ndarray,
    m: int,
    n: int,
    k: int,
    bm: int,
    bn: int,
    bk: int,
    tm: int = 1,
    tn: int = 1,
    nwarmup: int = 5,
    niterations: int = 100,
    raw_module_options: Iterable[str] = (),
) -> dict[str, Any]:
    import cupy as cu

    a_cu = cu.asarray(a)
    b_cu = cu.asarray(b)
    c_cu = cu.zeros((m, n), dtype=a.dtype)

    knl = knl.copy(target=lp.CudaTarget())
    dev_code = lp.generate_code_v2(knl).device_code()

    module = cu.RawModule(code=dev_code, options=tuple(raw_module_options))
    knl_cu = module.get_function("loopy_kernel")

    grid_dim = (m // bm, n // bn, 1)
    block_dim = (bm // tm, bn // tn, 1)

    for _ in range(nwarmup):
        knl_cu(grid_dim, block_dim, (a_cu, b_cu, c_cu))
    cu.cuda.Device().synchronize()

    start = cu.cuda.Event()
    end = cu.cuda.Event()

    start.record()
    for _ in range(niterations):
        knl_cu(grid_dim, block_dim, (a_cu, b_cu, c_cu))
    end.record()
    end.synchronize()

    total_elapsed_s = cu.cuda.get_elapsed_time(start, end) * 1e-3
    s_per_iter = total_elapsed_s / niterations

    total_flops = 2 * a.shape[0] * a.shape[1] * b.shape[1]
    gflops = (total_flops / s_per_iter) * 1e-9

    c_ref = a_cu @ b_cu
    error = cu.linalg.norm(c_cu - c_ref) / cu.linalg.norm(c_ref)

    print(f"================= CUDA Results =================")
    print(f"Kernel version = {kernel_version}")
    print(f"Global problem (m, n, k) = ({m}, {n}, {k})")
    print(f"Block problem size (bm, bn, bk) = ({bm}, {bn}, {bk})")
    print(f"Per-thread output tile (tm, tn) = ({tm}, {tn})")
    print(f"           Error = {float(error):.4}")
    print(f"   Total time (s): {total_elapsed_s:.4}")
    print(f"Time per iter (s): {s_per_iter:.4}")
    print(f"          GFLOP/s: {gflops:.4f}")
    print(f"================================================")

    return {
        "backend": "cuda",
        "kernel_version": kernel_version,
        "m": m,
        "n": n,
        "k": k,
        "bm": bm,
        "bn": bn,
        "bk": bk,
        "tm": tm,
        "tn": tn,
        "nwarmup": nwarmup,
        "niterations": niterations,
        "total_elapsed_s": total_elapsed_s,
        "s_per_iter": s_per_iter,
        "gflops": gflops,
        "error": float(error),
    }


# }}}


# {{{ loopy transforms from original script


def naive_matmul(
    knl: lp.TranslationUnit,
    bm: int,
    bn: int,
    bk: int,
) -> lp.TranslationUnit:
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

    compute_map_a = nisl.make_map(f"""{{
        [is, ks] -> [a_ii, io, a_ki, ko] :
            is = io * {bm} + a_ii and
            ks = ko * {bk} + a_ki
    }}""")

    compute_map_b = nisl.make_map(f"""{{
        [ks, js] -> [b_ki, ko, b_ji, jo] :
            js = jo * {bn} + b_ji and
            ks = ko * {bk} + b_ki
    }}""")

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
    # shared-memory-level split / compute
    knl = lp.split_iname(knl, "i", bm, inner_iname="ii", outer_iname="io")
    knl = lp.split_iname(knl, "j", bn, inner_iname="ji", outer_iname="jo")
    knl = lp.split_iname(knl, "k", bk, inner_iname="ki", outer_iname="ko")

    compute_map_a = nisl.make_map(f"""{{
        [is, ks] -> [a_ii, io, a_ki, ko] :
            is = io * {bm} + a_ii and
            ks = ko * {bk} + a_ki
    }}""")

    compute_map_b = nisl.make_map(f"""{{
        [ks, js] -> [b_ki, ko, b_ji, jo] :
            js = jo * {bn} + b_ji and
            ks = ko * {bk} + b_ki
    }}""")

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
    knl = lp.split_iname(
        knl,
        "a_ii",
        wg_size_i,
        inner_iname="a_local",
        outer_iname="a_tile",
    )

    knl = lp.split_iname(
        knl,
        "b_ji",
        wg_size_j,
        inner_iname="b_local",
        outer_iname="b_tile",
    )

    # register-level split / compute
    knl = lp.extract_subst(
        knl,
        "a_smem_",
        "a_smem[is, ks]",
        parameters="is, ks",
    )

    knl = lp.extract_subst(
        knl,
        "b_smem_",
        "b_smem[ks, js]",
        parameters="ks, js",
    )

    knl = lp.split_iname(knl, "ii", tm, inner_iname="ii_reg", outer_iname="ii_thr")
    knl = lp.split_iname(knl, "ji", tn, inner_iname="ji_reg", outer_iname="ji_thr")
    knl = lp.split_iname(knl, "ki", 8, inner_iname="dot", outer_iname="ki_outer")

    a_reg_tile = nisl.make_map(f"""{{
        [is, ks] -> [a_reg_i, ii_thr, ki_outer, dot] :
        is = ii_thr * {tm} + a_reg_i and
        ks = ki_outer * 8 + dot
    }}""")

    b_reg_tile = nisl.make_map(f"""{{
        [ks, js] -> [b_reg_j, ki_outer, dot, ji_thr] :
        ks = ki_outer * 8 + dot and
        js = ji_thr * {tn} + b_reg_j
    }}""")

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
        "io": "g.1",
        "jo": "g.0",
        "a_local": "l.1",
        "a_ki": "l.0",
        "b_local": "l.0",
        "b_ki": "l.1",
        "ii_thr": "l.1",
        "ji_thr": "l.0",
        "a_reg_i": "ilp",
        "b_reg_j": "ilp",
        "ii_reg": "ilp",
        "ji_reg": "ilp",
    }

    return lp.tag_inames(knl, iname_tags)


# }}}


def make_matmul_kernel(
    m: int,
    n: int,
    k: int,
    bm: int,
    bn: int,
    bk: int,
    tm: int,
    tn: int,
    dtype: lp.ToLoopyTypeConvertible,
    shared_memory_tiled: bool,
    register_tiled: bool,
) -> tuple[lp.TranslationUnit, str]:
    if register_tiled:
        kernel_version = "register_tiled"
    elif shared_memory_tiled:
        kernel_version = "shared_memory_tiled"
    else:
        kernel_version = "naive"

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
            lp.GlobalArg("c", shape=(m, n), is_output=True),
        ],
    )

    knl = lp.fix_parameters(knl, M=m, N=n, K=k)

    if shared_memory_tiled:
        knl = shared_memory_tiled_matmul(knl, bm, bn, bk)
    elif register_tiled:
        knl = register_tiled_matmul(knl, bm, bn, bk, tm, tn)
    else:
        knl = naive_matmul(knl, bm, bn, bk)

    return knl, kernel_version


def main(
    m: int = 1024,
    n: int = 1024,
    k: int = 1024,
    bm: int = 64,
    bn: int = 64,
    bk: int = 32,
    tm: int = 4,
    tn: int = 4,
    shared_memory_tiled: bool = False,
    register_tiled: bool = False,
    dtype: lp.ToLoopyTypeConvertible = np.float32,
    print_kernel: bool = False,
    print_device_code: bool = False,
    use_cuda: bool = False,
    no_cl: bool = False,
    nwarmup: int = 5,
    niterations: int = 100,
    dump_artifacts: bool = False,
    artifact_dir: str | Path = "compiler_artifacts",
    artifact_prefix: str | None = None,
    cuda_arch: str = "sm_90",
    cuda_nvcc_option: Iterable[str] = (),
    cuda_raw_option: Iterable[str] = (),
    opencl_build_option: Iterable[str] = (),
    skip_external_tools: bool = False,
    seed: int = 0,
) -> None:
    knl, kernel_version = make_matmul_kernel(
        m=m,
        n=n,
        k=k,
        bm=bm,
        bn=bn,
        bk=bk,
        tm=tm,
        tn=tn,
        dtype=dtype,
        shared_memory_tiled=shared_memory_tiled,
        register_tiled=register_tiled,
    )

    if print_kernel:
        print(knl)

    if print_device_code:
        print(lp.generate_code_v2(knl).device_code())

    rng = np.random.default_rng(seed)
    a = rng.standard_normal((m, k), dtype=dtype)
    b = rng.standard_normal((k, n), dtype=dtype)

    run_stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = artifact_prefix or f"{kernel_version}_m{m}_n{n}_k{k}_bm{bm}_bn{bn}_bk{bk}_tm{tm}_tn{tn}_{run_stamp}"
    run_dir = Path(artifact_dir) / prefix
    manifest: dict[str, Any] = {
        "kernel_version": kernel_version,
        "parameters": {
            "m": m,
            "n": n,
            "k": k,
            "bm": bm,
            "bn": bn,
            "bk": bk,
            "tm": tm,
            "tn": tn,
            "dtype": str(dtype),
            "nwarmup": nwarmup,
            "niterations": niterations,
            "seed": seed,
        },
        "benchmarks": [],
        "artifacts": {},
    }

    if dump_artifacts and not no_cl:
        manifest["pocl_capture_env"] = setup_pocl_capture_env(run_dir, enabled=True)

    need_opencl = not no_cl or (dump_artifacts and not no_cl)
    ctx: cl.Context | None = None
    queue: cl.CommandQueue | None = None
    if need_opencl:
        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

    if dump_artifacts:
        _mkdir(run_dir)
        _write_json(run_dir / "run_manifest.pre.json", manifest)
        if use_cuda:
            manifest["artifacts"]["cuda"] = dump_cuda_artifacts(
                knl,
                run_dir / "cuda",
                cuda_arch=cuda_arch,
                nvcc_options=cuda_nvcc_option,
                run_external_tools=not skip_external_tools,
            )

    if not no_cl:
        assert queue is not None
        manifest["benchmarks"].append(
            benchmark_kernel_with_cl(
                knl,
                kernel_version,
                queue,
                a,
                b,
                m,
                n,
                k,
                bm,
                bn,
                bk,
                tm,
                tn,
                nwarmup=nwarmup,
                niterations=niterations,
            )
        )

    if dump_artifacts and not no_cl:
        assert ctx is not None
        manifest["artifacts"]["opencl"] = dump_opencl_artifacts(
            knl,
            ctx,
            run_dir / "opencl",
            build_options=opencl_build_option,
            run_external_tools=not skip_external_tools,
            ptxas_arch=cuda_arch,
            collect_pocl_cache=True,
        )

    if use_cuda:
        manifest["benchmarks"].append(
            benchmark_kernel_with_cuda(
                knl,
                kernel_version,
                a,
                b,
                m,
                n,
                k,
                bm,
                bn,
                bk,
                tm,
                tn,
                nwarmup=nwarmup,
                niterations=niterations,
                raw_module_options=cuda_raw_option,
            )
        )

    if dump_artifacts:
        _write_json(run_dir / "run_manifest.json", manifest)
        print(f"Wrote compiler artifacts to: {run_dir}")
        print(f"Analyze with: python analyze_compiler_artifacts.py {run_dir} --color always | less -R")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    _ = parser.add_argument("--m", action="store", type=int, default=1024)
    _ = parser.add_argument("--n", action="store", type=int, default=1024)
    _ = parser.add_argument("--k", action="store", type=int, default=1024)

    _ = parser.add_argument("--bm", action="store", type=int, default=64)
    _ = parser.add_argument("--bn", action="store", type=int, default=64)
    _ = parser.add_argument("--bk", action="store", type=int, default=16)

    _ = parser.add_argument("--tm", action="store", type=int, default=4)
    _ = parser.add_argument("--tn", action="store", type=int, default=4)

    _ = parser.add_argument("--shared-memory-tiled", action="store_true")
    _ = parser.add_argument("--register-tiled", action="store_true")

    _ = parser.add_argument("--print-kernel", action="store_true")
    _ = parser.add_argument("--print-device-code", action="store_true")

    _ = parser.add_argument("--use-cuda", action="store_true")
    _ = parser.add_argument("--no-cl", action="store_true")

    _ = parser.add_argument("--nwarmup", action="store", type=int, default=5)
    _ = parser.add_argument("--niterations", action="store", type=int, default=100)
    _ = parser.add_argument("--seed", action="store", type=int, default=0)

    _ = parser.add_argument("--dump-artifacts", action="store_true")
    _ = parser.add_argument("--artifact-dir", action="store", default="compiler_artifacts")
    _ = parser.add_argument("--artifact-prefix", action="store", default=None)
    _ = parser.add_argument("--cuda-arch", action="store", default="sm_90")
    _ = parser.add_argument(
        "--cuda-nvcc-option",
        action="append",
        default=[],
        help="Additional option passed to offline nvcc artifact compilation; may be repeated.",
    )
    _ = parser.add_argument(
        "--cuda-raw-option",
        action="append",
        default=[],
        help="Additional option passed to cupy.RawModule; may be repeated.",
    )
    _ = parser.add_argument(
        "--opencl-build-option",
        action="append",
        default=[],
        help="Additional option passed to the manual OpenCL artifact build; may be repeated. Example: -cl-nv-verbose",
    )
    _ = parser.add_argument("--skip-external-tools", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    main(
        m=args.m,
        n=args.n,
        k=args.k,
        bm=args.bm,
        bn=args.bn,
        bk=args.bk,
        tm=args.tm,
        tn=args.tn,
        shared_memory_tiled=args.shared_memory_tiled,
        register_tiled=args.register_tiled,
        print_kernel=args.print_kernel,
        print_device_code=args.print_device_code,
        use_cuda=args.use_cuda,
        no_cl=args.no_cl,
        nwarmup=args.nwarmup,
        niterations=args.niterations,
        dump_artifacts=args.dump_artifacts,
        artifact_dir=args.artifact_dir,
        artifact_prefix=args.artifact_prefix,
        cuda_arch=args.cuda_arch,
        cuda_nvcc_option=args.cuda_nvcc_option,
        cuda_raw_option=args.cuda_raw_option,
        opencl_build_option=args.opencl_build_option,
        skip_external_tools=args.skip_external_tools,
        seed=args.seed,
    )

