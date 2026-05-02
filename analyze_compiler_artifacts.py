"""
Terminal-first compiler artifact analyzer for matmul_dump_artifacts.py.

The default view intentionally analyzes one canonical artifact set per backend:
  * CUDA: loopy_kernel.ptx + loopy_kernel.sass + ptxas/resource logs.
  * POCL/OpenCL: opencl/pocl_primary.ptx + opencl/pocl_primary.sass + logs.
For POCL runs, matmul_dump_artifacts.py forces a run-local cache at
  opencl/pocl-cache
and this analyzer ignores system-wide POCL caches by default.

Use --all-opencl-artifacts only when you deliberately want to sum every file in
an OpenCL/POCL cache snapshot.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Iterable


PTX_PATTERNS: dict[str, re.Pattern[str]] = {
    "ptx_fma": re.compile(r"\bfma(?:\.[a-z0-9]+)*\.f(?:16|32|64)\b", re.I),
    "ptx_mad": re.compile(r"\bmad(?:\.[a-z0-9]+)*\b", re.I),
    "ptx_mul": re.compile(r"\bmul(?:\.[a-z0-9]+)*\b", re.I),
    "ptx_add": re.compile(r"\badd(?:\.[a-z0-9]+)*\b", re.I),
    "ptx_ld_global": re.compile(r"\bld(?:\.volatile)?\.global\b", re.I),
    "ptx_st_global": re.compile(r"\bst(?:\.volatile)?\.global\b", re.I),
    "ptx_ld_shared": re.compile(r"\bld(?:\.volatile)?\.shared\b", re.I),
    "ptx_st_shared": re.compile(r"\bst(?:\.volatile)?\.shared\b", re.I),
    "ptx_ld_local": re.compile(r"\bld(?:\.volatile)?\.local\b", re.I),
    "ptx_st_local": re.compile(r"\bst(?:\.volatile)?\.local\b", re.I),
    "ptx_barrier": re.compile(r"\b(?:bar\.sync|barrier)\b", re.I),
    "ptx_mma": re.compile(r"\b(?:mma|wmma)\.", re.I),
}
PTX_REG_RE = re.compile(r"\.reg\s+\.[a-zA-Z0-9]+\s+%[A-Za-z_][\w$]*(?:<(?P<n>\d+)>)?", re.M)
PTX_ENTRY_RE = re.compile(r"\.visible\s+\.entry\s+([A-Za-z_][\w$]*)")
PTX_INST_RE = re.compile(r"^\s*(?!//|/\*|\.|$)(?:@!?%?p?\d+\s+)?[a-zA-Z_][\w.]*\b", re.M)

SASS_OPCODE_RE = re.compile(r"^\s*(?:/\*[^*]*\*/\s*)?(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]*)(?:\s|;|$)")
SASS_FAMILIES = (
    "FFMA", "FADD", "FMUL", "DFMA", "DADD", "DMUL", "HMMA", "MUFU",
    "LDG", "STG", "LDS", "STS", "LDL", "STL", "LDC", "LDSM",
    "BAR", "MEMBAR", "SHF", "SHFL", "IMAD", "IADD", "ISETP", "LEA",
    "BRA", "EXIT", "NOP", "MOV", "S2R", "CS2R", "R2UR", "UR2R",
)

PTXAS_USED_RE = re.compile(r"Used\s+(?P<regs>\d+)\s+registers(?:,\s*(?P<rest>.*))?", re.I)
PTXAS_SPILL_RE = re.compile(r"(?P<n>\d+)\s+bytes\s+spill\s+(?P<kind>stores|loads)", re.I)
PTXAS_STACK_RE = re.compile(r"(?P<n>\d+)\s+bytes\s+stack\s+frame", re.I)
PTXAS_CMEM_RE = re.compile(r"(?P<n>\d+)\s+bytes\s+cmem\[(?P<idx>\d+)\]", re.I)
PTXAS_SMEM_RE = re.compile(r"(?P<n>\d+)\s+bytes\s+smem", re.I)
PTXAS_LMEM_RE = re.compile(r"(?P<n>\d+)\s+bytes\s+lmem", re.I)
RESOURCE_RE = re.compile(r"\b(?P<key>REG|STACK|SHARED|LOCAL|CMEM|PARAM|MAX_THREADS_PER_BLOCK)\b\s*[:=]\s*(?P<val>\d+)", re.I)

METRICS: list[tuple[str, str, str, float]] = [
    ("gflops", "measured GFLOP/s", "higher", 1.15),
    ("s_per_iter", "seconds / iteration", "lower", 1.15),
    ("ptxas_registers_max", "ptxas registers/thread", "lower", 1.10),
    ("ptxas_spill_stores_bytes_max", "spill stores, bytes", "lower", 1.01),
    ("ptxas_spill_loads_bytes_max", "spill loads, bytes", "lower", 1.01),
    ("ptxas_lmem_bytes_max", "ptxas local memory, bytes", "lower", 1.01),
    ("ptxas_stack_frame_bytes_max", "stack frame, bytes", "lower", 1.01),
    ("sass_instruction_lines", "SASS instructions", "lower", 1.10),
    ("sass_family_stg", "SASS STG output stores", "context", 1.20),
    ("sass_inst_per_stg", "SASS instructions / STG", "lower", 1.10),
    ("sass_int_addr_ops", "SASS int/address ops", "lower", 1.10),
    ("sass_int_addr_ops_per_stg", "int/address ops / STG", "lower", 1.10),
    ("sass_ffma_per_stg", "static FFMA / STG", "context", 1.20),
    ("sass_family_ffma", "SASS FFMA", "context", 1.20),
    ("sass_family_fmul", "SASS FMUL", "lower", 1.10),
    ("sass_family_fadd", "SASS FADD", "lower", 1.10),
    ("sass_ffma_fraction", "FFMA / SASS instruction", "higher", 1.15),
    ("sass_inst_per_ffma", "SASS instructions / FFMA", "lower", 1.10),
    ("sass_global_mem_per_ffma", "global mem ops / FFMA", "lower", 1.10),
    ("sass_shared_mem_per_ffma", "shared mem ops / FFMA", "lower", 1.15),
    ("sass_local_mem_per_ffma", "local mem ops / FFMA", "lower", 1.01),
    ("sass_global_mem_per_stg", "global mem ops / STG", "lower", 1.10),
    ("sass_shared_mem_per_stg", "shared mem ops / STG", "lower", 1.15),
    ("sass_family_bar", "barriers", "lower", 1.01),
    ("ptx_virtual_reg_slots", "PTX virtual reg slots", "lower", 1.10),
    ("ptx_instruction_lines", "PTX instruction lines", "lower", 1.10),
]


class Palette:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
    def c(self, text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text
    def bold(self, text: str) -> str: return self.c(text, "1")
    def dim(self, text: str) -> str: return self.c(text, "2")
    def red(self, text: str) -> str: return self.c(text, "31")
    def green(self, text: str) -> str: return self.c(text, "32")
    def yellow(self, text: str) -> str: return self.c(text, "33")
    def cyan(self, text: str) -> str: return self.c(text, "36")


def read_text(path: Path) -> str:
    return path.read_bytes().decode("utf-8", errors="replace")


def classify_file(path: Path) -> str:
    name = path.name.lower()
    suffix = path.suffix.lower()
    if name.endswith(".cmd.json"):
        return "command_json"
    if suffix in {".cubin", ".bin"}:
        return "binary"
    if suffix == ".ptx":
        return "ptx"
    if suffix in {".sass"} or name.endswith(".nvdisasm.sass"):
        return "sass"
    if "resource_usage" in name:
        return "resource_usage"
    if name.endswith(".stderr.txt") or name.endswith(".stdout.txt") or "build_log" in name:
        return "compiler_log"
    if suffix in {".cl", ".cu"} or "translation_unit" in name:
        return "source"
    if suffix == ".json":
        return "json"
    return "other"


def backend_from_path(path: Path) -> str:
    parts = set(path.parts)
    if "opencl" in parts:
        return "opencl"
    if "cuda" in parts:
        return "cuda"
    return "unknown"


def merge_numeric(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for key, val in src.items():
        if isinstance(val, bool):
            dst[key] = val
        elif isinstance(val, int | float):
            if key.endswith("_max"):
                dst[key] = max(dst.get(key, val), val)
            else:
                dst[key] = dst.get(key, 0) + val
        elif isinstance(val, Counter):
            old = dst.get(key)
            if not isinstance(old, Counter):
                old = Counter(old or {})
            old.update(val)
            dst[key] = old
        elif isinstance(val, list):
            dst.setdefault(key, [])
            dst[key].extend(val)
        else:
            dst.setdefault(key, val)


def normalize_sass_opcode(op: str) -> str:
    return op.split(".")[0].upper()


def parse_ptx(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    entries = sorted(set(PTX_ENTRY_RE.findall(text)))
    out["ptx_entries"] = entries
    out["ptx_virtual_reg_decls"] = len(PTX_REG_RE.findall(text))
    out["ptx_virtual_reg_slots"] = sum(int(m.group("n") or "1") for m in PTX_REG_RE.finditer(text))
    out["ptx_instruction_lines"] = len(PTX_INST_RE.findall(text))
    for key, pat in PTX_PATTERNS.items():
        out[key] = len(pat.findall(text))
    out["ptx_global_load_store"] = out.get("ptx_ld_global", 0) + out.get("ptx_st_global", 0)
    out["ptx_shared_load_store"] = out.get("ptx_ld_shared", 0) + out.get("ptx_st_shared", 0)
    out["ptx_local_load_store"] = out.get("ptx_ld_local", 0) + out.get("ptx_st_local", 0)
    return out


def parse_sass(text: str) -> dict[str, Any]:
    counter: Counter[str] = Counter()
    for line in text.splitlines():
        m = SASS_OPCODE_RE.match(line)
        if not m:
            continue
        op = normalize_sass_opcode(m.group(1))
        counter[op] += 1
    out: dict[str, Any] = {"sass_opcode_counts": counter}
    out.update({f"sass_{k.lower()}": v for k, v in counter.items()})
    out["sass_instruction_lines"] = sum(counter.values())
    for fam in SASS_FAMILIES:
        out[f"sass_family_{fam.lower()}"] = sum(v for k, v in counter.items() if k.startswith(fam))
    out["sass_local_load_store"] = out.get("sass_family_ldl", 0) + out.get("sass_family_stl", 0)
    out["sass_global_load_store"] = out.get("sass_family_ldg", 0) + out.get("sass_family_stg", 0)
    out["sass_shared_load_store"] = out.get("sass_family_lds", 0) + out.get("sass_family_sts", 0) + out.get("sass_family_ldsm", 0)
    return out


def parse_compiler_log(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    regs = [int(m.group("regs")) for m in PTXAS_USED_RE.finditer(text)]
    if regs:
        out["ptxas_registers_max"] = max(regs)
        out["ptxas_registers_sum"] = sum(regs)
    stores = [int(m.group("n")) for m in PTXAS_SPILL_RE.finditer(text) if m.group("kind").lower() == "stores"]
    loads = [int(m.group("n")) for m in PTXAS_SPILL_RE.finditer(text) if m.group("kind").lower() == "loads"]
    if stores: out["ptxas_spill_stores_bytes_max"] = max(stores)
    if loads: out["ptxas_spill_loads_bytes_max"] = max(loads)
    stacks = [int(m.group("n")) for m in PTXAS_STACK_RE.finditer(text)]
    smem = [int(m.group("n")) for m in PTXAS_SMEM_RE.finditer(text)]
    lmem = [int(m.group("n")) for m in PTXAS_LMEM_RE.finditer(text)]
    if stacks: out["ptxas_stack_frame_bytes_max"] = max(stacks)
    if smem: out["ptxas_smem_bytes_max"] = max(smem)
    if lmem: out["ptxas_lmem_bytes_max"] = max(lmem)
    for m in PTXAS_CMEM_RE.finditer(text):
        out[f"ptxas_cmem{m.group('idx')}_bytes_max"] = max(out.get(f"ptxas_cmem{m.group('idx')}_bytes_max", 0), int(m.group("n")))
    return out


def parse_resource_usage(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for m in RESOURCE_RE.finditer(text):
        key = "resource_" + m.group("key").lower() + "_max"
        out[key] = max(out.get(key, 0), int(m.group("val")))
    return out


def summarize_backend(files: Iterable[Path], root: Path, backend: str) -> dict[str, Any]:
    summary: dict[str, Any] = {"backend": backend, "file_count": 0, "files_by_kind": Counter(), "files": []}
    for path in files:
        kind = classify_file(path)
        if kind in {"other", "binary"}:
            continue
        summary["file_count"] += 1
        summary["files_by_kind"][kind] += 1
        summary["files"].append(str(path.relative_to(root)) if path.is_relative_to(root) else str(path))
        if kind in {"json", "command_json"}:
            continue
        try:
            text = read_text(path)
        except Exception:
            continue
        parsed: dict[str, Any] = {}
        if kind == "ptx": parsed = parse_ptx(text)
        elif kind == "sass": parsed = parse_sass(text)
        elif kind == "compiler_log": parsed = parse_compiler_log(text)
        elif kind == "resource_usage": parsed = parse_resource_usage(text)
        elif kind == "source":
            parsed = {
                "source_lines": len(text.splitlines()),
                "source_barriers": len(re.findall(r"\b(?:barrier|__syncthreads)\b", text)),
                "source_local_or_shared_mentions": len(re.findall(r"\b(?:__local|__shared__)\b", text)),
            }
        merge_numeric(summary, parsed)
    summary["files_by_kind"] = dict(summary["files_by_kind"])
    if isinstance(summary.get("sass_opcode_counts"), Counter):
        summary["sass_opcode_counts"] = dict(summary["sass_opcode_counts"])
    return summary


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def load_manifest(root: Path) -> dict[str, Any] | None:
    for p in [root / "run_manifest.json", root / "run_manifest.pre.json"]:
        if p.exists():
            return load_json(p)
    return None


def load_artifact_manifest(root: Path, backend: str) -> dict[str, Any] | None:
    return load_json(root / backend / "artifact_manifest.json")


def collect_files(root: Path) -> dict[str, list[Path]]:
    grouped = {"opencl": [], "cuda": [], "unknown": []}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        kind = classify_file(path)
        if kind == "other":
            continue
        grouped[backend_from_path(path)].append(path)
    return grouped


def safe_stem(path: Path | str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(path))[:180]


def resolve_path(root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    p = Path(value)
    if p.is_absolute():
        return p.resolve()
    return (root / p).resolve()


def select_opencl_primary_ptx(root: Path, manifest: dict[str, Any] | None) -> tuple[Path | None, str]:
    if (root / "opencl" / "pocl_primary.ptx").exists():
        return (root / "opencl" / "pocl_primary.ptx").resolve(), "opencl/pocl_primary.ptx"
    snap = (manifest or {}).get("pocl_cache_snapshot") or {}
    ptx_items = snap.get("ptx_files") or []
    if not ptx_items:
        return None, "no POCL PTX entries in manifest"
    copied = snap.get("copied_files") or []
    source_by_path = {}
    for rec in copied:
        dst = resolve_path(root, rec.get("path"))
        if dst is not None:
            source_by_path[str(dst)] = str(rec.get("source", ""))
    runtime_cache = str((snap.get("env") or {}).get("POCL_CACHE_DIR") or "")
    candidates: list[tuple[float, Path, str]] = []
    for item in ptx_items:
        p = resolve_path(root, item.get("path"))
        if p is None or not p.exists():
            continue
        source = source_by_path.get(str(p), str(item.get("source", "")))
        score = 0.0
        if runtime_cache and source.startswith(runtime_cache): score += 10000
        if "pocl-cache" in source.lower() or "pocl-cache" in str(p).lower() or "pocl_cache_runtime" in source.lower() or "pocl_cache_runtime" in str(p).lower(): score += 8000
        if "program.bc" in p.name.lower() and p.name.lower().endswith(".ptx"): score += 1000
        try:
            txt = p.read_text(encoding="utf-8", errors="replace")
            if any(("loopy_kernel" in e or "gemm" in e) for e in PTX_ENTRY_RE.findall(txt)): score += 2000
            score += min(len(txt), 1_000_000) / 1_000_000
        except Exception:
            score += float(item.get("bytes") or 0) / 1_000_000
        candidates.append((score, p, source))
    if not candidates:
        return None, "PTX entries existed but files were missing"
    candidates.sort(key=lambda x: x[0], reverse=True)
    score, p, source = candidates[0]
    rel = str(p.relative_to(root)) if p.is_relative_to(root) else str(p)
    msg = f"{rel} score={score:.1f}"
    if source: msg += f" source={source}"
    if len(candidates) > 1: msg += f"; ignored {len(candidates)-1} other PTX candidates"
    return p.resolve(), msg


def select_files(root: Path, grouped_all: dict[str, list[Path]], all_opencl_artifacts: bool, explicit_opencl_ptx: Path | None) -> tuple[dict[str, list[Path]], list[str]]:
    selected: dict[str, list[Path]] = {}
    notes: list[str] = []
    for backend, files in grouped_all.items():
        if backend == "opencl" and not all_opencl_artifacts:
            manifest = load_artifact_manifest(root, "opencl")
            primary = explicit_opencl_ptx.resolve() if explicit_opencl_ptx else None
            why = "specified by --opencl-primary-ptx" if primary else ""
            if primary is None:
                primary, why = select_opencl_primary_ptx(root, manifest)
            if primary and primary.exists():
                keep: set[Path] = set()
                stem = "pocl_primary" if primary.name == "pocl_primary.ptx" else "pocl_cache_" + safe_stem(primary.name)
                for f in files:
                    kind, name = classify_file(f), f.name
                    if kind in {"json", "source"}:
                        keep.add(f)
                    elif f.resolve() == primary:
                        keep.add(f)
                    elif f.parent == root / "opencl" and name.startswith(stem):
                        if name.endswith(".nvdisasm.sass"):
                            continue
                        keep.add(f)
                selected[backend] = sorted(keep)
                notes.append(f"OpenCL summary uses one primary PTX: {why}")
                continue
            notes.append(f"OpenCL primary PTX not selected ({why}); parsing all OpenCL artifacts.")
        elif backend == "cuda":
            keep: set[Path] = set()
            for f in files:
                kind, name = classify_file(f), f.name
                if kind in {"json", "source"}:
                    keep.add(f)
                elif name in {
                    "loopy_kernel.ptx", "loopy_kernel.sass", "loopy_kernel.resource_usage.txt",
                    "ptxas_from_ptx.stderr.txt", "ptxas_from_ptx.stdout.txt", "nvcc_cubin.stderr.txt",
                    "ptxas_from_ptx.cmd.json",
                }:
                    keep.add(f)
            selected[backend] = sorted(keep)
            notes.append("CUDA summary uses canonical loopy_kernel PTX/SASS/resource files only.")
            continue
        selected[backend] = files
    return selected, notes


def attach_benchmarks(summaries: dict[str, dict[str, Any]], manifest: dict[str, Any] | None) -> None:
    if not manifest:
        return
    for bench in manifest.get("benchmarks", []):
        if not isinstance(bench, dict):
            continue
        backend = bench.get("backend")
        if not backend:
            continue
        summaries.setdefault(backend, {"backend": backend, "file_count": 0, "files_by_kind": {}, "files": []})
        for k in ("gflops", "s_per_iter", "total_elapsed_s", "error"):
            if k in bench:
                summaries[backend][k] = bench[k]


def add_derived(summary: dict[str, Any]) -> None:
    ffma = summary.get("sass_family_ffma", 0)
    stg = summary.get("sass_family_stg", 0)
    instr = summary.get("sass_instruction_lines", 0)

    # Integer/address overhead is often the useful signal for these Loopy GEMM
    # kernels. IADD catches IADD3 via the family-prefix logic in parse_sass().
    int_addr = (
        summary.get("sass_family_iadd", 0)
        + summary.get("sass_family_imad", 0)
        + summary.get("sass_family_lea", 0)
        + summary.get("sass_family_shf", 0)
        + summary.get("sass_family_isetp", 0)
    )
    summary["sass_int_addr_ops"] = int_addr

    if instr:
        summary["sass_ffma_fraction"] = ffma / instr
    if ffma:
        summary["sass_inst_per_ffma"] = instr / ffma
        summary["sass_global_mem_per_ffma"] = summary.get("sass_global_load_store", 0) / ffma
        summary["sass_shared_mem_per_ffma"] = summary.get("sass_shared_load_store", 0) / ffma
        summary["sass_local_mem_per_ffma"] = summary.get("sass_local_load_store", 0) / ffma
        summary["sass_barriers_per_ffma"] = summary.get("sass_family_bar", 0) / ffma
    if stg:
        summary["sass_inst_per_stg"] = instr / stg
        summary["sass_ffma_per_stg"] = ffma / stg
        summary["sass_int_addr_ops_per_stg"] = int_addr / stg
        summary["sass_global_mem_per_stg"] = summary.get("sass_global_load_store", 0) / stg
        summary["sass_shared_mem_per_stg"] = summary.get("sass_shared_load_store", 0) / stg
        summary["sass_local_mem_per_stg"] = summary.get("sass_local_load_store", 0) / stg
        summary["sass_barriers_per_stg"] = summary.get("sass_family_bar", 0) / stg


def build_payload(root: Path, all_opencl_artifacts: bool = False, opencl_primary_ptx: Path | None = None) -> dict[str, Any]:
    grouped_all = collect_files(root)
    grouped, selection_notes = select_files(root, grouped_all, all_opencl_artifacts, opencl_primary_ptx)
    summaries: dict[str, dict[str, Any]] = {}
    for backend, files in grouped.items():
        if not files:
            continue
        s = summarize_backend(files, root, backend)
        s["available_file_count"] = len(grouped_all.get(backend, []))
        s["available_files_by_kind"] = dict(Counter(classify_file(f) for f in grouped_all.get(backend, [])))
        summaries[backend] = s
    manifest = load_manifest(root)
    attach_benchmarks(summaries, manifest)
    for s in summaries.values():
        add_derived(s)
    notes = selection_notes + comparison_notes(summaries)
    return {"artifact_root": str(root), "manifest": manifest, "summaries": summaries, "comparison_notes": notes, "selection_notes": selection_notes}


def val_to_float(v: Any) -> float | None:
    if isinstance(v, int | float) and not isinstance(v, bool):
        f = float(v)
        return None if math.isnan(f) or math.isinf(f) else f
    return None


def fmt(v: Any) -> str:
    if v is None or v == "":
        return "-"
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        av = abs(v)
        if av == 0: return "0"
        if av >= 1000: return f"{v:,.1f}"
        if av >= 100: return f"{v:.2f}"
        if av >= 10: return f"{v:.3f}"
        if av >= 1: return f"{v:.4f}"
        return f"{v:.3g}"
    if isinstance(v, (dict, list)):
        return json.dumps(v, sort_keys=True)
    return str(v)


def ratio_raw(ov: Any, cv: Any) -> float | None:
    o, c = val_to_float(ov), val_to_float(cv)
    if o is None or c is None or c == 0:
        return None
    return o / c


def ansi_strip(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def vlen(s: str) -> int:
    return len(ansi_strip(s))


def trunc(s: str, width: int) -> str:
    if vlen(s) <= width:
        return s
    return ansi_strip(s)[:max(0, width-1)] + "…"


def table(headers: list[str], rows: list[list[str]], width: int | None = None) -> str:
    colw = [vlen(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            colw[i] = max(colw[i], vlen(cell))
    if width:
        fixed = sum(colw[1:]) + 3 * (len(colw) - 1)
        if fixed + colw[0] > width:
            colw[0] = max(18, width - fixed)
    def one(row: list[str]) -> str:
        out = []
        for i, cell in enumerate(row):
            cell = trunc(cell, colw[i])
            pad = colw[i] - vlen(cell)
            out.append(cell + " " * pad if i == 0 else " " * pad + cell)
        return " | ".join(out)
    return "\n".join([one(headers), "-+-".join("-"*w for w in colw), *[one(r) for r in rows]])


def ratio_cell(ov: Any, cv: Any, direction: str, threshold: float, pal: Palette) -> str:
    r = ratio_raw(ov, cv)
    if r is None: return "-"
    text = f"{r:.2f}x"
    if direction == "lower" and r >= threshold: return pal.red(text)
    if direction == "higher" and r <= 1.0 / threshold: return pal.red(text)
    if direction == "higher" and r >= threshold: return pal.green(text)
    return pal.dim(text)


def flag_cell(ov: Any, cv: Any, direction: str, threshold: float, pal: Palette) -> str:
    r = ratio_raw(ov, cv)
    if r is None: return ""
    if direction == "lower" and r >= threshold: return pal.red("worse")
    if direction == "higher" and r <= 1.0 / threshold: return pal.red("worse")
    if direction == "higher" and r >= threshold: return pal.green("better")
    return ""


def top_opcodes(summary: dict[str, Any], n: int) -> list[tuple[str, int]]:
    counts = summary.get("sass_opcode_counts", {})
    if not isinstance(counts, dict):
        return []
    return Counter({str(k): int(v) for k, v in counts.items()}).most_common(n)


def has_compiler_stats(s: dict[str, Any]) -> bool:
    return any(k in s for k in ("sass_instruction_lines", "ptx_instruction_lines", "ptxas_registers_max", "resource_reg_max"))


def comparison_notes(summaries: dict[str, dict[str, Any]]) -> list[str]:
    opencl, cuda = summaries.get("opencl", {}), summaries.get("cuda", {})
    if not opencl or not cuda:
        return ["Need both opencl/ and cuda/ artifacts for direct comparison."]
    notes: list[str] = []
    if not has_compiler_stats(opencl):
        notes.append("OpenCL compiler stats are missing.")
    for key, _, _, _ in METRICS:
        if key in opencl or key in cuda:
            r = ratio_raw(opencl.get(key), cuda.get(key))
            notes.append(f"{key}: OpenCL={fmt(opencl.get(key))}, CUDA={fmt(cuda.get(key))}" + (f", ratio={r:.2f}x." if r is not None else "."))
    if opencl.get("sass_local_load_store", 0) > cuda.get("sass_local_load_store", 0):
        notes.append("OpenCL has more local-memory traffic (LDL/STL), consistent with spills/stack traffic.")
    if opencl.get("ptxas_spill_stores_bytes_max", 0) or opencl.get("ptxas_spill_loads_bytes_max", 0):
        notes.append("OpenCL ptxas output reports explicit spill bytes.")
    if opencl.get("sass_inst_per_ffma", 0) > max(cuda.get("sass_inst_per_ffma", 0), 1) * 1.15:
        notes.append("OpenCL has materially more SASS instructions per FFMA.")
    if opencl.get("sass_inst_per_stg", 0) > max(cuda.get("sass_inst_per_stg", 0), 1) * 1.10:
        notes.append("OpenCL has more static SASS instructions per output store (STG).")
    ofps = opencl.get("sass_ffma_per_stg")
    cfps = cuda.get("sass_ffma_per_stg")
    if ofps and cfps and ofps < cfps / 1.2:
        notes.append("OpenCL has fewer static FFMA per output store than CUDA; this usually means different unrolling, so per-FFMA static ratios are less reliable than per-STG ratios.")
    if opencl.get("ptxas_registers_max") and cuda.get("ptxas_registers_max"):
        if abs(opencl["ptxas_registers_max"] - cuda["ptxas_registers_max"]) <= 8 and not opencl.get("ptxas_spill_stores_bytes_max", 0) and not opencl.get("ptxas_spill_loads_bytes_max", 0):
            notes.append("Register count and spills are comparable; the main gap is unlikely to be register spilling for this artifact.")
    return notes


def suspects(summaries: dict[str, dict[str, Any]], pal: Palette) -> list[str]:
    o, c = summaries.get("opencl", {}), summaries.get("cuda", {})
    if not o or not c:
        return [pal.yellow("Need both backends.")]
    hits: list[tuple[float, str]] = []
    perf = ratio_raw(o.get("gflops"), c.get("gflops"))
    if perf is not None and perf < 0.85:
        hits.append((1/max(perf,1e-9), f"OpenCL throughput is {perf:.2f}x CUDA throughput."))
    if not has_compiler_stats(o):
        hits.append((999, "OpenCL compiler stats are missing."))
    for key, label, threshold in [
        ("ptxas_registers_max", "registers/thread", 1.10),
        ("ptxas_spill_stores_bytes_max", "spill-store bytes", 1.01),
        ("ptxas_spill_loads_bytes_max", "spill-load bytes", 1.01),
        ("ptxas_stack_frame_bytes_max", "stack-frame bytes", 1.01),
        ("sass_local_mem_per_ffma", "local-memory ops per FFMA", 1.01),
        ("sass_inst_per_stg", "SASS instructions per output store", 1.10),
        ("sass_int_addr_ops_per_stg", "integer/address ops per output store", 1.10),
        ("sass_inst_per_ffma", "SASS instructions per FFMA", 1.10),
        ("sass_global_mem_per_ffma", "global-memory ops per FFMA", 1.10),
        ("sass_global_mem_per_stg", "global-memory ops per output store", 1.10),
        ("sass_family_bar", "barriers", 1.01),
    ]:
        r = ratio_raw(o.get(key), c.get(key))
        if r is not None and r >= threshold:
            hits.append((r, f"OpenCL has {r:.2f}x CUDA {label}."))
        elif c.get(key) in (0, None) and o.get(key):
            hits.append((5, f"OpenCL has {fmt(o.get(key))} {label}; CUDA has {fmt(c.get(key))}."))
    ofps, cfps = o.get("sass_ffma_per_stg"), c.get("sass_ffma_per_stg")
    if ofps and cfps and ofps < cfps / 1.2:
        hits.append((cfps / max(ofps, 1e-9), f"OpenCL has fewer static FFMA per output store ({ofps:.2f} vs CUDA {cfps:.2f}); likely a different unroll/loop structure."))
    if not hits:
        return [pal.green("No obvious smoking gun from parsed metrics.")]
    hits.sort(reverse=True, key=lambda x: x[0])
    return [pal.red(msg) if score >= 1.5 else pal.yellow(msg) for score, msg in hits[:8]]



def _flatten_mapping(prefix: str, value: Any, out: dict[str, Any]) -> None:
    """Flatten JSON-like mappings using dotted keys."""
    if isinstance(value, dict):
        for k, v in value.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            _flatten_mapping(key, v, out)
    else:
        out[prefix] = value


def _short_manifest_label(manifest: dict[str, Any] | None) -> str:
    if not manifest:
        return ""
    case = manifest.get("case") or manifest.get("kernel_version") or "-"
    md = manifest.get("metadata") or {}
    p = manifest.get("parameters") or {}
    # Prefer generic loopy_perf_artifacts metadata, but retain the old matmul fields.
    fields: list[str] = []
    for container in (md, p):
        if isinstance(container, dict):
            for k in ("m", "n", "k", "bm", "bn", "bk", "tm", "tn", "dtype", "variant", "kernel_version"):
                if k in container and f"{k}=" not in " ".join(fields):
                    fields.append(f"{k}={container[k]}")
    for k in ("niterations", "nwarmup"):
        if k in p:
            fields.append(f"{k}={p[k]}")
    return f"case={case}" + ("; " + ", ".join(fields) if fields else "")


def params_line(manifest: dict[str, Any] | None) -> str:
    return _short_manifest_label(manifest)


def diagnostic_lines(root: Path, summaries: dict[str, dict[str, Any]], pal: Palette) -> list[str]:
    lines: list[str] = []
    ocl = load_artifact_manifest(root, "opencl")
    if ocl:
        snap = ocl.get("pocl_cache_snapshot") or {}
        lines.append(f"opencl: is_pocl_context={ocl.get('is_pocl_context')}, binaries={len(ocl.get('binaries', []))}, run_local_pocl_ptx={len(snap.get('ptx_files', []))}, primary_ptx={snap.get('primary_ptx') or ('opencl/pocl_primary.ptx' if (root/'opencl'/'pocl_primary.ptx').exists() else '-')}")
        if snap.get("cache_dir"):
            lines.append("run-local POCL cache: " + str(snap.get("cache_dir")))
        if snap.get("searched_dirs"):
            lines.append("POCL cache dirs used: " + ", ".join(str(x) for x in snap.get("searched_dirs", [])[:4]))
        for w in (ocl.get("warnings", []) + snap.get("warnings", []))[:4]:
            lines.append(pal.yellow(f"warning: {w}"))
    cuda = load_artifact_manifest(root, "cuda")
    if cuda:
        lines.append(f"cuda: outputs={len(cuda.get('outputs', []))}, errors={len(cuda.get('errors', []))}")
    for b in ("opencl","cuda"):
        s = summaries.get(b, {})
        if s.get("available_file_count") and s.get("available_file_count") != s.get("file_count"):
            lines.append(pal.dim(f"{b}: parsed {s.get('file_count')} canonical files out of {s.get('available_file_count')} available files."))
    return lines


def render_terminal(payload: dict[str, Any], pal: Palette, top_n: int, print_files: bool) -> str:
    root = Path(payload["artifact_root"])
    summaries = payload["summaries"]
    width = shutil.get_terminal_size((120, 24)).columns
    lines: list[str] = []
    lines.append(pal.bold("Compiler artifact comparison"))
    lines.append(pal.dim(str(root)))
    pl = params_line(payload.get("manifest"))
    if pl: lines.append(pl)
    lines.append("")

    diag = diagnostic_lines(root, summaries, pal)
    if diag:
        lines.append(pal.bold("Collection diagnostics"))
        for d in diag: lines.append(f"  - {d}")
        lines.append("")

    rows: list[list[str]] = []
    for key,label,direction,threshold in METRICS:
        ov, cv = summaries.get("opencl",{}).get(key, ""), summaries.get("cuda",{}).get(key, "")
        if ov == "" and cv == "": continue
        rows.append([label, fmt(ov), fmt(cv), ratio_cell(ov, cv, direction, threshold, pal), flag_cell(ov, cv, direction, threshold, pal)])
    if rows:
        lines.append(pal.bold("Main comparison"))
        lines.append(table(["metric","OpenCL","CUDA","OCL/CUDA","flag"], rows, width))
        lines.append("")

    lines.append(pal.bold("Top suspects"))
    for i,msg in enumerate(suspects(summaries, pal),1): lines.append(f"  {i}. {msg}")
    lines.append("")

    oops, cops = top_opcodes(summaries.get("opencl",{}), top_n), top_opcodes(summaries.get("cuda",{}), top_n)
    if oops or cops:
        rows = []
        for i in range(max(len(oops), len(cops))):
            oo = oops[i] if i < len(oops) else ("",0)
            co = cops[i] if i < len(cops) else ("",0)
            rows.append([f"{oo[0]} {fmt(oo[1])}" if oo[0] else "-", f"{co[0]} {fmt(co[1])}" if co[0] else "-"])
        lines.append(pal.bold(f"Top SASS opcodes, per backend (top {top_n})"))
        lines.append(table(["OpenCL","CUDA"], rows, width))
        lines.append("")

    frows = []
    for b in ("opencl","cuda","unknown"):
        s = summaries.get(b)
        if not s: continue
        kinds = s.get("files_by_kind", {})
        avail = s.get("available_file_count")
        files = fmt(s.get("file_count", 0)) + (f" / {fmt(avail)} avail" if avail and avail != s.get("file_count") else "")
        frows.append([b, files, ", ".join(f"{k}:{v}" for k,v in sorted(kinds.items()))])
    if frows:
        lines.append(pal.bold("Parsed artifact files"))
        lines.append(table(["backend","files","kinds"], frows, width))
        lines.append("")

    if payload.get("comparison_notes"):
        lines.append(pal.bold("Raw comparison notes"))
        for note in payload["comparison_notes"][:20]:
            lines.append(f"  - {note}")
        lines.append("")

    if print_files:
        lines.append(pal.bold("Files used for metrics"))
        for b in ("opencl","cuda","unknown"):
            s = summaries.get(b)
            if not s: continue
            lines.append(pal.cyan(f"[{b}]"))
            for f in s.get("files", []): lines.append(f"  {f}")
        lines.append("")

    lines.append(pal.dim("Tip: use --color always with less -R. Use --suite-root plus --plot-metric for multi-run plots."))
    return "\n".join(lines)


def flatten_for_csv(summaries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    keys = sorted(k for s in summaries.values() for k in s if k not in {"files", "files_by_kind", "sass_opcode_counts"})
    rows = []
    for backend, s in summaries.items():
        row = {"backend": backend}
        for k in keys:
            v = s.get(k, "")
            if isinstance(v, (list, dict)):
                v = json.dumps(v, sort_keys=True)
            row[k] = v
        rows.append(row)
    return rows


def write_csv(path: Path, summaries: dict[str, dict[str, Any]]) -> None:
    rows = flatten_for_csv(summaries)
    if not rows: return
    fields = sorted({k for r in rows for k in r})
    with path.open("w", newline="", encoding="utf-8") as outf:
        w = csv.DictWriter(outf, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    s = payload["summaries"]
    lines = ["# Compiler artifact summary", "", _short_manifest_label(payload.get("manifest")), "", "| metric | OpenCL | CUDA |", "|---|---:|---:|"]
    for key,label,_,_ in METRICS:
        lines.append(f"| {label} | {fmt(s.get('opencl',{}).get(key,''))} | {fmt(s.get('cuda',{}).get(key,''))} |")
    lines.append("")
    lines.append("## Notes")
    for note in payload.get("comparison_notes", []):
        lines.append(f"- {note}")
    path.write_text("\n".join(lines)+"\n", encoding="utf-8")


def resolve_color(mode: str) -> Palette:
    if mode == "always": return Palette(True)
    if mode == "never": return Palette(False)
    return Palette(sys.stdout.isatty())


# {{{ suite/multi-run support


def _has_run_manifest(path: Path) -> bool:
    return (path / "run_manifest.json").exists() or (path / "run_manifest.pre.json").exists()


def discover_run_roots(paths: Iterable[Path], recursive: bool = True) -> list[Path]:
    """Return artifact run directories, accepting either run dirs or parent dirs."""
    roots: list[Path] = []
    seen: set[Path] = set()
    for raw in paths:
        p = raw.expanduser().resolve()
        candidates: list[Path] = []
        if _has_run_manifest(p):
            candidates.append(p)
        elif p.is_dir() and recursive:
            candidates.extend(sorted({q.parent for q in p.rglob("run_manifest.json")}))
            candidates.extend(sorted({q.parent for q in p.rglob("run_manifest.pre.json")}))
        elif p.is_dir():
            candidates.extend(sorted(child for child in p.iterdir() if child.is_dir() and _has_run_manifest(child)))
        for c in candidates:
            c = c.resolve()
            if c not in seen:
                seen.add(c)
                roots.append(c)
    return sorted(roots, key=lambda x: str(x))


def _stringify_cell(value: Any) -> Any:
    if isinstance(value, (str, int, float)) or value is None:
        return value
    if isinstance(value, bool):
        return value
    return json.dumps(value, sort_keys=True, default=str)


def suite_rows_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten one analyzed run into one row per backend."""
    root = Path(payload["artifact_root"])
    manifest = payload.get("manifest") or {}
    backend_errors = manifest.get("backend_errors") or {}
    artifact_errors = manifest.get("artifact_errors") or {}
    base: dict[str, Any] = {
        "artifact_root": str(root),
        "case": manifest.get("case") or manifest.get("kernel_version") or root.name,
        "created_at": manifest.get("created_at", ""),
        "run_dir": manifest.get("run_dir", str(root)),
        "run_status": manifest.get("status", ""),
        "backend_error_count": len(backend_errors) if isinstance(backend_errors, dict) else 0,
        "artifact_error_count": len(artifact_errors) if isinstance(artifact_errors, dict) else 0,
        "backend_errors": json.dumps(backend_errors, sort_keys=True, default=str) if backend_errors else "",
        "artifact_errors": json.dumps(artifact_errors, sort_keys=True, default=str) if artifact_errors else "",
    }
    _flatten_mapping("metadata", manifest.get("metadata") or {}, base)
    _flatten_mapping("parameters", manifest.get("parameters") or {}, base)
    # Older matmul_dump_artifacts.py stored shape/tile parameters directly under parameters.
    # Generic loopy_perf_artifacts.py expects these under metadata, but both are preserved.
    rows: list[dict[str, Any]] = []
    for backend, summary in sorted((payload.get("summaries") or {}).items()):
        if backend not in {"opencl", "cuda"}:
            continue
        row = dict(base)
        row["backend"] = backend
        if isinstance(backend_errors, dict) and backend in backend_errors:
            err = backend_errors[backend]
            if isinstance(err, dict):
                row["backend_error_type"] = err.get("type", "")
                row["backend_error_repr"] = err.get("repr", "")
        if isinstance(artifact_errors, dict) and backend in artifact_errors:
            err = artifact_errors[backend]
            if isinstance(err, dict):
                row["artifact_error_type"] = err.get("type", "")
                row["artifact_error_repr"] = err.get("repr", "")
        for key, value in summary.items():
            if key in {"files", "files_by_kind", "sass_opcode_counts"}:
                continue
            row[key] = _stringify_cell(value)
        rows.append(row)
    return rows


def build_suite_payload(
    roots: Iterable[Path],
    *,
    all_opencl_artifacts: bool = False,
    opencl_primary_ptx: Path | None = None,
) -> dict[str, Any]:
    analyzed: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for root in roots:
        try:
            payload = build_payload(root, all_opencl_artifacts, opencl_primary_ptx)
            analyzed.append(payload)
            rows.extend(suite_rows_from_payload(payload))
        except Exception as exc:
            errors.append({"artifact_root": str(root), "error": repr(exc)})
    return {"runs": analyzed, "rows": rows, "errors": errors}


def filter_suite_rows(rows: list[dict[str, Any]], filters: Sequence[str]) -> list[dict[str, Any]]:
    """Filter flattened rows with key=value or key!=value predicates."""
    out = rows
    for expr in filters:
        neg = "!=" in expr
        if neg:
            key, expected = expr.split("!=", 1)
        elif "=" in expr:
            key, expected = expr.split("=", 1)
        else:
            raise ValueError(f"invalid --where expression {expr!r}; use key=value or key!=value")
        key = key.strip()
        expected = expected.strip()
        if neg:
            out = [r for r in out if str(r.get(key, "")) != expected]
        else:
            out = [r for r in out if str(r.get(key, "")) == expected]
    return out


def write_suite_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as outf:
        writer = csv.DictWriter(outf, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _stringify_cell(row.get(k, "")) for k in fields})


def _maybe_float(value: Any) -> float | None:
    if isinstance(value, bool) or value in (None, ""):
        return None
    try:
        f = float(value)
    except Exception:
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _plot_safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "plot"


def plot_suite_metrics(
    rows: Sequence[dict[str, Any]],
    *,
    metrics: Sequence[str],
    x: str,
    series: str = "backend",
    out_dir: Path,
) -> list[Path]:
    """Plot queried suite metrics. Requires matplotlib only when called."""
    import matplotlib.pyplot as plt  # type: ignore

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for metric in metrics:
        points_by_series: dict[str, list[tuple[Any, float]]] = {}
        x_is_numeric = True
        for row in rows:
            y = _maybe_float(row.get(metric))
            if y is None:
                continue
            xv_raw = row.get(x, "")
            xv_num = _maybe_float(xv_raw)
            if xv_num is None:
                x_is_numeric = False
                xv: Any = str(xv_raw)
            else:
                xv = xv_num
            label = str(row.get(series, "")) or "series"
            points_by_series.setdefault(label, []).append((xv, y))
        if not points_by_series:
            continue

        # Stable categorical coordinate mapping if any x values are non-numeric.
        category_map: dict[str, int] = {}
        if not x_is_numeric:
            cats = sorted({str(xv) for pts in points_by_series.values() for xv, _ in pts})
            category_map = {c: i for i, c in enumerate(cats)}

        fig, ax = plt.subplots()
        for label, pts in sorted(points_by_series.items()):
            if not x_is_numeric:
                xs = [category_map[str(xv)] for xv, _ in pts]
            else:
                xs = [float(xv) for xv, _ in pts]
            ys = [yv for _, yv in pts]
            order = sorted(range(len(xs)), key=lambda i: xs[i])
            xs = [xs[i] for i in order]
            ys = [ys[i] for i in order]
            ax.plot(xs, ys, marker="o", label=label)
        ax.set_xlabel(x)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} vs {x}")
        if len(points_by_series) > 1:
            ax.legend()
        if not x_is_numeric:
            cats = [c for c, _ in sorted(category_map.items(), key=lambda kv: kv[1])]
            ax.set_xticks(list(range(len(cats))))
            ax.set_xticklabels(cats, rotation=30, ha="right")
        fig.tight_layout()
        path = out_dir / f"{_plot_safe_name(metric)}_vs_{_plot_safe_name(x)}_by_{_plot_safe_name(series)}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def render_suite_terminal(rows: Sequence[dict[str, Any]], *, metrics: Sequence[str]) -> str:
    width = shutil.get_terminal_size((120, 24)).columns
    shown_metrics = list(metrics) if metrics else ["run_status", "gflops", "s_per_iter", "ptxas_registers_max", "sass_instruction_lines", "backend_error_repr"]
    headers = ["case", "backend", *shown_metrics]
    table_rows: list[list[str]] = []
    for row in rows:
        table_rows.append([fmt(row.get(h, "")) for h in headers])
    return "\n".join([
        f"Suite summary: {len(rows)} backend rows",
        table(headers, table_rows, width) if table_rows else "No rows matched.",
    ])


# }}}


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Loopy OpenCL/CUDA compiler artifacts and benchmark manifests.")
    parser.add_argument("artifact_roots", nargs="*", type=Path, help="One run directory, or one/more parent directories in suite mode.")
    parser.add_argument("--format", choices=["terminal","json"], default="terminal")
    parser.add_argument("--json", dest="json_out", type=Path)
    parser.add_argument("--csv", dest="csv_out", type=Path)
    parser.add_argument("--markdown", dest="markdown_out", type=Path)
    parser.add_argument("--print-files", action="store_true")
    parser.add_argument("--top-opcodes", type=int, default=12)
    parser.add_argument("--color", choices=["auto","always","never"], default="auto")
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--all-opencl-artifacts", action="store_true", help="Sum every OpenCL artifact instead of selecting one primary POCL artifact.")
    parser.add_argument("--opencl-primary-ptx", type=Path, default=None, help="Explicit OpenCL/POCL PTX file to use for OpenCL metrics in single-run mode.")

    parser.add_argument("--suite-root", action="append", type=Path, default=[], help="Parent directory containing many run_manifest.json files. May be repeated.")
    parser.add_argument("--suite", action="store_true", help="Analyze multiple run directories and emit one row per backend.")
    parser.add_argument("--no-recursive", action="store_true", help="Do not recursively discover run_manifest.json under suite roots.")
    parser.add_argument("--where", action="append", default=[], help="Suite filter: key=value or key!=value, e.g. backend=cuda or metadata.variant=tiled.")
    parser.add_argument("--metric", action="append", default=[], help="Metric column to show/plot. May be repeated. Also accepts comma-separated names.")
    parser.add_argument("--plot-metric", action="append", default=[], help="Metric to plot. May be repeated. Also accepts comma-separated names.")
    parser.add_argument("--plot-x", default="metadata.m", help="X-axis column for suite plots, e.g. metadata.m, metadata.n, case.")
    parser.add_argument("--plot-series", default="backend", help="Series/grouping column for suite plots.")
    parser.add_argument("--plot-dir", type=Path, default=None, help="Directory for PNG plots. Enables plotting when --plot-metric is set.")
    args = parser.parse_args()

    requested_metrics = [m.strip() for item in args.metric for m in item.split(",") if m.strip()]
    requested_plot_metrics = [m.strip() for item in args.plot_metric for m in item.split(",") if m.strip()]

    suite_mode = args.suite or bool(args.suite_root) or len(args.artifact_roots) > 1
    if suite_mode:
        roots_input = [*args.suite_root, *args.artifact_roots]
        if not roots_input:
            parser.error("suite mode needs at least one --suite-root or artifact root")
        run_roots = discover_run_roots(roots_input, recursive=not args.no_recursive)
        suite_payload = build_suite_payload(
            run_roots,
            all_opencl_artifacts=args.all_opencl_artifacts,
            opencl_primary_ptx=None,
        )
        rows = filter_suite_rows(suite_payload["rows"], args.where)
        suite_payload["filtered_rows"] = rows

        if args.format == "json":
            print(json.dumps(suite_payload, indent=2, sort_keys=True, default=str))
        else:
            print(render_suite_terminal(rows, metrics=requested_metrics))
            if suite_payload.get("errors"):
                print("\nErrors:")
                for err in suite_payload["errors"]:
                    print(f"  - {err['artifact_root']}: {err['error']}")

        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(suite_payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
        if args.csv_out:
            write_suite_csv(args.csv_out, rows)
        if args.markdown_out:
            lines = ["# Loopy perf suite summary", "", f"Rows: {len(rows)}", ""]
            metric_cols = requested_metrics or ["gflops", "s_per_iter", "ptxas_registers_max"]
            lines.append("| case | backend | " + " | ".join(metric_cols) + " |")
            lines.append("|---|---" + "|---:" * len(metric_cols) + "|")
            for row in rows:
                lines.append("| " + " | ".join([fmt(row.get("case", "")), fmt(row.get("backend", "")), *[fmt(row.get(m, "")) for m in metric_cols]]) + " |")
            args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
            args.markdown_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

        if requested_plot_metrics:
            plot_dir = args.plot_dir or Path("suite_plots")
            paths = plot_suite_metrics(rows, metrics=requested_plot_metrics, x=args.plot_x, series=args.plot_series, out_dir=plot_dir)
            if args.format != "json":
                for path in paths:
                    print(f"wrote plot: {path}")
        return

    if not args.artifact_roots:
        parser.error("artifact_root is required unless --suite-root is used")
    root = args.artifact_roots[0].resolve()
    payload = build_payload(root, args.all_opencl_artifacts, args.opencl_primary_ptx)

    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        pal = resolve_color("never" if args.no_color else args.color)
        print(render_terminal(payload, pal, args.top_opcodes, args.print_files))

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        write_csv(args.csv_out, payload["summaries"])
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(args.markdown_out, payload)


if __name__ == "__main__":
    main()
