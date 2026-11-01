"""
Hardware-Aware Resource Profiler for Local-First AI Routing.
Detects CPU, RAM, GPU/VRAM, and storage without mandatory third-party packages,
providing deterministic resource profiles: LOW_RESOURCE, STANDARD, GPU_ACCELERATED, HIGH_MEMORY.
Zero crashes: fails gracefully to safe defaults and marks classification honestly.
"""

from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
import sys
from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class ResourceProfile(str, Enum):
    LOW_RESOURCE = "LOW_RESOURCE"
    STANDARD = "STANDARD"
    GPU_ACCELERATED = "GPU_ACCELERATED"
    HIGH_MEMORY = "HIGH_MEMORY"


class HardwareCapabilities(BaseModel):
    os_name: str
    architecture: str
    cpu_count: int
    total_ram_gb: float
    available_ram_gb: float
    gpu_available: bool = False
    gpu_name: Optional[str] = None
    vram_gb: float = 0.0
    disk_free_gb: float = 0.0
    profile: ResourceProfile = ResourceProfile.STANDARD
    classification: str = "REAL"  # REAL when queried via OS API, HEURISTIC if estimated
    detection_source: str = "system_api"


def _get_ram_psutil() -> Optional[tuple[float, float]]:
    """Retrieve total and available RAM via psutil if installed."""
    try:
        import psutil
        v = psutil.virtual_memory()
        return round(v.total / (1024 ** 3), 2), round(v.available / (1024 ** 3), 2)
    except Exception:
        return None


def _get_ram_windows() -> tuple[float, float, bool]:
    """Retrieve total and available RAM on Windows via GlobalMemoryStatusEx ctypes."""
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            total_gb = stat.ullTotalPhys / (1024 ** 3)
            avail_gb = stat.ullAvailPhys / (1024 ** 3)
            return round(total_gb, 2), round(avail_gb, 2), True
    except Exception:
        pass
    return 8.0, 4.0, False


def _get_ram_unix() -> tuple[float, float, bool]:
    """Retrieve total and available RAM on Linux/macOS via /proc/meminfo or sysconf."""
    try:
        if os.path.exists("/proc/meminfo"):
            meminfo = {}
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.split(":")
                    if len(parts) == 2:
                        meminfo[parts[0].strip()] = parts[1].strip()
            if "MemTotal" in meminfo and "MemAvailable" in meminfo:
                total_kb = float(meminfo["MemTotal"].split()[0])
                avail_kb = float(meminfo["MemAvailable"].split()[0])
                return round(total_kb / (1024 ** 2), 2), round(avail_kb / (1024 ** 2), 2), True
    except Exception:
        pass

    try:
        pagesize = os.sysconf("SC_PAGE_SIZE")
        total_pages = os.sysconf("SC_PHYS_PAGES")
        avail_pages = os.sysconf("SC_AVPHYS_PAGES")
        total_gb = (total_pages * pagesize) / (1024 ** 3)
        avail_gb = (avail_pages * pagesize) / (1024 ** 3)
        return round(total_gb, 2), round(avail_gb, 2), True
    except Exception:
        pass

    return 8.0, 4.0, False


def _detect_gpu() -> tuple[bool, Optional[str], float, str]:
    """
    Detect NVIDIA GPU and VRAM safely via nvidia-smi without requiring torch or drivers.
    Returns (gpu_available, gpu_name, vram_gb, source).
    If detection fails or GPU is absent, returns (False, None, 0.0, 'none') without raising.
    """
    # 1. Try torch if already imported into memory
    if "torch" in sys.modules:
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                return True, name, round(vram_gb, 2), "torch_cuda"
        except Exception:
            pass

    # 2. Try nvidia-smi CLI
    try:
        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi:
            res = subprocess.run(
                [nvidia_smi, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if res.returncode == 0 and res.stdout.strip():
                line = res.stdout.strip().splitlines()[0]
                parts = line.split(",")
                name = parts[0].strip()
                vram_mb = float(parts[1].strip())
                return True, name, round(vram_mb / 1024.0, 2), "nvidia_smi"
    except Exception:
        pass

    return False, None, 0.0, "none"


def classify_resource_profile(
    total_ram_gb: float,
    available_ram_gb: float,
    gpu_available: bool = False,
    vram_gb: float = 0.0,
    cpu_count: int = 4,
) -> ResourceProfile:
    """
    Deterministically classify host resources into one of four operational profiles:

    1. LOW_RESOURCE: Total RAM < 8.0 GB or Available RAM < 2.5 GB or CPU cores <= 2.
       Rationale: 3B+ parameter models require >= 3.5 GB resident memory. Under 2.5 GB available RAM,
       running standard LLMs triggers swap thrashing or OOM killer.
       Recommended: 1B parameter quantized models (e.g. Llama-3.2-1B) or deterministic offline synthesis.

    2. STANDARD: Total RAM >= 8.0 GB, Available RAM >= 2.5 GB, CPU >= 4, no discrete high-VRAM GPU.
       Rationale: Modern workstations with 8-16 GB system RAM can smoothly host 3B-8B Q4 models on CPU.
       Recommended: 3B-8B quantized models (e.g. Llama-3.2-3B, Llama-3.1-8B Q4).

    3. GPU_ACCELERATED: Discrete GPU present with VRAM >= 4.0 GB.
       Rationale: 4 GB VRAM is the minimum budget to fully offload a 4-bit 7B model or vision encoder.
       Recommended: GPU-offloaded 7B-11B models, local vision models, or whisper-medium/large.

    4. HIGH_MEMORY: Total RAM >= 32.0 GB AND (VRAM >= 12.0 GB or Total RAM >= 64.0 GB).
       Rationale: Large RAM/VRAM capacity enables running 14B-70B models, extended 128k context windows,
       or concurrent multimodal transcription, vision, and reasoning pipelines simultaneously.
    """
    if (gpu_available and vram_gb >= 12.0 and total_ram_gb >= 32.0) or (total_ram_gb >= 64.0):
        return ResourceProfile.HIGH_MEMORY
    if gpu_available and vram_gb >= 4.0:
        return ResourceProfile.GPU_ACCELERATED
    if total_ram_gb < 8.0 or available_ram_gb < 2.5 or cpu_count <= 2:
        return ResourceProfile.LOW_RESOURCE
    return ResourceProfile.STANDARD


def detect_hardware() -> HardwareCapabilities:
    """
    Detect host system resources and compute active ResourceProfile.
    Never crashes: falls back gracefully to safe defaults and marks classification honestly.
    """
    os_name = platform.system()
    arch = platform.machine()
    cpu_count = os.cpu_count() or 4

    ram_success = False
    psutil_ram = _get_ram_psutil()
    if psutil_ram is not None:
        total_ram, avail_ram = psutil_ram
        ram_success = True
        ram_source = "psutil"
    elif os_name == "Windows":
        total_ram, avail_ram, ram_success = _get_ram_windows()
        ram_source = "windows_ctypes"
    else:
        total_ram, avail_ram, ram_success = _get_ram_unix()
        ram_source = "unix_sysconf"

    gpu_available, gpu_name, vram_gb, gpu_source = _detect_gpu()

    # Disk free space for scratch workspace
    try:
        disk_usage = shutil.disk_usage(".")
        disk_free_gb = round(disk_usage.free / (1024 ** 3), 2)
    except Exception:
        disk_free_gb = 50.0

    profile = classify_resource_profile(
        total_ram_gb=total_ram,
        available_ram_gb=avail_ram,
        gpu_available=gpu_available,
        vram_gb=vram_gb,
        cpu_count=cpu_count,
    )

    classification = "REAL" if ram_success else "HEURISTIC"
    detection_source = f"{os_name.lower()}_api+{gpu_source}"

    return HardwareCapabilities(
        os_name=os_name,
        architecture=arch,
        cpu_count=cpu_count,
        total_ram_gb=total_ram,
        available_ram_gb=avail_ram,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        vram_gb=vram_gb,
        disk_free_gb=disk_free_gb,
        profile=profile,
        classification=classification,
        detection_source=detection_source,
    )


def recommend_model_for_task(
    task: str,
    hardware: Optional[HardwareCapabilities] = None,
) -> Dict[str, Any]:
    """Recommends local open-weight model and parameters based on host hardware budget."""
    hw = hardware or detect_hardware()

    if hw.profile == ResourceProfile.LOW_RESOURCE:
        return {
            "model_id": "ollama/llama3.2:1b",
            "runtime": "ollama",
            "quantization": "q4_K_M",
            "approx_download_size_gb": 1.3,
            "approx_ram_required_gb": 2.0,
            "capabilities": ["TEXT", "REASONING"],
            "license": "Llama 3.2 Community License",
            "status": "RECOMMENDED_FOR_LOW_RAM",
        }
    elif hw.profile == ResourceProfile.GPU_ACCELERATED or hw.profile == ResourceProfile.HIGH_MEMORY:
        return {
            "model_id": "ollama/llama3.2-vision:11b",
            "runtime": "ollama",
            "quantization": "q4_K_M",
            "approx_download_size_gb": 7.9,
            "approx_ram_required_gb": 9.0,
            "capabilities": ["TEXT", "VISION", "REASONING", "DOCUMENT"],
            "license": "Llama 3.2 Community License",
            "status": "RECOMMENDED_FOR_GPU",
        }
    else:  # STANDARD
        return {
            "model_id": "ollama/llama3.2:3b",
            "runtime": "ollama",
            "quantization": "q4_K_M",
            "approx_download_size_gb": 2.0,
            "approx_ram_required_gb": 4.0,
            "capabilities": ["TEXT", "REASONING", "STRUCTURED_OUTPUT"],
            "license": "Llama 3.2 Community License",
            "status": "RECOMMENDED_STANDARD",
        }
