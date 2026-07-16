# Copyright (c) ModelScope Contributors. All rights reserved.
"""Moore Threads MUSA platform adapter for Twinkle.

Minimal counterpart of ``npu.py`` for ``torch_musa``.
Validated against MTT S5000 + torch_musa 2.7.1:
- device prefix: ``musa``
- visible devices env: ``MUSA_VISIBLE_DEVICES``
- distributed backend: ``mccl``
"""
import hashlib
import os
import shutil
import socket
from functools import lru_cache
from typing import Optional

from .base import Platform


def ensure_musa_backend() -> None:
    """Import ``torch_musa`` so ``torch.musa`` is registered."""
    try:
        import torch_musa  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            'MUSA backend is not available. Please install torch_musa / Moore Threads PyTorch.') from exc


@lru_cache
def is_musa_available() -> bool:
    """Detect MUSA toolkit or a working torch_musa runtime."""
    if shutil.which('mthreads-gmi') or shutil.which('musaInfo'):
        return True
    try:
        import torch
        import torch_musa  # noqa: F401
        return bool(getattr(torch, 'musa', None) and torch.musa.is_available())
    except Exception:
        return False


def _resolve_musa_physical_device_id(device_id: int) -> int:
    """Map local MUSA index to physical id via ``MUSA_VISIBLE_DEVICES``."""
    visible = os.environ.get('MUSA_VISIBLE_DEVICES', '').strip()
    if not visible:
        return device_id
    parts = [p.strip() for p in visible.split(',') if p.strip()]
    if device_id < 0 or device_id >= len(parts):
        return device_id
    try:
        return int(parts[device_id])
    except ValueError:
        return device_id


class MUSA(Platform):

    @staticmethod
    def visible_device_env():
        # Moore Threads runtime uses MUSA_VISIBLE_DEVICES (not CUDA_VISIBLE_DEVICES).
        return 'MUSA_VISIBLE_DEVICES'

    @staticmethod
    def device_prefix():
        return 'musa'

    @staticmethod
    def get_local_device(idx, **kwargs) -> str:
        return f'musa:{idx}'

    @staticmethod
    def device_backend(platform: str = None):
        # torch_musa registers MCCL as the collective backend.
        return 'mccl'

    @staticmethod
    def get_vllm_device_uuid(device_id: int = 0) -> str:
        """Stable cross-process device id for sampler / IPC paths.

        vLLM-MUSA may be unavailable; prefer torch device name + physical id,
        then fall back to a deterministic hash (same idea as NPU).
        """
        try:
            ensure_musa_backend()
            import torch
            physical_id = _resolve_musa_physical_device_id(device_id)
            if hasattr(torch.musa, 'get_device_name'):
                name = torch.musa.get_device_name(physical_id)
                return f'{name}:{physical_id}'.replace(' ', '_')
        except Exception:
            pass

        visible = os.environ.get(MUSA.visible_device_env(), '')
        raw = f'{socket.gethostname()}:{visible}:{device_id}'
        return hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]
