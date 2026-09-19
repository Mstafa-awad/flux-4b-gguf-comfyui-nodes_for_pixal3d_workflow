# SPDX-License-Identifier: GPL-3.0-or-later
"""Character Turnaround (Front / Back / Left / Right) for ComfyUI.

One reference image in -> four 1024x1024 views out.

The FRONT slot passes the reference straight through (re-fit to 1024x1024),
so only THREE views are generated: back, left, right.

Built for 8GB VRAM:
  * strict stage isolation - text encoder, VAE and diffusion model are never
    resident at the same time
  * GGUF weights stream from RAM (patch_on_device=False) unless GPU >= 12GB
  * one diffusion-model load shared by all generated views
  * the reference is VAE-encoded once and reused by every view
  * tiled VAE decode with an automatic fallback ladder
  * per-view OOM retry at reduced resolution, upscaled back to the target size
"""

from __future__ import annotations

import gc
import hashlib
import inspect
import logging
import os
import time
import weakref
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

import comfy.model_management as model_management
import comfy.samplers
import comfy.utils
import folder_paths
import nodes as comfy_nodes

LOG = logging.getLogger("CharacterTurnaround")
CATEGORY = "Character Turnaround"

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

MODE_LOW_VRAM = "low_vram"
MODE_BALANCED = "balanced"
MODE_HIGH_SPEED = "high_speed"
MODE_CPU_ONLY = "cpu_only"
PERFORMANCE_MODES = [MODE_LOW_VRAM, MODE_BALANCED, MODE_HIGH_SPEED, MODE_CPU_ONLY]

DEFAULT_OUTPUT_SIZE = 1024

GENERATED_VIEWS = ("back", "left", "right")
ALL_VIEWS = ("front", "back", "left", "right")

VIEW_PHRASES: Dict[str, str] = {
    "front": (
        "FRONT VIEW: the character faces the camera directly, perfectly symmetrical "
        "front elevation, both eyes visible, arms relaxed at the sides"
    ),
    "back": (
        "BACK VIEW: seen from directly behind, the character has their back fully to "
        "the camera, back of the head and heels visible, no face visible at all, "
        "the character is turned exactly 180 degrees away"
    ),
    "left": (
        "LEFT SIDE VIEW: full side profile, we see the character's own LEFT side, "
        "the character's left shoulder and left ear are closest to the camera, "
        "the character's nose points toward the LEFT edge of the frame, "
        "the body is turned exactly 90 degrees"
    ),
    "right": (
        "RIGHT SIDE VIEW: full side profile, we see the character's own RIGHT side, "
        "the character's right shoulder and right ear are closest to the camera, "
        "the character's nose points toward the RIGHT edge of the frame, "
        "the body is turned exactly 90 degrees"
    ),
}

DEFAULT_PROMPT_TEMPLATE = (
    "Character model sheet, character turnaround reference, full body visible from "
    "head to feet, one single character only, centered in the frame. {view}. "
    "Exactly the same character as the reference image: identical face, identical "
    "hairstyle and hair colour, identical outfit, identical colours, identical "
    "fabrics, identical proportions, identical shoes and accessories. "
    "Standing upright in a relaxed neutral pose, legs together, arms hanging "
    "naturally. Plain flat white seamless studio background, soft even diffuse "
    "lighting, no cast shadow, sharp focus, high detail, same art style and same "
    "rendering quality as the reference image."
)

FRONT_FIT_MODES = ["fit_padding", "center_crop_square", "stretch"]
FRONT_SOURCES = ["generated_front_view", "reference_passthrough"]

# --------------------------------------------------------------------------- #
# Small caches
# --------------------------------------------------------------------------- #


class _LRU:
    def __init__(self, name: str, max_entries: int) -> None:
        self.name = name
        self.max_entries = max(1, int(max_entries))
        self._data: "OrderedDict[Any, Any]" = OrderedDict()

    def get(self, key: Any) -> Any:
        if key not in self._data:
            return None
        self._data.move_to_end(key)
        return self._data[key]

    def set(self, key: Any, value: Any) -> None:
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        while len(self._data) > self.max_entries:
            self._data.popitem(last=False)

    def clear(self) -> None:
        self._data.clear()


_SIGMA_CACHE = _LRU("sigmas", 32)
_ACTIVE_PATCHERS: Dict[str, List[Any]] = {"clip": [], "unet": [], "vae": []}


class _Runtime:
    device_mode_applied: Optional[bool] = None
    cpu_threads_configured: bool = False
    cpu_warning_shown: bool = False


RUNTIME = _Runtime()

# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #


def _unique(items: Any) -> List[str]:
    return sorted({x for x in items if isinstance(x, str)})


def _filenames(keys: Sequence[str], extension: Optional[str] = None) -> List[str]:
    out: List[str] = []
    for key in keys:
        try:
            out.extend(folder_paths.get_filename_list(key))
        except Exception:
            continue
    out = _unique(out)
    if extension:
        out = [x for x in out if x.lower().endswith(extension.lower())]
    return out


def _gguf_diffusion_names() -> List[str]:
    files = _filenames(("unet_gguf", "diffusion_models", "unet"), ".gguf")
    klein = [x for x in files if "klein" in x.lower()]
    return klein or files or ["FLUX.2-klein-4B-Q8_0.gguf"]


def _gguf_qwen_names() -> List[str]:
    files = _filenames(("clip_gguf", "text_encoders", "clip"))
    qwen = [x for x in files if "qwen" in x.lower()]
    return qwen or files or ["Qwen3-4B-Q8_0.gguf"]


def _vae_names() -> List[str]:
    files = _filenames(("vae",))
    flux = [
        x for x in files
        if "flux2" in x.lower() or "flux_2" in x.lower() or "flux-2" in x.lower()
    ]
    return flux or files or ["flux2-vae.safetensors"]


def _sampler_names() -> List[str]:
    try:
        names = list(comfy.samplers.KSampler.SAMPLERS)
    except Exception:
        names = ["euler"]
    if "euler" in names:
        names.remove("euler")
    names.insert(0, "euler")
    return names


def _file_size_gb(keys: Sequence[str], name: str) -> Optional[float]:
    for key in keys:
        try:
            path = folder_paths.get_full_path(key, name)
        except Exception:
            path = None
        if path and os.path.isfile(path):
            try:
                return os.path.getsize(path) / 1_000_000_000
            except Exception:
                return None
    return None


_SIGNATURE_CACHE: Dict[str, Optional[inspect.Signature]] = {}


def _signature_for(name: str, function: Any) -> Optional[inspect.Signature]:
    if name in _SIGNATURE_CACHE:
        return _SIGNATURE_CACHE[name]
    try:
        sig = inspect.signature(function)
    except Exception:
        sig = None
    _SIGNATURE_CACHE[name] = sig
    return sig


def _require_node(name: str):
    cls = comfy_nodes.NODE_CLASS_MAPPINGS.get(name)
    if cls is None:
        if "GGUF" in name:
            raise RuntimeError(
                f"Required node '{name}' is missing. Install/update ComfyUI-GGUF, "
                "update ComfyUI, then restart."
            )
        raise RuntimeError(
            f"Required ComfyUI core node '{name}' is missing. Update ComfyUI and restart."
        )
    return cls


def _normalize_output(result: Any) -> Tuple[Any, ...]:
    if hasattr(result, "args"):
        return tuple(result.args)
    if isinstance(result, dict) and "result" in result:
        wrapped = result["result"]
        return tuple(wrapped) if isinstance(wrapped, (list, tuple)) else (wrapped,)
    if isinstance(result, (list, tuple)):
        return tuple(result)
    return (result,)


def _invoke(name: str, **kwargs: Any) -> Tuple[Any, ...]:
    """Run a registered ComfyUI node, tolerating V3 output containers."""
    cls = _require_node(name)
    instance = cls()
    function = getattr(instance, getattr(cls, "FUNCTION"))
    sig = _signature_for(name, function)
    if sig is not None:
        var_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if not var_kwargs:
            allowed = set(sig.parameters.keys())
            kwargs = {k: v for k, v in kwargs.items() if k in allowed}
    return _normalize_output(function(**kwargs))


def _to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu")
    if isinstance(value, dict):
        return {k: _to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(v) for v in value)
    return value


def _clone(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: _clone(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clone(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_clone(v) for v in value)
    return value


def _round16(value: int) -> int:
    return max(64, int(round(int(value) / 16.0) * 16))


def _fit_megapixels(width: int, height: int, max_mp: float) -> Tuple[int, int]:
    width, height = int(width), int(height)
    budget = max(64 * 64, float(max_mp) * 1_000_000)
    scale = min(1.0, (budget / max(1, width * height)) ** 0.5)
    if scale < 1.0:
        width = max(64, int(width * scale) // 16 * 16)
        height = max(64, int(height * scale) // 16 * 16)
    else:
        width, height = _round16(width), _round16(height)
    return width, height

# --------------------------------------------------------------------------- #
# Memory / device helpers
# --------------------------------------------------------------------------- #


def _cuda_available() -> bool:
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _safe_empty_cache() -> None:
    try:
        if hasattr(model_management, "soft_empty_cache"):
            model_management.soft_empty_cache()
    except Exception:
        LOG.debug("soft_empty_cache failed", exc_info=True)


def _configure_cpu_threads() -> None:
    if RUNTIME.cpu_threads_configured:
        return
    RUNTIME.cpu_threads_configured = True
    cores: Optional[int] = None
    try:
        import psutil
        cores = psutil.cpu_count(logical=False)
    except Exception:
        cores = None
    if not cores:
        cores = os.cpu_count()
    threads = max(1, int(cores or 1))
    try:
        torch.set_num_threads(threads)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(max(1, min(4, threads // 2 or 1)))
    except Exception:
        pass


def _set_comfy_cpu_state(enabled: bool) -> bool:
    cpu_enum = getattr(model_management, "CPUState", None)
    current = getattr(model_management, "cpu_state", None)
    if cpu_enum is None or current is None:
        return not enabled
    target = cpu_enum.CPU if enabled else cpu_enum.GPU
    if current == target:
        return True
    if not enabled and not _cuda_available():
        return False
    try:
        model_management.unload_all_models()
        model_management.cpu_state = target
        _safe_empty_cache()
        return True
    except Exception:
        return False


def _apply_device_mode(mode: str) -> None:
    want_cpu = mode == MODE_CPU_ONLY
    if RUNTIME.device_mode_applied == want_cpu:
        return
    if want_cpu:
        _set_comfy_cpu_state(True)
        _configure_cpu_threads()
        if not _cuda_available() and not RUNTIME.cpu_warning_shown:
            LOG.warning("CPU mode requested; for reliability restart ComfyUI with --cpu.")
            RUNTIME.cpu_warning_shown = True
        RUNTIME.device_mode_applied = True
    else:
        if _cuda_available():
            _set_comfy_cpu_state(False)
        RUNTIME.device_mode_applied = False


def _resolve_mode(mode: str) -> str:
    resolved = str(mode or MODE_LOW_VRAM).strip().lower()
    if resolved not in PERFORMANCE_MODES:
        resolved = MODE_LOW_VRAM
    if resolved != MODE_CPU_ONLY and not _cuda_available():
        resolved = MODE_CPU_ONLY
    _apply_device_mode(resolved)
    return resolved


def _is_cpu_device() -> bool:
    try:
        device = model_management.get_torch_device()
        return getattr(device, "type", str(device)) == "cpu"
    except Exception:
        return False


def _gpu_total_gb() -> float:
    if not _cuda_available():
        return 0.0
    try:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        return props.total_memory / 2 ** 30
    except Exception:
        return 0.0


def _gpu_free_gb() -> float:
    if not _cuda_available():
        return 0.0
    try:
        return torch.cuda.mem_get_info()[0] / 2 ** 30
    except Exception:
        return 0.0


def _ram_free_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().available / 2 ** 30
    except Exception:
        return 0.0


def _free_memory_gb(mode: str) -> float:
    if mode == MODE_CPU_ONLY or _is_cpu_device():
        return _ram_free_gb()
    if _cuda_available():
        return _gpu_free_gb()
    return _ram_free_gb()


def _is_memory_error(exc: BaseException) -> bool:
    if isinstance(exc, MemoryError):
        return True
    text = str(exc).lower()
    markers = (
        "out of memory", "oom", "bad_alloc", "cannot allocate", "can't allocate",
        "cuda memory", "not enough memory", "cudnn", "cublas",
    )
    return any(m in text for m in markers)

# --------------------------------------------------------------------------- #
# Patcher tracking + staged cleanup
# --------------------------------------------------------------------------- #


def _get_patcher(obj: Any) -> Any:
    if obj is None:
        return None
    patcher = getattr(obj, "patcher", None)
    if patcher is not None:
        return patcher
    if hasattr(obj, "model") and hasattr(obj, "clone"):
        return obj
    return None


def _register_patcher(kind: str, obj: Any) -> None:
    patcher = _get_patcher(obj)
    if patcher is None:
        return
    registry = _ACTIVE_PATCHERS.setdefault(kind, [])
    for ref in registry:
        try:
            if ref() is patcher:
                return
        except Exception:
            continue
    try:
        registry.append(weakref.ref(patcher))
    except TypeError:
        pass


def _release_patcher(patcher: Any) -> None:
    if patcher is None:
        return
    try:
        if hasattr(model_management, "unload_model_and_clones"):
            model_management.unload_model_and_clones(patcher)
        elif hasattr(model_management, "unload_model_clones"):
            model_management.unload_model_clones(patcher)
        else:
            model_management.unload_all_models()
    except Exception:
        LOG.debug("targeted patcher release failed", exc_info=True)


def _release_kind(kind: str, empty_cache: bool = False) -> None:
    registry = _ACTIVE_PATCHERS.get(kind, [])
    if not registry:
        return
    for ref in list(registry):
        try:
            patcher = ref()
        except Exception:
            patcher = None
        if patcher is not None:
            _release_patcher(patcher)
    registry.clear()
    if empty_cache:
        _safe_empty_cache()


def _release_everything() -> None:
    for kind in list(_ACTIVE_PATCHERS.keys()):
        _release_kind(kind, empty_cache=False)
    try:
        model_management.unload_all_models()
    except Exception:
        pass
    _safe_empty_cache()
    gc.collect()


def _stage_cleanup(mode: str, aggressive: bool, release: Sequence[str]) -> None:
    """Free the listed model kinds. high_speed keeps the diffusion model warm."""
    if not aggressive and mode == MODE_LOW_VRAM:
        aggressive = True
    for kind in release:
        if kind == "unet" and mode == MODE_HIGH_SPEED and not aggressive:
            continue
        _release_kind(kind, empty_cache=False)
    _safe_empty_cache()
    if mode in (MODE_LOW_VRAM, MODE_CPU_ONLY) or aggressive:
        gc.collect()


def _resolve_patch_on_device(mode: str, file_size_gb: Optional[float]) -> bool:
    if mode == MODE_CPU_ONLY or not _cuda_available():
        return False
    if mode != MODE_HIGH_SPEED:
        return False
    total = _gpu_total_gb()
    if total < 12.0:
        return False
    if file_size_gb is None:
        return True
    return total >= (float(file_size_gb) * 2.0 + 4.0)


def _tile_for_mode(mode: str) -> int:
    if mode in (MODE_LOW_VRAM, MODE_CPU_ONLY):
        return 512
    if mode == MODE_BALANCED:
        return 768
    return 1024

# --------------------------------------------------------------------------- #
# Image helpers
# --------------------------------------------------------------------------- #


def _first_image(image: torch.Tensor) -> torch.Tensor:
    """Normalise to [1,H,W,3] float32 on CPU."""
    t = image.detach().to("cpu", dtype=torch.float32)
    if t.dim() == 3:
        t = t.unsqueeze(0)
    t = t[:1]
    if t.shape[-1] == 4:
        t = t[..., :3]
    elif t.shape[-1] == 1:
        t = t.repeat(1, 1, 1, 3)
    elif t.shape[-1] != 3:
        t = t[..., :3]
    return t.clamp(0.0, 1.0)


def _resize(image: torch.Tensor, width: int, height: int, method: str = "lanczos") -> torch.Tensor:
    channels_first = image.movedim(-1, 1)
    out = comfy.utils.common_upscale(channels_first, width, height, method, "disabled")
    return out.movedim(1, -1)


def _front_from_reference(image: torch.Tensor, size: int, fit_mode: str) -> torch.Tensor:
    """Re-fit the reference to size x size so it matches the generated views."""
    src = _first_image(image)
    h, w = int(src.shape[1]), int(src.shape[2])

    if fit_mode == "center_crop_square":
        side = min(h, w)
        top = (h - side) // 2
        left = (w - side) // 2
        src = src[:, top:top + side, left:left + side, :]
        return _resize(src, size, size, "lanczos")

    if fit_mode == "stretch":
        return _resize(src, size, size, "lanczos")

    # fit_padding (default): keep the whole character, pad to a white square.
    scale = size / max(h, w)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = _resize(src, nw, nh, "lanczos")
    canvas = torch.ones((1, size, size, 3), dtype=torch.float32)
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas[:, top:top + nh, left:left + nw, :] = resized
    return canvas


def _build_sheet(views: Dict[str, torch.Tensor]) -> torch.Tensor:
    """2x2 contact sheet: front | left / back | right."""
    order = ["front", "left", "back", "right"]
    tiles = [views[k] for k in order if k in views]
    if len(tiles) < 4:
        return tiles[0]
    top = torch.cat((tiles[0], tiles[1]), dim=2)
    bottom = torch.cat((tiles[2], tiles[3]), dim=2)
    return torch.cat((top, bottom), dim=1)


def _hash_image(image: torch.Tensor) -> str:
    t = image.detach().to("cpu", dtype=torch.float32).contiguous()
    hasher = hashlib.sha1()
    hasher.update(str(tuple(t.shape)).encode("utf-8"))
    try:
        hasher.update(memoryview(t.numpy()))
    except Exception:
        hasher.update(t.numpy().tobytes())
    return hasher.hexdigest()

# --------------------------------------------------------------------------- #
# Model loading helpers
# --------------------------------------------------------------------------- #


def _validate_flux_gguf(name: str) -> None:
    if not name:
        raise ValueError("No FLUX GGUF file selected.")
    low = name.lower()
    if "base" in low and "klein" in low:
        raise ValueError(
            "This node is tuned for the DISTILLED 4-step Klein checkpoint, not Base."
        )
    if any(tok in low for tok in ("9b", "12b", "14b", "32b")):
        raise ValueError(
            "Select FLUX.2 Klein 4B. Larger checkpoints will not fit an 8GB workflow."
        )


def _validate_qwen(name: str) -> None:
    if not name:
        raise ValueError("No Qwen GGUF text encoder selected.")
    low = name.lower()
    if "qwen" not in low:
        LOG.warning("Text encoder filename does not contain 'Qwen': %s", name)
    if any(tok in low for tok in ("8b", "14b", "30b", "32b")):
        raise ValueError("FLUX.2 Klein 4B needs Qwen3 4B. Do not use 8B or larger.")


def _load_unet_gguf(name: str, dequant: str, patch: str, on_device: bool) -> Any:
    if "UnetLoaderGGUFAdvanced" in comfy_nodes.NODE_CLASS_MAPPINGS:
        return _invoke(
            "UnetLoaderGGUFAdvanced",
            unet_name=name,
            dequant_dtype=dequant,
            patch_dtype=patch,
            patch_on_device=on_device,
        )[0]
    return _invoke("UnetLoaderGGUF", unet_name=name)[0]


def _empty_latent(width: int, height: int, batch: int = 1) -> Any:
    try:
        return _invoke(
            "EmptyFlux2LatentImage", width=width, height=height, batch_size=batch
        )[0]
    except RuntimeError:
        return _invoke(
            "EmptySD3LatentImage", width=width, height=height, batch_size=batch
        )[0]


def _sigmas_for(steps: int, width: int, height: int) -> Any:
    key = (int(steps), int(width), int(height))
    cached = _SIGMA_CACHE.get(key)
    if cached is not None:
        return cached.clone() if isinstance(cached, torch.Tensor) else cached
    sigmas = _invoke("Flux2Scheduler", steps=int(steps), width=int(width), height=int(height))[0]
    if isinstance(sigmas, torch.Tensor):
        _SIGMA_CACHE.set(key, sigmas.detach().clone())
    return sigmas

# --------------------------------------------------------------------------- #
# VAE encode / decode with fallbacks
# --------------------------------------------------------------------------- #


def _encode_reference(image: torch.Tensor, vae: Any, mode: str, budget_mp: float) -> Tuple[Any, int, int, str]:
    width, height = _fit_megapixels(int(image.shape[2]), int(image.shape[1]), budget_mp)
    resized = image
    if (int(image.shape[2]), int(image.shape[1])) != (width, height):
        resized = _resize(image, width, height, "area")
    if mode == MODE_CPU_ONLY:
        resized = _to_cpu(resized)

    _register_patcher("vae", vae)
    tiled = mode in (MODE_LOW_VRAM, MODE_CPU_ONLY)
    tile = _tile_for_mode(mode)
    attempts: List[Tuple[str, int, int]] = []
    if not tiled:
        attempts.append(("normal", 0, 0))
    attempts += [("tiled", tile, 64), ("tiled", max(256, tile // 2), 32), ("tiled", 256, 16)]

    last: Optional[BaseException] = None
    for idx, (kind, t, ov) in enumerate(attempts):
        try:
            if kind == "normal":
                latent = _invoke("VAEEncode", pixels=resized, vae=vae)[0]
                return _to_cpu(latent), width, height, "normal"
            latent = _invoke(
                "VAEEncodeTiled", pixels=resized, vae=vae,
                tile_size=t, overlap=ov, temporal_size=64, temporal_overlap=8,
            )[0]
            return _to_cpu(latent), width, height, f"tiled {t}px"
        except Exception as exc:
            last = exc
            if idx < len(attempts) - 1 and _is_memory_error(exc):
                _release_everything()
                _register_patcher("vae", vae)
                continue
            raise
    raise last if last is not None else RuntimeError("VAE reference encode failed.")


def _decode_latent(samples: Any, vae: Any, mode: str) -> Tuple[torch.Tensor, str]:
    _register_patcher("vae", vae)
    tile = _tile_for_mode(mode)
    attempts: List[Tuple[str, int, int]] = []
    if mode in (MODE_BALANCED, MODE_HIGH_SPEED) and _free_memory_gb(mode) >= 3.5:
        attempts.append(("normal", 0, 0))
    attempts += [("tiled", tile, 64), ("tiled", max(256, tile // 2), 32), ("tiled", 256, 16)]

    last: Optional[BaseException] = None
    for idx, (kind, t, ov) in enumerate(attempts):
        try:
            if kind == "normal":
                image = _invoke("VAEDecode", samples=samples, vae=vae)[0]
                return _first_image(image), "normal"
            image = _invoke(
                "VAEDecodeTiled", samples=samples, vae=vae,
                tile_size=t, overlap=ov, temporal_size=64, temporal_overlap=8,
            )[0]
            return _first_image(image), f"tiled {t}px"
        except Exception as exc:
            last = exc
            if idx < len(attempts) - 1 and _is_memory_error(exc):
                _release_everything()
                _register_patcher("vae", vae)
                continue
            raise
    raise last if last is not None else RuntimeError("VAE decode failed.")

# --------------------------------------------------------------------------- #
# The node
# --------------------------------------------------------------------------- #


class CharacterTurnaround:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_image": ("IMAGE",),
                "performance_mode": (PERFORMANCE_MODES, {"default": MODE_LOW_VRAM}),
                "flux_gguf": (_gguf_diffusion_names(),),
                "qwen_gguf": (_gguf_qwen_names(),),
                "vae_name": (_vae_names(),),
                "dequant_dtype": (["target", "float16", "bfloat16", "float32"], {"default": "target"}),
                "patch_dtype": (["default", "target", "float16", "bfloat16", "float32"], {"default": "default"}),
                "output_size": ("INT", {"default": DEFAULT_OUTPUT_SIZE, "min": 512, "max": 2048, "step": 64}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 12}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "sampler_name": (_sampler_names(), {"default": "euler"}),
                "seed": ("INT", {"default": 43, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
                "front_output_source": (FRONT_SOURCES, {"default": "generated_front_view"}),
                "front_fit_mode": (FRONT_FIT_MODES, {"default": "fit_padding"}),
                "swap_left_right": ("BOOLEAN", {"default": False}),
                "reference_budget_megapixels": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 2.0, "step": 0.05}),
                "auto_reduce_on_oom": ("BOOLEAN", {"default": True}),
                "aggressive_cleanup": ("BOOLEAN", {"default": True}),
                "prompt_template": ("STRING", {"multiline": True, "dynamicPrompts": True,
                                               "default": DEFAULT_PROMPT_TEMPLATE}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("front", "back", "left", "right", "sheet_2x2", "info")
    FUNCTION = "generate"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "One reference character in, four 1024x1024 views out. The FRONT slot passes the "
        "reference through, so only back/left/right are generated. Staged for 8GB VRAM: "
        "text encoder, VAE and diffusion model are never resident at the same time, and "
        "the diffusion model is loaded once for all views."
    )

    # ------------------------------------------------------------------ #

    def generate(
        self,
        reference_image=None,
        performance_mode=MODE_LOW_VRAM,
        flux_gguf="",
        qwen_gguf="",
        vae_name="",
        dequant_dtype="target",
        patch_dtype="default",
        output_size=DEFAULT_OUTPUT_SIZE,
        steps=4,
        cfg=1.0,
        sampler_name="euler",
        seed=43,
        front_output_source="generated_front_view",
        front_fit_mode="fit_padding",
        swap_left_right=False,
        reference_budget_megapixels=1.0,
        auto_reduce_on_oom=True,
        aggressive_cleanup=True,
        prompt_template=DEFAULT_PROMPT_TEMPLATE,
    ):
        t_total = time.perf_counter()
        mode = _resolve_mode(performance_mode)
        if reference_image is None:
            raise ValueError("A reference image is required.")

        _validate_flux_gguf(flux_gguf)
        _validate_qwen(qwen_gguf)

        size = _round16(int(output_size))
        steps = int(steps)
        timings: Dict[str, float] = {}
        notes: List[str] = []

        if steps != 4:
            notes.append(f"steps={steps} (Klein distilled is validated at 4)")
        if abs(float(cfg) - 1.0) > 0.001:
            notes.append(f"cfg={float(cfg):g} (validated at 1.0)")
        if sampler_name != "euler":
            notes.append(f"sampler={sampler_name} (validated euler)")
        if mode == MODE_CPU_ONLY:
            notes.append("CPU mode: this will take many minutes per view")

        reference = _first_image(reference_image)
        views: Dict[str, torch.Tensor] = {}
        views["front"] = _front_from_reference(reference, size, front_fit_mode)

        generate_front = front_output_source == "generated_front_view"
        targets: List[str] = ["front"] + list(GENERATED_VIEWS) if generate_front else list(GENERATED_VIEWS)

        template = prompt_template if "{view}" in prompt_template else DEFAULT_PROMPT_TEMPLATE
        prompts = {v: template.replace("{view}", VIEW_PHRASES[v]) for v in targets}

        # ---------------- Stage 1: text encode (CLIP resident, then freed) ---
        t0 = time.perf_counter()
        conditioning: Dict[str, Any] = {}
        negative: Any = None
        clip = _invoke("CLIPLoaderGGUF", clip_name=qwen_gguf, type="flux2")[0]
        _register_patcher("clip", clip)
        clip_patcher = _get_patcher(clip)
        try:
            if mode != MODE_CPU_ONLY and _cuda_available():
                try:
                    if clip_patcher is not None:
                        model_management.load_models_gpu([clip_patcher])
                except Exception as exc:
                    notes.append(f"Qwen GPU staging skipped ({exc.__class__.__name__})")
            for view in targets:
                cond = _to_cpu(_invoke("CLIPTextEncode", clip=clip, text=prompts[view])[0])
                conditioning[view] = cond
            negative = _to_cpu(_invoke("ConditioningZeroOut", conditioning=conditioning[targets[0]])[0])
        finally:
            _release_patcher(clip_patcher)
            _release_kind("clip", empty_cache=False)
            del clip
            _safe_empty_cache()
            if mode in (MODE_LOW_VRAM, MODE_CPU_ONLY) or aggressive_cleanup:
                gc.collect()
        timings["text_encode"] = time.perf_counter() - t0

        # ---------------- Stage 2: reference VAE encode (once, reused) -------
        t0 = time.perf_counter()
        vae = _invoke("VAELoader", vae_name=vae_name)[0]
        _register_patcher("vae", vae)
        try:
            ref_latent, ref_w, ref_h, ref_strategy = _encode_reference(
                reference, vae, mode, float(reference_budget_megapixels)
            )
        finally:
            if mode != MODE_HIGH_SPEED:
                _release_kind("vae", empty_cache=True)
            del vae
            if mode in (MODE_LOW_VRAM, MODE_CPU_ONLY) or aggressive_cleanup:
                gc.collect()
        timings["reference_encode"] = time.perf_counter() - t0

        # ---------------- Stage 3: sample all views on one model load --------
        t0 = time.perf_counter()
        flux_size_gb = _file_size_gb(("unet_gguf", "diffusion_models", "unet"), flux_gguf)
        on_device = _resolve_patch_on_device(mode, flux_size_gb)
        if mode == MODE_CPU_ONLY and dequant_dtype == "target":
            dequant_dtype, patch_dtype = "float32", "float32"

        model = _load_unet_gguf(flux_gguf, dequant_dtype, patch_dtype, on_device)
        _register_patcher("unet", model)

        latents: Dict[str, Any] = {}
        latent_sizes: Dict[str, Tuple[int, int]] = {}
        sampler = _invoke("KSamplerSelect", sampler_name=sampler_name)[0]
        noise = _invoke("RandomNoise", noise_seed=int(seed))[0]

        try:
            for view in targets:
                t_view = time.perf_counter()
                positive = _invoke(
                    "ReferenceLatent", conditioning=_clone(conditioning[view]), latent=ref_latent
                )[0]
                neg = _invoke("ReferenceLatent", conditioning=_clone(negative), latent=ref_latent)[0]
                guider = _invoke("CFGGuider", model=model, positive=positive, negative=neg, cfg=float(cfg))[0]

                attempts: List[Tuple[int, int]] = [(size, size)]
                if auto_reduce_on_oom:
                    attempts += [
                        (_round16(size * 0.875), _round16(size * 0.875)),
                        (_round16(size * 0.75), _round16(size * 0.75)),
                    ]

                produced: Optional[Any] = None
                used: Tuple[int, int] = (size, size)
                last_exc: Optional[BaseException] = None
                for idx, (w, h) in enumerate(attempts):
                    try:
                        sigmas = _sigmas_for(steps, w, h)
                        if mode == MODE_CPU_ONLY:
                            sigmas = _to_cpu(sigmas)
                        empty = _empty_latent(w, h, 1)
                        if mode == MODE_CPU_ONLY:
                            empty = _to_cpu(empty)
                            positive = _to_cpu(positive)
                            neg = _to_cpu(neg)
                            guider = _invoke(
                                "CFGGuider", model=model, positive=positive, negative=neg, cfg=float(cfg)
                            )[0]
                        result = _invoke(
                            "SamplerCustomAdvanced",
                            noise=noise, guider=guider, sampler=sampler,
                            sigmas=sigmas, latent_image=empty,
                        )
                        produced = result[0]
                        used = (w, h)
                        break
                    except Exception as exc:
                        last_exc = exc
                        if idx < len(attempts) - 1 and _is_memory_error(exc) and auto_reduce_on_oom:
                            notes.append(f"{view}: OOM at {w}x{h}, retrying at {attempts[idx+1][0]}px")
                            _release_everything()
                            _register_patcher("unet", model)
                            guider = _invoke(
                                "CFGGuider", model=model, positive=positive, negative=neg, cfg=float(cfg)
                            )[0]
                            continue
                        raise
                if produced is None:
                    raise last_exc if last_exc else RuntimeError(f"Sampling failed for {view}.")

                latents[view] = _to_cpu(produced)
                latent_sizes[view] = used
                if used != (size, size):
                    notes.append(f"{view}: generated at {used[0]}x{used[1]}, upscaled to {size}")
                timings[f"sample_{view}"] = time.perf_counter() - t_view

                del produced, guider, positive, neg, empty
                if mode in (MODE_LOW_VRAM, MODE_CPU_ONLY) or aggressive_cleanup:
                    _safe_empty_cache()
                    gc.collect()
        finally:
            if mode != MODE_HIGH_SPEED or aggressive_cleanup:
                _release_kind("unet", empty_cache=True)
            del model
            if mode in (MODE_LOW_VRAM, MODE_CPU_ONLY) or aggressive_cleanup:
                gc.collect()
        timings["sampling_total"] = time.perf_counter() - t0

        # ---------------- Stage 4: decode -----------------------------------
        t0 = time.perf_counter()
        vae = _invoke("VAELoader", vae_name=vae_name)[0]
        _register_patcher("vae", vae)
        decode_methods: List[str] = []
        try:
            for view in targets:
                if mode == MODE_CPU_ONLY:
                    latents[view] = _to_cpu(latents[view])
                image, method = _decode_latent(latents[view], vae, mode)
                decode_methods.append(f"{view}:{method}")
                if image.shape[1] != size or image.shape[2] != size:
                    image = _resize(image, size, size, "lanczos")
                views[view] = image
                del latents[view]
                if mode in (MODE_LOW_VRAM, MODE_CPU_ONLY) or aggressive_cleanup:
                    _safe_empty_cache()
        finally:
            _release_kind("vae", empty_cache=True)
            del vae
            gc.collect()
        timings["decode_total"] = time.perf_counter() - t0

        # ---------------- Assemble ------------------------------------------
        if swap_left_right and "left" in views and "right" in views:
            views["left"], views["right"] = views["right"], views["left"]
            notes.append("left/right outputs were swapped")

        if generate_front and "front" in views:
            notes.append("front slot holds a GENERATED view (reference passthrough disabled)")

        for key in ALL_VIEWS:
            if key not in views:
                views[key] = views["front"]

        sheet = _build_sheet(views)
        timings["total"] = time.perf_counter() - t_total

        timing_text = ", ".join(f"{k}={v:.1f}s" for k, v in timings.items())
        info = (
            f"mode={mode}; output {size}x{size}; generated views: {', '.join(targets)}; "
            f"reference {ref_w}x{ref_h} via {ref_strategy}; decode [{', '.join(decode_methods)}]; "
            f"patch_on_device={on_device}; seed={seed}; sheet order: front|left / back|right. "
            f"TIMINGS -> {timing_text}."
        )
        if notes:
            info += " NOTES -> " + " | ".join(notes) + "."

        return (
            views["front"],
            views["back"],
            views["left"],
            views["right"],
            sheet,
            info,
        )


class CharacterTurnaroundCacheClear:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"clear_sigmas": ("BOOLEAN", {"default": True})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "run"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    DESCRIPTION = "Clears the turnaround scheduler cache and reports GPU/RAM plus live tracked models."

    def run(self, clear_sigmas=True):
        if clear_sigmas:
            _SIGMA_CACHE.clear()
        alive = []
        for kind in ("clip", "unet", "vae"):
            count = 0
            for ref in _ACTIVE_PATCHERS.get(kind, []):
                try:
                    if ref() is not None:
                        count += 1
                except Exception:
                    continue
            alive.append(f"{kind}={count}")
        text = (
            f"free GPU {_gpu_free_gb():.2f}/{_gpu_total_gb():.2f} GiB; "
            f"free RAM {_ram_free_gb():.2f} GiB; tracked models: {', '.join(alive)}; "
            f"sigmas cache cleared={clear_sigmas}"
        )
        return {"ui": {"text": [text]}, "result": (text,)}


NODE_CLASS_MAPPINGS = {
    "CharacterTurnaround": CharacterTurnaround,
    "CharacterTurnaroundCacheClear": CharacterTurnaroundCacheClear,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CharacterTurnaround": "Character Turnaround (Front/Back/Left/Right) [8GB Staged]",
    "CharacterTurnaroundCacheClear": "Character Turnaround Cache + Memory Report",
}