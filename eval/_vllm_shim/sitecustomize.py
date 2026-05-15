"""
sitecustomize for vLLM toxicity runs.

Background
----------
The system flash_attn at /usr/local/lib/python3.12/dist-packages/flash_attn
was built against a different libtorch ABI than the torch 2.11.0+cu130 in
/home/work/.local/lib/python3.12/site-packages.  Any attempt to actually
import flash_attn from inside our process raises:

    ImportError: undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib

We can't uninstall the system flash_attn (it would need elevated permissions
and might affect other system processes), and rebuilding it is slow.

Both transformers 4.57.1 and vLLM 0.20.1 try to import flash_attn during
module-load time:

  1. transformers/integrations/flash_attention.py runs
     `_use_top_left_mask = flash_attn_supports_top_left_mask()` at import,
     which probes flash_attn → crash.
  2. transformers/utils/import_utils._is_package_available("flash_attn")
     is called from many call-sites and returns True (because flash_attn
     dist-info exists) which leads to later import attempts.
  3. vLLM's TRITON_ATTN / FLEX_ATTENTION backends do not need flash_attn,
     but vLLM imports `transformers` during config parsing, triggering (1).

This shim sidesteps both:

  - Pre-injects a dummy `transformers.integrations.flash_attention` module
    so the module-level probe is skipped.
  - Wraps `__import__` so that as soon as `transformers.utils.import_utils`
    is loaded, `_is_package_available("flash_attn")` and
    `_is_package_available("flash_attn_3")` are patched to return False.

Together these make transformers (and downstream vLLM) behave as if
flash_attn isn't installed, falling back to SDPA / Triton attention.

Activated via PYTHONPATH=<this dir>:... and is inherited by all
subprocesses vLLM spawns (EngineCore workers).
"""
import sys
import types
import builtins
import importlib.util as _il


# --- Step 0 : monkey-patch importlib.util.find_spec so any caller probing for
#              flash_attn (vLLM rotary_embedding, transformers _is_package_available,
#              etc.) sees it as "not installed". ---

_BLOCKED_PKGS = ("flash_attn", "flash_attn_3", "flash_attn_2_cuda")
_orig_find_spec = _il.find_spec


def _patched_find_spec(name, package=None):
    if name in _BLOCKED_PKGS or name.startswith("flash_attn."):
        return None
    return _orig_find_spec(name, package)


_il.find_spec = _patched_find_spec


# --- Step 1 : shim transformers.integrations.flash_attention ---

_fake_module_name = "transformers.integrations.flash_attention"
_fake = types.ModuleType(_fake_module_name)


def _shimmed_flash_attention_forward(*args, **kwargs):
    raise RuntimeError(
        "transformers flash_attention_forward was called, but flash_attn is "
        "shimmed out by KFinEval sitecustomize (incompatible ABI). Use a "
        "different attention implementation: sdpa, eager, or TRITON_ATTN."
    )


_fake.flash_attention_forward = _shimmed_flash_attention_forward
_fake._use_top_left_mask = False
sys.modules[_fake_module_name] = _fake


# --- Step 2 : patch transformers.utils.import_utils after it loads ---

_PATCHED_FLAG = "_kfineval_flash_attn_patched"

_original_import = builtins.__import__


def _patched_import(name, *args, **kwargs):
    module = _original_import(name, *args, **kwargs)
    target = sys.modules.get("transformers.utils.import_utils")
    if target is not None and not getattr(target, _PATCHED_FLAG, False):
        if hasattr(target, "_is_package_available"):
            _orig = target._is_package_available

            def _patched_is_pkg_available(pkg_name, return_version=False):
                if pkg_name in _BLOCKED_PKGS:
                    return (False, "N/A") if return_version else False
                return _orig(pkg_name, return_version)

            target._is_package_available = _patched_is_pkg_available
            setattr(target, _PATCHED_FLAG, True)
    return module


builtins.__import__ = _patched_import
