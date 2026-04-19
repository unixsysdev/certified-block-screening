__version__ = "0.0.1"


def _neutralize_broken_torchvision() -> None:
    """Your local torchvision is pinned to an older torch and raises at import
    time (`operator torchvision::nms does not exist`). Transformers 5.x probes
    torchvision as an optional dep — if the probe raises, so does Qwen3 loading.

    This helper leaves a clean torchvision alone, but if it's broken, stubs it
    with None in sys.modules so transformers sees it as "not available" and
    cleanly skips its vision features. No packages are installed or removed.
    """
    import importlib.util
    import sys

    if sys.modules.get("torchvision") not in (None, ...):
        # Either already imported cleanly, or absent. Nothing to do.
        if "torchvision" in sys.modules:
            return
    if importlib.util.find_spec("torchvision") is None:
        return  # not installed: transformers already handles that case

    try:
        import torchvision  # noqa: F401  # test if it imports cleanly
    except Exception:
        # Purge any half-registered submodules, then mark as unavailable.
        for k in list(sys.modules):
            if k == "torchvision" or k.startswith("torchvision."):
                del sys.modules[k]
        sys.modules["torchvision"] = None  # type: ignore[assignment]


_neutralize_broken_torchvision()
del _neutralize_broken_torchvision
