# canvas_api.py
"""
Canvas API.

These functions expose a minimal persistent canvas for proposing,
reading, and listing versioned modules. Scoring is system-controlled
and used to drive selection under learning.
"""

from typing import Any, Dict, List, Optional
from simple_canvas import Canvas

_canvas = Canvas()

def _err(msg: str) -> Dict[str, Any]:
    return {"success": False, "error": msg}

def PROPOSE(*, sig: str = "", doc: str = "", blob: str = "") -> Dict[str, Any]:
    """
    Propose a new module or a new version of an existing module.

    This creates a new version under the given signature. Versions are
    stored persistently and later selected based on score.

    Args:
        sig: Stable identifier for the module (e.g., function name).
        doc: Human-readable documentation describing the module.
        blob: The module body or content (code, text, proof, etc.).

    Returns:
        A dictionary with at least:
            - success: bool indicating whether the proposal was stored
    """
    if not isinstance(sig, str) or not sig.strip():
        return _err("PROPOSE: `sig` must be a non-empty string")
    if not isinstance(doc, str):
        doc = str(doc)
    if not isinstance(blob, str):
        blob = str(blob)
    try:
        return _canvas.PROPOSE(sig=sig.strip(), doc=doc, blob=blob)
    except Exception as e:
        return _err(f"PROPOSE failed: {e}")

def READ(*, sig: Optional[str] = None) -> Dict[str, Any]:
    """
    Read the currently winning version of a module.

    The winning version is selected by highest score, with ties broken
    by randomized neutral drift.

    Args:
        sig: Module signature to retrieve.

    Returns:
        A dictionary containing:
            - success: bool
            - data: dictionary with 
                - sig: The module signature.
                - doc: Documentation string for the version.
                - blob: The module body/content.
    """
    if sig is None:
        return _err("READ: missing `sig`")
    if not isinstance(sig, str) or not sig.strip():
        return _err("READ: `sig` must be a non-empty string")
    try:
        return _canvas.READ(sig=sig.strip())
    except Exception as e:
        return _err(f"READ failed: {e}")


def LIST(*, top_k: int = 8) -> Dict[str, Any]:
    """
    List module signatures ordered by their current winning score.

    Only the highest-scoring version of each module is considered.
    Optionally restricts the output to the top-k modules.

    Args:
        top_k: Maximum number of modules to return.

    Returns:
        A dictionary containing:
            - success: bool
            - data: list of dictionaries, each with:
                - sig: Module signature
                - doc: Documentation of the winning version
    """
    try:
        # allow missing/None, strings, etc.
        if top_k is None:
            top_k = 8
        top_k = int(top_k)
        if top_k <= 0:
            top_k = 8
        if top_k > 64:
            top_k = 64
        return _canvas.LIST(top_k=top_k)
    except Exception as e:
        return _err(f"LIST failed: {e}")


def update_score(*, sig: str, blob: str, delta: float) -> Dict[str, Any]:
    """
    System-only hook for applying selection pressure.

    This must never be exposed to the model policy.
    Called by verifier / RL loop after task evaluation.
    """
    return _canvas.update_score(sig=sig, blob=blob, delta=delta)
