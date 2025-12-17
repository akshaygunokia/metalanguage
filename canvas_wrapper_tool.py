# canvas_api.py
"""
Public Canvas API.

These functions expose a minimal persistent canvas for proposing,
reading, and listing versioned modules. Scoring is system-controlled
and used to drive selection under learning.
"""

from typing import Any, Dict, List, Optional
from simple_canvas import Canvas

_canvas = Canvas()


def PROPOSE(*, sig: str, doc: str, blob: str) -> str:
    """
    Propose a new module or a new version of an existing module.

    This creates a new version under the given signature. Versions are
    stored persistently and later selected based on score.

    Args:
        sig: Stable identifier for the module (e.g., function name).
        doc: Human-readable documentation describing the module.
        blob: The module body or content (code, text, proof, etc.).

    Returns:
        The module signature under which the version was stored.
    """
    return _canvas.PROPOSE(sig=sig, doc=doc, blob=blob)


def READ(*, sig: str) -> Dict[str, Any]:
    """
    Read the currently winning version of a module.

    The winning version is selected by highest score, with ties broken
    by randomized neutral drift.

    Args:
        sig: Module signature to retrieve.

    Returns:
        A dictionary containing:
            - sig: The module signature.
            - vid: The selected version identifier.
            - doc: Documentation string for the version.
            - blob: The module body/content.
    """
    return _canvas.READ(sig=sig)


def LIST(*, top_k: int) -> List[Dict[str, Any]]:
    """
    List module signatures ordered by their current winning score.

    Only the highest-scoring version of each module is considered.
    Optionally restricts the output to the top-k modules.

    Args:
        top_k: Optional maximum number of modules to return.

    Returns:
        A list of dictionaries, each containing:
            - sig: Module signature.
            - doc: Documentation of the current winning version.
    """
    return _canvas.LIST(top_k=top_k)
