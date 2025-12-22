# canvas_tool.py
from __future__ import annotations

from typing import Any, Dict, Optional
from uuid import uuid4
import threading

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op
import json
import contextlib
import io
# Import the canvas API functions (assumed to be available)
from simple_canvas import Canvas
_canvas = Canvas()

class CanvasTool(BaseTool):
    """
    verl tool wrapper around your canvas_api.

    Model-visible ops:
      - LIST(top_k)
      - READ(sig)
      - PROPOSE(sig, doc, blob)

    System-only scoring must be done outside the tool (your reward/verifier loop),
    e.g. by calling canvas_api.update_score(...) on used modules.
    """

    _lock = threading.Lock()  # helps if multiple threads call into a shared in-proc canvas

    def __init__(self, config: dict, tool_schema: Optional[OpenAIFunctionToolSchema] = None):
        with contextlib.redirect_stdout(io.StringIO()):
            super().__init__(config=config, tool_schema=tool_schema or self.get_openai_tool_schema())
        self._instance_dict: Dict[str, Dict[str, Any]] = {}

        # shaping rewards (optional)
        self.bad_call_penalty = float(config.get("bad_call_penalty", -0.05))
        self.good_call_reward = float(config.get("good_call_reward", 0.05))  # keep 0.0 if you only want penalties

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return OpenAIFunctionToolSchema.model_validate(
            {
                "type": "function",
                "function": {
                    "name": "canvas_tool",
                    "description": (
                        "Persistent versioned module canvas. Use LIST to see available module signatures, "
                        "READ to fetch the current winning version of a signature, "
                        "and PROPOSE to add a new module version."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "op": {
                                "type": "string",
                                "enum": ["LIST", "READ", "PROPOSE"],
                                "description": "Canvas operation to perform.",
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "For LIST: number of modules to return (default 8).",
                            },
                            "sig": {
                                "type": "string",
                                "description": "Required for READ/PROPOSE: stable identifier for the module",
                            },
                            "doc": {
                                "type": "string",
                                "description": "Required for PROPOSE: documentation string.",
                            },
                            "blob": {
                                "type": "string",
                                "description": "Required for PROPOSE: module body/content (code/text/proof/etc.).",
                            },
                        },
                        "required": ["op"],
                    },
                },
            }
        )

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """
        Create a per-trajectory tool instance.

        We keep per-trajectory bookkeeping here (e.g., what sigs were read/proposed)
        but the underlying canvas is persistent globally (your canvas_api uses a global Canvas()).
        """
        if instance_id is None:
            instance_id = str(uuid4())

        self._instance_dict[instance_id] = {
            "reads": [],
            "proposes": [],
            "lists": 0,
        }
        return instance_id, ToolResponse()
    
    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        op = parameters.get("op")
        if not op or not isinstance(op, str):
            return ToolResponse(text="Error: missing string parameter `op`."), self.bad_call_penalty, {"success": False}

        op = op.upper().strip()
        metrics: Dict[str, Any] = {"success": False, "op": op}

        # Ensure instance exists (verl usually calls create(), but be robust)
        if instance_id not in self._instance_dict:
            self._instance_dict[instance_id] = {"reads": [], "proposes": [], "lists": 0}

        try:
            with self._lock:
                if op == "LIST":
                    top_k = parameters.get("top_k", 8)
                    res = _canvas.LIST(top_k=top_k)
                    self._instance_dict[instance_id]["lists"] += 1
                    metrics["success"] = bool(res.get("success", False))
                    reward = self.good_call_reward if res.get("success") else self.bad_call_penalty
                    return ToolResponse(text=_as_pretty_json(res)), reward, metrics

                if op == "READ":
                    sig = parameters.get("sig", None)
                    if sig is None:
                        return ToolResponse(text="Error: 'sig' parameter required for READ."
                                            ), self.bad_call_penalty, {"success": False}
                    res = _canvas.READ(sig=sig)
                    if res.get("success"):
                        self._instance_dict[instance_id]["reads"].append(sig)
                    metrics["success"] = bool(res.get("success", False))
                    reward = self.good_call_reward if res.get("success") else self.bad_call_penalty
                    return ToolResponse(text=_as_pretty_json(res)), reward, metrics

                if op == "PROPOSE":
                    sig = parameters.get("sig", None)
                    doc = parameters.get("doc", "")
                    blob = parameters.get("blob", "")
                    if sig is None:
                        return ToolResponse(text="Error: 'sig' parameter required for PROPOSE."
                                            ), self.bad_call_penalty, {"success": False}
                    res = _canvas.PROPOSE(sig=sig, doc=doc, blob=blob)
                    if res.get("success"):
                        self._instance_dict[instance_id]["proposes"].append(sig)
                    metrics["success"] = bool(res.get("success", False))
                    reward = self.good_call_reward if res.get("success") else self.bad_call_penalty
                    return ToolResponse(text=_as_pretty_json(res)), reward, metrics

                else:
                    return ToolResponse(text=f"Error: unsupported operation '{op}'. Use LIST, READ, or PROPOSE."
                                        ), self.bad_call_penalty, {"success": False, "op": op}

        except Exception as e:
            return ToolResponse(
                text=f"CanvasTool internal error: {e}"
            ), self.bad_call_penalty, {"success": False, "op": op}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """
        Optional tool-scoped reward. For v0, keep it 0.0.

        If later you want tool-terminal shaping (e.g., penalize too many proposes),
        you can compute it from self._instance_dict[instance_id].
        """
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instance_dict.pop(instance_id, None)


def _as_pretty_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return str(obj)
