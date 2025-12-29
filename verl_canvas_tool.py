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

class _CanvasBaseTool(BaseTool):
    """Shared init/fields for all canvas tools."""

    def __init__(self, config: dict, tool_schema: Optional[OpenAIFunctionToolSchema] = None):
        with contextlib.redirect_stdout(io.StringIO()):
            super().__init__(config=config, tool_schema=tool_schema or self.get_openai_tool_schema())
        self._instance_dict: Dict[str, Dict[str, Any]] = {}

        # shaping rewards (optional)
        self.bad_call_penalty = float(config.get("bad_call_penalty", 0.0))
        self.good_call_reward = float(config.get("good_call_reward", 0.01))  # keep 0.0 if you only want penalties

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """
        Optional tool-scoped reward. For v0, keep it 0.0.

        If later you want tool-terminal shaping (e.g., penalize too many proposes),
        you can compute it from self._instance_dict[instance_id].
        """
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instance_dict.pop(instance_id, None)


class CanvasListTool(_CanvasBaseTool):
    """Tool: canvas_list(top_k=8)"""

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return OpenAIFunctionToolSchema.model_validate(
            {
                "type": "function",
                "function": {
                    "name": "canvas_list",
                    "description": "List top modules available in the persistent Canvas.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "top_k": {
                                "type": "integer",
                                "description": "Number of module cards to return (default 8).",
                            }
                        },
                        "required": [],
                    },
                },
            }
        )

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {"lists": 0}
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        top_k = parameters.get("top_k", 8)
        try:
            res = _canvas.LIST(top_k=top_k)
            self._instance_dict.setdefault(instance_id, {"lists": 0})["lists"] += 1
            ok = bool(res.get("success", False))
            reward = self.good_call_reward if ok else self.bad_call_penalty
            return ToolResponse(text=_as_pretty_json(res)), reward, {"success": ok, "op": "LIST", "top_k": top_k}
        except Exception as e:
            return ToolResponse(text=f"canvas_list error: {e}"), self.bad_call_penalty, {"success": False, "op": "LIST"}


class CanvasReadTool(_CanvasBaseTool):
    """Tool: canvas_read(sig)"""

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return OpenAIFunctionToolSchema.model_validate(
            {
                "type": "function",
                "function": {
                    "name": "canvas_read",
                    "description": "Read the current winning version of a module signature (sig) from the Canvas.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sig": {"type": "string", "description": "Module signature to read."}
                        },
                        "required": ["sig"],
                    },
                },
            }
        )

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {"reads": []}
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        sig = parameters.get("sig")
        if not isinstance(sig, str) or not sig:
            return ToolResponse(text="Error: 'sig' is required and must be a non-empty string."), self.bad_call_penalty, {
                "success": False,
                "op": "READ",
            }

        try:
            res = _canvas.READ(sig=sig)
            ok = bool(res.get("success", False))
            if ok:
                self._instance_dict.setdefault(instance_id, {"reads": []})["reads"].append(sig)
            reward = self.good_call_reward if ok else self.bad_call_penalty
            return ToolResponse(text=_as_pretty_json(res)), reward, {"success": ok, "op": "READ", "sig": sig}
        except Exception as e:
            return ToolResponse(text=f"canvas_read error: {e}"), self.bad_call_penalty, {"success": False, "op": "READ", "sig": sig}


class CanvasProposeTool(_CanvasBaseTool):
    """Tool: canvas_propose(sig, doc, blob)"""

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return OpenAIFunctionToolSchema.model_validate(
            {
                "type": "function",
                "function": {
                    "name": "canvas_propose",
                    "description": "Propose a new module version under a signature (sig) into the Canvas.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sig": {"type": "string", "description": "Module signature."},
                            "doc": {"type": "string", "description": "Short documentation."},
                            "blob": {"type": "string", "description": "Module content."},
                        },
                        "required": ["sig", "doc", "blob"],
                    },
                },
            }
        )

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {"proposes": []}
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        sig = parameters.get("sig")
        doc = parameters.get("doc")
        blob = parameters.get("blob")

        if not isinstance(sig, str) or not sig:
            return ToolResponse(text="Error: 'sig' is required and must be a non-empty string."), self.bad_call_penalty, {
                "success": False,
                "op": "PROPOSE",
            }
        if not isinstance(doc, str) or not isinstance(blob, str):
            return ToolResponse(text="Error: 'doc' and 'blob' must be strings."), self.bad_call_penalty, {
                "success": False,
                "op": "PROPOSE",
                "sig": sig,
            }

        try:
            res = _canvas.PROPOSE(sig=sig, doc=doc, blob=blob)
            ok = bool(res.get("success", False))
            if ok:
                self._instance_dict.setdefault(instance_id, {"proposes": []})["proposes"].append(sig)
            reward = self.good_call_reward if ok else self.bad_call_penalty
            return ToolResponse(text=_as_pretty_json(res)), reward, {"success": ok, "op": "PROPOSE", "sig": sig}
        except Exception as e:
            return ToolResponse(text=f"canvas_propose error: {e}"), self.bad_call_penalty, {
                "success": False,
                "op": "PROPOSE",
                "sig": sig,
            }

def _as_pretty_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return str(obj)
