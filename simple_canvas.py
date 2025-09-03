# simple_canvas.py
# Minimal callable Canvas with DEFINE / REVISE / READ / LIST / DELETE.
# Persistence: append-only JSONL at ./canvas.log (configurable).
# Blobs: any JSON-serializable object.

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4
from datetime import datetime, timezone
import json, os, threading

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

@dataclass(frozen=True)
class Version:
    version_id: str
    created_at: str
    delta: Optional[str]
    blob: Any
    meta: Dict[str, Any]
    size_bytes: int

@dataclass
class ModuleMeta:
    module_id: str
    sig: Optional[str]
    doc: Optional[str]
    tags: List[str]
    created_at: str
    updated_at: str
    last_version_id: str
    version_count: int

class Canvas:
    def __init__(self, log_path: str = "./canvas.log"):
        self.log_path = log_path
        self._lock = threading.RLock()
        self._modules: Dict[str, ModuleMeta] = {}
        self._versions: Dict[str, List[Version]] = {}
        if os.path.exists(self.log_path):
            self._restore()

    # ---------- Public API (call these directly) ----------
    def DEFINE(self, *, sig: Optional[str], doc: Optional[str], blob: Any,
               tags: Optional[List[str]] = None, meta: Optional[Dict[str, Any]] = None
               ) -> Tuple[str, str]:
        """Create a new module with an initial version (immutable)."""
        with self._lock:
            module_id = uuid4().hex
            version_id = uuid4().hex
            now = _now_iso()
            v = Version(
                version_id=version_id,
                created_at=now,
                delta="initial DEFINE",
                blob=blob,
                meta=meta or {},
                size_bytes=_sizeof(blob),
            )
            m = ModuleMeta(
                module_id=module_id, sig=sig, doc=doc, tags=tags or [],
                created_at=now, updated_at=now, last_version_id=version_id,
                version_count=1
            )
            self._modules[module_id] = m
            self._versions[module_id] = [v]
            self._append_log({"op":"DEFINE","module":asdict(m),"version":_v2dict(v)})
            return module_id, version_id

    def REVISE(self, *, module_id: str, blob: Any,
               delta: Optional[str] = None, meta: Optional[Dict[str, Any]] = None
               ) -> str:
        """Add a new version to an existing module."""
        with self._lock:
            m = self._require_module(module_id)
            version_id = uuid4().hex
            now = _now_iso()
            v = Version(
                version_id=version_id,
                created_at=now,
                delta=delta or "revision",
                blob=blob,
                meta=meta or {},
                size_bytes=_sizeof(blob),
            )
            self._versions[module_id].append(v)
            m.updated_at = now
            m.last_version_id = version_id
            m.version_count += 1
            self._append_log({"op":"REVISE","module_id":module_id,"version":_v2dict(v)})
            return version_id

    def READ(self, *, module_id: str, version_id: Optional[str] = None) -> Dict[str, Any]:
        """Return module meta and versions (blob included for requested version or 'latest')."""
        with self._lock:
            m = self._require_module(module_id)
            versions = self._versions[module_id]
            out_versions = []
            include_vid = None
            if version_id == "latest" or version_id is None:
                include_vid = m.last_version_id
            else:
                include_vid = version_id
            for v in versions:
                entry = {
                    "version_id": v.version_id,
                    "created_at": v.created_at,
                    "delta": v.delta,
                    "size_bytes": v.size_bytes,
                    "meta": v.meta,
                }
                if v.version_id == include_vid:
                    entry["blob"] = v.blob
                out_versions.append(entry)
            return {
                "module_id": m.module_id,
                "sig": m.sig, "doc": m.doc, "tags": m.tags,
                "created_at": m.created_at, "updated_at": m.updated_at,
                "last_version_id": m.last_version_id,
                "version_count": m.version_count,
                "versions": out_versions,
            }

    def LIST(self, *, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return brief cards for modules, most-recent first."""
        with self._lock:
            items = []
            for m in self._modules.values():
                items.append({
                    "module_id": m.module_id,
                    "sig": m.sig, "doc": m.doc, "tags": m.tags,
                    "created_at": m.created_at, "updated_at": m.updated_at,
                    "last_version_id": m.last_version_id,
                    "version_count": m.version_count,
                })
            items.sort(key=lambda x: x["updated_at"], reverse=True)
            return items[:top_k] if top_k else items

    def DELETE(self, *, module_id: str) -> None:
        """Remove a module and its versions (does not rewrite history in the log)."""
        with self._lock:
            self._require_module(module_id)
            del self._modules[module_id]
            del self._versions[module_id]
            self._append_log({"op":"DELETE","module_id":module_id})

    # ---------- Optional: tool specs for “function calling” LLMs ----------
    @staticmethod
    def tool_specs() -> List[Dict[str, Any]]:
        """Return JSON schemas describing the callable tools (OpenAI/Anthropic-style)."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "DEFINE",
                    "description": "Create a new canvas module with an initial blob.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sig": {"type":"string","nullable":True, "description":"Human-readable signature/name"},
                            "doc": {"type":"string","nullable":True, "description":"Short description"},
                            "blob": {"description":"Arbitrary JSON-serializable content"},
                            "tags": {"type":"array","items":{"type":"string"}, "nullable":True},
                            "meta": {"type":"object","nullable":True}
                        },
                        "required": ["blob"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "REVISE",
                    "description": "Add a new version to an existing module (immutable lineage).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "module_id": {"type":"string"},
                            "blob": {"description":"New JSON-serializable blob"},
                            "delta": {"type":"string","nullable":True},
                            "meta": {"type":"object","nullable":True}
                        },
                        "required": ["module_id","blob"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "READ",
                    "description": "Read module meta and versions; include blob for a specific version or 'latest'.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "module_id": {"type":"string"},
                            "version_id": {"type":"string","nullable":True, "description":"Specific version_id or 'latest' (default: latest)"}
                        },
                        "required": ["module_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "LIST",
                    "description": "List module cards, most recent first.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "top_k": {"type":"integer","nullable":True, "minimum":1, "maximum":100}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "DELETE",
                    "description": "Delete a module and its versions.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "module_id": {"type":"string"}
                        },
                        "required": ["module_id"]
                    }
                }
            }
        ]

    # ---------- Internals ----------
    def _append_log(self, rec: Dict[str, Any]) -> None:
        rec["_ts"] = _now_iso()
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _restore(self) -> None:
        with self._lock, open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                rec = json.loads(line)
                op = rec.get("op")
                if op == "DEFINE":
                    m = ModuleMeta(**rec["module"])
                    v = _dict2v(rec["version"])
                    self._modules[m.module_id] = m
                    self._versions[m.module_id] = [v]
                elif op == "REVISE":
                    mid = rec["module_id"]
                    if mid in self._modules:
                        v = _dict2v(rec["version"])
                        self._versions[mid].append(v)
                        m = self._modules[mid]
                        m.updated_at = v.created_at
                        m.last_version_id = v.version_id
                        m.version_count += 1
                elif op == "DELETE":
                    mid = rec["module_id"]
                    self._modules.pop(mid, None)
                    self._versions.pop(mid, None)

    def _require_module(self, module_id: str) -> ModuleMeta:
        m = self._modules.get(module_id)
        if not m:
            raise KeyError(f"module_id not found: {module_id}")
        return m

def _sizeof(obj: Any) -> int:
    try:
        return len(json.dumps(obj, ensure_ascii=False).encode("utf-8"))
    except Exception:
        return -1

def _v2dict(v: Version) -> Dict[str, Any]:
    d = asdict(v)
    return d

def _dict2v(d: Dict[str, Any]) -> Version:
    return Version(**d)

