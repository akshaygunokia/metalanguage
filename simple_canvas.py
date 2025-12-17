# simple_canvas.py
# Minimal callable Canvas with PROPOSE / READ / LIST.
# TODO: Variant-only reward by design. Revision behavior is implicitly
#       rewarded via the action chain (LIST -> READ -> PROPOSE/REVISE).
#       Consider adding explicit lineage back-pay or revision bonuses
#       if long repair chains or fork inflation become problematic.

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4
import threading
import random
import json, os
from pathlib import Path
import hashlib

def _sig_id(sig: str) -> str:
    return hashlib.sha256(sig.encode("utf-8")).hexdigest()[:24]  # short but safe

def _ensure(p: Path):
    p.mkdir(parents=True, exist_ok=True)

@dataclass
class Version:
    version_id: str
    blob: str
    score: float
    doc: str

@dataclass
class Module:
    sig: str
    versions: List[Version]

class Canvas:
    def __init__(self, root: str = "canvas"):
        self._lock = threading.RLock()
        self._modules: Dict[str, Module] = {}
        self._root = Path(root)
        self.load()

    def _mod_dir(self, sig: str) -> Path:
        return self._root / "modules" / _sig_id(sig)

    def _save_module_meta(self, sig: str):
        _ensure(self._mod_dir(sig))
        with open(self._mod_dir(sig) / "module.json", "w") as f:
            json.dump({"sig": sig}, f, indent=2, ensure_ascii=False)

    def _ver_dir(self, sig: str) -> Path:
        return self._mod_dir(sig) / "versions"

    def _ver_path(self, sig: str, vid: str) -> Path:
        return self._ver_dir(sig) / f"{vid}.json"

    def load(self):
        self._modules.clear()
        mods = self._root / "modules"
        if not mods.exists():
            return

        for md in mods.iterdir():
            if not md.is_dir():
                continue

            meta = md / "module.json"
            if not meta.exists():
                continue
            with open(meta) as f:
                sig = json.load(f)["sig"]
            versions = []
            vdir = md / "versions"
            if vdir.exists():
                for vf in vdir.glob("*.json"):
                    with open(vf) as f:
                        d = json.load(f)
                    versions.append(Version(
                        version_id=d["version_id"],
                        doc=d["doc"],
                        blob=d["blob"],
                        score=d.get("score", 0.0),
                    ))

            if versions:
                self._modules[sig] = Module(sig=sig, versions=versions)

    def _save_version(self, sig: str, v: Version):
        _ensure(self._ver_dir(sig))
        with open(self._ver_path(sig, v.version_id), "w") as f:
            json.dump({
                "version_id": v.version_id,
                "doc": v.doc,
                "blob": v.blob,
                "score": v.score,
            }, f, indent=2, ensure_ascii=False)

    def _pick_winning_version(self, module: Module) -> Optional[Version]:
        if not module.versions:
            return None
    
        max_score = max(v.score for v in module.versions)
        top = [v for v in module.versions if v.score == max_score]
        return random.choice(top)

    def PROPOSE(self, *, sig: str, doc: str, blob: str) -> Dict[str, Any]:
        with self._lock:
            try:
                version_id = uuid4().hex
                v = Version(version_id=version_id, doc=doc, blob=blob, score=0.0)
                if sig in self._modules:
                    m = self._modules[sig]
                    m.versions.append(v)
                else:
                    m = Module(sig=sig, versions=[v])
                self._modules[sig] = m
                self._save_module_meta(sig)
                self._save_version(sig, v)
                return {"success": True}
            except Exception as e:
                return {"success": False, "error": f"PROPOSE failed: {e}"}

    def READ(self, *, sig: str) -> Dict[str, Any]:
        with self._lock:
            m = self._modules.get(sig)
            if m is None or not m.versions:
                return {"success": False, "error": f"sig not found: {sig}"}
            version = self._pick_winning_version(m)
            if version is None:
                return {"success": False, "error": f"no versions for sig: {sig}"}
            return {"success": True, "data": {"sig": m.sig, "doc": version.doc, "blob": version.blob}}

    def LIST(self, *, top_k: Optional[int] = None) -> Dict[str, Any]:
        with self._lock:
            tmp: List[Tuple[float, str, str]] = []  # (score, sig, doc)
    
            for sig, m in self._modules.items():
                v = self._pick_winning_version(m)
                if v is None:
                    continue
                tmp.append((v.score, m.sig, v.doc))
    
            tmp.sort(key=lambda t: t[0], reverse=True)
    
            items = [{"sig": sig, "doc": doc} for (_, sig, doc) in tmp]
            if top_k:
                items = items[:top_k]
    
            return {"success": True, "data": items}

    def update_score(self, *, sig: str, version_id: str, delta: float) -> Dict[str, Any]:
        with self._lock:
            m = self._modules.get(sig)
            if m is None:
                return {"success": False, "error": f"sig not found: {sig}"}
    
            for v in m.versions:
                if v.version_id == version_id:
                    try:
                        v.score += delta
                        self._save_version(sig, v)
                        return {"success": True}
                    except Exception as e:
                        return {"success": False, "error": f"update_score failed: {e}"}
    
            return {"success": False, "error": f"version_id not found for sig={sig}"}

