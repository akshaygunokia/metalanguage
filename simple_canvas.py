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
import time
from pathlib import Path
import hashlib
from contextlib import contextmanager
import fcntl

def _sig_id(sig: str) -> str:
    return hashlib.sha256(sig.encode("utf-8")).hexdigest()[:24]  # short but safe

def _ver_id(blob: str) -> str:
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]  # short but safe

def _ensure(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _atomic_json_write(path: Path, obj: Dict[str, Any]):
    """
    Atomic write: write to temp file in same directory, then os.replace().
    Ensures readers never observe partially-written JSON.
    """
    _ensure(path.parent)
    tmp = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

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
        _ensure(self._root)
        self._log_path = self._root / "canvas.log"
        self._lock_path = self._root / ".lock"
        self.load()

    def _log(self, msg: str):
        try:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(f"{ts} {msg}\n")
        except Exception:
            pass

    @contextmanager
    def _fs_lock(self, *, shared: bool):
        """
        Cross-process lock for multi-GPU / multi-process training.
        Uses flock() on a single global lock file under the canvas root.
        """
        if fcntl is None:
            # Best-effort fallback: no cross-process locking available.
            yield
            return
        _ensure(self._root)
        with open(self._lock_path, "a+") as lf:
            try:
                fcntl.flock(lf.fileno(), fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
                yield
            finally:
                try:
                    fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass

    def _mod_dir(self, sig: str) -> Path:
        return self._root / "modules" / _sig_id(sig)

    def _save_module_meta(self, sig: str):
        meta_path = self._mod_dir(sig) / "module.json"
        _atomic_json_write(meta_path, {"sig": sig})

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
        path = self._ver_path(sig, v.version_id)
        _atomic_json_write(path, {
            "version_id": v.version_id,
            "doc": v.doc,
            "blob": v.blob,
            "score": v.score,
        })

    def _pick_winning_version(self, module: Module) -> Optional[Version]:
        if not module.versions:
            return None
    
        max_score = max(v.score for v in module.versions)
        top = [v for v in module.versions if v.score == max_score]
        return random.choice(top)

    def PROPOSE(self, *, sig: str, doc: str, blob: str) -> Dict[str, Any]:
        with self._lock, self._fs_lock(shared=False):
            try:
                version_id = _ver_id(blob)
                self._log(f"PROPOSE sig={sig} vid={version_id} bytes={len(blob)}")
                v = Version(version_id=version_id, doc=doc, blob=blob, score=0.0)
                if sig in self._modules:
                    m = self._modules[sig]
                    for existing in m.versions:
                        if existing.version_id == version_id:
                            self._log(f"PROPOSE dedupe sig={sig} vid={version_id}")
                            return {"success": True}
                    m.versions.append(v)
                else:
                    m = Module(sig=sig, versions=[v])
                self._modules[sig] = m
                self._save_module_meta(sig)
                self._save_version(sig, v)
                return {"success": True}
            except Exception as e:
                self._log(f"PROPOSE failed sig={sig} err={e}")
                return {"success": False, "error": f"PROPOSE failed: {e}"}

    def READ(self, *, sig: str) -> Dict[str, Any]:
        with self._lock, self._fs_lock(shared=True):
            self.load()
            m = self._modules.get(sig)
            if m is None or not m.versions:
                self._log(f"READ miss sig={sig}")
                return {"success": False, "error": f"sig not found: {sig}"}
            version = self._pick_winning_version(m)
            if version is None:
                self._log(f"READ no_versions sig={sig}")
                return {"success": False, "error": f"no versions for sig: {sig}"}
            self._log(f"READ hit sig={sig} vid={version.version_id} score={version.score}")
            return {"success": True, "data": {"sig": m.sig, "doc": version.doc, "blob": version.blob}}

    def LIST(self, *, top_k: Optional[int] = None) -> Dict[str, Any]:
        with self._lock, self._fs_lock(shared=True):
            self.load()
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
            self._log(f"LIST n={len(items)} top_k={top_k}")
            return {"success": True, "data": items}

    def update_score(self, *, sig: str, blob: str, delta: float) -> Dict[str, Any]:
        with self._lock, self._fs_lock(shared=False):
            self.load()
            m = self._modules.get(sig)
            if m is None:
                return {"success": False, "error": f"sig not found: {sig}"}
    
            for v in m.versions:
                if v.version_id == _ver_id(blob):
                    try:
                        v.score += delta
                        self._save_version(sig, v)
                        return {"success": True}
                    except Exception as e:
                        return {"success": False, "error": f"update_score failed: {e}"}
    
            return {"success": False, "error": f"version_id not found for sig={sig}"}

