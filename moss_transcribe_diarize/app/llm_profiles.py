# -*- coding: utf-8 -*-
"""LLM API 配置管理（ccswitch 式多 profile 切换）。

存储: {config_dir}/llm_profiles.json
结构: {"active_id": "...", "profiles": [{"id", "name", "base_url", "api_key", "model", "provider", "disable_thinking"}]}
"""
from __future__ import annotations

import json
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any


def mask_api_key(key: str | None) -> str:
    key = str(key or "")
    if not key:
        return ""
    if len(key) <= 10:
        return key[:3] + "****"
    return key[:5] + "****" + key[-4:]


class LlmProfileStore:
    def __init__(self, config_dir: str | Path):
        self.path = Path(config_dir) / "llm_profiles.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"active_id": None, "profiles": []}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {"active_id": None, "profiles": []}
        if not isinstance(data, dict):
            return {"active_id": None, "profiles": []}
        profiles = data.get("profiles")
        if not isinstance(profiles, list):
            profiles = []
        return {"active_id": data.get("active_id"), "profiles": profiles}

    def _save(self, data: dict[str, Any]) -> None:
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _clean(profile: dict[str, Any]) -> dict[str, Any]:
        provider = str(profile.get("provider") or "openai").strip()
        if provider not in {"openai", "ollama"}:
            provider = "openai"
        base_url = str(profile.get("base_url") or "").strip()
        if provider == "openai" and base_url and not re.search(r"/chat/completions$|/v1$", base_url.rstrip("/")):
            base_url = base_url.rstrip("/") + "/v1"
        return {
            "id": str(profile.get("id") or uuid.uuid4().hex[:12]),
            "name": str(profile.get("name") or "未命名").strip()[:40],
            "base_url": base_url,
            "api_key": str(profile.get("api_key") or "").strip(),
            "model": str(profile.get("model") or "").strip()[:80],
            "provider": provider,
            "disable_thinking": bool(profile.get("disable_thinking")),
            "created_at": float(profile.get("created_at") or time.time()),
        }

    def list_profiles(self, *, include_secrets: bool = False) -> dict[str, Any]:
        with self._lock:
            data = self._load()
        masked = []
        for item in data["profiles"]:
            cleaned = self._clean(item)
            entry = {**cleaned, "api_key_masked": mask_api_key(cleaned["api_key"])}
            # 明文 key 只服务后端内部流程(test connection 等),任何 HTTP 响应都不携带。
            if not include_secrets:
                entry.pop("api_key", None)
            masked.append(entry)
        return {"active_id": data.get("active_id"), "profiles": masked}

    def add_profile(self, payload: dict[str, Any]) -> dict[str, Any]:
        profile = self._clean({**payload, "id": None})
        with self._lock:
            data = self._load()
            data["profiles"].append(profile)
            if data.get("active_id") is None:
                data["active_id"] = profile["id"]
            self._save(data)
        return {
            **profile,
            "api_key_masked": mask_api_key(profile["api_key"]),
            "api_key": None,
        }

    def update_profile(self, profile_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            data = self._load()
            for index, item in enumerate(data["profiles"]):
                if str(item.get("id")) == str(profile_id):
                    merged = {**item, **payload, "id": item["id"]}
                    # api_key 留空 = 保持原值
                    if not str(merged.get("api_key") or "").strip():
                        merged["api_key"] = item.get("api_key")
                    profile = self._clean(merged)
                    profile["created_at"] = item.get("created_at") or profile["created_at"]
                    data["profiles"][index] = profile
                    self._save(data)
                    return {
                        **profile,
                        "api_key_masked": mask_api_key(profile["api_key"]),
                        "api_key": None,
                    }
        raise KeyError(f"Profile not found: {profile_id}")

    def delete_profile(self, profile_id: str) -> None:
        with self._lock:
            data = self._load()
            data["profiles"] = [p for p in data["profiles"] if str(p.get("id")) != str(profile_id)]
            if str(data.get("active_id")) == str(profile_id):
                data["active_id"] = data["profiles"][0]["id"] if data["profiles"] else None
            self._save(data)

    def set_active(self, profile_id: str) -> dict[str, Any]:
        with self._lock:
            data = self._load()
            if not any(str(p.get("id")) == str(profile_id) for p in data["profiles"]):
                raise KeyError(f"Profile not found: {profile_id}")
            data["active_id"] = str(profile_id)
            self._save(data)
        return self.list_profiles()

    def get_active(self) -> dict[str, Any] | None:
        data = self._load()
        active_id = data.get("active_id")
        for item in data["profiles"]:
            if str(item.get("id")) == str(active_id):
                return self._clean(item)
        return None
