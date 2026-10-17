"""直播用 Index 翻译器：独立模型、端口，只有已发生的上下文。"""
from __future__ import annotations

import json
import re
import urllib.request

from .hy_mt_translator import HyMtTranslator, _context_output_suspect


class IndexTranslator(HyMtTranslator):
    def __init__(self, model_path="models/Index-Translate-2B.Q4_K_M.gguf", **kwargs):
        super().__init__(model_path=model_path, port=8091, context_size=4096,
                         parallel_slots=1, concurrency=1, timeout=30, **kwargs)
        self.model = "Index-Translate-2B"

    def translate_live(self, text: str, context: list[str]) -> str:
        self._ensure_server()
        background = "\n".join(context[-2:])[-500:]
        source_prompt = ("Translate the following subtitle into Simplified Chinese, "
                         "without additional explanation. Keep proper names intact.\n\n" + text)
        prompt = source_prompt
        if background:
            prompt = (f"[Background Information]\n{background}\n\n"
                      "Use the background only to understand references. "
                      "Do not translate the background or repeat it in your answer.\n" + source_prompt)
        result = self._request_translation(prompt)
        if background and self._suspect(result, text):
            result = self._request_translation(source_prompt)
        if self._suspect(result, text):
            raise RuntimeError("Index 返回了提示词或过长译文，原文已保留")
        return result

    @staticmethod
    def _suspect(result, source):
        return (_context_output_suspect(result, source)
                or bool(re.search(r"当前字幕\s*[:：]|背景（仅供理解）|\[Background Information\]|<think>", result))
                or len(result) > max(80, len(source) * 2))

    def _request_translation(self, prompt):
        body = {"messages": [{"role": "user", "content": prompt}], "temperature": 0,
                "max_tokens": 384, "stream": False,
                "chat_template_kwargs": {"enable_thinking": False}}
        request = urllib.request.Request(self._base_url() + "/v1/chat/completions",
                                         data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read())
        choice = (payload.get("choices") or [{}])[0]
        result = str(choice.get("message", {}).get("content") or "").strip()
        if not result or choice.get("finish_reason") == "length":
            raise RuntimeError("Index 返回空译文或达到输出上限，原文已保留")
        return result
