"""字幕文本查找：字面模式 + 同音(拼音)模式。

同音模式把文本与查询都转成"逐字符 token 序列"再做连续匹配：
- CJK 字符 → 无声调拼音（多音字取 pypinyin 默认读音）
- 其他字符 → 小写原字符（拉丁字母查询即逐字符匹配，"neuro" 能命中 "Neuro"）

每个 token 记录它在原文字符串中的 [start, end) 偏移，替换时按偏移回写。
"""
from __future__ import annotations

from dataclasses import dataclass

_CJK_RANGES = (
    (0x3400, 0x4DBF),   # CJK 扩展 A
    (0x4E00, 0x9FFF),   # CJK 基本区
    (0xF900, 0xFAFF),   # CJK 兼容表意
    (0x20000, 0x2A6DF), # CJK 扩展 B
)

_pinyin_cache: dict[str, str] = {}
_lazy_pinyin = None
_pinyin_missing = False


def _require_lazy_pinyin():
    global _lazy_pinyin, _pinyin_missing
    if _lazy_pinyin is None and not _pinyin_missing:
        try:
            from pypinyin import lazy_pinyin
            _lazy_pinyin = lazy_pinyin
        except ImportError:
            _pinyin_missing = True
    if _lazy_pinyin is None:
        raise RuntimeError("同音搜索需要 pypinyin 依赖（uv pip install pypinyin）。")
    return _lazy_pinyin


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return any(lo <= code <= hi for lo, hi in _CJK_RANGES)


def _char_token(ch: str) -> str:
    if not _is_cjk(ch):
        return ch.lower()
    token = _pinyin_cache.get(ch)
    if token is None:
        lazy_pinyin = _require_lazy_pinyin()
        try:
            token = lazy_pinyin(ch)[0]
        except Exception:
            token = ch.lower()
        _pinyin_cache[ch] = token
    return token


@dataclass(frozen=True)
class Token:
    start: int
    end: int
    text: str


def tokenize(text: str) -> list[Token]:
    tokens: list[Token] = []
    for offset, ch in enumerate(str(text)):
        token_text = _char_token(ch)
        if not token_text:
            continue
        tokens.append(Token(offset, offset + 1, token_text))
    return tokens


def find_matches(tokens: list[Token], query_tokens: list[Token]) -> list[tuple[int, int]]:
    """返回所有互不重叠的连续匹配的 (token 起下标, token 止下标)（左闭右开）。

    命中后跳过整个匹配跨度（与字面模式步进语义一致）；
    若允许重叠，apply_replacements 按区间回写时会损坏文本。
    """
    matches: list[tuple[int, int]] = []
    if not query_tokens or len(query_tokens) > len(tokens):
        return matches
    first = query_tokens[0].text
    span = len(query_tokens)
    i = 0
    while i <= len(tokens) - span:
        if tokens[i].text == first and all(
            tokens[i + k].text == query_tokens[k].text for k in range(1, span)
        ):
            matches.append((i, i + span))
            i += span
        else:
            i += 1
    return matches


@dataclass(frozen=True)
class TextMatch:
    segment_id: str
    index: int
    char_start: int
    char_end: int
    snippet: str


def search_segment_texts(
    segments: list[dict],
    query: str,
    *,
    mode: str = "literal",
    limit: int = 500,
) -> list[TextMatch]:
    """在段落文本里找 query。mode: literal=子串, pinyin=同音序列。

    返回按段落顺序排列的匹配（含段内字符偏移与上下文片段），最多 limit 条。
    """
    query = str(query)
    if not query.strip():
        return []
    matches: list[TextMatch] = []
    use_pinyin = mode == "pinyin"
    if use_pinyin:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
    for index, segment in enumerate(segments):
        text = str(segment.get("text") or "")
        if use_pinyin:
            tokens = tokenize(text)
            for token_start, token_end in find_matches(tokens, query_tokens):
                char_start = tokens[token_start].start
                char_end = tokens[token_end - 1].end
                matches.append(TextMatch(str(segment.get("id")), index, char_start, char_end, _snippet(text, char_start, char_end)))
        else:
            cursor = 0
            lowered = text.lower()
            needle = query.lower()
            while len(matches) < limit:
                found = lowered.find(needle, cursor)
                if found < 0:
                    break
                matches.append(TextMatch(str(segment.get("id")), index, found, found + len(needle), _snippet(text, found, found + len(query))))
                cursor = found + max(1, len(needle))
        if len(matches) >= limit:
            break
    return matches[:limit]


def _snippet(text: str, start: int, end: int, width: int = 16) -> str:
    left = max(0, start - width)
    right = min(len(text), end + width)
    prefix = "…" if left > 0 else ""
    suffix = "…" if right < len(text) else ""
    return prefix + text[left:start] + "[" + text[start:end] + "]" + text[end:right] + suffix


def apply_replacements(text: str, ranges: list[tuple[int, int]], replacement: str) -> str:
    """把 [(start, end), ...] 区间按从右到左的顺序替换为 replacement。"""
    result = str(text)
    for start, end in sorted(ranges, reverse=True):
        result = result[:start] + replacement + result[end:]
    return result
