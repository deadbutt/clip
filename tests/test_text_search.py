import unittest

from moss_transcribe_diarize.app.text_search import (
    apply_replacements,
    search_segment_texts,
    tokenize,
)


def _segments(*texts: str) -> list[dict]:
    return [{"id": f"s{i}", "text": text} for i, text in enumerate(texts)]


class TokenizeTest(unittest.TestCase):
    def test_cjk_chars_map_to_toneless_pinyin(self):
        tokens = tokenize("纽罗")
        self.assertEqual([t.text for t in tokens], ["niu", "luo"])
        # 偏移必须指向原文字符
        self.assertEqual([(t.start, t.end) for t in tokens], [(0, 1), (1, 2)])

    def test_latin_chars_lowercase_and_keep_offsets(self):
        tokens = tokenize("aB c")
        self.assertEqual([t.text for t in tokens], ["a", "b", " ", "c"])
        self.assertEqual([(t.start, t.end) for t in tokens], [(0, 1), (1, 2), (2, 3), (3, 4)])


class SearchTest(unittest.TestCase):
    def test_literal_search_is_case_insensitive_with_offsets(self):
        matches = search_segment_texts(_segments("Hello NEURO world"), "neuro")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].char_start, 6)
        self.assertEqual(matches[0].char_end, 11)

    def test_pinyin_search_finds_homophone_typos(self):
        # "纽罗" 是 "Neuro" 的近音错听；同音模式按拼音 niu/luo… 命中
        segments = _segments("欢迎来到纽罗的直播间", "neuro 是正确写法")
        matches = search_segment_texts(segments, "纽罗", mode="pinyin")
        self.assertEqual([m.segment_id for m in matches], ["s0"])

    def test_pinyin_search_latin_query_matches_by_letters(self):
        segments = _segments("这是 Neuro 的频道")
        matches = search_segment_texts(segments, "纽罗", mode="pinyin")
        # "Neuro" 拼出的 n-e-u-r-o 与 "纽罗" 的 niu-luo 不同，不应误报
        self.assertEqual(matches, [])

    def test_pinyin_search_matches_do_not_overlap(self):
        # 重叠匹配会让批量替换按区间回写时损坏文本
        matches = search_segment_texts(_segments("哈哈哈哈"), "哈哈", mode="pinyin")
        self.assertEqual([(m.char_start, m.char_end) for m in matches], [(0, 2), (2, 4)])

    def test_pinyin_replace_repeated_pattern(self):
        segments = _segments("哈哈哈哈")
        matches = search_segment_texts(segments, "哈哈", mode="pinyin")
        ranges = [(m.char_start, m.char_end) for m in matches]
        self.assertEqual(apply_replacements("哈哈哈哈", ranges, "嘿"), "嘿嘿")

    def test_no_match_returns_empty(self):
        self.assertEqual(search_segment_texts(_segments("你好"), "world"), [])
        self.assertEqual(search_segment_texts(_segments("你好"), "  "), [])


class ReplaceTest(unittest.TestCase):
    def test_apply_replacements_right_to_left(self):
        self.assertEqual(apply_replacements("abcabc", [(0, 3), (3, 6)], "x"), "xx")

    def test_apply_replacements_empty_replacement_deletes(self):
        self.assertEqual(apply_replacements("a纽罗b", [(1, 3)], ""), "ab")


if __name__ == "__main__":
    unittest.main()
