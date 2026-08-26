from __future__ import annotations

import unittest

from moss_transcribe_diarize.subtitle import (
    SubtitleSegment,
    coerce_subtitle_segments,
    normalize_segments,
    regroup_sentences,
    regroup_sentences_from_words,
    subtitle_segments_from_transcript,
)


class SubtitlePostprocessTest(unittest.TestCase):
    def test_builds_subtitle_segments_from_transcript(self):
        segments = subtitle_segments_from_transcript("[0][S01]你好[1.5][2][S02]开始[3.5]")

        self.assertEqual([segment.speaker for segment in segments], ["S01", "S02"])
        self.assertEqual([segment.text for segment in segments], ["你好", "开始"])
        self.assertEqual(segments[0].id, "seg_0001")

    def test_can_build_raw_subtitle_segments_without_postprocess(self):
        segments = subtitle_segments_from_transcript(
            "[0][S01]短[0.4][0.2][S01]重叠但保留[0.8]",
            postprocess=False,
        )

        self.assertEqual([(s.start, s.end, s.text) for s in segments], [(0.0, 0.4, "短"), (0.2, 0.8, "重叠但保留")])

    def test_coerce_payload_does_not_reorder_or_fix_times(self):
        segments = coerce_subtitle_segments(
            [
                {"id": "b", "start": 3, "end": 2, "speaker": "S02", "text": ""},
                {"id": "a", "start": 0.2, "end": 0.1, "speaker": "S01", "text": "x"},
            ]
        )

        self.assertEqual([segment.id for segment in segments], ["b", "a"])
        self.assertEqual([(segment.start, segment.end) for segment in segments], [(3.0, 2.0), (0.2, 0.1)])
        self.assertEqual(segments[0].text, "")

    def test_merges_adjacent_same_speaker_short_gap(self):
        segments = normalize_segments(
            [
                SubtitleSegment("a", 0, 1.2, "S01", "你好"),
                SubtitleSegment("b", 1.3, 2.4, "S01", "世界"),
            ],
            merge_gap=0.3,
            max_chars=24,
        )

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].text, "你好世界")
        self.assertEqual(segments[0].end, 2.4)

    def test_fixes_overlaps_and_min_duration(self):
        segments = normalize_segments(
            [
                SubtitleSegment("a", 0, 0.4, "S01", "a"),
                SubtitleSegment("b", 0.2, 0.6, "S02", "b"),
            ],
            min_duration=1.0,
            merge_gap=0,
        )

        self.assertEqual([(s.start, s.end) for s in segments], [(0.0, 1.0), (1.0, 2.0)])

    def test_splits_long_segments(self):
        segments = normalize_segments(
            [
                SubtitleSegment("a", 0, 12, "S01", "第一句很长，需要切开。第二句也很长，需要继续切开。"),
            ],
            max_duration=6.0,
            max_chars=10,
            merge_gap=0,
        )

        self.assertGreater(len(segments), 1)
        self.assertTrue(all(segment.end > segment.start for segment in segments))
        self.assertEqual(segments[0].start, 0.0)


class RegroupSentencesTest(unittest.TestCase):
    def test_merges_fragments_without_sentence_final_punct(self):
        segments = regroup_sentences(
            [
                SubtitleSegment("a", 0, 2.0, "S01", "我们今天要讨论的"),
                SubtitleSegment("b", 2.1, 4.0, "S01", "是人工智能的"),
                SubtitleSegment("c", 4.2, 6.0, "S01", "未来发展方向。"),
            ]
        )

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].text, "我们今天要讨论的是人工智能的未来发展方向。")
        self.assertEqual(segments[0].start, 0.0)
        self.assertEqual(segments[0].end, 6.0)

    def test_does_not_merge_after_sentence_final_punct(self):
        segments = regroup_sentences(
            [
                SubtitleSegment("a", 0, 2.0, "S01", "第一句结束了。"),
                SubtitleSegment("b", 2.2, 4.0, "S01", "第二句开始了。"),
            ]
        )

        self.assertEqual([segment.text for segment in segments], ["第一句结束了。", "第二句开始了。"])

    def test_does_not_merge_across_large_gap(self):
        segments = regroup_sentences(
            [
                SubtitleSegment("a", 0, 2.0, "S01", "前半句还没有说完"),
                SubtitleSegment("b", 6.0, 8.0, "S01", "后半句"),
            ]
        )

        self.assertEqual(len(segments), 2)

    def test_respects_sentence_char_cap(self):
        segments = regroup_sentences(
            [
                SubtitleSegment("a", 0, 4.0, "S01", "这一句特别长特别长特别长特别长特别长特别长特别长，"),
                SubtitleSegment("b", 4.1, 8.0, "S01", "后面还有内容继续接上来。"),
            ],
            max_sentence_chars=20,
        )

        # 合并后会超过 max_sentence_chars(20)，保持两个片段
        self.assertEqual(len(segments), 2)

    def test_merges_overlapping_vad_segments(self):
        segments = regroup_sentences(
            [
                SubtitleSegment("a", 0, 2.2, "S01", "重叠的片段"),
                SubtitleSegment("b", 2.0, 4.0, "S01", "也要合并成句。"),
            ]
        )

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].end, 4.0)

    def test_sentence_final_inside_trailing_quote(self):
        segments = regroup_sentences(
            [
                SubtitleSegment("a", 0, 2.0, "S01", "他说“就这样吧”。"),
                SubtitleSegment("b", 2.2, 4.0, "S01", "然后离开了。"),
            ]
        )

        self.assertEqual(len(segments), 2)

    def test_splits_overlong_sentence_for_display(self):
        segments = regroup_sentences(
            [
                SubtitleSegment("a", 0, 8.0, "S01", "这句话非常非常长，超过了单行字幕的显示上限，需要在标点处切成两行才好看。"),
            ]
        )

        self.assertGreater(len(segments), 1)
        self.assertEqual(segments[0].start, 0.0)
        self.assertEqual(segments[-1].end, 8.0)
        for segment in segments:
            self.assertLessEqual(len(segment.text), 40)

    def test_merges_english_fragments_despite_char_count(self):
        # 英文字符数约是中文两倍，计权后不应撞上句子上限
        segments = regroup_sentences(
            [
                SubtitleSegment("a", 0, 2.8, "S01", "So today I want to talk about"),
                SubtitleSegment("b", 3.1, 5.9, "S01", "how we think about the future of"),
                SubtitleSegment("c", 6.2, 8.0, "S01", "artificial intelligence."),
            ]
        )

        # 合并成完整句；句子总权重 44 超过单行上限 32，会再切成 2 个显示行（在词边界处）
        self.assertEqual(len(segments), 2)
        self.assertEqual(
            "".join(segment.text for segment in segments).replace(" ", ""),
            "SotodayIwanttotalkabouthowwethinkaboutthefutureofartificialintelligence.",
        )
        self.assertFalse(
            segments[0].text.split()[-1].lower() in {"because", "and", "the", "of", "to"}
        )
        self.assertEqual(segments[-1].end, 8.0)

    def test_keeps_timing_for_short_sentence(self):
        segments = regroup_sentences(
            [SubtitleSegment("a", 1.5, 3.2, "S01", "短句结束。")]
        )

        self.assertEqual([(s.start, s.end, s.text) for s in segments], [(1.5, 3.2, "短句结束。")])
        self.assertEqual(segments[0].id, "seg_0001")


class RegroupSentencesFromWordsTest(unittest.TestCase):
    @staticmethod
    def _words(script):
        """把 [(text, start, end), ...] 转成词流；时间按词递增。"""
        words = []
        cursor = 0.0
        for text in script:
            duration = max(0.2, 0.12 * len(text))
            words.append((cursor, cursor + duration, text))
            cursor += duration
        return words

    def test_merges_across_segment_boundaries_with_exact_word_timing(self):
        # 一句话被 Whisper 切成两个 segment，词流重组应无视边界合并
        words = self._words([" I", " don't", " talk", " about", " this", " much", " because", " it", " felt", " bad."])
        segments = regroup_sentences_from_words(words)

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].text, "I don't talk about this much because it felt bad.")
        # 起止时间取自词的真实时间戳
        self.assertEqual(segments[0].start, words[0][0])
        self.assertEqual(segments[0].end, words[-1][1])

    def test_closes_sentence_unit_at_period_word(self):
        words = self._words([" First", " sentence.", " Second", " sentence."])
        segments = regroup_sentences_from_words(words)

        self.assertEqual([s.text for s in segments], ["First sentence.", "Second sentence."])

    def test_closes_unit_at_char_cap_without_punctuation(self):
        script = [" word"] * 30  # 无标点长流，靠字数上限收句
        segments = regroup_sentences_from_words(self._words(script))

        self.assertGreater(len(segments), 1)

    def test_long_unit_wraps_into_multiline_segment(self):
        # 整句权重超单行宽度(32)但低于硬切上限(64)且时长<8s：
        # 应作为一个 segment，内部用 \n 换行，不切成多个独立 segment
        script = ["我从前在大厂", "做技术负责人", "去年辞职了", "现在回想起来", "那段日子", "真的非常辛苦", "非常累。"]
        words = self._words(script)
        segments = regroup_sentences_from_words(words)

        self.assertEqual(len(segments), 1)
        self.assertIn("\n", segments[0].text)  # 整句超单行宽度 → 内部换行成多行
        # 整句文本完整保留（去掉 \n 等空白后等于原句拼接）
        self.assertEqual("".join(segments[0].text.split()), "".join(script))
        # 词级时间：整句 segment 起点取自首词
        self.assertEqual(segments[0].start, words[0][0])

    def test_overlong_unit_hard_splits_and_moves_dangling(self):
        # 整句超硬切上限（dur>8s）：切成多段，段内/段尾都不悬虚词，词级时间来自词
        script = [" I", " quit", " my", " job", " because", " the", " pay", " was", " bad", " and", " I", " hated", " every", " single", " day.", " Next", " line."]
        words = self._words(script)
        segments = regroup_sentences_from_words(words)

        # 第一句 dur>8s 硬切成多段 + "Next line." = 至少 3 个 segment
        self.assertGreaterEqual(len(segments), 3)
        # 任何行（含 \n 内部换行）行尾都不悬 because
        for segment in segments:
            for line in segment.text.split("\n"):
                if line.split():
                    self.assertNotEqual(line.split()[-1].lower().strip("',."), "because")
        # 词级时间：首段起点取自首词；末段是 "Next line."
        self.assertEqual(segments[0].start, words[0][0])
        self.assertEqual(segments[-1].text, "Next line.")

    def test_long_sentence_soft_breaks_at_comma(self):
        # 长句超重(weight>max_chars)时不在词中间硬切，等下一个逗号软切。
        # 解决 "the reason" 孤立开头类问题：切点落在语义断点（逗号）。
        from moss_transcribe_diarize.subtitle.postprocess import _sentence_units_from_words
        words = []
        t = 0.0
        for _ in range(45):  # 45 词 * 2.0 weight = 90 > 80(max_chars)，dur 8.8s < 15
            words.append((t, t + 0.3, "word")); t += 0.2
        words.append((t, t + 0.3, "comma,")); t += 0.2  # 逗号软切点
        for _ in range(5):
            words.append((t, t + 0.3, "more")); t += 0.2
        units = _sentence_units_from_words(words, max_chars=80, max_duration=15.0)
        self.assertGreaterEqual(len(units), 2)
        self.assertTrue(units[0][-1][2].endswith(','))  # 第一unit止于逗号（非词中间）

    def test_empty_words_returns_empty(self):
        self.assertEqual(regroup_sentences_from_words([]), [])

    def test_overlapping_word_times_clamped(self):
        # 两个时长超过 tiny 阈值的句末片段，时间重叠 → clamp 应让第二段起点不早于第一段终点
        words = [(0.0, 1.5, " hello there."), (1.3, 2.9, " world rotates.")]
        segments = regroup_sentences_from_words(words)

        self.assertEqual(len(segments), 2)
        self.assertGreaterEqual(segments[1].start, segments[0].end)

    def test_overlapping_tiny_words_merge_instead_of_clamp(self):
        # 两个时长都在 tiny 阈值内的句末短答 → 应被合并成一行而非各自独立
        words = [(0.0, 1.0, " hello."), (0.9, 1.5, " world.")]
        segments = regroup_sentences_from_words(words)

        self.assertEqual(len(segments), 1)
        self.assertLessEqual(segments[0].end, 1.5)

    def test_merges_tiny_sentence_final_fragments(self):
        # 问句后跟一连串句末短答（Yes./Easily./Very much.），不应各自独占一行
        words = self._words([
            " Six", " figures", " plus?", " Yes.", " Easily.", " Very", " much.",
        ])
        segments = regroup_sentences_from_words(words)
        # 至少把 400ms 量级的 "Yes." / "Easily." 并掉，行数显著少于词数
        self.assertLess(len(segments), 5)
        joined = " ".join(s.text for s in segments)
        self.assertIn("Yes", joined)
        self.assertIn("Easily", joined)
        # 不应出现只有 "Yes." 的孤立行
        for segment in segments:
            self.assertGreater(len(segment.text), 5)


if __name__ == "__main__":
    unittest.main()
