import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from moss_transcribe_diarize.app.speaker_embedding import (
    SHARED_EMBEDDING_ENV,
    _SharedEmbedding,
    enable_shared_speaker_embedding,
)

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "optional torch runtime is not installed")
class SharedEmbeddingTest(unittest.TestCase):
    def setUp(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.child = torch.nn.Identity()
                self.hparams = SimpleNamespace(dither=0.0)
                self.fbank_only = False
                self.batch_sizes = []

            def forward_frames(self, waveforms):
                self.batch_sizes.append(len(waveforms))
                return waveforms.squeeze(1).square()

            def forward_embedding(self, frames, weights=None):
                if weights is None:
                    return frames.mean(dim=1, keepdim=True)
                return (frames * weights).sum(dim=1, keepdim=True) / weights.sum(dim=1, keepdim=True)

        self.model = Model().eval()
        self.original = Mock(model_=self.model, device=torch.device("cpu"), dimension=1)
        self.original.side_effect = lambda waves, masks=None: self.model.forward_embedding(
            self.model.forward_frames(waves), masks
        ).numpy()
        self.shared = _SharedEmbedding(self.original)
        self.waves = torch.tensor([[[1., 2., 3.]], [[1., 2., 3.]], [[4., 5., 6.]], [[4., 5., 6.]]])
        self.masks = torch.tensor([[1., 0., 0.], [0., 1., 1.], [1., 1., 0.], [0., 0., 0.]])

    def test_duplicate_windows_keep_distinct_masks_order_and_nan_rows(self):
        expected = self.original(self.waves, masks=self.masks)
        self.original.reset_mock()
        self.model.batch_sizes.clear()
        actual = self.shared(self.waves, masks=self.masks)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(self.model.batch_sizes, [2])
        self.original.assert_not_called()
        self.assertNotEqual(actual[0, 0], actual[1, 0])
        self.assertTrue(np.isnan(actual[-1, 0]))

    def test_nearly_equal_windows_are_not_merged(self):
        self.waves[1, 0, 0] += 1e-6
        self.shared(self.waves, masks=self.masks)
        self.assertEqual(self.model.batch_sizes, [3])

    def test_maskless_unique_and_singleton_batches_use_original(self):
        for waves, masks in [(self.waves, None), (self.waves[::2], self.masks[::2]), (self.waves[:1], self.masks[:1])]:
            with self.subTest(size=len(waves), masks=masks is not None):
                self.original.reset_mock()
                self.shared(waves, masks=masks)
                self.original.assert_called_once_with(waves, masks=masks)

    def test_training_child_random_dither_and_fbank_only_use_original(self):
        for attr, value in [("training", True), ("dither", 0.1), ("fbank_only", True)]:
            with self.subTest(attr=attr):
                owner = self.model.child if attr == "training" else self.model.hparams if attr == "dither" else self.model
                old_value = getattr(owner, attr)
                setattr(owner, attr, value)
                self.original.reset_mock()
                self.shared(self.waves, masks=self.masks)
                self.original.assert_called_once_with(self.waves, masks=self.masks)
                setattr(owner, attr, old_value)

    def test_invalid_mask_batch_uses_original_validation(self):
        self.original.side_effect = ValueError("invalid mask batch")
        with self.assertRaisesRegex(ValueError, "invalid mask batch"):
            self.shared(self.waves, self.masks[:1])
        self.original.assert_called_once()

    def test_features_do_not_survive_between_calls_or_instances(self):
        first = self.shared(self.waves, masks=self.masks)
        second = self.shared(self.waves + 1, masks=self.masks)
        other = _SharedEmbedding(self.original)
        third = other(self.waves, masks=self.masks)
        self.assertEqual(self.model.batch_sizes, [2, 2, 2])
        self.assertNotEqual(first[0, 0], second[0, 0])
        np.testing.assert_array_equal(first, third)

    def test_properties_and_device_move_delegate(self):
        self.assertEqual(self.shared.dimension, 1)
        device = torch.device("cpu")
        self.assertIs(self.shared.to(device), self.shared)
        self.original.to.assert_called_once_with(device)

    def test_inference_errors_are_not_hidden(self):
        with patch.object(self.model, "forward_frames", side_effect=RuntimeError("inference failed")):
            with self.assertRaisesRegex(RuntimeError, "inference failed"):
                self.shared(self.waves, masks=self.masks)
        self.original.assert_not_called()


class SharedEmbeddingInstallTest(unittest.TestCase):
    def test_only_supported_embedding_is_wrapped_and_install_is_idempotent(self):
        class BaseInference:
            pass

        class WeSpeaker:
            hparams = SimpleNamespace(dither=0.0)
            fbank_only = False
            training = False

            def modules(self):
                return [self]

            def forward_frames(self, waves):
                return waves

            def forward_embedding(self, frames, weights=None):
                return frames

        class Embedding:
            def __init__(self):
                self.model_ = WeSpeaker()
                self.device = SimpleNamespace(type="cuda")

        modules = {
            "pyannote.audio.core.inference": SimpleNamespace(BaseInference=BaseInference),
            "pyannote.audio.models.embedding.wespeaker": SimpleNamespace(WeSpeakerResNet34=WeSpeaker),
            "pyannote.audio.pipelines.speaker_verification": SimpleNamespace(PyannoteAudioPretrainedSpeakerEmbedding=Embedding),
        }
        with patch.dict("sys.modules", modules), patch.dict("os.environ", {SHARED_EMBEDDING_ENV: "1"}), patch(
            "moss_transcribe_diarize.app.speaker_embedding.version", return_value="4.0.7"
        ):
            pipeline = SimpleNamespace(_embedding=Embedding())
            other = SimpleNamespace(_embedding=Embedding())
            original = pipeline._embedding
            self.assertTrue(enable_shared_speaker_embedding(pipeline))
            self.assertIsInstance(pipeline._embedding, BaseInference)
            self.assertIs(pipeline._embedding.original, original)
            self.assertIs(type(other._embedding), Embedding)
            adapter = pipeline._embedding
            self.assertTrue(enable_shared_speaker_embedding(pipeline))
            self.assertIs(pipeline._embedding, adapter)
            for case in ("embedding", "model", "device", "training"):
                with self.subTest(case=case):
                    unsupported = Embedding()
                    if case == "embedding":
                        unsupported = object()
                    elif case == "model":
                        unsupported.model_ = object()
                    elif case == "device":
                        unsupported.device.type = "cpu"
                    else:
                        unsupported.model_.training = True
                    candidate = SimpleNamespace(_embedding=unsupported)
                    self.assertFalse(enable_shared_speaker_embedding(candidate))
                    self.assertIs(candidate._embedding, unsupported)

    def test_opt_out_leaves_pipeline_untouched_without_loading_runtime(self):
        for value in ("0", "off", "FALSE", "no"):
            with self.subTest(value=value), patch.dict("os.environ", {SHARED_EMBEDDING_ENV: value}):
                pipeline = SimpleNamespace(_embedding=object())
                original = pipeline._embedding
                with patch("moss_transcribe_diarize.app.speaker_embedding.version") as get_version:
                    self.assertFalse(enable_shared_speaker_embedding(pipeline))
                self.assertIs(pipeline._embedding, original)
                get_version.assert_not_called()

    def test_unvalidated_runtime_keeps_original(self):
        pipeline = SimpleNamespace(_embedding=object())
        original = pipeline._embedding
        with patch.dict("os.environ", {SHARED_EMBEDDING_ENV: "1"}), patch(
            "moss_transcribe_diarize.app.speaker_embedding.version", return_value="99.0.0"
        ):
            self.assertFalse(enable_shared_speaker_embedding(pipeline))
        self.assertIs(pipeline._embedding, original)


if __name__ == "__main__":
    unittest.main()
