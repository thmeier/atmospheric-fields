import sys
import tempfile
import unittest
from pathlib import Path

import torch

from FeatureMetric.utils.sfno_embedding import SFNOEmbedding, _resolve_sfno_repo


class SFNOBundleResolutionTest(unittest.TestCase):
    def setUp(self):
        self.addCleanup(self._remove_test_paths)
        self._test_paths = []

    def _remove_test_paths(self):
        for value in self._test_paths:
            while value in sys.path:
                sys.path.remove(value)

    def test_legacy_checkout_layout_uses_src_and_weights_subdir(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").mkdir()
            (root / "weights_4fields").mkdir()
            resolved, weights = _resolve_sfno_repo(root)
            self.assertEqual(resolved, str(root))
            self.assertEqual(weights, str(root / "weights_4fields"))
            self.assertIn(str(root / "src"), sys.path)
            self._test_paths.append(str(root / "src"))

    def test_standalone_bundle_layout_uses_models_and_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "models").mkdir()
            (root / "weights").mkdir()
            resolved, weights = _resolve_sfno_repo(root)
            self.assertEqual(resolved, str(root))
            self.assertEqual(weights, str(root / "weights"))
            self.assertIn(str(root), sys.path)
            self._test_paths.append(str(root))

    def test_explicit_weights_subdir_overrides_standalone_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "models").mkdir()
            (root / "custom_weights").mkdir()
            _, weights = _resolve_sfno_repo(root, "custom_weights")
            self.assertEqual(weights, str(root / "custom_weights"))
            self._test_paths.append(str(root))

    def test_missing_layout_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "src/ or models/"):
                _resolve_sfno_repo(directory)

    @unittest.skipUnless(
        Path("sfno_8c_31x60_code_and_weights/weights/model_8c_31x60_4fields.pth").is_file(),
        "standalone SFNO bundle is not staged locally",
    )
    def test_standalone_bundle_loads_and_extracts_features(self):
        model = SFNOEmbedding(
            repo_root="sfno_8c_31x60_code_and_weights",
            embedding_channels=8, embedding_resolution=(31, 60),
            pooling="grid", pool_grid=(7, 8),
        ).eval()
        with torch.no_grad():
            features = model.extract_features(torch.zeros(1, 4, 121, 240))
        self.assertEqual(tuple(features.shape), (1, 448))


if __name__ == "__main__":
    unittest.main()
