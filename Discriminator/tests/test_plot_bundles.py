import json
import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from Discriminator.scripts.plot_bundles import plot_bundle_paths, save_figure_bundle


class PlotBundleTests(unittest.TestCase):
    def test_bundle_writes_png_pdf_and_safe_npz_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "plots" / "curve.png"
            figure, axis = plt.subplots()
            axis.plot([0.0, 1.0], [2.0, 3.0])
            paths = save_figure_bundle(
                figure, output, plot_type="unit_curve",
                payload={"raw_x": np.array([0.0, 1.0]), "raw_y": np.array([2.0, 3.0])},
            )
            plt.close(figure)
            self.assertEqual(paths, list(plot_bundle_paths(output)))
            self.assertTrue(all(path.is_file() for path in paths))
            with np.load(output.with_suffix(".npz"), allow_pickle=False) as data:
                self.assertTrue(np.array_equal(data["input_raw_x"], [0.0, 1.0]))
                self.assertTrue(np.array_equal(data["input_raw_y"], [2.0, 3.0]))
                metadata = json.loads(str(data["metadata_json"]))
            self.assertEqual(metadata["plot_type"], "unit_curve")
            manifest = json.loads((output.parent / "plot_data_manifest.json").read_text())
            self.assertEqual(manifest["plots"]["curve"]["npz"], "curve.npz")

    def test_bundle_rejects_pickle_requiring_payloads(self):
        figure, _ = plt.subplots()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(TypeError):
                save_figure_bundle(
                    figure, Path(directory) / "bad.png", plot_type="bad",
                    payload={"objects": np.array([object()], dtype=object)},
                )
        plt.close(figure)


if __name__ == "__main__":
    unittest.main()
