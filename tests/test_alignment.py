import unittest

import numpy as np

from latent_correction.alignment import fit_map
from latent_correction.diagnostics import evaluate
from pair_features import pair
from score_benchmarks import score


class AlignmentTests(unittest.TestCase):
    def test_rotation_recovers_held_out_shift_and_turn(self):
        rng = np.random.default_rng(10)
        a = rng.normal(size=(60, 3))
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        b = a @ q + np.array([2.0, -1.0, 0.5])
        fitted = fit_map(a[:40], b[:40], "rotation")
        np.testing.assert_allclose(fitted.transform(a[40:]), b[40:], atol=1e-10)

    def test_ridge_beats_identity_on_synthetic_linear_drift(self):
        rng = np.random.default_rng(11)
        a = rng.normal(size=(100, 3))
        b = a @ np.diag([2.0, 0.5, 1.5]) + np.array([1.0, 0.0, -2.0])
        identity_error = np.mean((fit_map(a[:80], b[:80], "identity")
                                  .transform(a[80:]) - b[80:]) ** 2)
        ridge_error = np.mean((fit_map(a[:80], b[:80], "ridge", ridge_alpha=0.01)
                               .transform(a[80:]) - b[80:]) ** 2)
        self.assertLess(ridge_error, identity_error)

    def test_held_out_diagnostic_rejects_leakage(self):
        data = {"anchor_id": np.array(["x", "y"]),
                "anchor_source": np.array([[1., 0.], [0., 1.]]),
                "anchor_target": np.array([[1., 0.], [0., 1.]]),
                "anchor_label": np.array(["A", "B"]),
                "eval_id": np.array(["x"]),
                "eval_source": np.array([[1., 0.]]),
                "eval_target": np.array([[1., 0.]]),
                "eval_label": np.array(["A"]),
                "eval_benchmark": np.array(["test"])}
        with self.assertRaisesRegex(ValueError, "overlap"):
            evaluate(data, ("identity",))
        data["eval_id"] = np.array(["z"])
        report = evaluate(data, ("identity",))
        self.assertEqual(report["methods"]["identity"]["benchmark_drop_count"], 0)

    def test_pair_aligns_rows_by_id(self):
        target = {"id": np.array(["a", "b"]), "split": np.array(["anchor", "eval"]),
                  "benchmark": np.array(["b1", "b2"]), "gold": np.array(["A", "B"]),
                  "video_sha256": np.array(["h1", "h2"]),
                  "processed_input_sha256": np.array(["i1", "i2"]),
                  "feature": np.array([[1., 0.], [0., 1.]])}
        source = {field: value[::-1].copy() for field, value in target.items()}
        paired, meta = pair(source, target)
        np.testing.assert_equal(paired["eval_source"], [[0., 1.]])
        self.assertEqual(meta["identical_processed_input_fraction"], 1.0)

    def test_actual_answer_score_counts_benchmark_drops(self):
        manifest = [{"id": "1", "gold": "A", "benchmark": "B1"},
                    {"id": "2", "gold": "B", "benchmark": "B2"}]
        answers = [{"id": "1", "method": "baseline", "answer": "A"},
                   {"id": "2", "method": "baseline", "answer": "A"},
                   {"id": "1", "method": "m", "answer": "B"},
                   {"id": "2", "method": "m", "answer": "B"}]
        result = score(manifest, answers)["methods"]["m"]
        self.assertEqual(result["benchmark_drop_count"], 1)
        self.assertEqual(result["worst_delta_pp"], -100.0)
        self.assertEqual(result["item_flips"], {"gained": 1, "lost": 1})


if __name__ == "__main__":
    unittest.main()
