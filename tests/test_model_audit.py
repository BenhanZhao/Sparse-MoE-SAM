"""Dataset-free checks of truthful parameter/source reporting."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import inspect_model
from sam_moe import SparseMoESAM


def tiny_model():
    return SparseMoESAM(encoder_size=320, channels=16,
                        encoder_config=dict(embed_dim=32, depth=2, num_heads=4,
                                            global_attn_indexes=(1,)))


class ModelAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model = tiny_model()

    def test_parameter_counts_are_real_partitioned_and_frozen(self):
        self.model.set_encoder_trainable(False)
        try:
            inventory = inspect_model.parameter_inventory(self.model)
            self.assertEqual(inventory["total"], sum(p.numel() for p in self.model.parameters()))
            self.assertEqual(inventory["total"], sum(group["total"] for group in inventory["groups"].values()))
            self.assertEqual(inventory["trainable"] + inventory["frozen"], inventory["total"])
            self.assertEqual(inventory["groups"]["encoder"]["trainable"], 0)
            self.assertEqual(inventory["groups"]["other"]["total"], 0)
        finally:
            self.model.set_encoder_trainable(True)

    def test_tied_parameters_are_not_double_counted(self):
        model = nn.Module()
        model.image_encoder = nn.Linear(3, 2)
        model.decoder = nn.Module()
        model.decoder.shared = model.image_encoder
        model.extra = nn.Parameter(torch.zeros(5))
        inventory = inspect_model.parameter_inventory(model)
        self.assertEqual(inventory["total"], 13)
        self.assertEqual(inventory["groups"]["encoder"]["total"], 8)
        self.assertEqual(inventory["groups"]["decoder"]["total"], 0)
        self.assertEqual(inventory["groups"]["other"]["total"], 5)

    def test_sha256_tracks_source_bytes_and_excludes_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "model.py").write_bytes(b"first\n")
            (root / "venv").mkdir()
            (root / "venv" / "thirdparty.py").write_bytes(b"excluded")
            first = inspect_model.source_fingerprints(root)
            self.assertEqual(first, {"model.py": hashlib.sha256(b"first\n").hexdigest()})
            (root / "model.py").write_bytes(b"second\n")
            self.assertNotEqual(first, inspect_model.source_fingerprints(root))

    def test_report_preserves_unresolved_claims_and_actual_settings(self):
        with patch.object(self.model, "forward", side_effect=AssertionError("must not run forward")):
            report = inspect_model.build_report(self.model)
        comparison = report["paper_comparison"]
        self.assertEqual(comparison["status"], "unresolved")
        self.assertEqual(comparison["reference_parameter_count"], 142_700_000)
        self.assertEqual(comparison["delta_actual_minus_reference"], report["parameters"]["total"] - 142_700_000)
        self.assertFalse(any(report["audit_scope"].values()))
        self.assertGreaterEqual(len(report["unresolved"]), 3)
        self.assertEqual(report["settings"]["encoder_embed_dim"], 32)
        self.assertEqual(report["settings"]["encoder_depth"], 2)
        self.assertEqual(report["settings"]["decoder_channels"], 16)
        self.assertIsNone(report["settings"]["encoder_variant_inferred_from_dimensions"])
        self.assertEqual(report["settings"]["stage1"]["num_experts"], len(self.model.decoder.stage1.experts))
        self.assertIn("sam_moe.py", report["source_sha256"])
        self.assertIn("decoder.stage1", [module["path"] for module in report["structure"]["registered_modules"]])
        self.assertEqual(json.loads(json.dumps(report))["parameters"], report["parameters"])

    def test_missing_optional_components_are_reported_as_unknown(self):
        settings = inspect_model.model_settings(nn.Module())
        self.assertIsNone(settings["boundary_mode"])
        self.assertIsNone(settings["refinement_se_class"])
        self.assertIsNone(settings["lateral_projection_count"])
        self.assertIsNone(settings["stage1"])

    def test_cli_output_and_constructor_do_not_load_weights_or_execute(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "subdirectory" / "audit.json"
            with patch.object(inspect_model, "SparseMoESAM", return_value=self.model) as constructor, \
                    patch.object(self.model, "forward", side_effect=AssertionError("must not run forward")), \
                    patch.object(torch, "load", side_effect=AssertionError("must not load weights")), \
                    contextlib.redirect_stdout(io.StringIO()) as stdout:
                report = inspect_model.main(["--sam-type", "vit_b", "--encoder-size", "320",
                                             "--decoder-channels", "16", "--output", str(output)])
            constructor.assert_called_once_with(model_type="vit_b", encoder_size=320, channels=16)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), report)
            self.assertEqual(json.loads(stdout.getvalue()), report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
