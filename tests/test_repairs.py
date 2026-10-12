"""Offline structural regressions. No pretrained-accuracy claims."""
import importlib.util
import json
from pathlib import Path
import random
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sparse_modules import SparseAttention, MixtureOfExperts, ExpertLayer, gather_patches, EnhancedASPP, topk_attention
from sam_moe import SparseMoESAM, SparseSAMAttention, predicted_signed_distance
from modeling.image_encoder import Attention
from token_clustering import CTM, TCBlock, merge_tokens
from data import PlantDiseaseDataset, prepare_splits
from losses import SegmentationLoss, BinaryMetrics
from deeplab_baseline import EnhancedDeepLabV3Plus
import run_code

torch.set_num_threads(2)


def tiny_model(channels=16, **kwargs):
    return SparseMoESAM(encoder_size=320, channels=channels,
                       encoder_config=dict(embed_dim=32, depth=2, num_heads=4, global_attn_indexes=(1,)), **kwargs)


class Repairs(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_attention_matches_masked_reference_and_exact_zeros(self):
        q, k, v = [torch.randn(2, 2, 16, 4) for _ in range(3)]
        output, l1, entropy, captured = topk_attention(q, k, v, .25, query_chunk_size=3, capture=True)
        scores = q @ k.transpose(-2, -1) / 2
        values, indices = scores.topk(4, dim=-1)
        masked = torch.full_like(scores, float("-inf")).scatter(-1, indices, values)
        weights = masked.softmax(-1)
        self.assertTrue(torch.allclose(output, weights @ v, atol=1e-6))
        self.assertTrue(((weights > 0).sum(-1) == 4).all())
        self.assertAlmostEqual(float(l1), 1., places=6)
        self.assertEqual(sum(item[0].shape[-2] for item in captured), 16)
        attention = SparseAttention(8, 2, .1).eval()
        attention.capture_attention = True
        for layer in (attention.query, attention.key):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        attention(torch.randn(1, 8, 4, 4))
        self.assertEqual(attention.last_attention[0][1].shape[-1], 1)
        self.assertTrue((attention.last_attention[0][1] == 1).all())

    def test_sam_attention_relative_positions_matches_upstream(self):
        dense = Attention(16, 4, use_rel_pos=True, input_size=(3, 4)).eval()
        nn.init.normal_(dense.rel_pos_h, std=.1)
        nn.init.normal_(dense.rel_pos_w, std=.1)
        sparse = SparseSAMAttention(dense, ratio=1, query_chunk_size=5).eval()
        x = torch.randn(2, 3, 4, 16)
        self.assertTrue(torch.allclose(dense(x), sparse(x), atol=2e-6))

    def test_gumbel_straight_through_hard_forward_and_soft_gradients(self):
        q, k, v = [torch.randn(1, 2, 6, 4, requires_grad=True) for _ in range(3)]
        torch.manual_seed(51)
        relaxed = topk_attention(q, k, v, .5, training=True, query_chunk_size=2)[0]
        torch.manual_seed(51)
        hard = topk_attention(q, k, v, .5, training=True, query_chunk_size=2, straight_through=False)[0]
        self.assertTrue(torch.allclose(relaxed, hard, atol=1e-6))
        relaxed.square().mean().backward()
        self.assertTrue(all(torch.isfinite(t.grad).all() and t.grad.abs().sum() > 0 for t in (q, k, v)))

    def test_selected_patch_convolutions_match_dense(self):
        x = torch.randn(2, 3, 7, 9, requires_grad=True)
        positions = torch.tensor([0, 8, 13, 62, 63, 79, 125])
        for dilation in (1, 2, 6):
            weight = torch.randn(4, 3, 3, 3)
            sparse = F.linear(gather_patches(x, positions, dilation).flatten(1), weight.flatten(1))
            dense = F.conv2d(x, weight, padding=dilation, dilation=dilation).permute(0, 2, 3, 1).reshape(-1, 4)
            self.assertTrue(torch.allclose(sparse, dense[positions], atol=3e-6))
        sparse.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_moe_batch_one_dispatch_and_weight_normalization(self):
        model = MixtureOfExperts(8, 8).eval()
        last = model.gate[-1]
        with torch.no_grad():
            last.weight.zero_()
            last.bias.copy_(torch.tensor([5., 0., -5., -6.]))
        calls = [0] * 4
        def hook(index):
            def record(module, args, output):
                calls[index] += args[1].numel()
            return record
        for i, expert in enumerate(model.experts):
            expert.register_forward_hook(hook(i))
        x = torch.randn(1, 8, 5, 7)
        actual = model(x)
        self.assertEqual(calls, [35, 35, 0, 0])
        self.assertAlmostEqual(float(model.last_weights[0, 0]), .993307, places=5)
        positions = torch.arange(35)
        with torch.no_grad():
            dense = sum(expert(x, positions) * model.last_weights[:, slot:slot + 1]
                        for slot, expert in enumerate(model.experts[:2]))
        self.assertTrue(torch.allclose(actual.permute(0, 2, 3, 1).reshape(35, 8), dense, atol=1e-6))
        actual.square().mean().backward()
        self.assertTrue(last.weight.grad.abs().sum() > 0)
        self.assertTrue(all(p.grad is None for p in model.experts[2].parameters()))

    def test_moe_spatial_routes_and_balance_gradients(self):
        model = MixtureOfExperts(1, 8).eval()
        gate = nn.Linear(1, 4)
        with torch.no_grad():
            gate.weight.copy_(torch.tensor([[3.], [1.], [-1.], [-3.]]))
            gate.bias.zero_()
        model.gate = gate
        x = torch.tensor([[[[-2., -1., 1., 2.]]]])
        model(x)
        self.assertFalse(torch.equal(model.last_indices[0], model.last_indices[-1]))
        self.assertEqual(int(model.last_dispatch_counts.sum()), 8)
        model.balance_loss.backward()
        self.assertTrue(torch.isfinite(gate.weight.grad).all())

    def test_stage_temperatures_balance_and_optimizer_ownership(self):
        model = tiny_model()
        self.assertEqual(model.decoder.stage1.gate[1].out_features, 512)
        self.assertIsInstance(model.decoder.stage1.gate[2], nn.ReLU)
        run_code.set_attention_temperature(model, .2)
        self.assertEqual(model.image_encoder.blocks[0].attn.temperature, .2)
        self.assertEqual(model.decoder.stage1.temperature, .5)
        self.assertEqual(model.decoder.stage2.temperature, .3)
        args = run_code.parser().parse_args(["--data-root", "."])
        optimizer, _ = run_code.make_optimizer(model, args)
        groups = {group["name"]: {id(p) for p in group["params"]} for group in optimizer.param_groups}
        expected_gate = {id(p) for stage in (model.decoder.stage1, model.decoder.stage2)
                         for p in stage.gate.parameters()}
        self.assertEqual(groups["gate"], expected_gate)
        self.assertTrue({id(p) for p in model.decoder.aspp.se.parameters()} <= groups["decoder"])
        self.assertTrue({id(p) for p in model.decoder.refinement_se.parameters()} <= groups["decoder"])
        routed = MixtureOfExperts(8, 8).eval()
        with torch.no_grad():
            routed.gate[-1].weight.zero_()
            routed.gate[-1].bias.copy_(torch.tensor([2., 1., -1., -2.]))
        routed(torch.randn(1, 8, 3, 4))
        self.assertAlmostEqual(float(routed.balance_loss.detach()), .25, places=7)
        routed.balance_loss.backward()
        self.assertGreater(float(routed.gate[-1].bias.grad.abs().sum()), 0)

    def test_predicted_distance_sign_and_empty_full_cases(self):
        confidence = torch.tensor([[[[.1, .1, .9, .9], [.1, .1, .9, .9]]]], requires_grad=True)
        distance = predicted_signed_distance(confidence)
        self.assertFalse(distance.requires_grad)
        self.assertTrue((distance[..., :2] < 0).all())
        self.assertTrue((distance[..., 2:] > 0).all())
        self.assertTrue(torch.allclose(distance[0, 0, 0], torch.tensor([-2., -1., 1., 2.]) / np.sqrt(20)))
        extremes = predicted_signed_distance(torch.cat([torch.zeros_like(confidence), torch.ones_like(confidence)]))
        self.assertTrue((extremes[0] == -1).all())
        self.assertTrue((extremes[1] == 1).all())
        self.assertEqual(distance.dtype, confidence.dtype)
        self.assertEqual(distance.shape, confidence.shape)
        with self.assertRaises(ValueError):
            predicted_signed_distance(torch.ones(1, 2, 3, 4))

    def test_aspp_singleton_and_noncentral_dilation_gradients(self):
        module = EnhancedASPP(16, 16).train()
        x = torch.randn(1, 16, 20, 20, requires_grad=True)
        module(x).square().mean().backward()
        for branch in module.branches[1:]:
            gradient = branch[0].weight.grad.clone()
            gradient[:, :, 1, 1] = 0
            self.assertGreater(int(torch.count_nonzero(gradient)), 0)

    def test_sam_end_to_end_singleton_dynamic_size_and_freeze(self):
        model = tiny_model().train()
        images = torch.randn(1, 3, 64, 80)
        logits, aux = model(images, return_aux=True)
        self.assertEqual(tuple(logits.shape), (1, 1, 64, 80))
        loss = SegmentationLoss()(logits, (torch.rand_like(logits) > .5).float(), aux)
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        self.assertGreater(float(model.image_encoder.blocks[1].attn.qkv.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.decoder.stage1.gate[-1].weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.decoder.offset_head.weight.grad.abs().sum()), 0)
        for projection in model.decoder.lateral_projections:
            self.assertGreater(float(projection[0].weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.decoder.refinement_se.gate[1].weight.grad.abs().sum()), 0)
        model.zero_grad(set_to_none=True)
        model.set_encoder_trainable(False)
        logits = model(images)
        logits.mean().backward()
        self.assertTrue(all(p.grad is None for p in model.image_encoder.parameters()))
        model.eval()
        with torch.no_grad():
            self.assertEqual(tuple(model(torch.randn(2, 3, 65, 97)).shape), (2, 1, 65, 97))

    def test_deeplab_ablation_singleton_training_and_dynamic_size(self):
        model = EnhancedDeepLabV3Plus(pretrained=False, channels=16).train()
        logits, aux = model(torch.randn(1, 3, 192, 208), return_aux=True)
        self.assertEqual(tuple(logits.shape), (1, 1, 192, 208))
        loss = SegmentationLoss()(logits, (torch.rand_like(logits) > .5).float(), aux)
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_checkpoint_adaptation_and_strict_reload(self):
        source = tiny_model()
        state = {"image_encoder." + k: v.clone() for k, v in source.image_encoder.state_dict().items()}
        state["image_encoder.pos_embed"] = torch.randn(1, 64, 64, 32)
        state["image_encoder.blocks.1.attn.rel_pos_h"] = torch.randn(127, 8)
        state["image_encoder.blocks.1.attn.rel_pos_w"] = torch.randn(127, 8)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sam.pt"
            torch.save(state, path)
            target = tiny_model(sam_checkpoint=path)
            self.assertTrue(target.pretrained_loaded)
            clone = tiny_model()
            clone.load_state_dict(target.state_dict(), strict=True)
            self.assertEqual(tuple(target.image_encoder.pos_embed.shape), (1, 20, 20, 32))

    def test_masks_manifest_groups_and_paired_resize(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images").mkdir()
            (root / "masks").mkdir()
            for i in range(10):
                image = np.zeros((4, 6, 3), dtype=np.uint8)
                image[:, 3:] = 255
                mask = np.zeros((4, 6), dtype=np.uint8)
                mask[:, 3:] = 1 if i % 2 else 255
                Image.fromarray(image).save(root / "images" / f"img{i}.PNG")
                Image.fromarray(mask).save(root / "masks" / f"img{i}.png")
            manifest = prepare_splits(root, root / "split.json", 42)
            self.assertEqual(manifest, prepare_splits(root, root / "split.json", 42))
            self.assertEqual([len(manifest[s]) for s in ("train", "val", "test")], [7, 1, 2])
            datasets = [PlantDiseaseDataset(root, (8, 12), pairs=manifest[s], augment=False) for s in ("train", "val", "test")]
            for dataset in datasets:
                image, mask = dataset[0]
                self.assertEqual(set(mask.unique().tolist()), {0., 1.})
                self.assertEqual(float(mask[:, :, :6].sum()), 0.)
                self.assertEqual(float(mask[:, :, 6:].sum()), 48.)
                self.assertAlmostEqual(float(image[0, 0, 0]), -0.485 / .229, places=5)
            group_path = root / "groups.json"
            groups = {f"img{i}.PNG": i // 2 for i in range(10)}
            group_path.write_text(json.dumps(groups))
            grouped = prepare_splits(root, root / "grouped.json", 42, group_path)
            sets = [{groups[Path(p["image"]).name] for p in grouped[s]} for s in ("train", "val", "test")]
            self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
            Image.fromarray(np.full((4, 6), 128, np.uint8)).save(root / "masks/img0.png")
            dataset = PlantDiseaseDataset(root, pairs=[{"image": "images/img0.PNG", "mask": "masks/img0.png"}], augment=False)
            with self.assertRaises(ValueError):
                dataset[0]

    def test_empty_metrics_and_loss_numerics(self):
        metrics = BinaryMetrics()
        metrics.update(torch.full((1, 1, 2, 2), -100.), torch.zeros(1, 1, 2, 2))
        self.assertEqual(metrics.compute()["dice"], 1.)
        self.assertEqual(metrics.compute()["iou"], 1.)
        logits = torch.tensor([[[[-1000., 1000.]]]], requires_grad=True)
        loss = SegmentationLoss()(logits, torch.tensor([[[[1., 0.]]]]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_clustering_padding_all_empty_and_trainable_block(self):
        tokens = {"x": torch.randn(2, 8, 4, requires_grad=True), "idx_token": torch.arange(8).expand(2, -1),
                  "agg_weight": torch.ones(2, 8, 1), "mask": torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0], [0] * 8])}
        merged, original = CTM(.5, 4, 8)(tokens)
        self.assertTrue(torch.isfinite(merged["x"]).all())
        self.assertTrue(torch.isfinite(merged["agg_weight"]).all())
        self.assertEqual(float(merged["x"][1].detach().abs().sum()), 0.)
        block = TCBlock(8, 2)
        out = block(merged)
        self.assertIsNot(out, merged)
        out["x"].square().mean().backward()
        self.assertGreater(float(block.attention.in_proj_weight.grad.abs().sum()), 0)
        simple = {"x": torch.tensor([[[1.], [3.]]]), "idx_token": torch.tensor([[0, 1]]),
                  "agg_weight": torch.ones(1, 2, 1), "mask": None}
        result = merge_tokens(simple, torch.tensor([[0, 0]]), 1)
        self.assertAlmostEqual(float(result["x"].item()), 2.)
        self.assertAlmostEqual(float(result["agg_weight"].max()), 1.)

    def test_package_import_and_native_sam_prediction(self):
        spec = importlib.util.spec_from_file_location("repaired_sam", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        from repaired_sam.modeling import ImageEncoderViT, PromptEncoder, MaskDecoder, Sam, TwoWayTransformer
        sam = Sam(image_encoder=ImageEncoderViT(img_size=32, patch_size=16, embed_dim=32, depth=1,
                  num_heads=4, out_chans=32), prompt_encoder=PromptEncoder(32, (2, 2), (32, 32), 16),
                  mask_decoder=MaskDecoder(transformer_dim=32, transformer=TwoWayTransformer(1, 32, 4, 64),
                                            iou_head_hidden_dim=32)).eval()
        predictor = module.SamPredictor(sam)
        predictor.set_image(np.zeros((25, 32, 3), dtype=np.uint8))
        masks, scores, logits = predictor.predict(point_coords=np.array([[16, 12]]), point_labels=np.array([1]))
        self.assertEqual(masks.shape[-2:], (25, 32))
        self.assertTrue(np.isfinite(logits).all())

    def test_training_entry_checkpoint_evaluation_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            (root / "images").mkdir(parents=True)
            (root / "masks").mkdir()
            for i in range(5):
                Image.fromarray(np.full((32, 40, 3), i * 20, np.uint8)).save(root / "images" / f"{i}.png")
                mask = np.zeros((32, 40), np.uint8)
                mask[8:24, 10:30] = 255
                Image.fromarray(mask).save(root / "masks" / f"{i}.png")
            output = Path(tmp) / "run"
            argv = ["--data-root", str(root), "--output-dir", str(output), "--allow-random-init",
                    "--encoder-size", "320", "--decoder-channels", "32", "--image-size", "64",
                    "--epochs", "2", "--batch-size", "2", "--accumulation-steps", "2", "--warmup-epochs", "0",
                    "--freeze-encoder-epochs", "1", "--device", "cpu"]
            second_output = Path(tmp) / "interrupted"
            second_argv = argv.copy()
            second_argv[second_argv.index(str(output))] = str(second_output)
            real_save = torch.save
            def capture_epoch_one(checkpoint, path, *args, **kwargs):
                real_save(checkpoint, path, *args, **kwargs)
                if checkpoint.get("epoch") == 0:
                    second_output.mkdir(exist_ok=True)
                    shutil.copy2(path, second_output / Path(path).name)
                    for name in ("split.json", "config.json", "source_manifest.json"):
                        shutil.copy2(output / name, second_output / name)
            # Four channels per pooled GN group avoids the near-zero-gradient
            # two-channel case; the other structural checks still use 16.
            with patch.object(run_code, "SparseMoESAM", side_effect=lambda *args, **kwargs: tiny_model(channels=32)):
                with patch.object(torch, "save", side_effect=capture_epoch_one):
                    result = run_code.main(argv)
                self.assertTrue((output / "best.pt").exists())
                last = torch.load(output / "last.pt", weights_only=True)
                self.assertEqual(last["epoch"], 1)
                self.assertEqual(len(last["history"]), 2)
                self.assertTrue(last["history"][0]["encoder_frozen"])
                self.assertFalse(last["history"][1]["encoder_frozen"])
                self.assertIn("sam_moe.py", last["source_sha256"])
                training_config = (output / "config.json").read_bytes()
                evaluated = run_code.main(argv + ["--eval-only"])
                self.assertAlmostEqual(result["dice"], evaluated["dice"], places=7)
                self.assertEqual(training_config, (output / "config.json").read_bytes())
                self.assertTrue((output / "eval_config.json").exists())
                resumed = run_code.main(argv + ["--resume", str(output / "last.pt")])
                self.assertAlmostEqual(result["dice"], resumed["dice"], places=7)
                self.assertEqual(training_config, (output / "config.json").read_bytes())
                self.assertTrue((output / "resume_config.json").exists())
                # Continue from the SAME saved epoch-one state as the reference
                # run. Independently repeating epoch one would also test CPU
                # floating-point/recomputation order, not just state restoration.
                run_code.main(second_argv + ["--resume", str(second_output / "last.pt")])
                continued = torch.load(second_output / "last.pt", weights_only=True)
                for name, value in last["model"].items():
                    self.assertTrue(torch.allclose(value, continued["model"][name], atol=2e-6), name)
                self.assertEqual(last["scheduler"], continued["scheduler"])
                self.assertEqual(last["optimizer"]["param_groups"], continued["optimizer"]["param_groups"])
                self.assertEqual(last["optimizer"]["state"].keys(), continued["optimizer"]["state"].keys())
                for key, expected in last["optimizer"]["state"].items():
                    actual = continued["optimizer"]["state"][key]
                    self.assertEqual(float(expected["step"]), float(actual["step"]))
                    for moment in ("exp_avg", "exp_avg_sq"):
                        self.assertTrue(torch.allclose(expected[moment], actual[moment], atol=2e-7), (key, moment))
                self.assertAlmostEqual(last["history"][-1]["train"]["loss"], continued["history"][-1]["train"]["loss"], places=6)
                # Separately verify that an actual interruption leaves a usable
                # checkpoint; this run is not the numerical reference above.
                interrupted_argv = argv.copy()
                interrupted_output = Path(tmp) / "interruption_check"
                interrupted_argv[interrupted_argv.index(str(output))] = str(interrupted_output)
                real_epoch, calls = run_code.run_epoch, [0]
                def interrupt_after_epoch_one(*args, **kwargs):
                    calls[0] += 1
                    if calls[0] == 3:
                        raise InterruptedError("simulated interruption")
                    return real_epoch(*args, **kwargs)
                with patch.object(run_code, "run_epoch", side_effect=interrupt_after_epoch_one):
                    with self.assertRaises(InterruptedError):
                        run_code.main(interrupted_argv)
                self.assertEqual(torch.load(interrupted_output / "last.pt", weights_only=True)["epoch"], 0)
                run_code.main(interrupted_argv + ["--resume", str(interrupted_output / "last.pt")])


if __name__ == "__main__":
    unittest.main(verbosity=2)
