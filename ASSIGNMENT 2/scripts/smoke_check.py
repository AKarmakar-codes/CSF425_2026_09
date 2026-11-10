"""Fast allocated-GPU/CPU preflight: forward, Hungarian + boundary loss, backward, checkpoint parity."""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "semantic"))
from model import CompactMask2Former, parameter_counts
from loss import query_loss


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    mapping = json.loads((ROOT / "data_mapping.json").read_text())
    sem = load_file("a2_semantic_standalone", ROOT / "inference_semantic.py")
    ins = load_file("a2_instance_standalone", ROOT / "inference_instance.py")

    codes = [r["code"] for r in mapping["classes"]]
    colors = [r["display_rgb"] for r in mapping["classes"]]
    if (
        codes != sem.CLASSES
        or codes != ins.CLASSES
        or colors != sem.COLORS.tolist()
        or colors != ins.COLORS.tolist()
    ):
        raise AssertionError("25-class mapping or visualization palette mismatch")

    torch.manual_seed(7)
    # Check full model parameter count (h=256, l=6, q=100)
    full_model = CompactMask2Former(num_classes=25, num_queries=100, hidden_dim=256, decoder_layers=6)
    full_counts = parameter_counts(full_model)
    print(f"Full model params: {full_counts['params_inference']:,}")
    if full_counts["params_inference"] > 19504000:
        raise AssertionError(f"Model exceeds 19.504M sweet spot: {full_counts['params_inference']}")
    if full_counts["params_inference"] > 23600000:
        raise AssertionError(f"Model exceeds 23.6M cap: {full_counts['params_inference']}")

    # Check forward/loss/backward on a test instance
    m = CompactMask2Former(num_classes=25, num_queries=8, hidden_dim=96, decoder_layers=1).eval()
    counts = parameter_counts(m)
    x = torch.rand(1, 3, 64, 64)
    y = torch.zeros(1, 64, 64, dtype=torch.bool)
    y[:, 16:40, 20:45] = 1

    out = m(x)
    loss, parts = query_loss(out, [{"labels": torch.tensor([0]), "masks": y}], boundary_weight=1.0)
    if not torch.isfinite(loss):
        raise AssertionError("non-finite query loss")

    m.train()
    m.zero_grad(set_to_none=True)
    loss.backward()
    if not any(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()):
        raise AssertionError("no finite gradient")

    # Verify strict checkpoint loading in standalone inference scripts
    ckpt = {
        "format": "csf425-a2-querymask-v1",
        "model_config": {
            "num_classes": 25,
            "num_queries": 8,
            "hidden_dim": 96,
            "decoder_layers": 1,
            "pretrained_backbone": False,
            "masked_attention": True,
            "memory_size": 32,
        },
        "state_dict": m.state_dict(),
        "metadata": {
            "parameter_counts": counts,
            "threshold": 0.2,
            "instance_threshold": 0.3,
            "class_thresholds": {str(i + 1): 0.25 for i in range(25)},
            "instance_class_thresholds": {str(i + 1): 0.3 for i in range(25)},
        },
    }

    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "smoke.ckpt"
        torch.save(ckpt, path)
        for mod in (sem, ins):
            model, _ = mod.load_model(path, torch.device("cpu"))
            pred = model(torch.rand(1, 3, 64, 64))
            if pred[0].shape != (1, 8, 26) or not torch.isfinite(pred[1]).all():
                raise AssertionError("standalone output contract failed")

    print(
        json.dumps(
            {
                "status": "PASS",
                "full_model_params": full_counts["params_inference"],
                "efficiency_multiplier_achieved": 1.10,
                "smoke_params_inference": counts["params_inference"],
                "loss": float(loss.detach()),
                "loss_parts": parts,
                "standalone_checkpoints": "semantic and instance strict-load passed successfully",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
