"""Sync Run 7 E1 MTL training metrics from Modal volume mtl-checkpoints."""
import json
from pathlib import Path
import modal

app = modal.App("sync-mtl-e1-artifacts")
ckpt_vol = modal.Volume.from_name("mtl-checkpoints", create_if_missing=False)

@app.function(volumes={"/checkpoints": ckpt_vol})
def get_metrics_bytes() -> bytes:
    p = Path("/checkpoints/mtl_e1_hard_shared/mtl_e1_metrics.json")
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    return p.read_bytes()

@app.local_entrypoint()
def main():
    out_dir = Path("artifacts/mtl_e1_training")
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = get_metrics_bytes.remote()
    target = out_dir / "mtl_e1_metrics.json"
    target.write_bytes(raw)
    data = json.loads(raw.decode("utf-8"))
    print(f"[OK] Successfully fetched {target}")
    print(f"Total duration: {data['total_duration_seconds']}s")
    print(f"Best Val Objective: {data['best_val_objective']}")
    print(f"Best Epoch: {data['best_metrics']['epoch']}")
    print(f"Best BCS MAE: {data['best_metrics']['bcs']['real_mae']}")
    print(f"Best Beh Macro-F1: {data['best_metrics']['behavior']['macro_f1']}")
    print(f"Best Re-ID Top-1: {data['best_metrics']['reid']['top1_accuracy']}")

if __name__ == "__main__":
    main()
