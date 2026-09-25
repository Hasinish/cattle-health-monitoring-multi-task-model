"""Sync Phase 3 E4 PCGrad MTL training metrics from Modal volume mtl-checkpoints."""
import json
import os
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import modal

app = modal.App("sync-mtl-e4-artifacts")
ckpt_vol = modal.Volume.from_name("mtl-checkpoints", create_if_missing=False)

@app.function(volumes={"/checkpoints": ckpt_vol})
def get_metrics_bytes() -> bytes:
    p = Path("/checkpoints/mtl_e4_pcgrad/mtl_e4_metrics.json")
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    return p.read_bytes()

@app.local_entrypoint()
def main():
    out_dir = Path("artifacts/mtl_e4_training")
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = get_metrics_bytes.remote()
    target = out_dir / "mtl_e4_metrics.json"
    target.write_bytes(raw)
    data = json.loads(raw.decode("utf-8"))
    print(f"[OK] Successfully fetched {target}")
    print(f"Total duration: {data['total_duration_seconds']}s")
    print(f"Best Val Objective: {data['best_val_objective']}")
    print(f"Best BCS MAE: {data['best_metrics']['bcs']['real_mae']}")
    print(f"Best Beh Macro-F1: {data['best_metrics']['behavior']['macro_f1']}")
    print(f"Best Re-ID Top-1: {data['best_metrics']['reid']['top1_accuracy']}")
    print(f"Total PCGrad Projections Triggered: {data['pcgrad_diagnostics']['overall_summary']['total_projections_triggered']}")

if __name__ == "__main__":
    main()
