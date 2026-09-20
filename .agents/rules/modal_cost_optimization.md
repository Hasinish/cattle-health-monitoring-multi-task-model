# Modal & Cloud Compute Cost Optimization Rule

## 1. Download & Data Transfer Containers
- **Absolute Minimum Resource Allocation**: When writing Modal functions for downloads, archive extraction, data ingestion, or preprocessing that does not require GPU acceleration, strictly allocate `cpu=1.0, memory=2048` (or `memory=1024` where sufficient).
- **No GPUs on I/O**: NEVER attach GPUs or allocate >1 CPU / >2GB RAM for network I/O, file downloads, or unzipping.
- **Periodic Volume Commits**: For long-running downloads or file transfers into Modal Volumes, always implement a periodic commit loop (e.g., every 60 seconds via `volume.commit()`) or a SIGINT/SIGTERM signal handler so that partial progress is preserved and resumable if cancelled or interrupted.

## 2. GPU Training & Feasibility Smoke Tests
- **Cheapest Hardware First**: Always default to the lowest-cost GPU tier (e.g. `gpu="T4"` at ~$0.59/hr) for smoke tests, sanity checks, and initial feasibility runs.
- **Explicit Approval for Upgrades**: Never upgrade to higher GPU tiers (L4, A10G, A100, H100) without explicit confirmation or hard VRAM necessity.
- **Low-Balance Account Awareness**: When running on burner or low-balance accounts (<$5.00), enforce tight timeouts and minimal sample sizes to protect remaining credits.
