# Research Log: Local Dataset Inventory Audit & MultiCamCows2024 Upstream Blocker

**Date**: 2026-09-20  
**Author**: Hasin Ishrak  
**Supervision**: Dr. Md. Khalilur Rahman  
**Project**: Cattle Health Monitoring Multi-Task Deep Learning Model (BRAC University)  
**Status**: AUDIT COMPLETE — LOCAL INVENTORY BASELINE ESTABLISHED  

---

## 1. Executive Summary
A comprehensive, read-only forensic physical audit of the local filesystem (`datasets/` and related directories) was conducted to verify actual dataset presence prior to initiating Phase 3 Step 1 modeling or registry generation. The audit revealed critical discrepancies between prior documentation claims and physical disk realities: **ScienceDB raw images are entirely absent locally** (the 53,566-row index CSV exists in Git but points to nonexistent paths); **MultiCamCows2024 download is BLOCKED upstream** due to persistent TCP connection resets from the official University of Bristol endpoint (reproduced locally and via Modal cloud); **Dryad BCS is physically present** (5,940 TIFF images across classes 2–7, though currently unindexed and diverging from prior expectations of ~5,923 images across classes 2–6); and **MmCows contains 213,686 verified valid behavior crops** (plus additional lying/standing folders totaling 427,390 JPGs; raw source videos were purged). Canonical scientific roles remain strictly intact.

---

## 2. Context & Motivation
Previous documentation in `memory/state.md` marked ScienceDB BCS as downloaded and restored. Furthermore, Step 1 of the Phase 3 Canonical Roadmap designated MultiCamCows2024 as the primary Re-ID benchmark and scheduled its download and indexing. Before constructing `datasets/dataset_registry.csv` or generating split manifests, `AGENTS.md` and scientific rigor require establishing ground truth on physical disk contents rather than assuming file presence based on committed index CSVs.

---

## 3. Forensic Findings & Physical Inventory

### 3.1 Local Dataset Physical Presence Table
| Dataset | Canonical Role | Local Physical Status | Exact Local Path | Images | Videos | Approx Size | Index Status | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ScienceDB BCS** | Primary In-Domain BCS | **NOT_FOUND (Metadata Only)** | `datasets/bcs/sciencedb_bcs/` *(Missing)* | 0 | 0 | 0 B | **STALE / BROKEN** (`sciencedb_bcs_index.csv`, 53,566 rows) | Raw image folder absent. 0/10 sample paths resolve. Index points to nonexistent local files. |
| **Dryad BCS** | Secondary External BCS | **AVAILABLE (Unindexed)** | `datasets/bcs/dryad_bcs/Total_sorted_DGE_images/` | 5,940 (`.tif`) | 0 | 854.26 MB | **UNINDEXED** (`bcs_index.csv` is 0 rows) | Present across folders 2, 3, 4, 5, 6, 7. Requires future audit regarding 5,940 vs 5,923 count and class 7 presence. |
| **MmCows** | Primary In-Domain Behavior | **AVAILABLE (Crops Only)** | `datasets/behavior/mmcows/cropped_bboxes/` | 427,390 (`.jpg`) | 0 | 12.64 GB | **VALID / ACTIVE** (`behavior_index.csv`, 213,686 rows) | 213,686 crops in `behaviors/` match 100% of index paths. Additional `lying/` (83.6k) & `standing/` (130.1k) crops exist. Raw videos purged. |
| **OpenCows2020** | Legacy Re-ID Baseline | **AVAILABLE** | `datasets/id/opencow2020-DatasetNinja/` | 4,736 (`.jpg`) | 0 | 91.15 MB | **VALID / ACTIVE** (`id_index.csv`, 4,736 rows) | 4,240 train + 496 test images across 46 unique cow identities. 100% path match. |
| **MultiCamCows2024** | Intended Primary Re-ID | **BLOCKED (Upstream)** | `datasets/id/raw/multicamcows2024/` *(Missing)* | 0 | 0 | 0 B | **NONE** | Official download endpoint fails with connection reset (local & cloud). 0 files locally. |
| **CattleLameness** | Historical P2 Audit | **AVAILABLE** | `datasets/lameness/` | 9,950 (`.jpg` frames) | 50 (`.mp4`) | 312.85 MB | **VALID / ACTIVE** (`lameness_index.csv`, 9,950 rows; manifest: 50 rows) | 50 MP4s (84.95 MB) + 9,950 extracted frames (227.9 MB). Excluded from Phase 3 MTL. |
| **Ruchay 2026** | Primary External BCS | **NOT_FOUND** | N/A | 0 | 0 | 0 B | **NONE** | 0 files locally. |
| **CBVD-5** | Primary External Behavior | **NOT_FOUND** | N/A | 0 | 0 | 0 B | **NONE** | 0 dataset files locally (only 1 demo clip in `workspaces/nusrat/`). |
| **SideViewCows2026**| Primary External Re-ID | **NOT_FOUND** | N/A | 0 | 0 | 0 B | **NONE** | 0 files locally. |
| **BECA-D / BECA-L** | Re-ID Scale/Stress | **NOT_FOUND** | N/A | 0 | 0 | 0 B | **NONE** | 0 files locally. |
| **Perception Models**| Cattle-Centered Upstream | **NOT_FOUND** | N/A | 0 | 0 | 0 B | **NONE** | CattleEyeView, SuperAnimal, SAM-2, MOO checkpoints not present locally. |

---

## 4. MultiCamCows2024 Blocker Investigation
- **Landing Page**: `https://data.bris.ac.uk/data/dataset/2inu67jru7a6821kkgehxg3cv2` (HTTP 200 OK).
- **Download Endpoint**: `https://data.bris.ac.uk/datasets/tar/2inu67jru7a6821kkgehxg3cv2.zip` (36.5 GiB reported).
- **Observed Behavior**:
  - Local browser (Chrome on Windows 11): `ERR_CONNECTION_RESET`.
  - Local CLI (`curl`, Python): `Recv failure: Connection was reset` / `Remote end closed connection without response`.
  - Remote cloud container (Modal US/EU datacenter): `RemoteDisconnected: Remote end closed connection without response`.
  - Other dataset download URLs under `data.bris.ac.uk/datasets/`: identically failed with connection reset.
  - Search across Zenodo, Hugging Face, Kaggle, Academic Torrents confirmed no verified secondary mirrors exist.
- **Decision**: MultiCamCows2024 remains the canonical primary Re-ID benchmark; its status is officially tracked as `BLOCKED — upstream download currently unavailable`.

---

## 5. Methodological Decisions & State Adjustments
1. **Separation of Roles vs Local Presence**: `memory/state.md` must strictly distinguish the theoretical/scientific role of a dataset from its physical presence on the executing machine.
2. **ScienceDB Status**: Marked as unpopulated locally. `sciencedb_bcs_index.csv` is retained as an unvalidated manifest, but all modeling on ScienceDB is blocked until raw data is restored.
3. **Dryad Discrepancy**: Physical scan revealed 5,940 TIFF images across classes 2 through 7 (classes 1, 8, 9 empty). Prior documentation recorded ~5,923 active images across classes 2–6. This discrepancy is flagged for a dedicated audit and is not artificially resolved.
4. **MmCows Sample Count**: The official active behavior subset corresponds strictly to the 213,686 images indexed in `behavior_index.csv` under `cropped_bboxes/behaviors/`. The presence of 83,620 lying and 130,084 standing crops (totaling 427,390 local images) does not redefine the behavior dataset size.
5. **Multi-Environment Awareness**: Physical presence on this local development laptop does not imply presence on cloud compute (Modal) or the BRACU Lab Research PC (RTX 5090). Future agents must verify disk presence before execution.

---

## 6. Next Steps
1. Proceed with Step 1: build canonical `datasets/dataset_registry.csv` incorporating verified metadata and availability statuses.
2. Maintain active tracking of the MultiCamCows2024 upstream download endpoint.
