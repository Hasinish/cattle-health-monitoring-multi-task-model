"""
Test suite for Real-Cattle Viewpoint Annotation Workspace.

Verifies:
1. Manifest integrity:
   - Exactly 1,000 rows
   - Exactly 334 ScienceDB, 333 MmCows, 333 SideViewCows2026
   - Zero duplicate source paths
   - Zero overlap with prior 160 review images
   - All 1,000 files exist locally
   - Initial values correctly set (UNREVIEWED, pending_human_review, none, none, no)
2. Server & API behavior:
   - GET / (serves HTML)
   - GET /api/status (returns correct initial status)
   - GET /api/sample?index=0 (returns sample without sensitive anchoring metadata)
   - GET /api/image?id=vp1k_0001 (returns valid image bytes with image/jpeg header)
   - POST /api/save (tests immediate autosave and resume on a temporary copy without touching real labels)
"""

import os
import sys
import shutil
import tempfile
import threading
import time
import urllib.request
import urllib.parse
import json
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MANIFEST_PATH = os.path.join(REPO_ROOT, "artifacts", "perception_audit", "viewpoint_1000_annotation_manifest.csv")
OLD_60_PATH = os.path.join(REPO_ROOT, "artifacts", "perception_audit", "viewpoint_manual_review_manifest.csv")
OLD_100_PATH = os.path.join(REPO_ROOT, "artifacts", "perception_audit", "viewpoint_expanded_agent_review_manifest.csv")

sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
from viewpoint_annotator import ManifestManager, AnnotationServerHandler
from http.server import HTTPServer


def run_checks():
    print("=== RUNNING WORKSPACE VERIFICATION CHECKS ===")
    
    # 1. Manifest Integrity
    assert os.path.exists(MANIFEST_PATH), f"Manifest not found: {MANIFEST_PATH}"
    df = pd.read_csv(MANIFEST_PATH)
    
    print(f"Total rows in manifest: {len(df)}")
    assert len(df) == 1000, f"Expected 1,000 rows, got {len(df)}"
    
    counts = df["dataset"].value_counts().to_dict()
    print(f"Dataset counts: {counts}")
    assert counts == {"ScienceDB": 334, "MmCows": 333, "SideViewCows2026": 333}, f"Incorrect counts: {counts}"
    
    assert df["source_image_path"].nunique() == 1000, "Duplicate source image paths found!"
    print("Zero duplicate source paths: PASSED")
    
    # 2. Check overlap with previous 160 review images
    old_paths = set()
    for p in [OLD_60_PATH, OLD_100_PATH]:
        if os.path.exists(p):
            old_df = pd.read_csv(p)
            for path_val in old_df["source_image_path"]:
                abs_p = os.path.abspath(os.path.join(REPO_ROOT, path_val)) if not os.path.isabs(path_val) else os.path.abspath(path_val)
                old_paths.add(abs_p)
                
    assert len(old_paths) == 160, f"Expected 160 prior paths, got {len(old_paths)}"
    
    new_paths = set(os.path.abspath(os.path.join(REPO_ROOT, p)) for p in df["source_image_path"])
    overlap = new_paths.intersection(old_paths)
    assert len(overlap) == 0, f"Fatal: {len(overlap)} overlapping paths found with prior 160 images!"
    print("Zero overlap with prior 160 review images: PASSED")
    
    # 3. Check that all 1,000 files exist locally
    missing = []
    for p in df["source_image_path"]:
        abs_p = os.path.join(REPO_ROOT, p)
        if not os.path.exists(abs_p):
            missing.append(abs_p)
    assert len(missing) == 0, f"Found {len(missing)} missing files! First few: {missing[:3]}"
    print("All 1,000 files exist locally: PASSED")
    
    # 4. Check initial values
    assert (df["viewpoint"] == "UNREVIEWED").all(), "Found non-UNREVIEWED initial viewpoint!"
    assert (df["review_status"] == "pending_human_review").all(), "Found non-pending initial review_status!"
    assert (df["occlusion"] == "none").all(), "Found non-none initial occlusion!"
    assert (df["body_cutoff"] == "none").all(), "Found non-none initial body_cutoff!"
    assert (df["multiple_cows"] == "no").all(), "Found non-no initial multiple_cows!"
    print("Initial default values: PASSED")
    
    # 5. Test Server & Autosave/Resume on a Temporary Manifest
    print("\n--- Testing Server API & Autosave/Resume ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_manifest = os.path.join(tmpdir, "test_manifest.csv")
        shutil.copy(MANIFEST_PATH, tmp_manifest)
        
        manager = ManifestManager(tmp_manifest)
        
        class TestHandler(AnnotationServerHandler):
            manager = None
        TestHandler.manager = manager
        
        test_port = 8999
        server = HTTPServer(("127.0.0.1", test_port), TestHandler)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        time.sleep(0.5)
        
        try:
            base_url = f"http://127.0.0.1:{test_port}"
            
            # Test GET /
            with urllib.request.urlopen(f"{base_url}/") as res:
                assert res.status == 200
                html = res.read().decode("utf-8")
                assert "CATTLE VIEWPOINT ANNOTATOR" in html
                assert "VP_MAP" in html
                print("  GET / (HTML): PASSED")
                
            # Test GET /api/status
            with urllib.request.urlopen(f"{base_url}/api/status") as res:
                assert res.status == 200
                status_data = json.loads(res.read().decode("utf-8"))
                assert status_data["total"] == 1000
                assert status_data["reviewed_count"] == 0
                assert status_data["first_unreviewed_index"] == 0
                print("  GET /api/status: PASSED")
                
            # Test GET /api/sample?index=0
            with urllib.request.urlopen(f"{base_url}/api/sample?index=0") as res:
                assert res.status == 200
                sample_data = json.loads(res.read().decode("utf-8"))
                assert sample_data["sample_id"] == "vp1k_0001"
                assert "dataset" not in sample_data
                assert "cow_id" not in sample_data
                assert "camera_id" not in sample_data
                assert "bcs" not in sample_data
                assert "behavior" not in sample_data
                print("  GET /api/sample (no anchoring metadata): PASSED")
                
            # Test GET /api/image?id=vp1k_0001
            with urllib.request.urlopen(f"{base_url}/api/image?id=vp1k_0001") as res:
                assert res.status == 200
                content_type = res.headers.get("Content-Type")
                assert "image" in content_type
                img_bytes = res.read()
                assert len(img_bytes) > 1000
                print(f"  GET /api/image ({len(img_bytes)} bytes, {content_type}): PASSED")
                
            # Test POST /api/save (autosave)
            save_payload = json.dumps({
                "index": 0,
                "sample_id": "vp1k_0001",
                "viewpoint": "side",
                "occlusion": "partial",
                "body_cutoff": "none",
                "multiple_cows": "yes"
            }).encode("utf-8")
            
            req = urllib.request.Request(
                f"{base_url}/api/save",
                data=save_payload,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req) as res:
                assert res.status == 200
                save_data = json.loads(res.read().decode("utf-8"))
                assert save_data["success"] is True
                assert save_data["next_index"] == 1
                print("  POST /api/save: PASSED")
                
            # Verify temporary manifest was written immediately
            saved_df = pd.read_csv(tmp_manifest)
            row0 = saved_df.iloc[0]
            assert row0["viewpoint"] == "side"
            assert row0["occlusion"] == "partial"
            assert row0["body_cutoff"] == "none"
            assert row0["multiple_cows"] == "yes"
            assert row0["review_status"] == "human_verified"
            print("  Autosave CSV persistence: PASSED")
            
            # Verify status now reflects 1 reviewed, and resumes at index 1
            with urllib.request.urlopen(f"{base_url}/api/status") as res:
                status_data2 = json.loads(res.read().decode("utf-8"))
                assert status_data2["reviewed_count"] == 1
                assert status_data2["first_unreviewed_index"] == 1
                print("  Resume at first unreviewed sample (index 1): PASSED")
                
        finally:
            server.shutdown()
            server.server_close()
            
    # Verify real manifest remains 100% untouched
    clean_df = pd.read_csv(MANIFEST_PATH)
    assert (clean_df["viewpoint"] == "UNREVIEWED").all(), "Real manifest was altered!"
    print("\nReal manifest confirmed completely untouched (all UNREVIEWED).")
    print("=== ALL CHECKS PASSED SUCCESSFULLY ===")


if __name__ == "__main__":
    run_checks()
