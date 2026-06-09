import subprocess
import os
import sys
import time

# List of scripts to run sequentially
SCRIPTS = [
    os.path.join("workspaces", "hasin", "train_lameness.py"),
    os.path.join("workspaces", "hasin", "train_spatiotemporal_lameness.py"),
    os.path.join("workspaces", "hasin", "train_id.py"),
    
    os.path.join("workspaces", "namira", "train_lameness.py"),
    os.path.join("workspaces", "namira", "train_spatiotemporal_lameness.py"),
    os.path.join("workspaces", "namira", "train_id.py"),
    
    os.path.join("workspaces", "bithi", "train_lameness.py"),
    os.path.join("workspaces", "bithi", "train_spatiotemporal_lameness.py"),
    os.path.join("workspaces", "bithi", "train_id.py"),
    
    os.path.join("workspaces", "shouvik", "train_lameness.py"),
    os.path.join("workspaces", "shouvik", "train_spatiotemporal_lameness.py"),
    os.path.join("workspaces", "shouvik", "train_id.py")
]

def main():
    print("=" * 70)
    print("CATTLE HEALTH MONITORING SYSTEM — MASTER TRAINING RUNNER")
    print("=" * 70)
    print(f"Total scripts queued: {len(SCRIPTS)}")
    print("Running sequentially to prevent GPU VRAM exhaustion...\n")
    
    results = {}
    overall_start_time = time.time()
    
    for i, script_path in enumerate(SCRIPTS, start=1):
        if not os.path.exists(script_path):
            print(f"[{i}/{len(SCRIPTS)}] Warning: Script not found at {script_path}. Skipping.")
            results[script_path] = "SKIPPED (File not found)"
            continue
            
        print("-" * 70)
        print(f"[{i}/{len(SCRIPTS)}] Running: {script_path}")
        print("-" * 70)
        
        script_start_time = time.time()
        
        try:
            # We run the script in a subprocess, piping stderr & stdout to our console in real-time
            res = subprocess.run([sys.executable, script_path], check=False)
            elapsed = (time.time() - script_start_time) / 60
            
            if res.returncode == 0:
                print(f"\n=> SUCCESS: Finished {script_path} in {elapsed:.2f} mins\n")
                results[script_path] = f"SUCCESS ({elapsed:.2f} mins)"
            else:
                print(f"\n=> FAILURE: {script_path} failed with exit code {res.returncode} after {elapsed:.2f} mins\n")
                results[script_path] = f"FAILED (exit code {res.returncode})"
                
        except Exception as e:
            elapsed = (time.time() - script_start_time) / 60
            print(f"\n=> ERROR running {script_path}: {e}\n")
            results[script_path] = f"ERROR ({str(e)})"
            
    total_elapsed = (time.time() - overall_start_time) / 60
    
    print("\n" + "=" * 70)
    print("ALL RUNS COMPLETED. SUMMARY OF RESULTS:")
    print("=" * 70)
    
    for script_path, status in results.items():
        print(f"{script_path:<60} : {status}")
        
    print("=" * 70)
    print(f"Total Execution Time: {total_elapsed:.2f} mins")
    print("=" * 70)

if __name__ == "__main__":
    main()
