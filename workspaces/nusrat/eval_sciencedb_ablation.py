import os
import torch
from ablation_bcs_ce_sciencedb import evaluate, BCSDataset, get_transforms, BCSModelCE, DataLoader
from sklearn.metrics import mean_absolute_error, accuracy_score

BASE_DIR = r"D:\T25301094 P2"
WORKSPACE_DIR = os.path.join(BASE_DIR, "workspaces", "nusrat")
SCIENCE_DB_CSV = os.path.join(BASE_DIR, "datasets", "bcs", "sciencedb_bcs_cropped_index.csv")
CHECKPOINT_PATH = os.path.join(WORKSPACE_DIR, "sciencedb_bcs_ce_best.pth")
RESULTS_TXT = os.path.join(WORKSPACE_DIR, "bcs_ce_sciencedb_ablation_results.txt")

DEVICE = torch.device("cuda")
BATCH_SIZE = 256
PERSON_NAME = "Nusrat"
BASE_MODEL = "EfficientNetB0"
EPOCHS = 5

def main():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint {CHECKPOINT_PATH} not found. Did the model save anything yet?")
        return

    print("Loading test dataset...")
    test_loader = DataLoader(BCSDataset(SCIENCE_DB_CSV, "test", get_transforms("test")),
                             batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    print("Loading best model from checkpoint...")
    best_model = BCSModelCE().to(DEVICE).to(memory_format=torch.channels_last)
    best_model.load_state_dict(torch.load(CHECKPOINT_PATH, weights_only=True))
    best_model.eval()

    print("Evaluating best model on Test set...")
    test_metrics = evaluate(best_model, test_loader, "Test Evaluation")

    report = f"""---CONTEXT 3 BCS CE SCIENCEDB ABLATION---
PERSON NAME: {PERSON_NAME}
BASE MODEL: {BASE_MODEL}
DATASET: ScienceDB (Cross-Entropy Loss)

EPOCHS TRAINED: {EPOCHS} (Manually stopped)
TEST MAE: {test_metrics['mae']:.6f}
TEST ACCURACY +-0: {test_metrics['acc_exact']:.6f}
TEST ACCURACY +-1: {test_metrics['acc_within_1']:.6f}
CHECKPOINT PATH: {CHECKPOINT_PATH}
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---
"""
    with open(RESULTS_TXT, "w", encoding="utf-8") as f:
        f.write(report)
    print("\n" + report)
    print(f"Saved results to {RESULTS_TXT}")

if __name__ == "__main__":
    main()
