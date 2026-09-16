# Deep Analysis of Raw Results

## Member: Nusrat
### bcs_ce_ablation_results.txt
`	ext
---CONTEXT 3 BCS CE ABLATION---
PERSON NAME: Nusrat
BASE MODEL: EfficientNetB0
DATASET: Dryad (Cross-Entropy Loss)

EPOCHS TRAINED: 30
LOSS AT EPOCH 10: 0.045874
LOSS AT EPOCH 20: 0.004258
LOSS AT EPOCH 30: 0.000347
FINAL TRAIN LOSS: 0.000347
VAL MAE: 1.112500
VAL ACCURACY +-0 (exact match): 0.222059
VAL ACCURACY +-1 (within 1 class): 0.679412
TEST MAE: 0.482500
TEST ACCURACY +-0: 0.645000
TEST ACCURACY +-1: 0.880000
CHECKPOINT PATH: D:\T25301094 P2\workspaces\nusrat\dryad_bcs_ce_best.pth
TRAINING TIME (mins): 6.82
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---
`

### bcs_ce_sciencedb_ablation_results.txt
`	ext
---CONTEXT 3 BCS CE SCIENCEDB ABLATION---
PERSON NAME: Nusrat
BASE MODEL: EfficientNetB0
DATASET: ScienceDB (Cross-Entropy Loss)

EPOCHS TRAINED: 5 (Manually stopped)
TEST MAE: 0.694023
TEST ACCURACY +-0: 0.463124
TEST ACCURACY +-1: 0.866233
CHECKPOINT PATH: D:\T25301094 P2\workspaces\nusrat\sciencedb_bcs_ce_best.pth
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---
`

### bcs_nocbam_ablation_results.txt
`	ext
---CONTEXT 3 BCS NO-CBAM ABLATION---
PERSON NAME: Nusrat
BASE MODEL: EfficientNetB0
DATASET: Dryad (CORAL Loss, No CBAM)

EPOCHS TRAINED: 5
LOSS AT EPOCH 10: N/A
LOSS AT EPOCH 20: N/A
LOSS AT EPOCH 30: N/A
FINAL TRAIN LOSS: 0.010362
VAL MAE: 1.339706
VAL ACCURACY +-0 (exact match): 0.152941
VAL ACCURACY +-1 (within 1 class): 0.637500
TEST MAE: 0.700000
TEST ACCURACY +-0: 0.525000
TEST ACCURACY +-1: 0.807500
CHECKPOINT PATH: D:\T25301094 P2\workspaces\nusrat\dryad_bcs_nocbam_best.pth
TRAINING TIME (mins): 0.98
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---
`

### bcs_results.txt
`	ext
---CONTEXT 3 BCS---
PERSON NAME: Nusrat
BASE MODEL: EfficientNetB0

DATASET: Dryad
EPOCHS TRAINED: 30
LOSS AT EPOCH 10: 0.005635
LOSS AT EPOCH 20: 0.000713
LOSS AT EPOCH 30: 0.000111
FINAL TRAIN LOSS: 0.000111
VAL MAE: 1.013971
VAL ACCURACY +-0 (exact match): 0.257353
VAL ACCURACY +-1 (within 1 class): 0.729412
TEST MAE: 0.617500
TEST ACCURACY +-0: 0.525000
TEST ACCURACY +-1: 0.857500
CHECKPOINT PATH: D:\T25301094 P2\workspaces\nusrat\dryad_bcs_best.pth
TRAINING TIME (mins): 4.78
ANY ISSUES ENCOUNTERED: None

DATASET: ScienceDB
EPOCHS TRAINED: 30
LOSS AT EPOCH 10: 0.032411
LOSS AT EPOCH 20: 0.007906
LOSS AT EPOCH 30: 0.002366
FINAL TRAIN LOSS: 0.002366
VAL MAE: 0.495778
VAL ACCURACY +-0 (exact match): 0.596834
VAL ACCURACY +-1 (within 1 class): 0.915699
TEST MAE: 0.556640
TEST ACCURACY +-0: 0.550615
TEST ACCURACY +-1: 0.905037
CHECKPOINT PATH: D:\T25301094 P2\workspaces\nusrat\sciencedb_bcs_best.pth
TRAINING TIME (mins): 26.24
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---
`

### behavior_ce_ablation_results.txt
`	ext
---CONTEXT 3 BEHAVIOR CE ABLATION---
PERSON NAME: Nusrat
BASE MODEL: EfficientNetB0
DATASET: MmCows (capped 3000/class) (Cross-Entropy Loss)
EPOCHS TRAINED: 5
LOSS AT EPOCH 10: N/A
LOSS AT EPOCH 20: N/A
LOSS AT EPOCH 30: N/A
FINAL TRAIN LOSS: 0.248379
VAL MACRO F1: 0.757026
VAL PER-CLASS ACCURACY:
  Class 1 (Walking): 63.96%
  Class 2 (Standing): 68.13%
  Class 3 (Feeding head up): 67.13%
  Class 4 (Feeding head down): 71.00%
  Class 5 (Licking): 94.21%
  Class 6 (Drinking): 74.55%
  Class 7 (Lying): 98.53%
TEST MACRO F1: 0.707367
TEST PER-CLASS ACCURACY:
  Class 1 (Walking): 42.84%
  Class 2 (Standing): 75.00%
  Class 3 (Feeding head up): 61.33%
  Class 4 (Feeding head down): 68.60%
  Class 5 (Licking): 75.94%
  Class 6 (Drinking): 65.75%
  Class 7 (Lying): 98.47%
CHECKPOINT PATH: D:\T25301094 P2\workspaces\nusrat\behavior_ce_best.pth
TRAINING TIME (mins): 9.22
EARLY STOPPING TRIGGERED AT EPOCH: N/A
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---
`

### behavior_results.txt
`	ext
---CONTEXT 3 BEHAVIOR---
PERSON NAME: Nusrat
BASE MODEL: EfficientNetB0
DATASET: MmCows (capped 3000/class)
EPOCHS TRAINED: 24
LOSS AT EPOCH 10: 0.013290
LOSS AT EPOCH 20: 0.003363
LOSS AT EPOCH 30: N/A
FINAL TRAIN LOSS: 0.000979
VAL MACRO F1: 0.763051
VAL PER-CLASS ACCURACY:
  Class 1 (Walking): 70.72%
  Class 2 (Standing): 71.60%
  Class 3 (Feeding head up): 76.23%
  Class 4 (Feeding head down): 63.00%
  Class 5 (Licking): 96.30%
  Class 6 (Drinking): 82.55%
  Class 7 (Lying): 98.53%
TEST MACRO F1: 0.744548
TEST PER-CLASS ACCURACY:
  Class 1 (Walking): 59.78%
  Class 2 (Standing): 70.00%
  Class 3 (Feeding head up): 71.97%
  Class 4 (Feeding head down): 66.07%
  Class 5 (Licking): 83.96%
  Class 6 (Drinking): 83.01%
  Class 7 (Lying): 98.77%
CHECKPOINT PATH: D:\T25301094 P2\workspaces\nusrat\behavior_best.pth
TRAINING TIME (mins): 8.95
EARLY STOPPING TRIGGERED AT EPOCH: 24
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---
`

### cbvd_behavior_results.txt
`	ext
---CONTEXT 3 CBVD-5 EVALUATION---
PERSON NAME: Nusrat
BASE MODEL: EfficientNetB0
DATASET: CBVD-5 (2000 balanced images)
VAL MACRO F1 ON CBVD-5: 0.124517
PER-CLASS ACCURACY ON CBVD-5:
  Class 2 (Standing): 90.34%
  Class 4 (Feeding head down): 8.39%
  Class 6 (Drinking): 2.99%
  Class 7 (Lying): 24.31%
TRAINING TIME (mins): N/A (Evaluation Only)
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---
`

### id_results.txt
`	ext
---CONTEXT 3 ID---
PERSON NAME: Nusrat
BASE MODEL: EfficientNetB0
DATASET: OpenCows2020
EPOCHS TRAINED: 10
LOSS AT EPOCH 10: 0.550795
FINAL TRAIN LOSS: 0.550795
VAL TOP-1 ACCURACY: 87.06%
TEST TOP-1 ACCURACY: 86.49%
CHECKPOINT PATH: D:\T25301094 P2\workspaces\nusrat\id_best.pth
TRAINING TIME (mins): 1.15
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---
`

### lameness_results.txt
`	ext
---CONTEXT 3 LAMENESS---
PERSON NAME: Nusrat
BASE MODEL: EfficientNetB0
DATASET: CattleLameness (20 frames/video)
EPOCHS TRAINED: 10
FINAL TRAIN LOSS: 0.000011
VAL AUC: 0.994655
VAL ACCURACY: 82.70%
VAL F1 SCORE: 0.842615
VAL PER-CLASS ACCURACY:
  Class 0 (Normal): 99.76%
  Class 1 (Lame): 72.91%
TEST AUC: 0.982942
TEST ACCURACY: 93.05%
TEST F1 SCORE: 0.948293
TEST PER-CLASS ACCURACY:
  Class 0 (Normal): 90.10%
  Class 1 (Lame): 94.48%
CHECKPOINT PATH: D:\T25301094 P2\workspaces\nusrat\lameness_best.pth
TRAINING TIME (mins): 4.26
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---
`

### multitask_results.txt
`	ext
Multi-Task Model Evaluation Results
========================================

BCS - MAE: 0.7266, Exact Acc: 0.4078
Behavior - Macro F1: 0.3771
Lameness - Acc: 0.9528, AUC: 0.9921
Cow ID - Acc: 0.9496
`

### multitask_temporal_results.txt
`	ext
Spatiotemporal Multi-Task Model Evaluation Results
========================================

BCS - MAE: 0.7827, Exact Acc: 0.3931, ±1 Acc: 0.8482
Behavior - Macro F1: 0.4948
Lameness - Acc: 1.0000, AUC: 1.0000
Cow ID - Acc: 0.9758
`

### spatiotemporal_lameness_efficientnet_results.txt
`	ext
---CONTEXT 3 SPATIOTEMPORAL LAMENESS---
PERSON NAME: Nusrat
BASE MODEL: EfficientNetB0-LSTM
DATASET: CattleLameness (20 frames video sequences)
EPOCHS TRAINED: 15
FINAL TRAIN LOSS: 0.106081
VAL AUC: 1.000000
VAL ACCURACY: 100.00%
VAL F1 SCORE: 1.000000
VAL PER-CLASS ACCURACY:
  Class 0 (Normal): 100.00%
  Class 1 (Lame): 100.00%
TEST AUC: 0.840000
TEST ACCURACY: 80.00%
TEST F1 SCORE: 0.800000
TEST PER-CLASS ACCURACY:
  Class 0 (Normal): 80.00%
  Class 1 (Lame): 80.00%
CHECKPOINT PATH: D:\T25301094 P2\workspaces\nusrat\spatiotemporal_lameness_efficientnet_best.pth
TRAINING TIME (mins): 6.29
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---
`

### spatiotemporal_lameness_results.txt
`	ext
---CONTEXT 3 SPATIOTEMPORAL LAMENESS---
PERSON NAME: Nusrat
BASE MODEL: ResNet18-LSTM
DATASET: CattleLameness (20 frames video sequences)
EPOCHS TRAINED: 15
FINAL TRAIN LOSS: 0.252091
VAL AUC: 1.000000
VAL ACCURACY: 50.00%
VAL F1 SCORE: 0.666667
VAL PER-CLASS ACCURACY:
  Class 0 (Normal): 0.00%
  Class 1 (Lame): 100.00%
TEST AUC: 0.960000
TEST ACCURACY: 60.00%
TEST F1 SCORE: 0.714286
TEST PER-CLASS ACCURACY:
  Class 0 (Normal): 20.00%
  Class 1 (Lame): 100.00%
CHECKPOINT PATH: D:\T25301094 P2\workspaces\nusrat\spatiotemporal_lameness_best.pth
TRAINING TIME (mins): 2.35
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---
`

## Member: Shouvik
### bcs_results.txt
`	ext
---CONTEXT 3 BCS---
PERSON NAME: Shouvik
BASE MODEL: DenseNet121

DATASET: Dryad
EPOCHS TRAINED: 30
LOSS AT EPOCH 10: 0.0295
LOSS AT EPOCH 20: 0.0059
LOSS AT EPOCH 30: 0.0027
FINAL TRAIN LOSS: 0.0027
VAL MAE: 0.7338
VAL ACCURACY +-0 (exact match): 17.65%
VAL ACCURACY +-1 (within 1 class): 63.09%
TEST MAE: 0.5875
TEST ACCURACY +-0: 54.75%
TEST ACCURACY +-1: 86.50%
CHECKPOINT PATH: D:\T25301094 P2\workspaces\shouvik\dryad_bcs_best.pth
TRAINING TIME (mins): 15.2
ANY ISSUES ENCOUNTERED: None

DATASET: ScienceDB
EPOCHS TRAINED: 30
LOSS AT EPOCH 10: 0.1200
LOSS AT EPOCH 20: 0.0210
LOSS AT EPOCH 30: 0.0066
FINAL TRAIN LOSS: 0.0066
VAL MAE: 0.5925
VAL ACCURACY +-0 (exact match): 52.78%
VAL ACCURACY +-1 (within 1 class): 89.01%
TEST MAE: 0.6292
TEST ACCURACY +-0: 49.51%
TEST ACCURACY +-1: 88.78%
CHECKPOINT PATH: D:\T25301094 P2\workspaces\shouvik\sciencedb_bcs_best.pth
TRAINING TIME (mins): 55.5
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---
`

### behavior_results.txt
`	ext
---CONTEXT 3 BEHAVIOR---
PERSON NAME: Shouvik
BASE MODEL: DenseNet121
DATASET: MmCows (capped 3000/class)
EPOCHS TRAINED: 27
LOSS AT EPOCH 10: 0.020973
LOSS AT EPOCH 20: 0.004873
LOSS AT EPOCH 30: N/A
FINAL TRAIN LOSS: 0.001219
VAL MACRO F1: 0.7371
VAL PER-CLASS ACCURACY:
  Class 1 (Walking): 53.83%
  Class 2 (Standing): 67.00%
  Class 3 (Feeding head up): 71.53%
  Class 4 (Feeding head down): 69.47%
  Class 5 (Licking): 88.66%
  Class 6 (Drinking): 75.64%
  Class 7 (Lying): 98.33%
TEST MACRO F1: 0.7366
TEST PER-CLASS ACCURACY:
  Class 1 (Walking): 41.64%
  Class 2 (Standing): 74.70%
  Class 3 (Feeding head up): 67.40%
  Class 4 (Feeding head down): 70.70%
  Class 5 (Licking): 87.26%
  Class 6 (Drinking): 73.42%
  Class 7 (Lying): 99.37%
CHECKPOINT PATH: D:\T25301094 P2\workspaces\shouvik\behavior_best.pth
TRAINING TIME (mins): 13.81
EARLY STOPPING TRIGGERED AT EPOCH: 27
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---
`

### id_results.txt
`	ext
---CONTEXT 3 ID---
PERSON NAME: Shouvik
BASE MODEL: DenseNet121
DATASET: OpenCows2020
EPOCHS TRAINED: 10
LOSS AT EPOCH 10: 0.751364
FINAL TRAIN LOSS: 0.751364
VAL TOP-1 ACCURACY: 82.04%
TEST TOP-1 ACCURACY: 82.46%
CHECKPOINT PATH: D:\T25301094 P2\workspaces\shouvik\id_best.pth
TRAINING TIME (mins): 1.08
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---
`

### lameness_results.txt
`	ext
---CONTEXT 3 LAMENESS---
PERSON NAME: Shouvik
BASE MODEL: DenseNet121
DATASET: CattleLameness (20 frames/video)
EPOCHS TRAINED: 10
FINAL TRAIN LOSS: 0.000055
VAL AUC: 0.951221
VAL ACCURACY: 82.34%
VAL F1 SCORE: 0.877236
VAL PER-CLASS ACCURACY:
  Class 0 (Normal): 52.80%
  Class 1 (Lame): 99.30%
TEST AUC: 0.794431
TEST ACCURACY: 73.98%
TEST F1 SCORE: 0.837723
TEST PER-CLASS ACCURACY:
  Class 0 (Normal): 20.92%
  Class 1 (Lame): 99.60%
CHECKPOINT PATH: D:\T25301094 P2\workspaces\shouvik\lameness_best.pth
TRAINING TIME (mins): 15.05
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---
`

### spatiotemporal_lameness_results.txt
`	ext
---CONTEXT 3 SPATIOTEMPORAL LAMENESS---
PERSON NAME: Shouvik
BASE MODEL: DenseNet121-LSTM
DATASET: CattleLameness (20 frames video sequences)
EPOCHS TRAINED: 15
FINAL TRAIN LOSS: 0.247507
VAL AUC: 1.000000
VAL ACCURACY: 100.00%
VAL F1 SCORE: 1.000000
VAL PER-CLASS ACCURACY:
  Class 0 (Normal): 100.00%
  Class 1 (Lame): 100.00%
TEST AUC: 0.920000
TEST ACCURACY: 90.00%
TEST F1 SCORE: 0.909091
TEST PER-CLASS ACCURACY:
  Class 0 (Normal): 80.00%
  Class 1 (Lame): 100.00%
CHECKPOINT PATH: D:\T25301094 P2\workspaces\shouvik\spatiotemporal_lameness_best.pth
TRAINING TIME (mins): 9.10
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---
`

## Member: Bithi
### bcs_results.txt
`	ext
---CONTEXT 3 BCS---
PERSON NAME: Bithi
BASE MODEL: ResNet-50

DATASET: Dryad
EPOCHS TRAINED: 30
LOSS AT EPOCH 10: 0.0104
LOSS AT EPOCH 20: 0.0094
LOSS AT EPOCH 30: 0.0008
FINAL TRAIN LOSS: 0.0008
VAL MAE: 0.7485
VAL ACCURACY +-0 (exact match): 42.13%
VAL ACCURACY +-1 (within 1 class): 85.81%
TEST MAE: 0.6300
TEST ACCURACY +-0: 47.75%
TEST ACCURACY +-1: 89.25%
CHECKPOINT PATH: D:\T25301094 P2\workspaces\bithi\dryad_bcs_best.pth
TRAINING TIME (mins): 11.03
ANY ISSUES ENCOUNTERED: None

DATASET: ScienceDB
EPOCHS TRAINED: 30
LOSS AT EPOCH 10: 0.0521
LOSS AT EPOCH 20: 0.0122
LOSS AT EPOCH 30: 0.0035
FINAL TRAIN LOSS: 0.0035
VAL MAE: 0.6009
VAL ACCURACY +-0 (exact match): 52.48%
VAL ACCURACY +-1 (within 1 class): 89.51%
TEST MAE: 0.6485
TEST ACCURACY +-0: 48.43%
TEST ACCURACY +-1: 88.20%
CHECKPOINT PATH: D:\T25301094 P2\workspaces\bithi\sciencedb_bcs_best.pth
TRAINING TIME (mins): 60.51
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---
`

### behavior_results.txt
`	ext
---CONTEXT 3 BEHAVIOR---
PERSON NAME: Bithi
BASE MODEL: ResNet-50
DATASET: MmCows (capped 3000/class)
EPOCHS TRAINED: 22
LOSS AT EPOCH 10: 0.012949
LOSS AT EPOCH 20: 0.003129
LOSS AT EPOCH 30: N/A
FINAL TRAIN LOSS: 0.001347
VAL MACRO F1: 0.7210
VAL PER-CLASS ACCURACY:
  Class 1 (Walking): 64.86%
  Class 2 (Standing): 58.20%
  Class 3 (Feeding head up): 71.20%
  Class 4 (Feeding head down): 64.97%
  Class 5 (Licking): 94.68%
  Class 6 (Drinking): 84.36%
  Class 7 (Lying): 98.53%
TEST MACRO F1: 0.7037
TEST PER-CLASS ACCURACY:
  Class 1 (Walking): 54.97%
  Class 2 (Standing): 65.37%
  Class 3 (Feeding head up): 66.77%
  Class 4 (Feeding head down): 68.20%
  Class 5 (Licking): 73.11%
  Class 6 (Drinking): 79.18%
  Class 7 (Lying): 99.03%
CHECKPOINT PATH: D:\T25301094 P2\workspaces\bithi\behavior_best.pth
TRAINING TIME (mins): 10.37
EARLY STOPPING TRIGGERED AT EPOCH: 22
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---
`

### id_results.txt
`	ext
---CONTEXT 3 ID---
PERSON NAME: Bithi
BASE MODEL: ResNet-50
DATASET: OpenCows2020
EPOCHS TRAINED: 10
LOSS AT EPOCH 10: 2.121322
FINAL TRAIN LOSS: 2.121322
VAL TOP-1 ACCURACY: 52.97%
TEST TOP-1 ACCURACY: 53.02%
CHECKPOINT PATH: D:\T25301094 P2\workspaces\bithi\id_best.pth
TRAINING TIME (mins): 1.20
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---
`

### lameness_results.txt
`	ext
---CONTEXT 3 LAMENESS---
PERSON NAME: Bithi
BASE MODEL: ResNet-50
DATASET: CattleLameness (20 frames/video)
EPOCHS TRAINED: 10
FINAL TRAIN LOSS: 0.000052
VAL AUC: 0.932106
VAL ACCURACY: 85.71%
VAL F1 SCORE: 0.887018
VAL PER-CLASS ACCURACY:
  Class 0 (Normal): 81.27%
  Class 1 (Lame): 88.27%
TEST AUC: 0.744353
TEST ACCURACY: 80.97%
TEST F1 SCORE: 0.875186
TEST PER-CLASS ACCURACY:
  Class 0 (Normal): 43.79%
  Class 1 (Lame): 98.92%
CHECKPOINT PATH: D:\T25301094 P2\workspaces\bithi\lameness_best.pth
TRAINING TIME (mins): 8.05
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---
`

### spatiotemporal_lameness_results.txt
`	ext
---CONTEXT 3 SPATIOTEMPORAL LAMENESS---
PERSON NAME: Bithi
BASE MODEL: ResNet50-LSTM
DATASET: CattleLameness (20 frames video sequences)
EPOCHS TRAINED: 15
FINAL TRAIN LOSS: 0.191912
VAL AUC: 1.000000
VAL ACCURACY: 100.00%
VAL F1 SCORE: 1.000000
VAL PER-CLASS ACCURACY:
  Class 0 (Normal): 100.00%
  Class 1 (Lame): 100.00%
TEST AUC: 1.000000
TEST ACCURACY: 80.00%
TEST F1 SCORE: 0.833333
TEST PER-CLASS ACCURACY:
  Class 0 (Normal): 60.00%
  Class 1 (Lame): 100.00%
CHECKPOINT PATH: D:\T25301094 P2\workspaces\bithi\spatiotemporal_lameness_best.pth
TRAINING TIME (mins): 9.18
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---
`

## Member: Namira
### bcs_results.txt
`	ext
---CONTEXT 3 BCS---
PERSON NAME: Namira
BASE MODEL: MobileNetV3-Small

DATASET: Dryad
EPOCHS TRAINED: 30
LOSS AT EPOCH 10: 0.0748
LOSS AT EPOCH 20: 0.0458
LOSS AT EPOCH 30: 0.0112
FINAL TRAIN LOSS: 0.0112
VAL MAE: 0.6324
VAL ACCURACY +-0 (exact match): 43.38%
VAL ACCURACY +-1 (within 1 class): 93.60%
TEST MAE: 0.5250
TEST ACCURACY +-0: 61.50%
TEST ACCURACY +-1: 86.00%
CHECKPOINT PATH: D:\T25301094 P2\workspaces\namira\dryad_bcs_best.pth
TRAINING TIME (mins): 4.42
ANY ISSUES ENCOUNTERED: None

DATASET: ScienceDB
EPOCHS TRAINED: 30
LOSS AT EPOCH 10: 0.3237
LOSS AT EPOCH 20: 0.1983
LOSS AT EPOCH 30: 0.1219
FINAL TRAIN LOSS: 0.1219
VAL MAE: 0.6641
VAL ACCURACY +-0 (exact match): 47.90%
VAL ACCURACY +-1 (within 1 class): 86.21%
TEST MAE: 0.7090
TEST ACCURACY +-0: 44.67%
TEST ACCURACY +-1: 86.47%
CHECKPOINT PATH: D:\T25301094 P2\workspaces\namira\sciencedb_bcs_best.pth
TRAINING TIME (mins): 132.40
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---
`

### behavior_results.txt
`	ext
---CONTEXT 3 BEHAVIOR---
PERSON NAME: Namira
BASE MODEL: MobileNetV3-Small
DATASET: MmCows (capped 3000/class)
EPOCHS TRAINED: 30
LOSS AT EPOCH 10: 0.040425
LOSS AT EPOCH 20: 0.015269
LOSS AT EPOCH 30: 0.004966
FINAL TRAIN LOSS: 0.004966
VAL MACRO F1: 0.7271
VAL PER-CLASS ACCURACY:
  Class 1 (Walking): 72.97%
  Class 2 (Standing): 64.13%
  Class 3 (Feeding head up): 74.13%
  Class 4 (Feeding head down): 56.47%
  Class 5 (Licking): 94.91%
  Class 6 (Drinking): 81.82%
  Class 7 (Lying): 96.10%
TEST MACRO F1: 0.6810
TEST PER-CLASS ACCURACY:
  Class 1 (Walking): 61.42%
  Class 2 (Standing): 60.93%
  Class 3 (Feeding head up): 65.83%
  Class 4 (Feeding head down): 66.93%
  Class 5 (Licking): 59.43%
  Class 6 (Drinking): 80.82%
  Class 7 (Lying): 94.77%
CHECKPOINT PATH: D:\T25301094 P2\workspaces\namira\behavior_best.pth
TRAINING TIME (mins): 6.97
EARLY STOPPING TRIGGERED AT EPOCH: N/A
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---
`

### id_results.txt
`	ext
---CONTEXT 3 ID---
PERSON NAME: Namira
BASE MODEL: MobileNetV3-Small
DATASET: OpenCows2020
EPOCHS TRAINED: 10
LOSS AT EPOCH 10: 1.102187
FINAL TRAIN LOSS: 1.102187
VAL TOP-1 ACCURACY: 80.21%
TEST TOP-1 ACCURACY: 78.83%
CHECKPOINT PATH: D:\T25301094 P2\workspaces\namira\id_best.pth
TRAINING TIME (mins): 0.90
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---
`

### lameness_results.txt
`	ext
---CONTEXT 3 LAMENESS---
PERSON NAME: Namira
BASE MODEL: MobileNetV3-Small
DATASET: CattleLameness (20 frames/video)
EPOCHS TRAINED: 10
FINAL TRAIN LOSS: 0.000024
VAL AUC: 0.998179
VAL ACCURACY: 94.85%
VAL F1 SCORE: 0.960969
VAL PER-CLASS ACCURACY:
  Class 0 (Normal): 86.37%
  Class 1 (Lame): 99.72%
TEST AUC: 0.722759
TEST ACCURACY: 75.66%
TEST F1 SCORE: 0.839425
TEST PER-CLASS ACCURACY:
  Class 0 (Normal): 36.96%
  Class 1 (Lame): 94.34%
CHECKPOINT PATH: D:\T25301094 P2\workspaces\namira\lameness_best.pth
TRAINING TIME (mins): 7.09
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---
`

### spatiotemporal_lameness_results.txt
`	ext
---CONTEXT 3 SPATIOTEMPORAL LAMENESS---
PERSON NAME: Namira
BASE MODEL: MobileNetV3-LSTM
DATASET: CattleLameness (20 frames video sequences)
EPOCHS TRAINED: 15
FINAL TRAIN LOSS: 0.305853
VAL AUC: 0.888889
VAL ACCURACY: 66.67%
VAL F1 SCORE: 0.666667
VAL PER-CLASS ACCURACY:
  Class 0 (Normal): 66.67%
  Class 1 (Lame): 66.67%
TEST AUC: 0.920000
TEST ACCURACY: 70.00%
TEST F1 SCORE: 0.727273
TEST PER-CLASS ACCURACY:
  Class 0 (Normal): 60.00%
  Class 1 (Lame): 80.00%
CHECKPOINT PATH: D:\T25301094 P2\workspaces\namira\spatiotemporal_lameness_best.pth
TRAINING TIME (mins): 9.02
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---
`

## Member: Hasin
### bcs_results.txt
`	ext
---CONTEXT 3 BCS---
PERSON NAME: Hasin
BASE MODEL: ResNet-18

DATASET: Dryad
EPOCHS TRAINED: 30
LOSS AT EPOCH 10: 0.008938
LOSS AT EPOCH 20: 0.001953
LOSS AT EPOCH 30: 0.001819
FINAL TRAIN LOSS: 0.001819
VAL MAE: 0.703676
VAL ACCURACY +-0 (exact match): 31.25%
VAL ACCURACY +-1 (within 1 class): 98.38%
TEST MAE: 0.867500
TEST ACCURACY +-0: 24.50%
TEST ACCURACY +-1: 88.75%
CHECKPOINT PATH: D:\T25301094 P2\workspaces\hasin\dryad_bcs_best.pth
TRAINING TIME (mins): 2.21
ANY ISSUES ENCOUNTERED: None

DATASET: ScienceDB
EPOCHS TRAINED: 30
LOSS AT EPOCH 10: 0.043623
LOSS AT EPOCH 20: 0.011708
LOSS AT EPOCH 30: 0.003786
FINAL TRAIN LOSS: 0.003786
VAL MAE: 0.518865
VAL ACCURACY +-0 (exact match): 58.72%
VAL ACCURACY +-1 (within 1 class): 90.40%
TEST MAE: 0.580019
TEST ACCURACY +-0: 54.81%
TEST ACCURACY +-1: 89.02%
CHECKPOINT PATH: D:\T25301094 P2\workspaces\hasin\sciencedb_bcs_best.pth
TRAINING TIME (mins): 21.87
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---
`

### behavior_results.txt
`	ext
---CONTEXT 3 BEHAVIOR---
PERSON NAME: Hasin
BASE MODEL: ResNet-18
DATASET: MmCows (capped 3000/class)
EPOCHS TRAINED: 18
LOSS AT EPOCH 10: 0.0139
LOSS AT EPOCH 20: N/A
LOSS AT EPOCH 30: N/A
FINAL TRAIN LOSS: 0.0034
VAL MACRO F1: 0.7241
VAL PER-CLASS ACCURACY:
  Class 1 (Walking): 60.14%
  Class 2 (Standing): 59.23%
  Class 3 (Feeding head up): 66.10%
  Class 4 (Feeding head down): 73.13%
  Class 5 (Licking): 91.20%
  Class 6 (Drinking): 77.09%
  Class 7 (Lying): 98.13%
TEST MACRO F1: 0.7134
TEST PER-CLASS ACCURACY:
  Class 1 (Walking): 47.21%
  Class 2 (Standing): 66.90%
  Class 3 (Feeding head up): 64.63%
  Class 4 (Feeding head down): 73.97%
  Class 5 (Licking): 69.34%
  Class 6 (Drinking): 81.51%
  Class 7 (Lying): 96.90%
CHECKPOINT PATH: D:\T25301094 P2\workspaces\hasin\behavior_best.pth
TRAINING TIME (mins): 4.62
EARLY STOPPING TRIGGERED AT EPOCH: 18
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---
`

### id_results.txt
`	ext
---CONTEXT 3 ID---
PERSON NAME: Hasin
BASE MODEL: ResNet-18
DATASET: OpenCows2020
EPOCHS TRAINED: 10
LOSS AT EPOCH 10: 2.415831
FINAL TRAIN LOSS: 2.415831
VAL TOP-1 ACCURACY: 42.01%
TEST TOP-1 ACCURACY: 45.56%
CHECKPOINT PATH: D:\T25301094 P2\workspaces\hasin\id_best.pth
TRAINING TIME (mins): 1.66
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---
`

### lameness_results.txt
`	ext
---CONTEXT 3 LAMENESS---
PERSON NAME: Hasin
BASE MODEL: ResNet-18
DATASET: CattleLameness (20 frames/video)
EPOCHS TRAINED: 10
FINAL TRAIN LOSS: 0.000117
VAL AUC: 0.994238
VAL ACCURACY: 95.30%
VAL F1 SCORE: 0.964310
VAL PER-CLASS ACCURACY:
  Class 0 (Normal): 87.10%
  Class 1 (Lame): 100.00%
TEST AUC: 0.720048
TEST ACCURACY: 75.30%
TEST F1 SCORE: 0.845191
TEST PER-CLASS ACCURACY:
  Class 0 (Normal): 24.13%
  Class 1 (Lame): 100.00%
CHECKPOINT PATH: D:\T25301094 P2\workspaces\hasin\lameness_best.pth
TRAINING TIME (mins): 7.95
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---
`

### spatiotemporal_lameness_results.txt
`	ext
---CONTEXT 3 SPATIOTEMPORAL LAMENESS---
PERSON NAME: Hasin
BASE MODEL: ResNet18-LSTM
DATASET: CattleLameness (20 frames video sequences)
EPOCHS TRAINED: 15
FINAL TRAIN LOSS: 0.582468
VAL AUC: 0.888889
VAL ACCURACY: 83.33%
VAL F1 SCORE: 0.857143
VAL PER-CLASS ACCURACY:
  Class 0 (Normal): 66.67%
  Class 1 (Lame): 100.00%
TEST AUC: 0.880000
TEST ACCURACY: 70.00%
TEST F1 SCORE: 0.769231
TEST PER-CLASS ACCURACY:
  Class 0 (Normal): 40.00%
  Class 1 (Lame): 100.00%
CHECKPOINT PATH: D:\T25301094 P2\workspaces\hasin\spatiotemporal_lameness_best.pth
TRAINING TIME (mins): 8.81
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---
`

