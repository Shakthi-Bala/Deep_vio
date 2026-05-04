# DeepVIO Experiment Results & Analysis
## Date: May 3, 2026

---

## Baseline Results (First Run)

| Model | Params | Best Epoch | Val Loss | Test Loss | Test Pos | Test Rot | ATE (m) |
|-------|--------|-----------|----------|-----------|----------|----------|---------|
| **IMU** | 832K | 3/13 | 0.3618 | 0.5430 | 0.5321 | 0.0109 | 30.25 |
| **Visual** | 735K | 21/31 | 0.2527 | 0.3017 | 0.2815 | 0.0202 | 42.32 |
| **Combined** | 2.18M | 1/11 | 0.3637 | 0.5129 | 0.4970 | 0.0159 | 30.37 |

---

## Key Findings

### 🔴 What DIDN'T Work

#### 1. Combined Model is WORSE than Individual Models
- **Expected**: Fused VIO > Visual-only > IMU-only
- **Actual**: Combined (0.5129 test loss) ≈ IMU (0.5430) > Visual (0.3017)
- **Root Cause**: Massive overfitting. Train loss dropped to 0.06 by epoch 8 but val never improved past epoch 1
- **The model memorized training data instead of learning generalizable fusion**
- **Fix**: Lower LR, dropout increase, possibly curriculum learning

#### 2. IMU Model Overfits Extremely Fast
- **Best at epoch 3**, early stopped at 13
- **Train loss 0.073 vs Val loss 0.43** — 6x gap = severe overfitting
- **IMU sequences are very similar within trajectory families** — model memorizes patterns
- **Fix**: More regularization, data augmentation on IMU (noise injection), lower LR

#### 3. Combined ATE ≈ IMU ATE, Not Visual ATE
- **Combined ATE 30.37m ≈ IMU ATE 30.25m** — the fusion is basically ignoring vision
- **The IMU encoder dominates** because it converges faster and has lower training loss
- **The visual branch gets suppressed** during joint training
- **Fix**: Pretrain visual encoder separately, freeze, then finetune jointly. Or use loss balancing.

#### 4. High ATE Despite Low Loss (Visual)
- **Visual has LOWEST test loss (0.3017) but HIGHEST ATE (42.32m)**
- **This is a classic issue**: small per-step errors accumulate during trajectory integration
- **Visual model makes consistent small biases** that add up over 500 steps
- **IMU has higher loss but lower ATE** because its errors are more random (cancel out)

### 🟢 What DID Work

#### 1. Visual Model Generalizes Well
- **Val/Test gap is small**: Val=0.2527, Test=0.3017 (1.2x ratio)
- **Trained for 21 epochs** without early collapse — good learning dynamics
- **Rotation loss (0.0202) is reasonable** — visual cues give strong orientation info

#### 2. Data Pipeline Works at Scale
- **240 train / 40 val / 26 test sequences** — 47,760 training samples
- **RTX 5060 handled all models** without OOM at bs=32
- **~8 min/epoch** for visual/combined, ~8 min for IMU at bs=64

#### 3. Rotation Estimation is Much Easier Than Translation
- **All models**: rotation loss (0.01-0.02) << position loss (0.2-0.5)
- **~20x easier to predict rotation** than position
- **This matches VIO literature** — rotation from gyro is accurate, position from double-integration drifts

#### 4. Early Stopping Worked Correctly
- **Caught all overfitting cases** within patience=10
- **IMU: stopped at 13, best=3** (saved 10 wasted epochs)
- **Combined: stopped at 11, best=1** (saved 10 wasted epochs)
- **Visual: stopped at 31, best=21** (still learning slowly — could benefit from more epochs)

---

## Diagnosis: Why Combined Model Fails

### The Modality Imbalance Problem
```
Epoch 1:  IMU train=0.27  Visual would be ~0.47
Epoch 5:  IMU train=0.07  Visual would be ~0.40
Epoch 10: IMU train=0.06  Visual would be ~0.31
```

The IMU encoder converges **5-10x faster** than the visual encoder because:
1. IMU features are lower-dimensional (6D vs 6×224×224)
2. LSTM processes sequential IMU directly — strong inductive bias
3. Conv visual encoder needs many epochs to learn useful features

**Result**: Gradients flow primarily through IMU path → visual branch gets negligible updates → fusion = IMU with extra parameters to overfit.

---

## Overnight Experiments (Running Now)

| # | Experiment | Hypothesis | Changes |
|---|-----------|-----------|---------|
| 1 | Combined LR=1e-4 | Slower training prevents IMU domination | LR 1e-3→1e-4, 80 epochs |
| 2 | Combined No Attention | Attention on single embedding is wasteful | Remove attention, LR=5e-4 |
| 3 | Visual LR=5e-4 | Visual can improve more with fine-tuning | LR 1e-3→5e-4, 80 epochs |
| 4 | IMU No Attention, bs=128 | Test regularization effect | No attention, larger batch |
| 5 | Combined λ_q=10 | Force more rotation learning | Balance loss weighting |
| 6 | Combined λ_q=5, no attn | Combined fix attempt | Lower LR + balanced loss |

### Estimated Timeline (~8 hrs total)
- Exp 1: ~80 min (80 epochs × 1 min/epoch if stops early, ~8 min/epoch if full)
- Exp 2: ~80 min
- Exp 3: ~80 min  
- Exp 4: ~40 min (IMU is faster)
- Exp 5: ~80 min
- Exp 6: ~80 min
- Final eval: ~10 min

---

## Data Summary

### Generated Dataset
```
output/
├── train/  240 sequences (15 traj types × ~16 textures)
├── val/     40 sequences  
├── test/    26 sequences
└── seq_001-005 (old flat format, unused)
```

- **Each sequence**: 20s, 200 camera frames, 20000 IMU samples, 199 relative pose pairs
- **Total samples**: Train=47,760 | Val=7,960 | Test=4,975

### Hardware
- **GPU**: NVIDIA GeForce RTX 5060 Laptop GPU (8GB VRAM)
- **CUDA**: 12.8
- **Throughput**: ~8 min/epoch for visual/combined at bs=32

---

## Next Steps (After Overnight Results)

### If Combined Still Overfits:
1. **Two-stage training**: Pretrain visual encoder with DeepVO, freeze, then fine-tune combined
2. **Gradient scaling**: Scale visual branch gradients up by 3-5x
3. **Curriculum learning**: Train on easy trajectories first, add harder ones

### If Visual ATE is Still High:
1. **Temporal modeling**: Add LSTM over sequence of visual features (not just pairs)
2. **Optical flow input**: Replace raw image pairs with precomputed flow
3. **Scale-aware loss**: Add ATE-like metric directly to training loss

### If IMU Still Overfits:
1. **IMU augmentation**: Random bias injection, noise scaling, temporal jittering
2. **Smaller LSTM**: Reduce hidden size from 128 to 64
3. **Dropout increase**: 0.2 → 0.4

---

## How to Check Results in 8 Hours

```bash
# SSH to lab laptop
ssh adipat@192.168.1.185
# password: 4829

# Check if experiments are still running
ps aux | grep overnight

# Check progress
tail -50 ~/Documents/Spring_26/CV/p4/DeepVIO/overnight_full.log

# See the full summary (at the bottom of the log when done)
tail -30 ~/Documents/Spring_26/CV/p4/DeepVIO/overnight_results.log

# Quick results
cat ~/Documents/Spring_26/CV/p4/DeepVIO/overnight_results.log | grep -A 20 "OVERNIGHT EXPERIMENT SUMMARY"
```

---

**Status**: Overnight experiments RUNNING (PID 305140)  
**Started**: Sun May 3 11:23:00 EDT 2026  
**Expected Done**: ~Sun May 3 19:00-20:00 EDT 2026
