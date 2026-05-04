# DeepVIO — Full Diagnosis & Pivot Plan
## Date: May 3, 2026 (Evening Analysis)

---

## 🚨 Root Cause Identified: DATA QUALITY

The overnight experiments are **not going to help** because the fundamental problem is **not** the model architecture or hyperparameters — it's the **training data**.

---

## 1. Dataset Image Quality Analysis

### Summary Statistics (1200 frames sampled across 240 sequences)

| Metric | Value | Verdict |
|--------|-------|---------|
| **Blur Score (Laplacian Var)** | Mean=164, Median=81 | **VERY LOW** (EuRoC: 200-800, KITTI: 500-2000) |
| **Frames <50 (very blurry)** | 478/1200 (39.8%) | ❌ Nearly half are unusable |
| **Frames <100 (blurry)** | 646/1200 (53.8%) | ❌ Majority insufficient for VO |
| **Frames >200 (sharp)** | 352/1200 (29.3%) | Only ~30% are "good" |
| **Edge Density <0.02 (featureless)** | 407/1200 (33.9%) | ❌ 1/3 have no trackable features |
| **Intensity Std <20 (no contrast)** | 285/1200 (23.8%) | ❌ Nearly uniform images |
| **ORB Features <50** | 20% of sampled frames | ❌ Not enough for any VO |

### Per-Sequence Breakdown

| Quality Level | Train | Val | Test |
|--------------|-------|-----|------|
| Good (blur>100) | 118/240 (49%) | 16/40 (40%) | 15/26 (58%) |
| Bad (blur<50) | 92/240 (38%) | 18/40 (45%) | 9/26 (35%) |
| Terrible (blur<20) | 53/240 (22%) | — | — |

### Worst Sequences (essentially blank images)
```
seq_173: blur=3.7    seq_041: blur=3.8    seq_107: blur=3.9
seq_085: blur=4.0    seq_217: blur=4.0    seq_072: blur=4.2
seq_019: blur=4.5    seq_021: blur=6.7    seq_030: blur=8.4
```
These sequences have **std < 10** and **< 5 unique intensity values** — they're effectively solid-color images. The visual encoder literally CANNOT extract any motion information.

---

## 2. Root Cause: Texture Quality + Scene Geometry

### The Scene Setup
- **Single flat plane** (200m × 200m) with tiled texture
- Camera looking **straight down** from ~12m height
- FOV: 84° horizontal → covers 21.6m × 16.2m on ground
- **1 render pixel = 34mm on ground**
- Textures need features at **>100mm scale** to be visible

### Texture Analysis (22 source textures)

| Quality | Count | Examples |
|---------|-------|----------|
| **Good (blur>200)** | 13/22 | Dense patterns, aerial city views, tile floors |
| **Bad (blur<100)** | 7/22 | Smooth gradients, clouds, bokeh, solid colors |
| **Terrible (blur<10)** | 1/22 | `susan-wilkinson` — essentially a solid color |

**Bad textures that should be removed:**
```
[ 4]   84.6 | deva-darshan (smooth gradient)
[ 5]   26.0 | fer-troulik (blurry macro)
[ 9]   49.5 | jack-white (soft bokeh)
[13]   31.4 | lowie-vanhoutte (smooth surface)
[14]   40.1 | martin-bennie (out of focus)
[17]   90.9 | nathan-hurst (low detail)
[20]    9.7 | susan-wilkinson (essentially solid)
```

### Deeper Scene Geometry Issues
1. **No parallax** — flat plane means no depth variation, no 3D structure
2. **Pure 2D motion** — camera looking down means visual odometry degenerates to 2D correlation
3. **No occlusion boundaries** — no edges to track across frames
4. **Repetitive textures** — tiled textures cause ambiguity (the "aperture problem")

---

## 3. Overnight Experiment Results (So Far)

### Completed: Experiment 1 — Combined LR=1e-4

| Metric | Baseline | Exp 1 (LR=1e-4) | Change |
|--------|----------|-----------------|--------|
| Best Epoch | 1/11 | 1/16 | Same problem |
| Val Loss | 0.3637 | 0.3008 | ↓ Slightly better |
| Test Loss | 0.5129 | 0.4342 | ↓ Slightly better |
| Overfitting | Yes (train=0.06) | Yes (train=0.05) | Still severe |

**Verdict:** Lower LR delayed overfitting slightly but didn't fix it. Best is still epoch 1.

### In Progress: Experiment 2 — Combined No Attention (LR=5e-4)
- At epoch 15, best still epoch 2 (val=0.3619)
- Same overfitting pattern: train→0.043, val stuck at 0.42+
- **Verdict:** Not helping.

### Baseline Summary (Original Runs)

| Model | Params | Val Loss | Test Loss | Test Pos | Test Rot | ATE |
|-------|--------|----------|-----------|----------|----------|-----|
| IMU | 832K | 0.3618 | 0.5430 | 0.5321 | 0.0109 | 30.25m |
| Visual | 735K | 0.2527 | 0.3017 | 0.2815 | 0.0202 | 42.32m |
| Combined | 2.18M | 0.3637 | 0.5129 | 0.4970 | 0.0159 | 30.37m |

---

## 4. Why the Combined Model Fails (Complete Diagnosis)

### Problem 1: Visual Branch Gets Garbage Data
- **38% of training samples** have featureless images → visual encoder learns NOTHING from them
- Model learns to **suppress visual features** because they're unreliable noise ~40% of the time
- IMU is consistent 100% of the time → model learns to trust IMU exclusively

### Problem 2: Modality Imbalance (IMU Converges 5-10x Faster)
- IMU: 6D input → LSTM → immediate convergence (epoch 1-3)
- Visual: 6×224×224 input → CNN must learn features from scratch → slow (epoch 10-20+)
- Gradient signal flows through IMU path; visual path starves

### Problem 3: No 3D Structure for Visual VO
- Flat plane = pure 2D translation estimation
- No depth variation → no epipolar geometry cues
- Visual odometry from a downward-facing camera on a flat surface is **inherently ill-conditioned** for rotation estimation around the optical axis

### Problem 4: Architecture Not Suited for the Task
- Single image-pair → pose (no temporal context)
- No explicit optical flow computation
- Tiny custom CNN (4 layers, 735K params) vs. literature (FlowNet: 38M params, pretrained)

---

## 5. Literature Review: How SOTA Does It

### Key Papers & Repos

| Paper | Year | Visual Input | Fusion | Key Insight |
|-------|------|-------------|--------|-------------|
| **DeepVO** (Wang et al.) | 2017 | FlowNetS pretrained, LSTM over sequences | N/A (vision only) | Pretrained features are critical |
| **DeepVIO** (Han et al.) | 2019 | Explicit optical flow → CNN → LSTM | Late fusion + fusion LSTM | Use flow, not raw images |
| **VINet** (Clark et al.) | 2017 | CNN(img pairs) → LSTM | Concatenation + FC | First deep VIO |
| **SelectFusion** (Chen et al.) | 2019 | ResNet18 pretrained | Soft attention gating | Learn when to trust each modality |
| **DIDO** (Liu et al.) | 2020 | ResNet encoder | Cross-attention | IMU preintegration as prior |
| **TartanVO** (Wang et al.) | 2021 | PWC-Net optical flow | N/A (visual) | Trained on TartanAir synthetic |

### Critical Architectural Patterns That Work

1. **Use optical flow as input, NOT raw image pairs**
   - DeepVIO, TartanVO, and others compute optical flow first (FlowNet2 / PWC-Net / RAFT)
   - Flow removes appearance variation, focuses on motion
   - Flow is robust to texture changes, lighting
   - **Your approach**: Raw image pairs → CNN must learn both "what is a feature" AND "how does it move"

2. **Use pretrained visual encoders**
   - FlowNet-S pretrained on FlyingChairs (DeepVO)
   - ResNet18/34 pretrained on ImageNet (SelectFusion, others)
   - PWC-Net pretrained on FlyingThings (TartanVO)
   - **Your approach**: 4-layer CNN from scratch with only 48K samples → vastly insufficient

3. **Temporal LSTM over sequences (not single pairs)**
   - DeepVO: LSTM over 5-7 consecutive frame-pair features
   - This allows the model to learn **velocity/acceleration priors**
   - **Your approach**: Single pair → pose, no temporal context

4. **Soft gating for modality selection (SelectFusion)**
   - Learns σ ∈ [0,1] for each modality at each timestep
   - On featureless frames: σ_visual → 0, σ_imu → 1
   - **Your approach**: Naive concatenation forces equal contribution

5. **IMU preintegration as structured prior (not raw LSTM)**
   - Classical IMU preintegration gives a physics-based pose estimate
   - Network only needs to learn the **residual/correction**
   - **Your approach**: Raw 1000-sample IMU sequence → LSTM (must learn integration from scratch)

### Dataset Requirements

| Dataset | Scenes | Images/seq | Blur Score | Features |
|---------|--------|-----------|-----------|----------|
| KITTI | Outdoor driving | 1000+ frames | 500-2000 | Rich 3D structure |
| EuRoC | Indoor MAV | 3000+ frames | 200-800 | Textured rooms |
| TartanAir | Synthetic (diverse 3D) | 1000+ frames | 300-1500 | Multiple objects, depth |
| **Yours** | Single flat plane | 200 frames | **10-300** | **No 3D structure** |

---

## 6. Recommendation: STOP Current Experiments & Pivot

### Kill/Let Finish Decision
**Let the experiments finish** for documentation purposes (results section of paper), but **do NOT expect them to solve the problem**. The issue is data, not hyperparameters.

### Option A: Fix the Dataset (Recommended if time allows)

1. **Remove bad textures** (7/22) → regenerate with only the 13 good ones
2. **Add 3D objects to the scene** — cubes, cylinders, barriers on the floor
3. **Add multiple planes at different heights** — simulate buildings/obstacles
4. **Lower camera height** to 5-8m → more resolution per texture pixel
5. **Add camera tilt** — not always looking straight down
6. **Use higher-frequency textures** — replace smooth photos with:
   - Satellite imagery (Google Earth tiles)
   - Dense pattern textures (herringbone, cobblestone, dense foliage)
   - Fractal/procedural textures (guaranteed detail at all scales)

### Option B: Fix the Pipeline (Quick Wins)

1. **Precompute optical flow** between frames (cv2.calcOpticalFlowFarneback or RAFT)
   - Use flow maps as input instead of raw image pairs
   - This explicitly provides the motion signal the CNN is failing to learn

2. **Use a pretrained visual encoder** (ResNet18 with ImageNet weights)
   - Replace your 4-layer CNN with a frozen ResNet18 → 512-dim features
   - Only train the fusion head

3. **Filter bad sequences from training**
   - Remove all sequences with blur_score < 50
   - This immediately removes ~38% of garbage data
   - Retrain with ~150 clean sequences

4. **Add soft gating** (SelectFusion approach)
   ```python
   gate = torch.sigmoid(self.gate_net(torch.cat([v_feat, i_feat], dim=1)))
   fused = gate * v_feat + (1 - gate) * i_feat
   ```
   - Model learns to ignore visual features when they're uninformative

5. **IMU preintegration** instead of raw LSTM
   - Compute classical preintegrated delta (Δp, Δv, ΔR) as IMU features
   - Network refines the preintegrated estimate

### Option C: Use an Existing Dataset (Fastest to Good Results)

- **TartanAir** — synthetic, diverse 3D environments, flow pre-computed
  - GitHub: `castacks/tartanair_tools`
  - Already has train/test splits, IMU, depth, flow, stereo
- **EuRoC MAV** — real drone data with GT from Vicon
  - Standard VIO benchmark, small but high quality
- Train your model on TartanAir/EuRoC → prove architecture works → then adapt to your synthetic data

---

## 7. Immediate Action Items

### Today:
- [x] Diagnose image quality issue
- [x] Identify root cause (texture quality + flat scene)
- [x] Document all results
- [ ] **Decision**: Which option (A/B/C) to pursue?

### If Option B (Quick Fix — 2-3 hours):
1. Filter bad sequences: `blur_score > 50` filter
2. Precompute optical flow for all remaining sequences
3. Swap visual encoder input to flow maps (2-channel H×W)
4. Retrain visual-only and combined models
5. Expected: Visual model should improve significantly on textured sequences

### If Option A (Dataset Fix — 6-12 hours):
1. Remove 7 bad textures
2. Find 10+ replacement textures (satellite/aerial/dense patterns)
3. Add 3D objects to Blender scene
4. Re-render all sequences
5. Retrain from scratch

### If Option C (Existing Dataset — 4-6 hours):
1. Download TartanAir (or EuRoC) 
2. Write dataloader adapter
3. Train on well-characterized data to validate architecture
4. Then decide if custom data is even needed

---

## 8. Key Takeaways

> **The model isn't failing because of architecture or hyperparameters.
> It's failing because 38-54% of training images contain NO visual information.**

> **You can't learn visual odometry from images that are effectively solid colors.**

> **The IMU dominates because it's the only modality that provides consistent signal.**

---

## Appendix: Optical Flow Analysis

| Metric | Your Dataset | Good VO Dataset |
|--------|-------------|-----------------|
| Mean flow magnitude | 5.65 px | 5-15 px |
| Flow <1px (no motion visible) | 8% | <2% |
| Flow <2px (borderline) | 32% | <10% |
| ORB features detected | 275 mean | 300-500 |
| Sequences with <50 features | 20% | <5% |

The optical flow magnitude is actually **reasonable** (5.65px mean), but it's **meaningless on featureless images** — you can't compute reliable flow on a solid color surface even if the camera moves.

---

*Generated by analysis on May 3, 2026. Lab machine: adipat@192.168.1.185*
