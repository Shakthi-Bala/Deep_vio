# Bradley's Experiment List — DeepVIO Project

> **Branch:** `bradley/experiments`
> **Base:** `aditya/p4` (has all infra code, models, training scripts)
> **Due:** [fill in date]

---

## Setup (Do This First)

```bash
# Clone and checkout this branch
git clone https://github.com/Shakthi-Bala/Deep_vio.git
cd Deep_vio
git checkout bradley/experiments
cd DeepVIO

# Install dependencies
pip install torch torchvision numpy scipy opencv-python matplotlib

# Data is generated on the lab machine (192.168.1.185)
# Copy it or regenerate using:
#   blender --background --python blender_script.py
#   python3 cleanup_bad_sequences.py
```

---

## Experiments

Each experiment is **independent** — do them in any order. Even completing 1-2 is valuable.

---

### Experiment 1: Pretrained ResNet Visual Encoder ⭐ HIGH PRIORITY

**Goal:** Replace the tiny 4-layer CNN with a pretrained ResNet18 to improve visual features.

**What to do:**
1. In `models.py`, create a new class `VisualEncoderResNet`:
```python
import torchvision.models as models

class VisualEncoderResNet(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        # Modify first conv to accept 6-channel input (image pair)
        resnet.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Copy weights for first 3 channels, random init for last 3
        with torch.no_grad():
            resnet.conv1.weight[:, :3] = models.resnet18(pretrained=True).conv1.weight
            resnet.conv1.weight[:, 3:] = models.resnet18(pretrained=True).conv1.weight
        # Remove final FC, use avgpool output (512-dim)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.proj = nn.Linear(512, 256)  # Project to 256 to match IMU encoder
    
    def forward(self, x):
        feat = self.backbone(x).flatten(1)  # (B, 512)
        return self.proj(feat)  # (B, 256)
```

2. In `models.py`, make `DeepVO` and `DeepVIO` accept `encoder_type='resnet'` parameter
3. Train with:
```bash
python train.py --data output/ --model visual --epochs 50 --attention \
    --batch-size 32 --lr 5e-4 --patience 15 \
    --checkpoint-dir checkpoints/exp_resnet_visual
```

**Expected:** Visual test loss should drop significantly (current: 0.18, target: <0.12)

**Deliverable:** Updated `models.py` + training log + checkpoint

---

### Experiment 2: Optical Flow as Visual Input ⭐ HIGH PRIORITY

**Goal:** Precompute optical flow between frame pairs and use flow maps instead of raw images.

**What to do:**
1. Write `precompute_flow.py`:
```python
"""Precompute optical flow for all sequences."""
import cv2, os, numpy as np

for split in ['train', 'val', 'test']:
    split_dir = f'output/{split}'
    for seq in sorted(os.listdir(split_dir)):
        img_dir = os.path.join(split_dir, seq, 'images')
        flow_dir = os.path.join(split_dir, seq, 'flow')
        os.makedirs(flow_dir, exist_ok=True)
        
        imgs = sorted(os.listdir(img_dir))
        for i in range(len(imgs) - 1):
            f1 = cv2.imread(os.path.join(img_dir, imgs[i]), cv2.IMREAD_GRAYSCALE)
            f2 = cv2.imread(os.path.join(img_dir, imgs[i+1]), cv2.IMREAD_GRAYSCALE)
            flow = cv2.calcOpticalFlowFarneback(f1, f2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            np.save(os.path.join(flow_dir, f'{i:05d}.npy'), flow.astype(np.float16))
        print(f'{split}/{seq}: {len(imgs)-1} flow fields')
```

2. Modify `dataset.py` to load flow maps instead of image pairs:
   - Change `__getitem__` to load `flow/{idx:05d}.npy` (shape: H×W×2)
   - The input becomes `(2, H, W)` instead of `(6, H, W)`

3. Modify `VisualEncoder` in `models.py`: change first conv from `in_channels=6` to `in_channels=2`

4. Train:
```bash
python precompute_flow.py
python train.py --data output/ --model visual --epochs 50 --attention \
    --batch-size 32 --lr 1e-3 --patience 15 \
    --checkpoint-dir checkpoints/exp_flow_visual
```

**Expected:** Flow removes appearance variation, focuses on pure motion → better generalization

**Deliverable:** `precompute_flow.py` + modified dataset/model + training log

---

### Experiment 3: Soft Fusion Gating (SelectFusion-style)

**Goal:** Let the model learn WHEN to trust vision vs IMU instead of naive concatenation.

**What to do:**
1. In `models.py`, modify `DeepVIO.forward()`:
```python
class DeepVIO(nn.Module):
    def __init__(self, use_attention=False):
        super().__init__()
        self.vis_enc = VisualEncoder()   # → 256
        self.imu_enc = IMUEncoder()      # → 256
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # output in [0, 1]
        )
        # ... rest stays the same but use gated fusion:
    
    def forward(self, img_pair, imu_seq):
        v_feat = self.vis_enc(img_pair)   # (B, 256)
        i_feat = self.imu_enc(imu_seq)    # (B, 256)
        
        # Learn when to trust vision vs IMU
        gate_input = torch.cat([v_feat, i_feat], dim=1)
        alpha = self.gate(gate_input)  # (B, 1) — how much to trust vision
        
        fused = alpha * v_feat + (1 - alpha) * i_feat  # (B, 256)
        # ... pass fused through FC heads
```

2. Log the gate values during training to see if the model learns to downweight vision on bad frames

3. Train:
```bash
python train.py --data output/ --model combined --epochs 50 --attention \
    --batch-size 32 --lr 1e-3 --patience 15 \
    --checkpoint-dir checkpoints/exp_gated_combined
```

**Expected:** Combined model should finally beat visual-only (current combined is WORSE than visual)

**Deliverable:** Modified `models.py` + gate value analysis + training log

---

### Experiment 4: Two-Stage Training (Freeze Visual, Then Finetune)

**Goal:** Prevent IMU from dominating by pretraining the visual encoder first.

**What to do:**
1. First, train visual-only model (already done — use existing checkpoint)
2. Then load those visual weights into the combined model and freeze them:
```python
# Load pretrained visual encoder
visual_checkpoint = torch.load('checkpoints/newdata_visual/visual_best.pt')
combined_model.vis_enc.load_state_dict(visual_checkpoint['vis_enc'])

# Freeze visual encoder for first 10 epochs
for param in combined_model.vis_enc.parameters():
    param.requires_grad = False

# After 10 epochs, unfreeze and finetune with low LR
```

3. Modify `train.py` to add `--freeze-visual-epochs 10` flag

4. Train:
```bash
python train.py --data output/ --model combined --epochs 50 --attention \
    --batch-size 32 --lr 5e-4 --patience 15 \
    --freeze-visual-epochs 10 \
    --resume-visual checkpoints/newdata_visual/visual_best.pt \
    --checkpoint-dir checkpoints/exp_twostage_combined
```

**Expected:** Visual branch retains good features, IMU learns to complement rather than dominate

**Deliverable:** Modified `train.py` + training log

---

### Experiment 5: Loss Weighting & Rotation Emphasis

**Goal:** The rotation loss is 20× smaller than position loss — rebalance.

**What to do:**
1. Already supported in `train.py` with `--lambda-q` flag
2. Run these experiments:
```bash
# Experiment 5a: lambda_q = 5
python train.py --data output/ --model combined --epochs 50 --attention \
    --batch-size 32 --lr 1e-3 --patience 15 --lambda-q 5.0 \
    --checkpoint-dir checkpoints/exp_lambdaq5

# Experiment 5b: lambda_q = 10
python train.py --data output/ --model combined --epochs 50 --attention \
    --batch-size 32 --lr 1e-3 --patience 15 --lambda-q 10.0 \
    --checkpoint-dir checkpoints/exp_lambdaq10
```

**Expected:** Better rotation accuracy, may help overall loss

**Deliverable:** Training logs for both runs

---

### Experiment 6: Data Augmentation

**Goal:** Reduce overfitting with augmented training data.

**What to do:**
1. In `dataset.py`, the `augment=True` flag already adds `ColorJitter`
2. Add more augmentations:
```python
if augment:
    self.img_tf = transforms.Compose([
        resize,
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        normalize,
    ])
```
3. Add IMU noise augmentation in `__getitem__`:
```python
if self.augment:
    # Add random noise to IMU (simulates sensor variance)
    imu_noise = torch.randn_like(imu_seq) * 0.02
    imu_seq = imu_seq + imu_noise
```

4. Make sure `augment=True` is passed for train set in `train.py`

5. Train:
```bash
python train.py --data output/ --model combined --epochs 50 --attention \
    --batch-size 32 --lr 1e-3 --patience 15 --augment \
    --checkpoint-dir checkpoints/exp_augmented
```

**Deliverable:** Modified `dataset.py` + training log

---

## Priority Order (if short on time)

1. **Exp 1** (ResNet encoder) — biggest expected impact, straightforward
2. **Exp 3** (Soft gating) — fixes the core fusion problem
3. **Exp 2** (Optical flow) — good improvement but more work
4. **Exp 5** (Loss weighting) — easiest, just run commands
5. **Exp 4** (Two-stage) — moderate difficulty
6. **Exp 6** (Augmentation) — easy but less impactful

---

## Results Template

After each experiment, fill in:

| Experiment | Best Epoch | Val Loss | Test Loss | Test Pos | Test Rot |
|-----------|-----------|----------|-----------|----------|----------|
| Baseline Visual | 17 | 0.1877 | 0.1822 | 0.1622 | 0.0200 |
| Baseline IMU | ? | ? | ? | ? | ? |
| Baseline Combined | ? | ? | ? | ? | ? |
| Exp 1: ResNet | | | | | |
| Exp 2: Flow | | | | | |
| Exp 3: Gated | | | | | |
| Exp 4: Two-stage | | | | | |
| Exp 5a: λ_q=5 | | | | | |
| Exp 5b: λ_q=10 | | | | | |
| Exp 6: Augment | | | | | |

---

## Notes

- All training is on the **new dataset** (bad textures removed, 3D objects added)
- Lab machine: `ssh adipat@192.168.1.185` (pw: 4829)
- GPU: RTX 5060 (8GB) — batch size 32 is safe for combined model
- Each training run takes ~15-30 min with early stopping
- Checkpoints save automatically — just note which `checkpoints/exp_*` dir to look at
