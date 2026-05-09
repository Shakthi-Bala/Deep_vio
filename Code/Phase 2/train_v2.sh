#!/bin/bash
# Train improved V2 models on the new dataset
# Run all 4 experiments sequentially

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cv_p3
cd ~/Documents/Spring_26/CV/p4/DeepVIO

LOG="v2_training.log"
echo "=== V2 MODEL TRAINING — $(date) ===" | tee "$LOG"

# Exp A: DeepVO with ResNet18 (pretrained)
echo "" | tee -a "$LOG"
echo "▸ EXP A: DeepVO_V2 (ResNet18 pretrained)" | tee -a "$LOG"
python3 -c "
import torch, json, time, os
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from models_v2 import DeepVO_V2
from models import CombinedVIOLoss
from dataset import VIODataset

device = torch.device('cuda')
model = DeepVO_V2(visual_encoder='resnet').to(device)
print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')

train_ds = VIODataset('output/train', augment=True)
val_ds = VIODataset('output/val')
test_ds = VIODataset('output/test')
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size=32, num_workers=4)
test_dl = DataLoader(test_ds, batch_size=32, num_workers=4)

criterion = CombinedVIOLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
scaler = GradScaler()

best_val = float('inf')
patience_counter = 0
os.makedirs('checkpoints/v2_resnet_visual', exist_ok=True)

for epoch in range(1, 51):
    t0 = time.time()
    model.train()
    train_loss = 0
    for img, imu, gt_p, gt_q in train_dl:
        img, gt_p, gt_q = img.to(device), gt_p.to(device), gt_q.to(device)
        optimizer.zero_grad()
        with autocast():
            pred_p, pred_q = model(img)
            loss, _, _ = criterion(pred_p, pred_q, gt_p, gt_q)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()
    train_loss /= len(train_dl)
    scheduler.step()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for img, imu, gt_p, gt_q in val_dl:
            img, gt_p, gt_q = img.to(device), gt_p.to(device), gt_q.to(device)
            with autocast():
                pred_p, pred_q = model(img)
                loss, _, _ = criterion(pred_p, pred_q, gt_p, gt_q)
            val_loss += loss.item()
    val_loss /= len(val_dl)

    dt = time.time() - t0
    marker = ''
    if val_loss < best_val:
        best_val = val_loss
        patience_counter = 0
        torch.save({'epoch': epoch, 'model': model.state_dict(), 'best_val': best_val}, 'checkpoints/v2_resnet_visual/best.pt')
        marker = ' << BEST'
    else:
        patience_counter += 1

    print(f'Ep {epoch:02d} | Train {train_loss:.4f} | Val {val_loss:.4f} | {dt:.0f}s{marker}')
    if patience_counter >= 15:
        print(f'Early stopping at epoch {epoch}')
        break

# Test
ckpt = torch.load('checkpoints/v2_resnet_visual/best.pt')
model.load_state_dict(ckpt['model'])
model.eval()
test_loss = test_pos = test_rot = 0
with torch.no_grad():
    for img, imu, gt_p, gt_q in test_dl:
        img, gt_p, gt_q = img.to(device), gt_p.to(device), gt_q.to(device)
        with autocast():
            pred_p, pred_q = model(img)
            loss, pl, rl = criterion(pred_p, pred_q, gt_p, gt_q)
        test_loss += loss.item(); test_pos += pl.item(); test_rot += rl.item()
test_loss /= len(test_dl); test_pos /= len(test_dl); test_rot /= len(test_dl)
print(f'TEST: loss={test_loss:.4f} pos={test_pos:.4f} rot={test_rot:.4f}')
" 2>&1 | tee -a "$LOG"

# Exp B: DeepVIO_FiLM (ResNet + FiLM fusion)
echo "" | tee -a "$LOG"
echo "▸ EXP B: DeepVIO_FiLM (ResNet + FiLM + Soft Gate)" | tee -a "$LOG"
python3 -c "
import torch, time, os
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from models_v2 import DeepVIO_FiLM
from models import CombinedVIOLoss
from dataset import VIODataset

device = torch.device('cuda')
model = DeepVIO_FiLM(visual_encoder='resnet').to(device)
print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')

train_ds = VIODataset('output/train', augment=True)
val_ds = VIODataset('output/val')
test_ds = VIODataset('output/test')
train_dl = DataLoader(train_ds, batch_size=24, shuffle=True, num_workers=4, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size=24, num_workers=4)
test_dl = DataLoader(test_ds, batch_size=24, num_workers=4)

criterion = CombinedVIOLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
scaler = GradScaler()

best_val = float('inf')
patience_counter = 0
os.makedirs('checkpoints/v2_film_combined', exist_ok=True)

for epoch in range(1, 51):
    t0 = time.time()
    model.train()
    train_loss = 0
    for img, imu, gt_p, gt_q in train_dl:
        img, imu, gt_p, gt_q = img.to(device), imu.to(device), gt_p.to(device), gt_q.to(device)
        optimizer.zero_grad()
        with autocast():
            pred_p, pred_q = model(img, imu)
            loss, _, _ = criterion(pred_p, pred_q, gt_p, gt_q)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()
    train_loss /= len(train_dl)
    scheduler.step()

    model.eval()
    val_loss = 0
    gate_vals = []
    with torch.no_grad():
        for img, imu, gt_p, gt_q in val_dl:
            img, imu, gt_p, gt_q = img.to(device), imu.to(device), gt_p.to(device), gt_q.to(device)
            with autocast():
                pred_p, pred_q, gate = model(img, imu, return_gate=True)
                loss, _, _ = criterion(pred_p, pred_q, gt_p, gt_q)
            val_loss += loss.item()
            gate_vals.extend(gate.cpu().numpy().flatten().tolist())
    val_loss /= len(val_dl)
    import numpy as np
    mean_gate = np.mean(gate_vals)

    dt = time.time() - t0
    marker = ''
    if val_loss < best_val:
        best_val = val_loss
        patience_counter = 0
        torch.save({'epoch': epoch, 'model': model.state_dict(), 'best_val': best_val}, 'checkpoints/v2_film_combined/best.pt')
        marker = ' << BEST'
    else:
        patience_counter += 1

    print(f'Ep {epoch:02d} | Train {train_loss:.4f} | Val {val_loss:.4f} | Gate={mean_gate:.3f} | {dt:.0f}s{marker}')
    if patience_counter >= 15:
        print(f'Early stopping at epoch {epoch}')
        break

ckpt = torch.load('checkpoints/v2_film_combined/best.pt')
model.load_state_dict(ckpt['model'])
model.eval()
test_loss = test_pos = test_rot = 0
with torch.no_grad():
    for img, imu, gt_p, gt_q in test_dl:
        img, imu, gt_p, gt_q = img.to(device), imu.to(device), gt_p.to(device), gt_q.to(device)
        with autocast():
            pred_p, pred_q = model(img, imu)
            loss, pl, rl = criterion(pred_p, pred_q, gt_p, gt_q)
        test_loss += loss.item(); test_pos += pl.item(); test_rot += rl.item()
test_loss /= len(test_dl); test_pos /= len(test_dl); test_rot /= len(test_dl)
print(f'TEST: loss={test_loss:.4f} pos={test_pos:.4f} rot={test_rot:.4f}')
" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== V2 TRAINING COMPLETE — $(date) ===" | tee -a "$LOG"
