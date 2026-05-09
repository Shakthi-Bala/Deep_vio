"""Two-stage training: Load pretrained ResNet visual weights into FiLM model,
freeze visual encoder for 10 epochs, then unfreeze and finetune jointly."""
import torch, time, os, numpy as np
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from models_v2 import DeepVIO_FiLM
from models import CombinedVIOLoss
from dataset import VIODataset

device = torch.device('cuda')
model = DeepVIO_FiLM(visual_encoder='resnet').to(device)

# Load pretrained ResNet visual weights from DeepVO_V2
print("Loading pretrained visual encoder from DeepVO_V2...")
v2_ckpt = torch.load('checkpoints/v2_resnet_visual/best.pt', map_location=device)
v2_state = v2_ckpt['model']

# Map DeepVO_V2 visual weights -> DeepVIO_FiLM visual weights
vis_state = {}
for k, v in v2_state.items():
    if k.startswith('vis_enc.'):
        vis_state[k] = v

model_state = model.state_dict()
matched = 0
for k, v in vis_state.items():
    if k in model_state and model_state[k].shape == v.shape:
        model_state[k] = v
        matched += 1
model.load_state_dict(model_state)
print(f"Transferred {matched} visual encoder tensors")

# Freeze visual encoder
for name, param in model.named_parameters():
    if 'vis_enc' in name:
        param.requires_grad = False
frozen_params = sum(1 for p in model.parameters() if not p.requires_grad)
trainable_params = sum(1 for p in model.parameters() if p.requires_grad)
print(f"Phase 1: {trainable_params} trainable, {frozen_params} frozen (visual)")

train_ds = VIODataset('output/train', augment=True)
val_ds = VIODataset('output/val')
test_ds = VIODataset('output/test')
train_dl = DataLoader(train_ds, batch_size=24, shuffle=True, num_workers=4, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size=24, num_workers=4)
test_dl = DataLoader(test_ds, batch_size=24, num_workers=4)

criterion = CombinedVIOLoss()
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4, weight_decay=1e-4)
scaler = GradScaler()

best_val = float('inf')
patience_counter = 0
FREEZE_EPOCHS = 10
os.makedirs('checkpoints/v2_twostage_film', exist_ok=True)

for epoch in range(1, 51):
    # Unfreeze visual encoder after FREEZE_EPOCHS
    if epoch == FREEZE_EPOCHS + 1:
        print(f"\n=== UNFREEZING visual encoder at epoch {epoch} ===")
        for name, param in model.named_parameters():
            if 'vis_enc' in name:
                param.requires_grad = True
        # Lower LR for finetuning
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        patience_counter = 0  # reset patience

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
    mean_gate = np.mean(gate_vals)

    dt = time.time() - t0
    phase = "FROZEN" if epoch <= FREEZE_EPOCHS else "JOINT"
    marker = ''
    if val_loss < best_val:
        best_val = val_loss
        patience_counter = 0
        torch.save({'epoch': epoch, 'model': model.state_dict(), 'best_val': best_val},
                   'checkpoints/v2_twostage_film/best.pt')
        marker = ' << BEST'
    else:
        patience_counter += 1

    print(f'Ep {epoch:02d} [{phase}] | Train {train_loss:.4f} | Val {val_loss:.4f} | Gate={mean_gate:.3f} | {dt:.0f}s{marker}')
    if patience_counter >= 15:
        print(f'Early stopping at epoch {epoch}')
        break

# Test
ckpt = torch.load('checkpoints/v2_twostage_film/best.pt')
model.load_state_dict(ckpt['model'])
model.eval()
test_loss = test_pos = test_rot = 0
gate_vals = []
with torch.no_grad():
    for img, imu, gt_p, gt_q in test_dl:
        img, imu, gt_p, gt_q = img.to(device), imu.to(device), gt_p.to(device), gt_q.to(device)
        with autocast():
            pred_p, pred_q, gate = model(img, imu, return_gate=True)
            loss, pl, rl = criterion(pred_p, pred_q, gt_p, gt_q)
        test_loss += loss.item(); test_pos += pl.item(); test_rot += rl.item()
        gate_vals.extend(gate.cpu().numpy().flatten().tolist())
test_loss /= len(test_dl); test_pos /= len(test_dl); test_rot /= len(test_dl)
print(f'\nTEST: loss={test_loss:.4f} pos={test_pos:.4f} rot={test_rot:.4f} gate={np.mean(gate_vals):.3f}')
