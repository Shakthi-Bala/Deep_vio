import os
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from dataset.dataset import SyntheticVIODataset
from models import VisionOnlyNet, IMUOnlyNet, VIOFusionNet


def make_augmentation():
    return T.Compose([
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
    ])


def loss_function(pred, target, trans_weight=1.0, rot_weight=100.0):
    trans_loss = torch.mean((pred[:, :3] - target[:, :3]) ** 2)
    dot = torch.abs(torch.sum(pred[:, 3:] * target[:, 3:], dim=1))
    rot_loss = torch.mean(1.0 - dot)
    return trans_weight * trans_loss + rot_weight * rot_loss, trans_loss, rot_loss


def build_model(mode, device):
    if mode == "vision":
        return VisionOnlyNet().to(device)
    if mode == "imu":
        return IMUOnlyNet().to(device)
    if mode == "vio":
        return VIOFusionNet().to(device)
    raise ValueError("Unknown mode")


def collate_fn(batch):
    out = {}
    for key in batch[0]:
        out[key] = torch.stack([item[key] for item in batch], dim=0)
    return out


def _forward(model, batch, mode, device):
    if mode == "vision":
        return model(batch["img0"].to(device), batch["img1"].to(device))
    if mode == "imu":
        return model(batch["imu"].to(device))
    return model(batch["img0"].to(device), batch["img1"].to(device), batch["imu"].to(device))


def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    aug = make_augmentation() if args.mode in ("vision", "vio") else None
    train_ds = SyntheticVIODataset(args.data_dir, split="train", mode=args.mode,
                                   transforms=aug, sample_split=True)
    val_ds   = SyntheticVIODataset(args.data_dir, split="val",   mode=args.mode,
                                   sample_split=True)
    print(f"Train samples: {len(train_ds)}  |  Val samples: {len(val_ds)}")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=2)
    model     = build_model(args.mode, device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    best_val  = float("inf")

    os.makedirs(args.out_dir, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = train_tl = train_rl = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            optimizer.zero_grad()
            pred = _forward(model, batch, args.mode, device)
            target = batch["pose"].to(device)
            loss, tl, rl = loss_function(pred, target)
            loss.backward()
            optimizer.step()
            n = target.shape[0]
            train_loss += loss.item() * n
            train_tl   += tl.item()   * n
            train_rl   += rl.item()   * n
        train_loss /= len(train_ds)
        train_tl   /= len(train_ds)
        train_rl   /= len(train_ds)
        scheduler.step()

        model.eval()
        val_loss = val_tl = val_rl = 0.0
        with torch.no_grad():
            for batch in val_loader:
                pred   = _forward(model, batch, args.mode, device)
                target = batch["pose"].to(device)
                loss, tl, rl = loss_function(pred, target)
                n = target.shape[0]
                val_loss += loss.item() * n
                val_tl   += tl.item()   * n
                val_rl   += rl.item()   * n
        if len(val_ds) > 0:
            val_loss /= len(val_ds); val_tl /= len(val_ds); val_rl /= len(val_ds)
        else:
            val_loss = float("inf")

        lr_now = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch}: "
              f"train={train_loss:.6f} (t={train_tl:.6f} r={train_rl:.6f})  "
              f"val={val_loss:.6f} (t={val_tl:.6f} r={val_rl:.6f})  "
              f"lr={lr_now:.2e}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best.pth"))
            print(f"  >> Saved best model (val={best_val:.6f})")


def parse_common_args(default_mode, default_out):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/synthetic")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out-dir", type=str, default=default_out)
    args = parser.parse_args()
    args.mode = default_mode
    return args
