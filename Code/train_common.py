import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from dataset.dataset import SyntheticVIODataset
from models import VisionOnlyNet, IMUOnlyNet, VIOFusionNet


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


def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = SyntheticVIODataset(args.data_dir, split="train", mode=args.mode)
    val_ds = SyntheticVIODataset(args.data_dir, split="val", mode=args.mode)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)
    model = build_model(args.mode, device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    best_val = float("inf")

    os.makedirs(args.out_dir, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Train epoch {epoch}"):
            optimizer.zero_grad()
            if args.mode == "vision":
                pred = model(batch["img0"].to(device), batch["img1"].to(device))
            elif args.mode == "imu":
                pred = model(batch["imu"].to(device))
            else:
                pred = model(batch["img0"].to(device), batch["img1"].to(device), batch["imu"].to(device))
            target = batch["pose"].to(device)
            loss, _, _ = loss_function(pred, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * target.shape[0]
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if args.mode == "vision":
                    pred = model(batch["img0"].to(device), batch["img1"].to(device))
                elif args.mode == "imu":
                    pred = model(batch["imu"].to(device))
                else:
                    pred = model(batch["img0"].to(device), batch["img1"].to(device), batch["imu"].to(device))
                target = batch["pose"].to(device)
                loss, _, _ = loss_function(pred, target)
                val_loss += loss.item() * target.shape[0]
        val_loss /= len(val_ds)
        print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best.pth"))
            print(f"Saved best model to {args.out_dir}/best.pth")


def parse_common_args(default_mode, default_out):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/synthetic")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out-dir", type=str, default=default_out)
    args = parser.parse_args()
    args.mode = default_mode
    return args
