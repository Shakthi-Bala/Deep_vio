import argparse
import glob
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from dataset.dataset import SyntheticVIODataset
from models import VisionOnlyNet, IMUOnlyNet, VIOFusionNet
from utils import dead_reckon


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


def evaluate_sequence(model, dataset, device):
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    rel_preds = []
    rel_gt = []
    with torch.no_grad():
        for batch in loader:
            if dataset.mode == "vision":
                pred = model(batch["img0"].to(device), batch["img1"].to(device))
            elif dataset.mode == "imu":
                pred = model(batch["imu"].to(device))
            else:
                pred = model(batch["img0"].to(device), batch["img1"].to(device), batch["imu"].to(device))
            rel_preds.append(pred.cpu().numpy())
            rel_gt.append(batch["pose"].numpy())
    rel_preds = np.concatenate(rel_preds, axis=0)
    rel_gt = np.concatenate(rel_gt, axis=0)
    return rel_preds, rel_gt


def plot_trajectory(pred, gt, out_path):
    pred_traj = dead_reckon(pred)
    gt_traj = dead_reckon(gt)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(gt_traj[:, 0], gt_traj[:, 1], label="ground truth", linewidth=2)
    ax.plot(pred_traj[:, 0], pred_traj[:, 1], label="prediction", linestyle="--")
    ax.set_title("Trajectory comparison")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend()
    fig.savefig(out_path)
    print(f"Saved trajectory plot to {out_path}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.mode, device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    dataset = SyntheticVIODataset(args.data_dir, split=args.split, mode=args.mode)
    rel_preds, rel_gt = evaluate_sequence(model, dataset, device)
    plot_trajectory(rel_preds, rel_gt, args.output_plot)
    np.savez(os.path.join(os.path.dirname(args.output_plot), "eval_results.npz"), pred=rel_preds, gt=rel_gt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/synthetic")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["vision", "imu", "vio"], default="vio")
    parser.add_argument("--output-plot", type=str, default="trajectory.png")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="val")
    args = parser.parse_args()
    main(args)
