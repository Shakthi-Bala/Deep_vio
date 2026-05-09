"""
Evaluate all three models on the TRAINING split and compare against saved val results.
Produces per-model metrics, an overfitting summary, and combined plots.

Usage:
    python evaluate_train.py --data-dir /path/to/data
"""

import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader

from dataset.dataset import SyntheticVIODataset
from models import VisionOnlyNet, IMUOnlyNet, VIOFusionNet
from utils import dead_reckon

DATA_DIR = "/home/sbalamurugan/cv_p4/blender_splats/new_data_out"

CHECKPOINTS = {
    "vision": "checkpoints/vision/best.pth",
    "imu":    "checkpoints/imu/best.pth",
    "vio":    "checkpoints/vio/best.pth",
}

VAL_RESULTS = {
    "vision": "checkpoints/vision/eval_results.npz",
    "imu":    "checkpoints/imu/eval_results.npz",
    "vio":    "checkpoints/vio/eval_results.npz",
}

OUT_DIR = "checkpoints/train_eval"


def build_model(mode, device):
    if mode == "vision":
        return VisionOnlyNet().to(device)
    if mode == "imu":
        return IMUOnlyNet().to(device)
    return VIOFusionNet().to(device)


def collate_fn(batch):
    return {k: torch.stack([item[k] for item in batch]) for k in batch[0]}


def run_inference(model, dataset, device):
    loader = DataLoader(dataset, batch_size=32, shuffle=False,
                        collate_fn=collate_fn, num_workers=2)
    preds, gts = [], []
    with torch.no_grad():
        for batch in loader:
            if dataset.mode == "vision":
                pred = model(batch["img0"].to(device), batch["img1"].to(device))
            elif dataset.mode == "imu":
                pred = model(batch["imu"].to(device))
            else:
                pred = model(batch["img0"].to(device),
                             batch["img1"].to(device),
                             batch["imu"].to(device))
            preds.append(pred.cpu().numpy())
            gts.append(batch["pose"].numpy())
    return np.concatenate(preds), np.concatenate(gts)


def compute_metrics(pred, gt):
    trans_err = np.linalg.norm(pred[:, :3] - gt[:, :3], axis=1)
    dot = np.abs(np.sum(pred[:, 3:] * gt[:, 3:], axis=1)).clip(0, 1)
    rot_err_deg = np.degrees(2 * np.arccos(dot))
    drift = np.linalg.norm(
        np.cumsum(pred[:, :3], axis=0) - np.cumsum(gt[:, :3], axis=0), axis=1
    )
    return {
        "n":              len(pred),
        "trans_mean":     float(trans_err.mean()),
        "trans_std":      float(trans_err.std()),
        "trans_max":      float(trans_err.max()),
        "rot_mean_deg":   float(rot_err_deg.mean()),
        "rot_std_deg":    float(rot_err_deg.std()),
        "rot_max_deg":    float(rot_err_deg.max()),
        "rot_under1_pct": float((rot_err_deg < 1.0).mean() * 100),
        "rot_under5_pct": float((rot_err_deg < 5.0).mean() * 100),
        "drift_final":    float(drift[-1]),
        "drift_max":      float(drift.max()),
    }


def print_metrics(label, m):
    print(f"  {label}:")
    print(f"    samples          : {m['n']}")
    print(f"    trans err (m)    : mean={m['trans_mean']:.5f}  std={m['trans_std']:.5f}  max={m['trans_max']:.5f}")
    print(f"    rot err (deg)    : mean={m['rot_mean_deg']:.4f}  std={m['rot_std_deg']:.4f}  max={m['rot_max_deg']:.4f}")
    print(f"    rot < 1deg       : {m['rot_under1_pct']:.1f}%")
    print(f"    rot < 5deg       : {m['rot_under5_pct']:.1f}%")
    print(f"    traj drift final : {m['drift_final']:.4f} m")
    print(f"    traj drift max   : {m['drift_max']:.4f} m")


def overfit_ratio(train_val, val_val):
    if val_val == 0:
        return float("inf")
    return train_val / val_val


def plot_comparison(mode, train_pred, train_gt, val_pred, val_gt, out_path):
    fig = plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(1, 3, figure=fig)

    # --- Trajectory: train ---
    ax1 = fig.add_subplot(gs[0])
    tr_traj = dead_reckon(train_pred)
    tr_gt_t = dead_reckon(train_gt)
    ax1.plot(tr_gt_t[:, 0], tr_gt_t[:, 1], label="GT", linewidth=2)
    ax1.plot(tr_traj[:, 0],  tr_traj[:, 1],  label="Pred", linestyle="--")
    ax1.set_title(f"{mode.upper()} — Train trajectory")
    ax1.set_xlabel("x (m)"); ax1.set_ylabel("y (m)"); ax1.legend()

    # --- Trajectory: val ---
    ax2 = fig.add_subplot(gs[1])
    va_traj = dead_reckon(val_pred)
    va_gt_t = dead_reckon(val_gt)
    ax2.plot(va_gt_t[:, 0], va_gt_t[:, 1], label="GT", linewidth=2)
    ax2.plot(va_traj[:, 0],  va_traj[:, 1],  label="Pred", linestyle="--")
    ax2.set_title(f"{mode.upper()} — Val trajectory")
    ax2.set_xlabel("x (m)"); ax2.set_ylabel("y (m)"); ax2.legend()

    # --- Per-frame translation error histogram ---
    ax3 = fig.add_subplot(gs[2])
    tr_err = np.linalg.norm(train_pred[:, :3] - train_gt[:, :3], axis=1)
    va_err = np.linalg.norm(val_pred[:, :3]   - val_gt[:, :3],   axis=1)
    bins = np.linspace(0, max(tr_err.max(), va_err.max()), 50)
    ax3.hist(tr_err, bins=bins, alpha=0.6, label=f"Train (μ={tr_err.mean():.4f})")
    ax3.hist(va_err, bins=bins, alpha=0.6, label=f"Val   (μ={va_err.mean():.4f})")
    ax3.set_title(f"{mode.upper()} — Trans error distribution")
    ax3.set_xlabel("Trans error (m)"); ax3.set_ylabel("Count"); ax3.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved plot → {out_path}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(OUT_DIR, exist_ok=True)

    all_results = {}

    for mode, ckpt_path in CHECKPOINTS.items():
        print(f"\n{'='*60}")
        print(f"  Mode: {mode.upper()}")
        print(f"{'='*60}")

        if not os.path.exists(ckpt_path):
            print(f"  [SKIP] checkpoint not found: {ckpt_path}")
            continue

        model = build_model(mode, device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        train_ds = SyntheticVIODataset(args.data_dir, split="train", mode=mode)
        print(f"  Train samples: {len(train_ds)}")
        train_pred, train_gt = run_inference(model, train_ds, device)

        train_m = compute_metrics(train_pred, train_gt)
        print_metrics("TRAIN", train_m)

        # Save train results
        np.savez(os.path.join(OUT_DIR, f"{mode}_train_results.npz"),
                 pred=train_pred, gt=train_gt)

        # Load val results for comparison
        val_npz = VAL_RESULTS[mode]
        if os.path.exists(val_npz):
            d = np.load(val_npz)
            val_pred, val_gt = d["pred"], d["gt"]
            val_m = compute_metrics(val_pred, val_gt)
            print_metrics("VAL  ", val_m)

            # Overfitting summary
            ratio = overfit_ratio(train_m["trans_mean"], val_m["trans_mean"])
            print(f"\n  Overfitting check (train/val ratio — closer to 1 = no overfit):")
            print(f"    trans_err ratio  : {ratio:.3f}  ", end="")
            if ratio < 0.5:
                print("(train << val → UNDERFITTING or different distributions)")
            elif ratio > 2.0:
                print("(train >> val → OVERFITTING)")
            else:
                print("(similar → well-generalised)")

            rot_ratio = overfit_ratio(train_m["rot_mean_deg"], val_m["rot_mean_deg"])
            print(f"    rot_err  ratio   : {rot_ratio:.3f}  ", end="")
            if rot_ratio < 0.5:
                print("(train << val → UNDERFITTING or different distributions)")
            elif rot_ratio > 2.0:
                print("(train >> val → OVERFITTING)")
            else:
                print("(similar → well-generalised)")

            all_results[mode] = {
                "train": train_m, "val": val_m,
                "trans_ratio": ratio, "rot_ratio": rot_ratio,
            }

            plot_comparison(mode, train_pred, train_gt, val_pred, val_gt,
                            os.path.join(OUT_DIR, f"{mode}_train_vs_val.png"))
        else:
            print(f"  [INFO] No val results found at {val_npz}, skipping comparison.")
            all_results[mode] = {"train": train_m}

    # Final summary table
    print(f"\n{'='*60}")
    print("  FINAL SUMMARY — Train vs Val")
    print(f"{'='*60}")
    header = f"  {'Mode':<8} {'Split':<6} {'TransErr':>10} {'RotErr(°)':>11} {'Drift(m)':>10} {'<1°%':>7} {'<5°%':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for mode, res in all_results.items():
        for split, m in res.items():
            if split in ("trans_ratio", "rot_ratio"):
                continue
            print(f"  {mode:<8} {split:<6} "
                  f"{m['trans_mean']:>10.5f} "
                  f"{m['rot_mean_deg']:>11.4f} "
                  f"{m['drift_final']:>10.4f} "
                  f"{m['rot_under1_pct']:>7.1f} "
                  f"{m['rot_under5_pct']:>7.1f}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    args = parser.parse_args()
    main(args)
