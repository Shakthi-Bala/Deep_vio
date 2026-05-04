"""
evaluate.py — Evaluate trained models and generate trajectory plots + comparison table.

Usage
-----
    python evaluate.py --data output/ --checkpoint-dir checkpoints/ --output-dir results/

Generates:
    results/
        comparison_table.txt    — Side-by-side metrics for all 3 models
        imu_trajectory.png      — IMU-only predicted vs GT trajectory
        visual_trajectory.png   — Vision-only predicted vs GT trajectory
        combined_trajectory.png — Fused predicted vs GT trajectory
        metrics_plot.png        — Training curves for all models
"""

import os
import json
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from models import DeepIO, DeepVO, DeepVIO, CombinedVIOLoss
from dataset import make_dataloaders


def load_model(model_type, checkpoint_path, device, use_attention=False):
    """Load a trained model from checkpoint."""
    if model_type == "visual":
        model = DeepVO(use_attention=use_attention)
    elif model_type == "imu":
        model = DeepIO(use_attention=use_attention)
    else:
        model = DeepVIO(use_attention=use_attention)

    ck = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ck["model"])
    model = model.to(device)
    model.eval()

    epoch = ck.get("epoch", "?")
    best_val = ck.get("best_val", "?")
    print(f"  Loaded {model_type} from epoch {epoch}, best_val={best_val}")
    return model


@torch.no_grad()
def predict_trajectory(model, model_type, dataloader, device):
    """Run model on dataloader, return predicted + GT relative poses."""
    all_pred_p, all_pred_q = [], []
    all_gt_p, all_gt_q = [], []

    for img_pair, imu_seq, gt_p, gt_q in dataloader:
        img_pair = img_pair.to(device, non_blocking=True)
        imu_seq  = imu_seq.to(device, non_blocking=True)

        if model_type == "visual":
            pred_p, pred_q = model(img_pair)
        elif model_type == "imu":
            pred_p, pred_q = model(imu_seq)
        else:
            pred_p, pred_q = model(img_pair, imu_seq)

        all_pred_p.append(pred_p.cpu().numpy())
        all_pred_q.append(pred_q.cpu().numpy())
        all_gt_p.append(gt_p.numpy())
        all_gt_q.append(gt_q.numpy())

    return (
        np.concatenate(all_pred_p),
        np.concatenate(all_pred_q),
        np.concatenate(all_gt_p),
        np.concatenate(all_gt_q),
    )


def integrate_relative_poses(rel_positions, max_steps=500):
    """Integrate relative translations to get absolute trajectory."""
    n = min(len(rel_positions), max_steps)
    traj = np.zeros((n + 1, 3))
    for i in range(n):
        traj[i + 1] = traj[i] + rel_positions[i]
    return traj


def plot_trajectory_comparison(gt_traj, pred_traj, title, save_path):
    """Plot 3D and 2D trajectory comparison."""
    fig = plt.figure(figsize=(16, 5))

    # 3D plot
    ax1 = fig.add_subplot(131, projection="3d")
    ax1.plot(gt_traj[:, 0], gt_traj[:, 1], gt_traj[:, 2], "b-", label="Ground Truth", linewidth=1.5)
    ax1.plot(pred_traj[:, 0], pred_traj[:, 1], pred_traj[:, 2], "r--", label="Predicted", linewidth=1.5)
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")
    ax1.set_title(f"{title} — 3D")
    ax1.legend()

    # XY plane
    ax2 = fig.add_subplot(132)
    ax2.plot(gt_traj[:, 0], gt_traj[:, 1], "b-", label="GT", linewidth=1.5)
    ax2.plot(pred_traj[:, 0], pred_traj[:, 1], "r--", label="Pred", linewidth=1.5)
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_title("XY Plane")
    ax2.legend()
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)

    # XZ plane
    ax3 = fig.add_subplot(133)
    ax3.plot(gt_traj[:, 0], gt_traj[:, 2], "b-", label="GT", linewidth=1.5)
    ax3.plot(pred_traj[:, 0], pred_traj[:, 2], "r--", label="Pred", linewidth=1.5)
    ax3.set_xlabel("X (m)")
    ax3.set_ylabel("Z (m)")
    ax3.set_title("XZ Plane")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_training_curves(ckpt_dir, save_path):
    """Plot training/val loss curves for all models."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    model_types = ["imu", "visual", "combined"]
    colors = {"train": "#2196F3", "val": "#FF5722"}

    for ax, mtype in zip(axes, model_types):
        metrics_path = os.path.join(ckpt_dir, f"{mtype}_metrics.json")
        if not os.path.isfile(metrics_path):
            ax.set_title(f"{mtype.upper()} — no data")
            continue

        with open(metrics_path) as f:
            history = json.load(f)

        for split in ["train", "val"]:
            if history.get(split):
                epochs = [m["epoch"] for m in history[split]]
                losses = [m["loss"] for m in history[split]]
                ax.plot(epochs, losses, color=colors[split], label=split.capitalize(), linewidth=1.5)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"{mtype.upper()}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Training Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def compute_ate(gt_traj, pred_traj):
    """Absolute Trajectory Error (ATE) — RMSE of position differences."""
    n = min(len(gt_traj), len(pred_traj))
    diff = gt_traj[:n] - pred_traj[:n]
    return np.sqrt(np.mean(np.sum(diff**2, axis=1)))


def main():
    parser = argparse.ArgumentParser(description="Evaluate DeepVIO models")
    parser.add_argument("--data", required=True, help="Path to output/ directory")
    parser.add_argument("--checkpoint-dir", dest="checkpoint_dir", default="checkpoints")
    parser.add_argument("--output-dir", dest="output_dir", default="results")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=32)
    parser.add_argument("--img-size", dest="img_size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--attention", action="store_true")
    parser.add_argument("--max-steps", dest="max_steps", type=int, default=500,
                        help="Max trajectory steps to integrate for plotting")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else
                          "cpu")
    print(f"Device: {device}")

    # ── Load data ─────────────────────────────────────────────────────────────
    loaders = make_dataloaders(
        root_dir=args.data,
        batch_size=args.batch_size,
        num_workers=args.workers,
        img_size=(args.img_size, args.img_size),
    )
    if len(loaders) == 3:
        _, val_dl, test_dl = loaders
    else:
        _, val_dl = loaders
        test_dl = None

    eval_dl = test_dl if test_dl else val_dl
    eval_name = "Test" if test_dl else "Val"
    print(f"Evaluating on: {eval_name} set ({len(eval_dl)} batches)")

    # ── Evaluate each model ───────────────────────────────────────────────────
    criterion = CombinedVIOLoss()
    model_types = ["imu", "visual", "combined"]
    results = {}

    for mtype in model_types:
        ckpt_path = os.path.join(args.checkpoint_dir, f"{mtype}_best.pt")
        if not os.path.isfile(ckpt_path):
            print(f"\n⚠ Skipping {mtype} — no checkpoint found at {ckpt_path}")
            continue

        print(f"\n── {mtype.upper()} ──")
        model = load_model(mtype, ckpt_path, device, args.attention)

        # Get predictions
        pred_p, pred_q, gt_p, gt_q = predict_trajectory(model, mtype, eval_dl, device)

        # Compute metrics
        pred_p_t = torch.from_numpy(pred_p)
        pred_q_t = torch.from_numpy(pred_q)
        gt_p_t   = torch.from_numpy(gt_p)
        gt_q_t   = torch.from_numpy(gt_q)

        total_loss, pos_loss, rot_loss = criterion(pred_p_t, pred_q_t, gt_p_t, gt_q_t)

        # Integrate trajectories
        gt_traj   = integrate_relative_poses(gt_p, args.max_steps)
        pred_traj = integrate_relative_poses(pred_p, args.max_steps)
        ate = compute_ate(gt_traj, pred_traj)

        results[mtype] = {
            "total_loss": total_loss.item(),
            "pos_loss":   pos_loss.item(),
            "rot_loss":   rot_loss.item(),
            "ate_rmse":   ate,
            "n_samples":  len(pred_p),
        }

        print(f"  Loss={total_loss:.4f} (pos={pos_loss:.4f} rot={rot_loss:.4f}) ATE={ate:.4f}m")

        # Plot trajectory
        plot_trajectory_comparison(
            gt_traj, pred_traj,
            f"{mtype.upper()} — {eval_name} Trajectory",
            os.path.join(args.output_dir, f"{mtype}_trajectory.png"),
        )

    # ── Comparison table ──────────────────────────────────────────────────────
    table_path = os.path.join(args.output_dir, "comparison_table.txt")
    with open(table_path, "w") as f:
        header = f"{'Model':<12} {'Total Loss':>12} {'Pos Loss':>12} {'Rot Loss':>12} {'ATE (m)':>12} {'Samples':>10}"
        sep = "-" * len(header)
        f.write(f"DeepVIO Model Comparison — {eval_name} Set\n")
        f.write(sep + "\n")
        f.write(header + "\n")
        f.write(sep + "\n")
        for mtype in model_types:
            if mtype in results:
                r = results[mtype]
                f.write(f"{mtype:<12} {r['total_loss']:>12.4f} {r['pos_loss']:>12.4f} "
                        f"{r['rot_loss']:>12.4f} {r['ate_rmse']:>12.4f} {r['n_samples']:>10d}\n")
        f.write(sep + "\n")

    # Print table
    with open(table_path) as f:
        print(f"\n{f.read()}")

    # ── Training curves ───────────────────────────────────────────────────────
    plot_training_curves(args.checkpoint_dir, os.path.join(args.output_dir, "training_curves.png"))

    # ── Save results JSON ─────────────────────────────────────────────────────
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nAll results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
