"""
train.py — Train DeepIO (IMU-only), DeepVO (vision-only), DeepVIO (fused).

Usage
-----
# Train all three in sequence:
    python train.py --data output/ --model all --epochs 50

# Train a single model:
    python train.py --data output/ --model visual --epochs 50 --attention

# Resume from checkpoint:
    python train.py --data output/ --model imu --resume checkpoints/imu_best.pt

# Test only (no training):
    python train.py --data output/ --model combined --test-only --resume checkpoints/combined_best.pt
"""

import os
import argparse
import time
import json
import multiprocessing as mp

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from models import DeepIO, DeepVO, DeepVIO, CombinedVIOLoss
from dataset import make_dataloaders


# ─────────────────────────────────────────────────────────────────────────────
# Metrics tracker
# ─────────────────────────────────────────────────────────────────────────────
class MetricsLog:
    """Accumulates per-epoch metrics and saves to JSON."""

    def __init__(self, path):
        self.path = path
        self.history = {"train": [], "val": [], "test": None}

    def append(self, split, epoch_metrics):
        self.history[split].append(epoch_metrics)
        self._save()

    def set_test(self, test_metrics):
        self.history["test"] = test_metrics
        self._save()

    def _save(self):
        with open(self.path, "w") as f:
            json.dump(self.history, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Forward pass helper (avoids repeating model-type dispatch)
# ─────────────────────────────────────────────────────────────────────────────
def forward_pass(model, model_type, img_pair, imu_seq):
    if model_type == "visual":
        return model(img_pair)
    elif model_type == "imu":
        return model(imu_seq)
    else:
        return model(img_pair, imu_seq)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation on a dataloader (val or test)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, model_type, dataloader, criterion, device, tag="VAL"):
    model.eval()
    total_loss = total_pos = total_rot = 0.0
    n_batches = 0

    for img_pair, imu_seq, gt_p, gt_q in dataloader:
        img_pair = img_pair.to(device, non_blocking=True)
        imu_seq  = imu_seq.to(device, non_blocking=True)
        gt_p     = gt_p.to(device, non_blocking=True)
        gt_q     = gt_q.to(device, non_blocking=True)

        with autocast(enabled=(device.type == "cuda")):
            pred_p, pred_q = forward_pass(model, model_type, img_pair, imu_seq)
            loss, p_loss, q_loss = criterion(pred_p, pred_q, gt_p, gt_q)

        total_loss += loss.item()
        total_pos  += p_loss.item()
        total_rot  += q_loss.item()
        n_batches  += 1

    if n_batches == 0:
        return {"loss": 0.0, "pos_loss": 0.0, "rot_loss": 0.0}

    return {
        "loss":     total_loss / n_batches,
        "pos_loss": total_pos  / n_batches,
        "rot_loss": total_rot  / n_batches,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Core training logic
# ─────────────────────────────────────────────────────────────────────────────
def train_one_model(model_type: str, args, gpu_id: int = 0):
    """Full training + validation + test loop for one model."""

    # ── Device ────────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    tag = f"[{model_type.upper()}@{device}]"
    print(f"{tag} Starting — {args.epochs} epochs, lr={args.lr}, bs={args.batch_size}")

    # ── Data ──────────────────────────────────────────────────────────────────
    loaders = make_dataloaders(
        root_dir=args.data,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.workers,
        img_size=(args.img_size, args.img_size),
    )

    # Handle both 2-tuple (legacy) and 3-tuple (split-based) returns
    if len(loaders) == 3:
        train_dl, val_dl, test_dl = loaders
    else:
        train_dl, val_dl = loaders
        test_dl = None

    print(f"{tag} Data: train={len(train_dl)} batches, val={len(val_dl)} batches"
          + (f", test={len(test_dl)} batches" if test_dl else ", test=None"))

    # ── Model ─────────────────────────────────────────────────────────────────
    if model_type == "visual":
        model = DeepVO(use_attention=args.attention)
    elif model_type == "imu":
        model = DeepIO(use_attention=args.attention)
    else:
        model = DeepVIO(use_attention=args.attention)

    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{tag} Parameters: {n_params:,}")

    # ── Optimiser / scheduler / loss ──────────────────────────────────────────
    criterion = CombinedVIOLoss(lambda_p=args.lambda_p, lambda_q=args.lambda_q)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler    = GradScaler(enabled=(device.type == "cuda"))

    # ── Metrics logger ────────────────────────────────────────────────────────
    ckpt_dir = args.checkpoint_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    metrics_log = MetricsLog(os.path.join(ckpt_dir, f"{model_type}_metrics.json"))

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    best_val    = float("inf")

    resume_path = args.resume if args.resume else os.path.join(ckpt_dir, f"{model_type}_best.pt")
    if args.resume and os.path.isfile(resume_path):
        ck = torch.load(resume_path, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        start_epoch = ck["epoch"] + 1
        best_val    = ck.get("best_val", float("inf"))
        print(f"{tag} Resumed from epoch {start_epoch}, best_val={best_val:.4f}")

    # ── Test-only mode ────────────────────────────────────────────────────────
    if args.test_only:
        if not os.path.isfile(resume_path):
            print(f"{tag} ERROR: --test-only requires a checkpoint (--resume)")
            return
        ck = torch.load(resume_path, map_location=device)
        model.load_state_dict(ck["model"])

        print(f"\n{tag} ── TEST-ONLY EVALUATION ──")
        val_result = evaluate(model, model_type, val_dl, criterion, device, "VAL")
        print(f"{tag} Val  Loss={val_result['loss']:.4f} "
              f"(pos={val_result['pos_loss']:.4f} rot={val_result['rot_loss']:.4f})")

        if test_dl:
            test_result = evaluate(model, model_type, test_dl, criterion, device, "TEST")
            print(f"{tag} Test Loss={test_result['loss']:.4f} "
                  f"(pos={test_result['pos_loss']:.4f} rot={test_result['rot_loss']:.4f})")
            metrics_log.set_test(test_result)
        return

    # ── Training loop ─────────────────────────────────────────────────────────
    patience_counter = 0
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        tr_loss = tr_pos = tr_rot = 0.0

        for img_pair, imu_seq, gt_p, gt_q in train_dl:
            img_pair = img_pair.to(device, non_blocking=True)
            imu_seq  = imu_seq.to(device, non_blocking=True)
            gt_p     = gt_p.to(device, non_blocking=True)
            gt_q     = gt_q.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=(device.type == "cuda")):
                pred_p, pred_q = forward_pass(model, model_type, img_pair, imu_seq)
                loss, p_loss, q_loss = criterion(pred_p, pred_q, gt_p, gt_q)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            tr_loss += loss.item()
            tr_pos  += p_loss.item()
            tr_rot  += q_loss.item()

        n = len(train_dl)
        tr_loss /= n;  tr_pos /= n;  tr_rot /= n

        # ── Validate ──────────────────────────────────────────────────────────
        val_result = evaluate(model, model_type, val_dl, criterion, device, "VAL")

        scheduler.step()
        elapsed = time.time() - t0

        # ── Log ───────────────────────────────────────────────────────────────
        train_metrics = {
            "epoch": epoch + 1, "loss": tr_loss,
            "pos_loss": tr_pos, "rot_loss": tr_rot,
            "lr": scheduler.get_last_lr()[0],
        }
        val_metrics = {"epoch": epoch + 1, **val_result}

        metrics_log.append("train", train_metrics)
        metrics_log.append("val", val_metrics)

        print(
            f"{tag} Ep {epoch+1:03d}/{args.epochs} | "
            f"Train {tr_loss:.4f} (p={tr_pos:.4f} q={tr_rot:.4f}) | "
            f"Val {val_result['loss']:.4f} (p={val_result['pos_loss']:.4f} q={val_result['rot_loss']:.4f}) | "
            f"LR {scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}s"
        )

        # ── Checkpoint ────────────────────────────────────────────────────────
        state = {
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val":  best_val,
            "args":      vars(args),
        }

        torch.save(state, os.path.join(ckpt_dir, f"{model_type}_last.pt"))

        if val_result["loss"] < best_val:
            best_val = val_result["loss"]
            state["best_val"] = best_val
            torch.save(state, os.path.join(ckpt_dir, f"{model_type}_best.pt"))
            print(f"{tag} ✓ New best val={best_val:.4f} — checkpoint saved")
            patience_counter = 0
        else:
            patience_counter += 1

        # ── Early stopping ────────────────────────────────────────────────────
        if args.patience > 0 and patience_counter >= args.patience:
            print(f"{tag} Early stopping at epoch {epoch+1} (patience={args.patience})")
            break

    # ── Final test evaluation ─────────────────────────────────────────────────
    print(f"\n{tag} ── FINAL EVALUATION ──")

    # Load best checkpoint for final eval
    best_path = os.path.join(ckpt_dir, f"{model_type}_best.pt")
    if os.path.isfile(best_path):
        ck = torch.load(best_path, map_location=device)
        model.load_state_dict(ck["model"])
        print(f"{tag} Loaded best checkpoint (epoch {ck['epoch']+1})")

    val_result = evaluate(model, model_type, val_dl, criterion, device, "VAL")
    print(f"{tag} Val  Loss={val_result['loss']:.4f} "
          f"(pos={val_result['pos_loss']:.4f} rot={val_result['rot_loss']:.4f})")

    if test_dl:
        test_result = evaluate(model, model_type, test_dl, criterion, device, "TEST")
        print(f"{tag} Test Loss={test_result['loss']:.4f} "
              f"(pos={test_result['pos_loss']:.4f} rot={test_result['rot_loss']:.4f})")
        metrics_log.set_test(test_result)
    else:
        print(f"{tag} No test set available.")

    print(f"{tag} Done. Best val loss: {best_val:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Multiprocessing wrapper
# ─────────────────────────────────────────────────────────────────────────────
def _worker(model_type, args, gpu_id):
    train_one_model(model_type, args, gpu_id)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="DeepVIO training")

    p.add_argument("--data",     required=True,  help="Path to output/ directory")
    p.add_argument("--model",    default="all",  choices=["visual", "imu", "combined", "all"])
    p.add_argument("--epochs",   type=int,   default=50)
    p.add_argument("--lr",       type=float, default=1e-3)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=32)
    p.add_argument("--img-size", dest="img_size",    type=int, default=224)
    p.add_argument("--val-split", dest="val_split",  type=float, default=0.15)
    p.add_argument("--workers",  type=int,   default=4)
    p.add_argument("--lambda-p", dest="lambda_p",    type=float, default=1.0)
    p.add_argument("--lambda-q", dest="lambda_q",    type=float, default=1.0)
    p.add_argument("--attention", action="store_true", help="Enable multi-head attention")
    p.add_argument("--checkpoint-dir", dest="checkpoint_dir", default="checkpoints")
    p.add_argument("--resume",   default=None,  help="Path to checkpoint to resume")
    p.add_argument("--patience", type=int, default=10, help="Early stopping patience (0=off)")
    p.add_argument("--test-only", dest="test_only", action="store_true",
                   help="Only run test evaluation (requires --resume)")
    p.add_argument("--device",   default=None,
                   help="Force device, e.g. 'cpu' or 'cuda:0'")

    return p.parse_args()


def assign_gpus(model_types):
    n_gpus = torch.cuda.device_count()
    return {m: i % max(n_gpus, 1) for i, m in enumerate(model_types)}


def main():
    args = parse_args()

    model_types = (
        ["visual", "imu", "combined"] if args.model == "all" else [args.model]
    )

    if len(model_types) == 1 or not torch.cuda.is_available():
        gpu_id = 0
        if args.device and args.device.startswith("cuda:"):
            gpu_id = int(args.device.split(":")[1])
        for m in model_types:
            train_one_model(m, args, gpu_id)

    else:
        gpu_map = assign_gpus(model_types)
        n_gpus  = torch.cuda.device_count()

        print(f"Launching {len(model_types)} parallel training processes")
        print(f"GPU map: {gpu_map}  ({n_gpus} GPU(s) available)")
        if n_gpus == 1:
            print("WARNING: All models sharing cuda:0. Reduce --batch-size if OOM.")

        mp.set_start_method("spawn", force=True)
        processes = []
        for m in model_types:
            proc = mp.Process(target=_worker, args=(m, args, gpu_map[m]))
            proc.start()
            processes.append(proc)

        for proc in processes:
            proc.join()

        exit_codes = [p.exitcode for p in processes]
        failed = [model_types[i] for i, ec in enumerate(exit_codes) if ec != 0]
        if failed:
            print(f"FAILED models: {failed}")
        else:
            print("All training processes completed successfully.")


if __name__ == "__main__":
    main()
