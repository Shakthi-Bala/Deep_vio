#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# overnight_experiments.sh — Run overnight experiments on RTX 5060
# Started: $(date)
# Expected completion: ~8 hours
# ═══════════════════════════════════════════════════════════════════════
set -e

PROJ_DIR="/home/adipat/Documents/Spring_26/CV/p4/DeepVIO"
cd "$PROJ_DIR"

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cv_p3

LOG="overnight_results.log"
echo "═══════════════════════════════════════════════════════════" | tee "$LOG"
echo "OVERNIGHT EXPERIMENT SUITE — $(date)" | tee -a "$LOG"
echo "═══════════════════════════════════════════════════════════" | tee -a "$LOG"

# ─────────────────────────────────────────────────────────────────
# EXPERIMENT 1: Combined model with lower LR (fix overfitting)
# Issue: Combined model overfits by epoch 2, best=epoch 1
# Hypothesis: LR too high for larger model (2.1M params)
# ─────────────────────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "▸ EXP 1: Combined — Lower LR (1e-4)" | tee -a "$LOG"
python train.py --data output/ --model combined --epochs 80 --attention \
    --batch-size 32 --lr 1e-4 --patience 15 \
    --checkpoint-dir checkpoints/exp1_combined_lowlr 2>&1 | tee -a "$LOG"

# ─────────────────────────────────────────────────────────────────
# EXPERIMENT 2: Combined model without attention
# Hypothesis: Attention on single-step embedding may not help
# ─────────────────────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "▸ EXP 2: Combined — No Attention" | tee -a "$LOG"
python train.py --data output/ --model combined --epochs 80 \
    --batch-size 32 --lr 5e-4 --patience 15 \
    --checkpoint-dir checkpoints/exp2_combined_noattn 2>&1 | tee -a "$LOG"

# ─────────────────────────────────────────────────────────────────
# EXPERIMENT 3: Visual model with lower LR + more epochs
# Visual had best val loss but highest ATE — investigate
# ─────────────────────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "▸ EXP 3: Visual — Lower LR (5e-4), 80 epochs" | tee -a "$LOG"
python train.py --data output/ --model visual --epochs 80 --attention \
    --batch-size 32 --lr 5e-4 --patience 15 \
    --checkpoint-dir checkpoints/exp3_visual_lowlr 2>&1 | tee -a "$LOG"

# ─────────────────────────────────────────────────────────────────
# EXPERIMENT 4: IMU with larger batch + no attention
# IMU model early stopped at ep 13, heavy overfitting
# ─────────────────────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "▸ EXP 4: IMU — No Attention, LR 5e-4" | tee -a "$LOG"
python train.py --data output/ --model imu --epochs 80 \
    --batch-size 128 --lr 5e-4 --patience 15 \
    --checkpoint-dir checkpoints/exp4_imu_noattn 2>&1 | tee -a "$LOG"

# ─────────────────────────────────────────────────────────────────
# EXPERIMENT 5: Combined with balanced loss weighting
# Rotation loss is very small — increase lambda_q
# ─────────────────────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "▸ EXP 5: Combined — lambda_q=10" | tee -a "$LOG"
python train.py --data output/ --model combined --epochs 80 --attention \
    --batch-size 32 --lr 5e-4 --patience 15 --lambda-q 10.0 \
    --checkpoint-dir checkpoints/exp5_combined_highq 2>&1 | tee -a "$LOG"

# ─────────────────────────────────────────────────────────────────
# EXPERIMENT 6: Combined with balanced loss + no attention
# ─────────────────────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "▸ EXP 6: Combined — lambda_q=5, no attn, lr=3e-4" | tee -a "$LOG"
python train.py --data output/ --model combined --epochs 80 \
    --batch-size 32 --lr 3e-4 --patience 15 --lambda-q 5.0 \
    --checkpoint-dir checkpoints/exp6_combined_balanced 2>&1 | tee -a "$LOG"

# ─────────────────────────────────────────────────────────────────
# FINAL: Run evaluation on ALL experiment checkpoints
# ─────────────────────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "═══════════════════════════════════════════════════════════" | tee -a "$LOG"
echo "RUNNING FINAL EVALUATIONS" | tee -a "$LOG"
echo "═══════════════════════════════════════════════════════════" | tee -a "$LOG"

for exp_dir in checkpoints/exp*; do
    exp_name=$(basename "$exp_dir")
    echo "" | tee -a "$LOG"
    echo "▸ Evaluating: $exp_name" | tee -a "$LOG"
    
    # Determine model type from dir name
    if [[ "$exp_name" == *"combined"* ]]; then
        model_type="combined"
    elif [[ "$exp_name" == *"visual"* ]]; then
        model_type="visual"
    elif [[ "$exp_name" == *"imu"* ]]; then
        model_type="imu"
    fi
    
    python train.py --data output/ --model "$model_type" --test-only \
        --resume "$exp_dir/${model_type}_best.pt" \
        --checkpoint-dir "$exp_dir" \
        $(if [[ "$exp_name" == *"noattn"* ]]; then echo ""; else echo "--attention"; fi) \
        2>&1 | tee -a "$LOG"
done

# ─────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "═══════════════════════════════════════════════════════════" | tee -a "$LOG"
echo "ALL EXPERIMENTS COMPLETE — $(date)" | tee -a "$LOG"
echo "═══════════════════════════════════════════════════════════" | tee -a "$LOG"

# Collect all results into one summary
python3 << 'PYEOF' 2>&1 | tee -a "$LOG"
import json, os, glob

print("\n" + "="*80)
print("OVERNIGHT EXPERIMENT SUMMARY")
print("="*80)
print(f"{'Experiment':<35} {'Val Loss':>10} {'Test Loss':>10} {'Test Pos':>10} {'Test Rot':>10}")
print("-"*80)

# Baseline results
baseline = "checkpoints"
for model in ["imu", "visual", "combined"]:
    mf = os.path.join(baseline, f"{model}_metrics.json")
    if os.path.isfile(mf):
        with open(mf) as f:
            d = json.load(f)
        best_val = min(d["val"], key=lambda x: x["loss"])["loss"] if d["val"] else "N/A"
        test = d.get("test", {})
        tl = test.get("loss", "N/A") if test else "N/A"
        tp = test.get("pos_loss", "N/A") if test else "N/A"
        tr = test.get("rot_loss", "N/A") if test else "N/A"
        print(f"baseline_{model:<29} {best_val:>10.4f} {tl:>10.4f} {tp:>10.4f} {tr:>10.4f}")

# Experiment results
for exp_dir in sorted(glob.glob("checkpoints/exp*")):
    exp_name = os.path.basename(exp_dir)
    for mf in glob.glob(os.path.join(exp_dir, "*_metrics.json")):
        with open(mf) as f:
            d = json.load(f)
        best_val = min(d["val"], key=lambda x: x["loss"])["loss"] if d["val"] else "N/A"
        test = d.get("test", {})
        tl = test.get("loss", "N/A") if test else "N/A"
        tp = test.get("pos_loss", "N/A") if test else "N/A"
        tr = test.get("rot_loss", "N/A") if test else "N/A"
        if isinstance(tl, float):
            print(f"{exp_name:<35} {best_val:>10.4f} {tl:>10.4f} {tp:>10.4f} {tr:>10.4f}")
        else:
            print(f"{exp_name:<35} {best_val:>10.4f} {'N/A':>10} {'N/A':>10} {'N/A':>10}")

print("="*80)
PYEOF

echo "Log saved to: $PROJ_DIR/$LOG"
