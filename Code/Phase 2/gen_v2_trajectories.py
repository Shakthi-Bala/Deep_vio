#!/usr/bin/env python3
"""
Generate 3D trajectory visualizations for V2 models (ResNet18 + Two-Stage FiLM).
Run on lab laptop:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate cv_p3
    cd ~/Documents/Spring_26/CV/p4/DeepVIO
    python3 gen_v2_trajectories.py
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os, sys

sys.path.insert(0, '.')
from models_v2 import DeepVO_V2, DeepVIO_FiLM
from dataset import VIODataset
from torch.utils.data import DataLoader

fm._load_fontmanager(try_read_cache=False)
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Roboto'],
    'font.size': 11,
    'figure.dpi': 150,
})

BG = '#1a1a2e'
COLORS = {
    'gt': '#ffffff',
    'resnet': '#4dd0e1',
    'film_onestage': '#ce93d8',
    'film_twostage': '#66bb6a',
}
LABELS = {
    'gt': 'Ground Truth',
    'resnet': 'ResNet18 Visual',
    'film_onestage': 'FiLM (one-shot)',
    'film_twostage': 'FiLM (two-stage)',
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
viz_dir = 'visualizations'
os.makedirs(viz_dir, exist_ok=True)

# ── Helpers ──────────────────────────────────────────────────
def quat_to_rotmat(q):
    qx, qy, qz, qw = q
    return np.array([
        [1-2*(qy**2+qz**2), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw), 1-2*(qx**2+qz**2), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)],
    ])

def integrate_trajectory(rel_positions, rel_quats, start_pos):
    n = len(rel_positions)
    positions = np.zeros((n+1, 3))
    positions[0] = start_pos
    R_abs = np.eye(3)
    for i in range(n):
        dp_world = R_abs @ rel_positions[i]
        positions[i+1] = positions[i] + dp_world
        R_abs = R_abs @ quat_to_rotmat(rel_quats[i])
    return positions

def predict_sequence(model, model_type, dataset, indices):
    model.eval()
    pred_pos, pred_quat, gt_pos, gt_quat = [], [], [], []
    with torch.no_grad():
        for idx in indices:
            sample = dataset[idx]
            img = sample[0].unsqueeze(0).to(device)
            imu = sample[1].unsqueeze(0).to(device)
            if model_type == 'visual':
                p, q = model(img)
            else:
                p, q = model(img, imu)
            pred_pos.append(p.cpu().numpy()[0])
            pred_quat.append(q.cpu().numpy()[0])
            gt_pos.append(sample[2].numpy())
            gt_quat.append(sample[3].numpy())
    return np.array(pred_pos), np.array(pred_quat), np.array(gt_pos), np.array(gt_quat)

# ── Load models ──────────────────────────────────────────────
print("Loading models...")
models = {}

# ResNet18 Visual
ckpt_path = 'checkpoints/v2_resnet_visual/best.pt'
if os.path.exists(ckpt_path):
    m = DeepVO_V2(visual_encoder='resnet').to(device)
    m.load_state_dict(torch.load(ckpt_path, map_location=device)['model'])
    models['resnet'] = (m, 'visual')
    print(f"  Loaded ResNet18 visual")

# FiLM one-shot
ckpt_path = 'checkpoints/v2_film_combined/best.pt'
if os.path.exists(ckpt_path):
    m = DeepVIO_FiLM(visual_encoder='resnet').to(device)
    m.load_state_dict(torch.load(ckpt_path, map_location=device)['model'])
    models['film_onestage'] = (m, 'combined')
    print(f"  Loaded FiLM one-shot")

# FiLM two-stage
ckpt_path = 'checkpoints/v2_twostage_film/best.pt'
if os.path.exists(ckpt_path):
    m = DeepVIO_FiLM(visual_encoder='resnet').to(device)
    m.load_state_dict(torch.load(ckpt_path, map_location=device)['model'])
    models['film_twostage'] = (m, 'combined')
    print(f"  Loaded FiLM two-stage")

if not models:
    print("ERROR: No checkpoints found! Check paths.")
    sys.exit(1)

# ── Load test data ───────────────────────────────────────────
print("Loading test dataset...")
test_ds = VIODataset('output/test', img_size=(224, 224), augment=False)
test_dir = 'output/test'
test_seqs = sorted(os.listdir(test_dir))

seq_lengths = []
for seq in test_seqs:
    rel_path = os.path.join(test_dir, seq, 'relative_poses.csv')
    if os.path.exists(rel_path):
        seq_lengths.append(len(np.loadtxt(rel_path, delimiter=',', skiprows=1)))

seq_ranges = []
offset = 0
for length in seq_lengths:
    seq_ranges.append((offset, offset + length))
    offset += length

# Pick 3 sequences
viz_indices = [0, min(4, len(test_seqs)-1), min(8, len(test_seqs)-1)]
print(f"Test: {len(test_seqs)} seqs, visualizing indices {viz_indices}")

# ── Run inference ────────────────────────────────────────────
all_results = {}
for seq_idx in viz_indices:
    if seq_idx >= len(seq_ranges):
        continue
    start, end = seq_ranges[seq_idx]
    seq_name = test_seqs[seq_idx]
    indices = list(range(start, end))
    print(f"\nProcessing {seq_name} ({end-start} samples)...")

    gt_abs = np.loadtxt(os.path.join(test_dir, seq_name, 'groundtruth.csv'),
                        delimiter=',', skiprows=1)[:, 1:4]
    results = {'gt': gt_abs, 'seq_name': seq_name}

    for name, (model, mtype) in models.items():
        pred_p, pred_q, gt_p, gt_q = predict_sequence(model, mtype, test_ds, indices)
        pred_traj = integrate_trajectory(pred_p, pred_q, gt_abs[0])
        results[name] = pred_traj
        min_len = min(len(pred_traj), len(gt_abs))
        ate = np.sqrt(np.mean(np.sum((pred_traj[:min_len] - gt_abs[:min_len])**2, axis=1)))
        results[f'{name}_ate'] = ate
        print(f"  {name}: ATE = {ate:.2f}m")

    all_results[seq_idx] = results

# ── Generate figures ─────────────────────────────────────────

# FIG 14: V2 3D Trajectories (one per sequence)
for seq_idx, results in all_results.items():
    fig = plt.figure(figsize=(14, 10), facecolor=BG)
    gt = results['gt']
    seq_name = results['seq_name']

    # 3D view
    ax = fig.add_subplot(221, projection='3d', facecolor=BG)
    ax.plot(gt[:,0], gt[:,1], gt[:,2], '-', color=COLORS['gt'], lw=2.5, label='Ground Truth', alpha=0.9)
    for name in ['resnet', 'film_onestage', 'film_twostage']:
        if name in results:
            t = results[name]
            ate = results[f'{name}_ate']
            ax.plot(t[:,0], t[:,1], t[:,2], '-', color=COLORS[name], lw=1.8,
                    label=f'{LABELS[name]} (ATE={ate:.1f}m)', alpha=0.85)
    ax.scatter(*gt[0], color='#66bb6a', s=100, marker='^', zorder=5)
    ax.scatter(*gt[-1], color='#ef5350', s=100, marker='s', zorder=5)
    ax.set_xlabel('X (m)', color='white', fontsize=9)
    ax.set_ylabel('Y (m)', color='white', fontsize=9)
    ax.set_zlabel('Z (m)', color='white', fontsize=9)
    ax.set_title(f'3D Trajectory -- {seq_name}', color='white', fontsize=13, pad=10)
    ax.legend(fontsize=8, loc='upper left', facecolor='#25253d', edgecolor='#444466', labelcolor='white')
    ax.tick_params(colors='#888899', labelsize=7)
    for p in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        p.fill = False; p.set_edgecolor('#333355')
    ax.grid(True, alpha=0.2)

    # XY top-down
    ax2 = fig.add_subplot(222, facecolor=BG)
    ax2.plot(gt[:,0], gt[:,1], '-', color=COLORS['gt'], lw=2.5)
    for name in ['resnet', 'film_onestage', 'film_twostage']:
        if name in results:
            t = results[name]
            ax2.plot(t[:,0], t[:,1], '-', color=COLORS[name], lw=1.5, alpha=0.85)
    ax2.scatter(gt[0,0], gt[0,1], color='#66bb6a', s=80, marker='^', zorder=5)
    ax2.scatter(gt[-1,0], gt[-1,1], color='#ef5350', s=80, marker='s', zorder=5)
    ax2.set_title('Top-Down (XY)', color='white', fontsize=11)
    ax2.set_xlabel('X (m)', color='white', fontsize=9); ax2.set_ylabel('Y (m)', color='white', fontsize=9)
    ax2.set_aspect('equal'); ax2.tick_params(colors='#888899', labelsize=7)
    ax2.grid(True, alpha=0.15, color='#333355')
    for s in ax2.spines.values(): s.set_color('#333355')

    # XZ side
    ax3 = fig.add_subplot(223, facecolor=BG)
    ax3.plot(gt[:,0], gt[:,2], '-', color=COLORS['gt'], lw=2.5)
    for name in ['resnet', 'film_onestage', 'film_twostage']:
        if name in results:
            t = results[name]
            ax3.plot(t[:,0], t[:,2], '-', color=COLORS[name], lw=1.5, alpha=0.85)
    ax3.set_title('Side View (XZ)', color='white', fontsize=11)
    ax3.set_xlabel('X (m)', color='white', fontsize=9); ax3.set_ylabel('Z (m)', color='white', fontsize=9)
    ax3.tick_params(colors='#888899', labelsize=7)
    ax3.grid(True, alpha=0.15, color='#333355')
    for s in ax3.spines.values(): s.set_color('#333355')

    # Drift error
    ax4 = fig.add_subplot(224, facecolor=BG)
    for name in ['resnet', 'film_onestage', 'film_twostage']:
        if name in results:
            t = results[name]
            ml = min(len(t), len(gt))
            err = np.sqrt(np.sum((t[:ml] - gt[:ml])**2, axis=1))
            ax4.plot(err, '-', color=COLORS[name], lw=1.5, label=LABELS[name], alpha=0.85)
    ax4.set_title('Cumulative Drift Error', color='white', fontsize=11)
    ax4.set_xlabel('Frame', color='white', fontsize=9); ax4.set_ylabel('Error (m)', color='white', fontsize=9)
    ax4.legend(fontsize=8, facecolor='#25253d', edgecolor='#444466', labelcolor='white')
    ax4.tick_params(colors='#888899', labelsize=7)
    ax4.grid(True, alpha=0.15, color='#333355')
    for s in ax4.spines.values(): s.set_color('#333355')

    plt.tight_layout()
    fname = f'fig14_v2_trajectory_{seq_name}.png'
    plt.savefig(os.path.join(viz_dir, fname), bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"  Saved {fname}")

# FIG 15: Multi-sequence grid (V2 models)
n_seqs = len(all_results)
model_keys = [k for k in ['resnet', 'film_onestage', 'film_twostage'] if k in list(all_results.values())[0]]
n_models = len(model_keys)

if n_seqs > 0 and n_models > 0:
    fig, axes = plt.subplots(n_seqs, n_models, figsize=(5*n_models, 4.5*n_seqs), facecolor=BG)
    if n_seqs == 1: axes = axes.reshape(1, -1)
    if n_models == 1: axes = axes.reshape(-1, 1)

    fig.suptitle('V2 Model Trajectories (GT white vs Predicted)', color='white', fontsize=15, fontweight='bold', y=0.98)

    for row, (seq_idx, results) in enumerate(all_results.items()):
        gt = results['gt']
        for col, name in enumerate(model_keys):
            ax = axes[row, col]
            ax.set_facecolor(BG)
            ax.plot(gt[:,0], gt[:,1], '-', color=COLORS['gt'], lw=2.0, label='GT', alpha=0.9)
            if name in results:
                t = results[name]
                ate = results[f'{name}_ate']
                ax.plot(t[:,0], t[:,1], '-', color=COLORS[name], lw=1.5,
                        label=f'Pred (ATE={ate:.1f}m)', alpha=0.85)
            ax.scatter(gt[0,0], gt[0,1], color='#66bb6a', s=50, marker='^', zorder=5)
            ax.scatter(gt[-1,0], gt[-1,1], color='#ef5350', s=50, marker='s', zorder=5)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.15, color='#333355')
            ax.tick_params(colors='#888899', labelsize=7)
            for sp in ax.spines.values(): sp.set_color('#333355')
            ax.legend(fontsize=7, facecolor='#25253d', edgecolor='#444466', labelcolor='white')
            if row == 0:
                ax.set_title(LABELS[name], color=COLORS[name], fontsize=12, fontweight='bold', pad=8)
            if col == 0:
                ax.set_ylabel(f'{results["seq_name"]}\nY (m)', color='white', fontsize=9)
            if row == n_seqs - 1:
                ax.set_xlabel('X (m)', color='white', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'fig15_v2_multi_sequence.png'), bbox_inches='tight', facecolor=BG)
    plt.close()
    print("  Saved fig15_v2_multi_sequence.png")

# FIG 16: ATE bar chart
if all_results:
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
    ax.set_facecolor(BG)
    seq_names = [r['seq_name'] for r in all_results.values()]
    x = np.arange(len(seq_names))
    width = 0.25
    for i, name in enumerate(model_keys):
        ates = [r.get(f'{name}_ate', 0) for r in all_results.values()]
        bars = ax.bar(x + i*width - width*(n_models-1)/2, ates, width,
                      label=LABELS[name], color=COLORS[name], alpha=0.85)
        for bar, val in zip(bars, ates):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9, color='white')
    ax.set_xticks(x); ax.set_xticklabels(seq_names, color='white', fontsize=10)
    ax.set_ylabel('ATE (m)', color='white', fontsize=12)
    ax.set_title('V2 Models: Absolute Trajectory Error', color='white', fontsize=13)
    ax.tick_params(colors='#888899')
    ax.legend(fontsize=10, facecolor='#25253d', edgecolor='#444466', labelcolor='white')
    ax.grid(True, alpha=0.15, axis='y', color='#333355')
    for sp in ax.spines.values(): sp.set_color('#333355')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'fig16_v2_ate_comparison.png'), bbox_inches='tight', facecolor=BG)
    plt.close()
    print("  Saved fig16_v2_ate_comparison.png")

print("\n=== Done! New figures in ./visualizations/ ===")
