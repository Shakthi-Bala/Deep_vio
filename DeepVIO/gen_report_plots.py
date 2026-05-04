"""Generate all report figures: data gen, results comparison, training curves."""
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import cv2
import os
import json

fm._load_fontmanager(try_read_cache=False)
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Roboto'],
    'font.size': 11,
    'figure.dpi': 150,
    'axes.titleweight': 'bold',
})

BG = '#1a1a2e'
CARD = '#25253d'
WHITE = '#ffffff'
GRAY = '#bbbbcc'
BLUE = '#64b5f6'
RED = '#ef5350'
GREEN = '#66bb6a'
ORANGE = '#ffa726'
PURPLE = '#ce93d8'
CYAN = '#4dd0e1'

viz_dir = 'visualizations'
os.makedirs(viz_dir, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# FIG 10: Data Generation Pipeline — Sample Images + Trajectories
# ═══════════════════════════════════════════════════════════════
print("Fig 10: Data generation showcase...")

fig = plt.figure(figsize=(16, 8), facecolor=BG)
fig.suptitle('Synthetic Data Generation Pipeline', color='white', fontsize=16, fontweight='bold', y=0.98)

# Row 1: Sample rendered images from different textures (new dataset)
train_dir = 'output/train'
seqs = sorted(os.listdir(train_dir))
sample_seqs = [seqs[i] for i in range(0, min(len(seqs), 48), max(1, len(seqs)//8))][:8]

for i, seq in enumerate(sample_seqs):
    ax = fig.add_subplot(2, 8, i+1)
    ax.set_facecolor(BG)
    img_path = os.path.join(train_dir, seq, 'images', '00050.png')
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
    ax.axis('off')
    ax.set_title(seq.replace('seq_', 'S'), fontsize=8, color=GRAY)

# Row 2: Sample trajectories (GT from different sequences)
traj_colors = [BLUE, ORANGE, GREEN, PURPLE, CYAN, RED, '#ff80ab', '#b388ff']
ax_traj_xy = fig.add_subplot(2, 4, 5, facecolor=BG)
ax_traj_3d = fig.add_subplot(2, 4, 6, projection='3d', facecolor=BG)
ax_traj_xz = fig.add_subplot(2, 4, 7, facecolor=BG)
ax_flow = fig.add_subplot(2, 4, 8, facecolor=BG)

for i, seq in enumerate(sample_seqs[:5]):
    gt_path = os.path.join(train_dir, seq, 'groundtruth.csv')
    if not os.path.exists(gt_path):
        continue
    gt = np.loadtxt(gt_path, delimiter=',', skiprows=1)
    pos = gt[:, 1:4]
    c = traj_colors[i % len(traj_colors)]
    
    ax_traj_xy.plot(pos[:, 0], pos[:, 1], '-', color=c, linewidth=1.2, alpha=0.8)
    ax_traj_3d.plot(pos[:, 0], pos[:, 1], pos[:, 2], '-', color=c, linewidth=1.0, alpha=0.8)
    ax_traj_xz.plot(pos[:, 0], pos[:, 2], '-', color=c, linewidth=1.2, alpha=0.8)

for ax, title in [(ax_traj_xy, 'Trajectories (XY)'), (ax_traj_xz, 'Trajectories (XZ)')]:
    ax.set_title(title, color='white', fontsize=10)
    ax.tick_params(colors='#666677', labelsize=7)
    ax.grid(True, alpha=0.15, color='#333355')
    for s in ax.spines.values(): s.set_color('#333355')
    ax.set_xlabel('X (m)', color=GRAY, fontsize=8)
    ax.set_ylabel('Y (m)' if 'XY' in title else 'Z (m)', color=GRAY, fontsize=8)

ax_traj_3d.set_title('3D Trajectories', color='white', fontsize=10)
ax_traj_3d.tick_params(colors='#666677', labelsize=6)
for pane in [ax_traj_3d.xaxis.pane, ax_traj_3d.yaxis.pane, ax_traj_3d.zaxis.pane]:
    pane.fill = False; pane.set_edgecolor('#333355')

# Show optical flow sample
flow_path = os.path.join(train_dir, seqs[5] if len(seqs) > 5 else seqs[0], 'flow', '00050.npy')
if os.path.exists(flow_path):
    flow = np.load(flow_path).astype(np.float32)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    ax_flow.imshow(flow_rgb)
    ax_flow.set_title('Optical Flow', color='white', fontsize=10)
ax_flow.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(viz_dir, 'fig10_data_generation.png'), bbox_inches='tight', facecolor=BG)
plt.close()
print("  Saved fig10_data_generation.png")

# ═══════════════════════════════════════════════════════════════
# FIG 11: Results Comparison — All Models Bar Chart
# ═══════════════════════════════════════════════════════════════
print("Fig 11: Results comparison...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)

# Test loss comparison
models = [
    'CNN\n(old data)', 'CNN\n(new data)', 'ResNet18\n(V2)',
    'IMU\n(old)', 'IMU\n(new)',
    'Concat\n(old)', 'Concat\n(new)', 'FiLM\n(V2)'
]
test_losses = [0.3017, 0.1822, 0.0295, 0.5430, 0.2437, 0.5129, 0.2055, 0.1331]
colors_bar = [BLUE, BLUE, CYAN, ORANGE, ORANGE, PURPLE, PURPLE, '#ff80ab']
alphas = [0.4, 0.7, 1.0, 0.4, 0.7, 0.4, 0.7, 1.0]

ax1.set_facecolor(BG)
bars = ax1.bar(range(len(models)), test_losses, color=colors_bar, edgecolor='#444466', linewidth=0.5)
for bar, a in zip(bars, alphas):
    bar.set_alpha(a)
for i, (bar, val) in enumerate(zip(bars, test_losses)):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.4f}', ha='center', va='bottom', fontsize=8, color='white', fontweight='bold')

ax1.set_xticks(range(len(models)))
ax1.set_xticklabels(models, fontsize=8, color=GRAY)
ax1.set_ylabel('Test Loss', color='white', fontsize=12)
ax1.set_title('Test Loss Across All Experiments', color='white', fontsize=13)
ax1.tick_params(colors='#888899')
ax1.grid(True, alpha=0.15, axis='y', color='#333355')
for s in ax1.spines.values(): s.set_color('#333355')

# Improvement waterfall
ax2.set_facecolor(BG)
stages = ['Baseline\nCNN', '+New Data\n(-40%)', '+ResNet18\n(-84%)', '+FiLM\nFusion']
values = [0.3017, 0.1822, 0.0295, 0.1331]
stage_colors = [RED, ORANGE, GREEN, CYAN]

for i in range(len(stages)):
    ax2.bar(i, values[i], color=stage_colors[i], alpha=0.85, edgecolor='#444466')
    ax2.text(i, values[i] + 0.008, f'{values[i]:.4f}', ha='center', fontsize=9, color='white', fontweight='bold')
    if i > 0:
        pct = (1 - values[i]/values[0]) * 100
        ax2.annotate(f'-{pct:.0f}%', xy=(i, values[i]/2), fontsize=11, color='white',
                     ha='center', fontweight='bold')

ax2.set_xticks(range(len(stages)))
ax2.set_xticklabels(stages, fontsize=9, color=GRAY)
ax2.set_ylabel('Test Loss', color='white', fontsize=12)
ax2.set_title('Progressive Improvement', color='white', fontsize=13)
ax2.tick_params(colors='#888899')
ax2.grid(True, alpha=0.15, axis='y', color='#333355')
for s in ax2.spines.values(): s.set_color('#333355')

plt.tight_layout()
plt.savefig(os.path.join(viz_dir, 'fig11_results_comparison.png'), bbox_inches='tight', facecolor=BG)
plt.close()
print("  Saved fig11_results_comparison.png")

# ═══════════════════════════════════════════════════════════════
# FIG 12: V2 Training Curves (ResNet + FiLM)
# ═══════════════════════════════════════════════════════════════
print("Fig 12: V2 training curves...")

# Parse v2 training log
resnet_train, resnet_val = [], []
film_train, film_val, film_gate = [], [], []

log_path = 'v2_training.log'
if os.path.exists(log_path):
    current_model = None
    with open(log_path) as f:
        for line in f:
            if 'EXP A' in line: current_model = 'resnet'
            elif 'EXP B' in line: current_model = 'film'
            elif line.startswith('Ep '):
                parts = line.split('|')
                train_val = float(parts[1].split()[-1])
                val_val = float(parts[2].split()[-1])
                if current_model == 'resnet':
                    resnet_train.append(train_val)
                    resnet_val.append(val_val)
                elif current_model == 'film':
                    film_train.append(train_val)
                    film_val.append(val_val)
                    gate_str = parts[3].split('=')[-1].strip()
                    film_gate.append(float(gate_str))

fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=BG)

# ResNet training curves
ax = axes[0]
ax.set_facecolor(BG)
if resnet_train:
    eps = range(1, len(resnet_train)+1)
    ax.plot(eps, resnet_train, '-', color=BLUE, linewidth=2, label='Train')
    ax.plot(eps, resnet_val, '--', color=CYAN, linewidth=2, label='Val')
    ax.fill_between(eps, resnet_train, resnet_val, alpha=0.1, color=BLUE)
    best_ep = np.argmin(resnet_val) + 1
    ax.axvline(best_ep, color=GREEN, linestyle=':', alpha=0.5)
    ax.plot(best_ep, min(resnet_val), '*', color=GREEN, markersize=12)
ax.set_title('DeepVO_V2 (ResNet18)', color=BLUE, fontsize=12)
ax.set_xlabel('Epoch', color=GRAY); ax.set_ylabel('Loss', color=GRAY)
ax.legend(facecolor=CARD, edgecolor='#444466', labelcolor='white', fontsize=9)
ax.tick_params(colors='#888899'); ax.grid(True, alpha=0.15, color='#333355')
for s in ax.spines.values(): s.set_color('#333355')

# FiLM training curves
ax = axes[1]
ax.set_facecolor(BG)
if film_train:
    eps = range(1, len(film_train)+1)
    ax.plot(eps, film_train, '-', color=PURPLE, linewidth=2, label='Train')
    ax.plot(eps, film_val, '--', color='#ff80ab', linewidth=2, label='Val')
    ax.fill_between(eps, film_train, film_val, alpha=0.1, color=PURPLE)
ax.set_title('DeepVIO FiLM (Still Overfits)', color=PURPLE, fontsize=12)
ax.set_xlabel('Epoch', color=GRAY); ax.set_ylabel('Loss', color=GRAY)
ax.legend(facecolor=CARD, edgecolor='#444466', labelcolor='white', fontsize=9)
ax.tick_params(colors='#888899'); ax.grid(True, alpha=0.15, color='#333355')
for s in ax.spines.values(): s.set_color('#333355')

# FiLM gate values
ax = axes[2]
ax.set_facecolor(BG)
if film_gate:
    eps = range(1, len(film_gate)+1)
    ax.plot(eps, film_gate, 'o-', color=ORANGE, linewidth=2, markersize=4)
    ax.axhline(0.5, color=GREEN, linestyle='--', alpha=0.5, label='Balanced (0.5)')
    ax.fill_between(eps, film_gate, 0.5, alpha=0.15, color=RED)
ax.set_title('FiLM Gate (Vision Weight)', color=ORANGE, fontsize=12)
ax.set_xlabel('Epoch', color=GRAY); ax.set_ylabel('Gate Value', color=GRAY)
ax.set_ylim(0, 0.6)
ax.legend(facecolor=CARD, edgecolor='#444466', labelcolor='white', fontsize=9)
ax.tick_params(colors='#888899'); ax.grid(True, alpha=0.15, color='#333355')
ax.annotate('Vision suppressed\n(gate < 0.2)', xy=(8, 0.1), fontsize=10, color=RED, ha='center', fontweight='bold')
for s in ax.spines.values(): s.set_color('#333355')

plt.tight_layout()
plt.savefig(os.path.join(viz_dir, 'fig12_v2_training_curves.png'), bbox_inches='tight', facecolor=BG)
plt.close()
print("  Saved fig12_v2_training_curves.png")

# ═══════════════════════════════════════════════════════════════
# FIG 13: Position & Rotation Loss Breakdown
# ═══════════════════════════════════════════════════════════════
print("Fig 13: Loss breakdown...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG)

models_names = ['CNN\n(old)', 'CNN\n(new)', 'ResNet\n(V2)', 'IMU\n(old)', 'IMU\n(new)', 'Concat\n(old)', 'Concat\n(new)', 'FiLM\n(V2)']
pos_losses = [0.2815, 0.1622, 0.0269, 0.5321, 0.2211, 0.4970, 0.1868, 0.1194]
rot_losses = [0.0202, 0.0200, 0.0026, 0.0109, 0.0226, 0.0159, 0.0187, 0.0137]
bar_colors = [BLUE, BLUE, CYAN, ORANGE, ORANGE, PURPLE, PURPLE, '#ff80ab']
bar_alphas = [0.4, 0.7, 1.0, 0.4, 0.7, 0.4, 0.7, 1.0]

x = np.arange(len(models_names))

ax1.set_facecolor(BG)
bars = ax1.bar(x, pos_losses, color=bar_colors, edgecolor='#444466')
for b, a in zip(bars, bar_alphas): b.set_alpha(a)
for i, v in enumerate(pos_losses):
    ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=7, color='white')
ax1.set_xticks(x); ax1.set_xticklabels(models_names, fontsize=7, color=GRAY)
ax1.set_title('Position Loss (L1)', color='white', fontsize=12)
ax1.set_ylabel('Test Position Loss', color=GRAY)
ax1.tick_params(colors='#888899'); ax1.grid(True, alpha=0.15, axis='y', color='#333355')
for s in ax1.spines.values(): s.set_color('#333355')

ax2.set_facecolor(BG)
bars = ax2.bar(x, rot_losses, color=bar_colors, edgecolor='#444466')
for b, a in zip(bars, bar_alphas): b.set_alpha(a)
for i, v in enumerate(rot_losses):
    ax2.text(i, v + 0.0005, f'{v:.4f}', ha='center', fontsize=7, color='white')
ax2.set_xticks(x); ax2.set_xticklabels(models_names, fontsize=7, color=GRAY)
ax2.set_title('Rotation Loss (Geodesic)', color='white', fontsize=12)
ax2.set_ylabel('Test Rotation Loss', color=GRAY)
ax2.tick_params(colors='#888899'); ax2.grid(True, alpha=0.15, axis='y', color='#333355')
for s in ax2.spines.values(): s.set_color('#333355')

plt.tight_layout()
plt.savefig(os.path.join(viz_dir, 'fig13_loss_breakdown.png'), bbox_inches='tight', facecolor=BG)
plt.close()
print("  Saved fig13_loss_breakdown.png")

print("\n=== All new figures saved ===")
for f in sorted(os.listdir(viz_dir)):
    if 'fig1' in f and len(f) > 20:
        print(f"  {f}")
