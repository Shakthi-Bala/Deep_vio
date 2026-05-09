#!/usr/bin/env python3
"""
Complete evaluation: DeepVO, DeepIO, DeepVIO (all variants) with trajectory plots.
Run on lab laptop:
    source /media/adipat/02EC0603EC05F1A9/archive/home/miniconda3/etc/profile.d/conda.sh
    conda activate cv_p3
    cd ~/Documents/Spring_26/CV/p4/DeepVIO
    CUDA_VISIBLE_DEVICES="" python3 eval_all_models.py
"""
import torch, numpy as np, matplotlib.pyplot as plt, matplotlib.font_manager as fm, os, sys

sys.path.insert(0, '.')
from models import DeepVO, DeepIO, DeepVIO
from models_v2 import DeepVO_V2, DeepVIO_FiLM
from dataset import VIODataset

fm._load_fontmanager(try_read_cache=False)
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Roboto'],'font.size':11,'figure.dpi':150})

BG='#1a1a2e'
COLORS = {
    'gt':'#ffffff',
    'vo_cnn':'#64b5f6',
    'io':'#ffa726',
    'vio_concat':'#ce93d8',
    'vo_resnet':'#4dd0e1',
    'vio_film_2stage':'#66bb6a',
}
LABELS = {
    'vo_cnn':'DeepVO (CNN)',
    'io':'DeepIO (LSTM)',
    'vio_concat':'DeepVIO (Concat)',
    'vo_resnet':'DeepVO (ResNet18)',
    'vio_film_2stage':'DeepVIO FiLM (2-stage)',
}
viz_dir = 'visualizations'

def quat_to_R(q):
    x,y,z,w=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

def integrate(rp, rq, s):
    n=len(rp); p=np.zeros((n+1,3)); p[0]=s; R=np.eye(3)
    for i in range(n): p[i+1]=p[i]+R@rp[i]; R=R@quat_to_R(rq[i])
    return p

def predict(model, mt, ds, idx):
    model.eval(); pp,pq=[],[]
    with torch.no_grad():
        for i in idx:
            s=ds[i]; im=s[0].unsqueeze(0); imu=s[1].unsqueeze(0)
            if mt=='visual': p,q=model(im)
            elif mt=='imu': p,q=model(imu)
            else: p,q=model(im,imu)
            pp.append(p.numpy()[0]); pq.append(q.numpy()[0])
    return np.array(pp), np.array(pq)

def get_ranges(data_dir):
    seqs=sorted(os.listdir(data_dir)); ranges=[]; off=0
    for seq in seqs:
        rp=os.path.join(data_dir,seq,'relative_poses.csv')
        if os.path.exists(rp):
            l=len(np.loadtxt(rp,delimiter=',',skiprows=1))
            ranges.append((off,off+l)); off+=l
    return seqs, ranges

# ── Load ALL models ──────────────────────────────────────────
print("Loading all models...")
models = {}

checkpoints = {
    'vo_cnn': ('checkpoints/newdata_visual/visual_best.pt', DeepVO, 'visual', {'use_attention':True}),
    'io': ('checkpoints/newdata_imu/imu_best.pt', DeepIO, 'imu', {'use_attention':True}),
    'vio_concat': ('checkpoints/newdata_combined/combined_best.pt', DeepVIO, 'combined', {'use_attention':True}),
    'vo_resnet': ('checkpoints/v2_resnet_visual/best.pt', DeepVO_V2, 'visual', {'visual_encoder':'resnet'}),
    'vio_film_2stage': ('checkpoints/v2_twostage_film/best.pt', DeepVIO_FiLM, 'combined', {'visual_encoder':'resnet'}),
}

for name, (path, cls, mtype, kwargs) in checkpoints.items():
    if not os.path.exists(path):
        print(f"  SKIP {name}: {path} not found")
        continue
    m = cls(**kwargs)
    ckpt = torch.load(path, map_location='cpu')
    m.load_state_dict(ckpt['model'])
    models[name] = (m, mtype)
    print(f"  Loaded {name}")

# ── Evaluate on test set ─────────────────────────────────────
print("\n=== TEST SET EVALUATION ===")
test_ds = VIODataset('output/test', img_size=(224,224), augment=False)
test_dir = 'output/test'
test_seqs, test_ranges = get_ranges(test_dir)

test_indices = [0, min(5,len(test_seqs)-1), min(10,len(test_seqs)-1)]
test_results = {}

for si in test_indices:
    if si >= len(test_ranges): continue
    s,e = test_ranges[si]; sn=test_seqs[si]
    gt=np.loadtxt(os.path.join(test_dir,sn,'groundtruth.csv'),delimiter=',',skiprows=1)[:,1:4]
    res={'gt':gt,'seq_name':sn}
    print(f"\n{sn}:")
    for mn,(m,mt) in models.items():
        pp,pq=predict(m,mt,test_ds,list(range(s,e)))
        traj=integrate(pp,pq,gt[0]); ml=min(len(traj),len(gt))
        ate=np.sqrt(np.mean(np.sum((traj[:ml]-gt[:ml])**2,axis=1)))
        res[mn]=traj; res[f'{mn}_ate']=ate
        print(f"  {LABELS[mn]:30s} ATE = {ate:.2f}m")
    test_results[si]=res

# ── Evaluate on val set (unseen trajectory) ──────────────────
print("\n=== VAL SET (linear - UNSEEN trajectory) ===")
val_ds = VIODataset('output/val', img_size=(224,224), augment=False)
val_dir = 'output/val'
val_seqs, val_ranges = get_ranges(val_dir)

val_indices = [0, min(5,len(val_seqs)-1), min(10,len(val_seqs)-1)]
val_results = {}

for si in val_indices:
    if si >= len(val_ranges): continue
    s,e = val_ranges[si]; sn=val_seqs[si]
    gt=np.loadtxt(os.path.join(val_dir,sn,'groundtruth.csv'),delimiter=',',skiprows=1)[:,1:4]
    res={'gt':gt,'seq_name':sn}
    print(f"\n{sn}:")
    for mn,(m,mt) in models.items():
        pp,pq=predict(m,mt,val_ds,list(range(s,e)))
        traj=integrate(pp,pq,gt[0]); ml=min(len(traj),len(gt))
        ate=np.sqrt(np.mean(np.sum((traj[:ml]-gt[:ml])**2,axis=1)))
        res[mn]=traj; res[f'{mn}_ate']=ate
        print(f"  {LABELS[mn]:30s} ATE = {ate:.2f}m")
    val_results[si]=res

# ── FIG 19: ALL models comparison on test ────────────────────
print("\nGenerating figures...")

model_keys = [k for k in ['vo_cnn','io','vio_concat','vo_resnet','vio_film_2stage'] if k in models]

# 3-panel: one per test sequence, all models overlaid
n = len(test_results)
fig, axes = plt.subplots(1, n, figsize=(5.5*n, 5), facecolor=BG)
if n == 1: axes = [axes]
fig.suptitle('All Models: GT vs Predicted Trajectories (Test Set)', color='white', fontsize=15, fontweight='bold')

for ax, (si, res) in zip(axes, test_results.items()):
    ax.set_facecolor(BG)
    gt = res['gt']
    ax.plot(gt[:,0], gt[:,1], '-', color=COLORS['gt'], lw=2.5, label='GT', alpha=0.9)
    for mn in model_keys:
        if mn in res:
            t=res[mn]; ate=res[f'{mn}_ate']
            ax.plot(t[:,0],t[:,1],'-',color=COLORS[mn],lw=1.3,
                    label=f'{LABELS[mn]} ({ate:.1f}m)',alpha=0.8)
    ax.scatter(gt[0,0],gt[0,1],color='#66bb6a',s=80,marker='^',zorder=5)
    ax.scatter(gt[-1,0],gt[-1,1],color='#ef5350',s=80,marker='s',zorder=5)
    ax.set_title(res['seq_name'],color='white',fontsize=11)
    ax.set_xlabel('X (m)',color='#bbbbcc',fontsize=9)
    ax.set_ylabel('Y (m)',color='#bbbbcc',fontsize=9)
    ax.set_aspect('equal')
    ax.legend(fontsize=7,facecolor='#25253d',edgecolor='#444466',labelcolor='white',loc='best')
    ax.tick_params(colors='#888899',labelsize=7)
    ax.grid(True,alpha=0.15,color='#333355')
    for sp in ax.spines.values(): sp.set_color('#333355')

plt.tight_layout()
plt.savefig(os.path.join(viz_dir,'fig19_all_models_test.png'),bbox_inches='tight',facecolor=BG)
plt.close()
print("  Saved fig19_all_models_test.png")

# ── FIG 20: ATE bar chart — ALL models, test + val ──────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
fig.suptitle('ATE Comparison: All Models on Test (seen) vs Val (unseen)', color='white', fontsize=14, fontweight='bold')

for ax, (title, results) in zip([ax1,ax2], [('Test (figure8 - seen type)', test_results), ('Val (linear - unseen type)', val_results)]):
    ax.set_facecolor(BG)
    seq_names = [r['seq_name'].replace('seq_','S') for r in results.values()]
    x = np.arange(len(seq_names))
    n_models = len(model_keys)
    w = 0.8 / n_models
    
    for i, mn in enumerate(model_keys):
        ates = [r.get(f'{mn}_ate', 0) for r in results.values()]
        offset = (i - n_models/2 + 0.5) * w
        bars = ax.bar(x + offset, ates, w, label=LABELS[mn], color=COLORS[mn], alpha=0.85)
        for b, v in zip(bars, ates):
            if v < 50:  # don't label huge values
                ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.2, f'{v:.1f}',
                        ha='center', fontsize=6, color='white')
    
    ax.set_xticks(x); ax.set_xticklabels(seq_names, color='white', fontsize=9)
    ax.set_title(title, color='white', fontsize=12)
    ax.set_ylabel('ATE (m)', color='#bbbbcc')
    ax.tick_params(colors='#888899')
    ax.grid(True, alpha=0.15, axis='y', color='#333355')
    for sp in ax.spines.values(): sp.set_color('#333355')
    ax.legend(fontsize=7, facecolor='#25253d', edgecolor='#444466', labelcolor='white', loc='best')

plt.tight_layout()
plt.savefig(os.path.join(viz_dir,'fig20_all_models_ate.png'),bbox_inches='tight',facecolor=BG)
plt.close()
print("  Saved fig20_all_models_ate.png")

# ── FIG 21: Per-model 3D trajectory (IO standalone) ─────────
# Show DeepIO separately since it's IMU-only
if 'io' in models:
    res = list(test_results.values())[0]
    gt = res['gt']
    fig = plt.figure(figsize=(14, 5), facecolor=BG)
    fig.suptitle('Individual Model Trajectories (Test Sequence)', color='white', fontsize=14, fontweight='bold')
    
    for i, (mn, title) in enumerate([('vo_cnn','DeepVO (CNN)'),('io','DeepIO (IMU-only)'),('vio_film_2stage','DeepVIO FiLM')]):
        ax = fig.add_subplot(1, 3, i+1, facecolor=BG)
        ax.plot(gt[:,0],gt[:,1],'-',color=COLORS['gt'],lw=2.5,label='GT')
        if mn in res:
            t=res[mn]; ate=res[f'{mn}_ate']
            ax.plot(t[:,0],t[:,1],'-',color=COLORS[mn],lw=1.5,label=f'Pred (ATE={ate:.1f}m)')
        ax.scatter(gt[0,0],gt[0,1],color='#66bb6a',s=80,marker='^',zorder=5)
        ax.scatter(gt[-1,0],gt[-1,1],color='#ef5350',s=80,marker='s',zorder=5)
        ax.set_title(title,color=COLORS.get(mn,'white'),fontsize=12,fontweight='bold')
        ax.set_xlabel('X (m)',color='#bbbbcc',fontsize=9)
        ax.set_ylabel('Y (m)',color='#bbbbcc',fontsize=9)
        ax.set_aspect('equal')
        ax.legend(fontsize=9,facecolor='#25253d',edgecolor='#444466',labelcolor='white')
        ax.tick_params(colors='#888899',labelsize=7)
        ax.grid(True,alpha=0.15,color='#333355')
        for sp in ax.spines.values(): sp.set_color('#333355')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir,'fig21_vo_io_vio_individual.png'),bbox_inches='tight',facecolor=BG)
    plt.close()
    print("  Saved fig21_vo_io_vio_individual.png")

# ── Print summary table ──────────────────────────────────────
print("\n" + "="*70)
print("COMPLETE ATE RESULTS (meters)")
print("="*70)
print(f"{'Model':35s} {'Test Avg':>10s} {'Val Avg':>10s}")
print("-"*70)
for mn in model_keys:
    test_ates = [r.get(f'{mn}_ate',0) for r in test_results.values()]
    val_ates = [r.get(f'{mn}_ate',0) for r in val_results.values()]
    print(f"{LABELS[mn]:35s} {np.mean(test_ates):>10.2f} {np.mean(val_ates):>10.2f}")
print("="*70)

print("\n=== All done! ===")
