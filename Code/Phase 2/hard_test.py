"""Test best models on all trajectory types: train (fig8/spiral/liss), val (linear-unseen), test (fig8)."""
import torch, numpy as np, matplotlib.pyplot as plt, matplotlib.font_manager as fm, os, sys

sys.path.insert(0, '.')
from models_v2 import DeepVO_V2, DeepVIO_FiLM
from dataset import VIODataset

fm._load_fontmanager(try_read_cache=False)
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Roboto'],'font.size':11,'figure.dpi':150})

BG='#1a1a2e'; COLORS={'gt':'#ffffff','resnet':'#4dd0e1','film_twostage':'#66bb6a'}
LABELS={'resnet':'ResNet18 Visual','film_twostage':'FiLM Two-Stage'}
viz_dir='visualizations'

def quat_to_R(q):
    x,y,z,w=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

def integrate(rp, rq, s):
    n=len(rp); p=np.zeros((n+1,3)); p[0]=s; R=np.eye(3)
    for i in range(n): p[i+1]=p[i]+R@rp[i]; R=R@quat_to_R(rq[i])
    return p

def predict(model, mt, ds, idx):
    model.eval(); pp,pq,gp,gq=[],[],[],[]
    with torch.no_grad():
        for i in idx:
            s=ds[i]; im=s[0].unsqueeze(0); imu=s[1].unsqueeze(0)
            if mt=='visual': p,q=model(im)
            else: p,q=model(im,imu)
            pp.append(p.numpy()[0]); pq.append(q.numpy()[0]); gp.append(s[2].numpy()); gq.append(s[3].numpy())
    return np.array(pp),np.array(pq),np.array(gp),np.array(gq)

def get_ranges(data_dir):
    seqs=sorted(os.listdir(data_dir)); lengths=[]; ranges=[]; off=0
    for seq in seqs:
        rp=os.path.join(data_dir,seq,'relative_poses.csv')
        if os.path.exists(rp): l=len(np.loadtxt(rp,delimiter=',',skiprows=1)); lengths.append(l)
    for l in lengths: ranges.append((off,off+l)); off+=l
    return seqs, ranges

def eval_split(name, data_dir, models_dict, sample_indices):
    print(f"\n=== {name} ===")
    ds = VIODataset(data_dir, img_size=(224,224), augment=False)
    seqs, ranges = get_ranges(data_dir)
    results = {}
    for si in sample_indices:
        if si >= len(ranges): continue
        s,e = ranges[si]; sn=seqs[si]
        gt=np.loadtxt(os.path.join(data_dir,sn,'groundtruth.csv'),delimiter=',',skiprows=1)[:,1:4]
        res={'gt':gt,'seq_name':sn}
        print(f"  {sn}:")
        for mn,(m,mt) in models_dict.items():
            pp,pq,_,_=predict(m,mt,ds,list(range(s,e)))
            traj=integrate(pp,pq,gt[0]); ml=min(len(traj),len(gt))
            ate=np.sqrt(np.mean(np.sum((traj[:ml]-gt[:ml])**2,axis=1)))
            res[mn]=traj; res[f'{mn}_ate']=ate
            print(f"    {mn}: ATE = {ate:.2f}m")
        results[si]=res
    return results

# Load models
print("Loading models...")
mdls={}
for n,c,t,p in [('resnet',DeepVO_V2,'visual','checkpoints/v2_resnet_visual/best.pt'),
                 ('film_twostage',DeepVIO_FiLM,'combined','checkpoints/v2_twostage_film/best.pt')]:
    if os.path.exists(p):
        m=c(visual_encoder='resnet'); m.load_state_dict(torch.load(p,map_location='cpu')['model'])
        mdls[n]=(m,t); print(f"  {n} loaded")

# Evaluate all splits
n_train=len(os.listdir('output/train'))
train_res=eval_split('TRAIN (fig8/spiral/lissajous)','output/train',mdls,[0,n_train//3,2*n_train//3])
val_res=eval_split('VAL (linear - UNSEEN trajectory)','output/val',mdls,[0,5,10])
test_res=eval_split('TEST (figure8)','output/test',mdls,[0,5,10])

# FIG 17: ATE comparison across splits
print("\nGenerating figures...")
all_sets=[('Train\n(seen types)',train_res),('Val\n(linear-unseen)',val_res),('Test\n(figure8)',test_res)]

fig,axes=plt.subplots(1,3,figsize=(16,5),facecolor=BG)
fig.suptitle('Generalization: Seen vs Unseen Trajectory Types',color='white',fontsize=15,fontweight='bold')
for ax,(sn,res) in zip(axes,all_sets):
    ax.set_facecolor(BG)
    sns=[r['seq_name'].replace('seq_','S') for r in res.values()]
    x=np.arange(len(sns)); w=0.35
    for i,mn in enumerate(['resnet','film_twostage']):
        ates=[r.get(f'{mn}_ate',0) for r in res.values()]
        bars=ax.bar(x+i*w-w/2,ates,w,label=LABELS.get(mn),color=COLORS.get(mn),alpha=0.85)
        for b,v in zip(bars,ates): ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.1,f'{v:.1f}',ha='center',fontsize=8,color='white')
    ax.set_xticks(x); ax.set_xticklabels(sns,color='white',fontsize=9)
    ax.set_title(sn,color='white',fontsize=12); ax.set_ylabel('ATE (m)',color='#bbbbcc')
    ax.tick_params(colors='#888899'); ax.grid(True,alpha=0.15,axis='y',color='#333355')
    for sp in ax.spines.values(): sp.set_color('#333355')
    if ax==axes[0]: ax.legend(fontsize=9,facecolor='#25253d',edgecolor='#444466',labelcolor='white')
plt.tight_layout()
plt.savefig(os.path.join(viz_dir,'fig17_cross_trajectory_ate.png'),bbox_inches='tight',facecolor=BG); plt.close()
print("  Saved fig17_cross_trajectory_ate.png")

# FIG 18: Trajectory paths across splits
fig,axes=plt.subplots(1,3,figsize=(16,5),facecolor=BG)
fig.suptitle('Best Models on Different Trajectory Types',color='white',fontsize=15,fontweight='bold')
for ax,(sn,res) in zip(axes,all_sets):
    ax.set_facecolor(BG)
    r=list(res.values())[0]; gt=r['gt']
    ax.plot(gt[:,0],gt[:,1],'-',color=COLORS['gt'],lw=2.5,label='Ground Truth')
    for mn in ['resnet','film_twostage']:
        if mn in r:
            t=r[mn]; ate=r[f'{mn}_ate']
            ls='-' if mn=='resnet' else '--'
            ax.plot(t[:,0],t[:,1],ls,color=COLORS[mn],lw=1.5,label=f'{LABELS[mn]} ({ate:.1f}m)')
    ax.scatter(gt[0,0],gt[0,1],color='#66bb6a',s=80,marker='^',zorder=5)
    ax.scatter(gt[-1,0],gt[-1,1],color='#ef5350',s=80,marker='s',zorder=5)
    ax.set_title(sn,color='white',fontsize=12); ax.set_aspect('equal')
    ax.set_xlabel('X (m)',color='#bbbbcc',fontsize=9); ax.set_ylabel('Y (m)',color='#bbbbcc',fontsize=9)
    ax.legend(fontsize=8,facecolor='#25253d',edgecolor='#444466',labelcolor='white')
    ax.tick_params(colors='#888899',labelsize=7); ax.grid(True,alpha=0.15,color='#333355')
    for sp in ax.spines.values(): sp.set_color('#333355')
plt.tight_layout()
plt.savefig(os.path.join(viz_dir,'fig18_cross_trajectory_paths.png'),bbox_inches='tight',facecolor=BG); plt.close()
print("  Saved fig18_cross_trajectory_paths.png")

print("\n=== All done! ===")
