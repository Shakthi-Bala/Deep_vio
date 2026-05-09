import os,sys,numpy as np
sys.path.insert(0,'.')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'sans-serif','font.size':11,'figure.dpi':150})
BG='#1a1a2e'
C={'gt':'#ffffff','resnet':'#4dd0e1','film_onestage':'#ce93d8','film_twostage':'#66bb6a'}
L={'resnet':'ResNet18','film_onestage':'FiLM 1-shot','film_twostage':'FiLM 2-stage'}
import torch
from models_v2 import DeepVO_V2,DeepVIO_FiLM
from dataset import VIODataset
viz='visualizations'
def qr(q):
    x,y,z,w=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def integ(rp,rq,s):
    n=len(rp);p=np.zeros((n+1,3));p[0]=s;R=np.eye(3)
    for i in range(n):p[i+1]=p[i]+R@rp[i];R=R@qr(rq[i])
    return p
def pred(m,mt,ds,ix):
    m.eval();pp,pq=[],[]
    with torch.no_grad():
        for i in ix:
            s=ds[i];img=s[0].unsqueeze(0);imu=s[1].unsqueeze(0)
            if mt=='visual':p,q=m(img)
            else:p,q=m(img,imu)
            pp.append(p[0].numpy());pq.append(q[0].numpy())
    return np.array(pp),np.array(pq)
print("Loading...")
ms={}
for nm,cl,mt,pt in [('resnet',lambda:DeepVO_V2(visual_encoder='resnet'),'visual','checkpoints/v2_resnet_visual/best.pt'),('film_onestage',lambda:DeepVIO_FiLM(visual_encoder='resnet'),'combined','checkpoints/v2_film_combined/best.pt'),('film_twostage',lambda:DeepVIO_FiLM(visual_encoder='resnet'),'combined','checkpoints/v2_twostage_film/best.pt')]:
    if os.path.exists(pt):m=cl();m.load_state_dict(torch.load(pt,map_location='cpu',weights_only=False)['model']);ms[nm]=(m,mt);print(f"  {nm}")
ds=VIODataset('output/test',img_size=(224,224));td='output/test';ts=sorted(os.listdir(td))
sr=[];off=0
for s in ts:
    rp=os.path.join(td,s,'relative_poses.csv')
    if os.path.exists(rp):n=len(np.loadtxt(rp,delimiter=',',skiprows=1));sr.append((off,off+n));off+=n
vi=[0,min(4,len(ts)-1),min(8,len(ts)-1)]
ar={}
for si in vi:
    if si>=len(sr):continue
    s,e=sr[si];sn=ts[si];ix=list(range(s,e))
    gt=np.loadtxt(os.path.join(td,sn,'groundtruth.csv'),delimiter=',',skiprows=1)[:,1:4]
    r={'gt':gt,'sn':sn};print(f"{sn}...")
    for nm,(m,mt) in ms.items():
        pp,pq=pred(m,mt,ds,ix);tj=integ(pp,pq,gt[0])
        ml=min(len(tj),len(gt));ate=np.sqrt(np.mean(np.sum((tj[:ml]-gt[:ml])**2,axis=1)))
        r[nm]=tj;r[f'{nm}_ate']=ate;print(f"  {nm}: ATE={ate:.2f}m")
    ar[si]=r
# Plot per-sequence
for si,r in ar.items():
    gt=r['gt'];sn=r['sn']
    fig,axes=plt.subplots(1,3,figsize=(16,5),facecolor=BG)
    for i,(xi,yi,xl,yl) in enumerate([(0,1,'X(m)','Y(m)'),(0,2,'X(m)','Z(m)'),(1,2,'Y(m)','Z(m)')]):
        ax=axes[i];ax.set_facecolor(BG)
        ax.plot(gt[:,xi],gt[:,yi],'-',color=C['gt'],lw=2.5,label='GT')
        for nm in ['resnet','film_onestage','film_twostage']:
            if nm in r:
                t=r[nm];ate=r[f'{nm}_ate']
                ax.plot(t[:,xi],t[:,yi],'-',color=C[nm],lw=1.5,label=f'{L[nm]} ({ate:.1f}m)',alpha=0.85)
        ax.scatter(gt[0,xi],gt[0,yi],color='#66bb6a',s=80,marker='^',zorder=5)
        ax.scatter(gt[-1,xi],gt[-1,yi],color='#ef5350',s=80,marker='s',zorder=5)
        ax.set_xlabel(xl,color='white');ax.set_ylabel(yl,color='white')
        ax.set_title(['Top-Down (XY)','Side (XZ)','Front (YZ)'][i],color='white',fontsize=12)
        ax.legend(fontsize=7,facecolor='#25253d',edgecolor='#444466',labelcolor='white')
        ax.tick_params(colors='#888899');ax.grid(True,alpha=0.15,color='#333355')
        if i==0:ax.set_aspect('equal')
        for sp in ax.spines.values():sp.set_color('#333355')
    fig.suptitle(f'V2 Trajectories -- {sn}',color='white',fontsize=14,fontweight='bold')
    plt.tight_layout();plt.savefig(f'{viz}/fig14_v2_trajectory_{sn}.png',bbox_inches='tight',facecolor=BG);plt.close()
    print(f"  Saved fig14_v2_trajectory_{sn}.png")
# Grid
nk=[k for k in ['resnet','film_onestage','film_twostage'] if k in list(ar.values())[0]]
ns=len(ar)
fig,axes=plt.subplots(ns,len(nk),figsize=(5*len(nk),4*ns),facecolor=BG)
if ns==1:axes=axes.reshape(1,-1)
for row,(si,r) in enumerate(ar.items()):
    gt=r['gt']
    for col,nm in enumerate(nk):
        ax=axes[row,col];ax.set_facecolor(BG)
        ax.plot(gt[:,0],gt[:,1],'-',color='#ffffff',lw=2,label='GT')
        if nm in r:t=r[nm];ate=r[f'{nm}_ate'];ax.plot(t[:,0],t[:,1],'-',color=C[nm],lw=1.5,label=f'ATE={ate:.1f}m')
        ax.scatter(gt[0,0],gt[0,1],color='#66bb6a',s=50,marker='^',zorder=5)
        ax.set_aspect('equal');ax.grid(True,alpha=0.15,color='#333355');ax.tick_params(colors='#888899',labelsize=7)
        ax.legend(fontsize=7,facecolor='#25253d',edgecolor='#444466',labelcolor='white')
        for sp in ax.spines.values():sp.set_color('#333355')
        if row==0:ax.set_title(L[nm],color=C[nm],fontsize=12,fontweight='bold')
        if col==0:ax.set_ylabel(r['sn'],color='white',fontsize=9)
fig.suptitle('V2: GT (white) vs Predicted',color='white',fontsize=14,fontweight='bold',y=1.0)
plt.tight_layout();plt.savefig(f'{viz}/fig15_v2_multi_sequence.png',bbox_inches='tight',facecolor=BG);plt.close()
print("  Saved fig15_v2_multi_sequence.png")
# ATE bars
fig,ax=plt.subplots(figsize=(10,5),facecolor=BG);ax.set_facecolor(BG)
sn2=[r2['sn'] for r2 in ar.values()];x=np.arange(len(sn2));w=0.25
for i,nm in enumerate(nk):
    ates=[r2.get(f'{nm}_ate',0) for r2 in ar.values()]
    bars=ax.bar(x+i*w-w,ates,w,label=L[nm],color=C[nm],alpha=0.85)
    for b,v in zip(bars,ates):ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.1,f'{v:.1f}',ha='center',fontsize=9,color='white')
ax.set_xticks(x);ax.set_xticklabels(sn2,color='white');ax.set_ylabel('ATE (m)',color='white',fontsize=12)
ax.set_title('V2: Absolute Trajectory Error',color='white',fontsize=13)
ax.tick_params(colors='#888899');ax.legend(fontsize=10,facecolor='#25253d',edgecolor='#444466',labelcolor='white')
ax.grid(True,alpha=0.15,axis='y',color='#333355')
for sp in ax.spines.values():sp.set_color('#333355')
plt.tight_layout();plt.savefig(f'{viz}/fig16_v2_ate_comparison.png',bbox_inches='tight',facecolor=BG);plt.close()
print("  Saved fig16_v2_ate_comparison.png")
print("DONE")
