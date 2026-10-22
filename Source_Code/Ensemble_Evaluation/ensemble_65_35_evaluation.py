"""Actual fixed-weight score fusion of saved CLEAN ConvNeXt-Tiny-Hybrid (final alpha=.60 L1) and V3.
Inputs are real saved validation model outputs; no synthetic predictions and no train-time labels in scores.
Report both original V3 cohort and distinct already-defined 500 query cohort; optional all 3462.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['OMP_NUM_THREADS']='4'
import json,time,argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0,'/mnt/data/ensemble_next')
from visual_query.model import RetrievalHeads

ROOT=Path('/mnt/data'); NOW=time.time()
torch.set_num_threads(4)
ca=np.load(ROOT/'validation_cache.npz');Q=np.asarray(ca['queries'],np.float32);ids=np.asarray(ca['image_query_ids'],np.int32)
labels=np.load(ROOT/'val_labels.npy.bin',mmap_mode='r')
assert labels.shape==(33407,40) and len(Q)==3462
assert np.array_equal(ca['gt_labels'],labels)
assert np.array_equal(Q[ids],labels), 'Query and image IDs do not align'
H=np.asarray(ca['probs'],np.float32);hz=np.asarray(ca['image_z'],np.float32);hqz=np.asarray(ca['query_z'],np.float32)
assert np.allclose(np.linalg.norm(hz,axis=1),1,atol=3e-3) and np.allclose(np.linalg.norm(hqz,axis=1),1,atol=3e-3)
assert H.shape==(33407,40) and hz.shape==(33407,256) and hqz.shape==(3462,256)
# Construct the previously saved V3 image and query embedding scores from the trained head.
checkpoint=torch.load(ROOT/'best_retrieval_head.pt',map_location='cpu',weights_only=False)
head=RetrievalHeads(feature_dim=int(checkpoint['feature_dim']),emb_dim=256)
head.load_state_dict(checkpoint['head_state'],strict=True)
head.eval()
features=np.load(ROOT/'val_features.npy.bin',mmap_mode='r')
with torch.inference_mode():
    z=np.concatenate([head.image(torch.from_numpy(np.array(features[s:s+2048]))) .numpy() for s in range(0,len(features),2048)],axis=0)
    qz=head.query(torch.from_numpy(Q)).numpy()
logits=np.load(ROOT/'val_logits.npy.bin',mmap_mode='r')
assert logits.shape==(33407,40)
# V3 Bernoulli NLL scores, no val labels or metadata at inference.
bv=np.logaddexp(0,np.asarray(logits,np.float32)).mean(axis=1)
dv=np.asarray(logits,np.float32)/40
# Hybrid final scoring from notebook CELL 55: .60*(L1 distance normalized over 40) + .40*(.5*(1-cos)).
# L1 to binary query: sum(p)/40 + q @ (1-2p).T /40
bh=H.mean(axis=1);dh=(1-2*H)/40
# Optional prior ensemble configuration from Oct 18 report, for transparent apples-to-apples comparison
hp=np.clip(H,1e-5,1-1e-5)
bhn=-np.log1p(-hp).mean(axis=1);dhn=(np.log1p(-hp)-np.log(hp))/40
counts=np.bincount(ids,minlength=len(Q));N=len(ids)
assert np.all(counts>0)

def zstd(s):
    return (s-s.mean())/max(s.std(),1e-8)

def individual_scores(qi):
    q=Q[qi]
    v_nll=bv-dv@q
    v_emb=.5*(1-z@qz[qi]); v3=.5*v_nll+.5*v_emb
    h_l1=bh+dh@q
    h_emb=.5*(1-hz@hqz[qi]); notebook=.6*h_l1+.4*h_emb
    h_nll=bhn+dhn@q
    previous=.25*h_nll+.75*h_emb
    return v3,notebook,previous

def metric(qi,score):
    order=np.argsort(score,kind='stable');rel=ids[order]==qi
    d=1-(Q!=Q[qi]).sum(axis=1)/40
    avg=d@counts/N
    dn=np.maximum(0,(d-avg)/max(1-avg,1e-12))
    return float((np.cumsum(dn[ids[order]],dtype=np.float64)/np.arange(1,N+1))[rel].mean())

def evaluate(indices,name):
    out={'v3':[], 'clean_hybrid_notebook':[], 'fusion_65v3_35clean_notebook':[], 'prior_65v3_35hybrid_nllcos':[]}
    for i,qi in enumerate(indices,1):
        qi=int(qi)
        v,h,prior=individual_scores(qi)
        vectors=(v,h,.65*zstd(v)+.35*zstd(h),.65*zstd(v)+.35*zstd(prior))
        for key,s in zip(out,vectors):out[key].append(metric(qi,s))
        if i%100==0:print(name,i,len(indices),'elapsed',round(time.time()-NOW,1),'s',flush=True)
    vals={k:np.asarray(v) for k,v in out.items()};means={k:float(a.mean()) for k,a in vals.items()}
    rng=np.random.default_rng(20271018)
    diffs={}
    for k in ['fusion_65v3_35clean_notebook','prior_65v3_35hybrid_nllcos']:
        d=vals[k]-vals['v3']
        ix=rng.integers(0,len(d),(3000,len(d)))
        diffs[k]={'delta_pp':float(100*d.mean()),'ci95_pp':list(map(float,100*np.quantile(d[ix].mean(1),[.025,.975])))}
    print(name,'RESULT',json.dumps(means),flush=True)
    return {'num_queries':len(indices),'query_indices':list(map(int,indices)),'madm':means,'delta_vs_v3':diffs}

p=np.random.default_rng(2027).permutation(len(Q))
# Original cohort used in V3: 192 dev then 600 holdout from the same permutation.
orig=p[192:792]
# Existing previous ensemble selection set of 180, previous independent 500 query cohort.
ind=p[792+180:792+180+500]
assert len(set(orig)&set(ind))==0
report={'experiment':'35% uploaded CLEAN notebook final ConvNeXt-Tiny-Hybrid + 65% V3 using per-query z-score fusion',
 'source_model':'Uploaded Copy of UPAR_Track2_CLEAN_Colab_Runner.ipynb cell 55, selected alpha=.60, mode=l1, visual checkpoint epoch=8',
 'scores':'V3 .5 Bernoulli NLL+.5 cosine distance; CLEAN hybrid .6 L1 attributes+.4 cosine distance, standardize per query over gallery, blend .65/.35',
 'reference_previous':'previous ensemble used hybrid .25 Bernoulli NLL + .75 embedding, different from uploaded notebook final scoring',
 'data':'33407 aligned real validation predictions and embeddings; 3462 real queries',
 'no_retraining_needed':True,
 'original600':evaluate(orig,'ORIGINAL600'),
 'distinct500':evaluate(ind,'DISTINCT500'),
 'limitations':'Both datasets are subsets of the repeatedly used public validation set; official hidden Codabench score/inference-time not measured.'}
if '--all' in sys.argv:
    report['all3462']=evaluate(np.arange(len(Q)),'ALL3462')
out=ROOT/'UPAR_CLEAN_HYBRID_V3_65_35_results.json'
out.write_text(json.dumps(report,indent=2))
print('SAVED',out,flush=True)
