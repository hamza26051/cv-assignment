
from __future__ import annotations
import json, os, sys
from typing import Any
import numpy as np
import torch

HERE=os.path.dirname(os.path.abspath(__file__))
VENDOR=os.path.join(HERE,"vendor")
if os.path.isdir(VENDOR):
    sys.path.insert(0,VENDOR)
from PIL import Image

ASSETS=os.path.join(HERE,"assets")
MODEL=None; QMODEL=None; CFG=None; DEVICE=None

def load_model():
    global MODEL,QMODEL,CFG,DEVICE
    if MODEL is not None: return
    with open(os.path.join(ASSETS,"config.json"),"r") as f: CFG=json.load(f)
    DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL=torch.jit.load(os.path.join(ASSETS,"image_model.ts"),map_location=DEVICE).eval()
    qp=os.path.join(ASSETS,"query_model.ts")
    if os.path.exists(qp): QMODEL=torch.jit.load(qp,map_location=DEVICE).eval()

def _image(path):
    with Image.open(path) as im:
        im=im.convert("RGB")
        rs=Image.Resampling.BILINEAR if hasattr(Image,"Resampling") else Image.BILINEAR
        im=im.resize((CFG["width"],CFG["height"]),rs)
        a=np.asarray(im,dtype=np.float32).transpose(2,0,1)/255.0
    mean=np.asarray(CFG["mean"],np.float32)[:,None,None]
    std=np.asarray(CFG["std"],np.float32)[:,None,None]
    return torch.from_numpy((a-mean)/std)

@torch.inference_mode()
def _predict_gallery(gallery):
    ps=[]; zs=[]; bs=CFG.get("batch_size",64)
    for s in range(0,len(gallery),bs):
        x=torch.stack([_image(g["image_path"]) for g in gallery[s:s+bs]]).to(DEVICE)
        logits,z=MODEL(x)
        ps.append(torch.sigmoid(logits).cpu().numpy())
        if z.shape[1]>0: zs.append(z.cpu().numpy())
    return np.concatenate(ps), (np.concatenate(zs) if zs else None)

def _distance(q,p,w,mode):
    if mode=="l1":
        base=(p*w[None,:]).sum(1)[None,:]
        delta=((1-2*p)*w[None,:]).T
        return (base+q@delta)/float(w.sum())
    p=np.clip(p,1e-5,1-1e-5)
    lp=np.log(p); ln=np.log(1-p)
    base=-(ln*w[None,:]).sum(1)[None,:]
    delta=(w[None,:]*(ln-lp)).T
    return (base+q@delta)/float(w.sum())

def rank_gallery(sample: dict[str,Any]) -> dict[str,Any]:
    if MODEL is None: load_model()
    incoming=list(sample["attribute_names"])
    pos={n:i for i,n in enumerate(incoming)}
    order=[pos[n] for n in CFG["attribute_names"]]
    q=np.asarray(sample["queries"],np.float32)[:,order]
    p,z=_predict_gallery(sample["gallery"])
    if CFG["weights"]=="balanced":
        w=np.asarray(CFG["attribute_weights"],np.float32)
    else:
        w=np.ones(40,np.float32)
    d=_distance(q,p,w,CFG["mode"])
    a=float(CFG["alpha"])
    if a<1:
        if QMODEL is None or z is None: raise RuntimeError("Hybrid config missing embedding model")
        with torch.inference_mode():
            qz=QMODEL(torch.from_numpy(q).to(DEVICE)).cpu().numpy()
        ed=.5*(1-qz@z.T)
        d=a*d+(1-a)*ed
    # Official sample contract: smaller distance = better.
    return {"distances":d.astype(np.float32)}
