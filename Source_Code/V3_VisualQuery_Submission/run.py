"""UPAR Track 2: eager-mode torch inference with trained visual-query alignment.

All dependencies are in Codabench runtime: torch, torchvision, numpy, Pillow.
"""
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import convnext_tiny
from PIL import Image

HERE = Path(__file__).resolve().parent
MODEL = None
HEAD = None
CFG = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone=convnext_tiny(weights=None)
        self.backbone.classifier[2]=nn.Identity()
        self.attr_head=nn.Sequential(nn.Dropout(.2),nn.Linear(768,40))

    def forward(self, x):
        f=self.backbone(x)
        return self.attr_head(f),f


class Heads(nn.Module):
    def __init__(self, feature_dim=768):
        super().__init__()
        self.image_head=nn.Sequential(nn.LayerNorm(feature_dim),nn.Linear(feature_dim,512),
                                      nn.GELU(),nn.Dropout(.1),nn.Linear(512,256))
        self.query_head=nn.Sequential(nn.Linear(40,256),nn.GELU(),nn.Linear(256,256))

    def image(self,f):
        return F.normalize(self.image_head(f.float()),dim=1)

    def query(self,q):
        return F.normalize(self.query_head(q.float()*2.-1.),dim=1)


def load_model():
    global MODEL,HEAD,CFG
    if MODEL is not None:
        return
    CFG=json.loads((HERE/'config.json').read_text())
    try:
        weight=torch.load(str(HERE/'weights.pt'),map_location='cpu',weights_only=True)
    except TypeError:
        weight=torch.load(str(HERE/'weights.pt'),map_location='cpu')
    MODEL=Baseline()
    MODEL.load_state_dict(weight['backbone_and_attributes'],strict=True)
    MODEL=MODEL.to(DEVICE).eval()
    HEAD=Heads(int(CFG['feature_dim']))
    HEAD.load_state_dict(weight['heads'],strict=True)
    HEAD=HEAD.to(DEVICE).eval()
    print('UPAR V3 model loaded',DEVICE,flush=True)


def _image(path):
    with Image.open(path) as img:
        im=img.convert('RGB')
        im=im.resize((128,256),Image.Resampling.BILINEAR if hasattr(Image,'Resampling') else Image.BILINEAR)
        a=np.asarray(im,dtype=np.float32).transpose(2,0,1)/255.0
    mean=np.array((.485,.456,.406),np.float32)[:,None,None]
    std=np.array((.229,.224,.225),np.float32)[:,None,None]
    return torch.from_numpy((a-mean)/std)


@torch.inference_mode()
def rank_gallery(sample):
    if MODEL is None:
        load_model()
    names=list(sample['attribute_names'])
    mapping={n:i for i,n in enumerate(names)}
    trained=list(CFG['attribute_names'])
    if len(names)!=40 or len(mapping)!=40 or set(names)!=set(trained):
        raise ValueError('Invalid attribute names')
    q=np.asarray(sample['queries'],np.float32)[:,[mapping[name] for name in trained]]
    if q.ndim!=2 or q.shape[1]!=40 or not np.isin(q,[0.,1.]).all():
        raise ValueError('Queries must be binary Qx40')
    gallery=sample['gallery'];p_list=[];z_list=[]
    bs=CFG.get('batch_size',64)
    for start in range(0,len(gallery),bs):
        x=torch.stack([_image(g['image_path']) for g in gallery[start:start+bs]]).to(DEVICE)
        with torch.no_grad():
            logits,f=MODEL(x)
            p_list.append(torch.sigmoid(logits).cpu())
            z_list.append(HEAD.image(f).cpu())
        if start % max(1024, bs)==0:
            print('Inference',min(start+bs,len(gallery)),'/',len(gallery),flush=True)
    p=torch.cat(p_list).clamp(1e-5,1-1e-5)
    z=torch.cat(z_list)
    query=torch.tensor(q,dtype=torch.float32)
    alpha=float(CFG['alpha'])
    # The official interface requires QxG outputs; calculate them in query chunks
    # to avoid allocating multiple gigabyte-sized matrices simultaneously.
    distances=np.empty((len(q),len(gallery)),dtype=np.float32)
    nll_base=-torch.log1p(-p).sum(1)[None,:]/40.
    nll_delta=(torch.log1p(-p)-torch.log(p)).T/40.
    for start in range(0,len(q),32):
        qb=query[start:start+32]
        d=nll_base+qb@nll_delta
        if alpha<1:
            with torch.inference_mode():
                qz=HEAD.query(qb.to(DEVICE)).cpu()
            d=alpha*d+(1-alpha)*.5*(1.-qz@z.T)
        if not bool(torch.isfinite(d).all()):
            raise ValueError('Non-finite distances')
        distances[start:start+len(qb)]=d.numpy()
    if distances.shape!=(len(q),len(gallery)):
        raise ValueError('Wrong QxG output shape')
    return {'distances':distances}
