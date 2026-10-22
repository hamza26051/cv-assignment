"""UPAR 2027 Track 2: genuine two-ConvNeXt fixed weighted retrieval ensemble.

Official ingestion imports run.py and calls load_model(), rank_gallery(sample).
All parameters are trained weights shipped under ./assets; no Internet or labels needed.
Lower returned distances are better matches.
"""
from __future__ import annotations
from pathlib import Path
import json
import os
from typing import Any
import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
ASSETS = HERE / 'assets'
_CONFIG = None
_MODELS = None
_DEVICE = None


def load_model() -> None:
    """Called once by organizer before rank_gallery; lazy loading is also supported."""
    global _MODELS, _CONFIG, _DEVICE
    if _MODELS is not None:
        return
    _CONFIG = json.loads((ASSETS / 'config.json').read_text())
    _DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    names = ['v3_image.ts', 'v3_query.ts', 'hybrid_image.ts', 'hybrid_query.ts']
    # TorchScript removes torchvision as a runtime dependency.
    _MODELS = [torch.jit.load(str(ASSETS / name), map_location=_DEVICE).eval() for name in names]
    print('UPAR 65/35 ensemble loaded on', _DEVICE, flush=True)


def _image(image_path: str) -> torch.Tensor:
    with Image.open(image_path) as f:
        im = f.convert('RGB').resize((_CONFIG['width'], _CONFIG['height']), Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR)
        a = np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0
    mu = np.asarray(_CONFIG['imagenet_mean'], dtype=np.float32)[:, None, None]
    sd = np.asarray(_CONFIG['imagenet_std'], dtype=np.float32)[:, None, None]
    return torch.from_numpy((a - mu) / sd)


def _gallery_vectors(gallery: list[dict[str, Any]]):
    """Per-image logits and embeddings; avoids train/validation image metadata at inference."""
    img_v, _, img_h, _ = _MODELS
    v_logits, v_embeddings, h_probs, h_embeddings = [], [], [], []
    # Avoid CPU/GPU OOM; each model image network operates on the same image batch.
    bs = int(os.environ.get('UPAR_BATCH_SIZE', '48' if _DEVICE.type == 'cuda' else '8'))
    for start in range(0, len(gallery), bs):
        x = torch.stack([_image(item['image_path']) for item in gallery[start:start + bs]]).to(_DEVICE)
        with torch.inference_mode():
            lv, zv = img_v(x)
            lh, zh = img_h(x)
            v_logits.append(lv.detach().cpu().numpy().astype(np.float32))
            v_embeddings.append(zv.detach().cpu().numpy().astype(np.float32))
            h_probs.append(torch.sigmoid(lh).detach().cpu().numpy().astype(np.float32))
            h_embeddings.append(zh.detach().cpu().numpy().astype(np.float32))
        if start % (bs * 100) == 0:
            print(f'UPAR inference: {min(len(gallery),start + bs)}/{len(gallery)} images', flush=True)
    return (np.concatenate(v_logits), np.concatenate(v_embeddings),
            np.concatenate(h_probs), np.concatenate(h_embeddings))


def _zrows(d):
    # Match offline study: population std, normalized independently over full gallery, per query.
    mean = d.mean(axis=1, keepdims=True)
    std = d.std(axis=1, keepdims=True)
    return (d - mean) / np.maximum(std, np.float32(1e-8))


def _score_from_vectors(queries, v_logits, v_z, h_probs, h_z, v_qz, h_qz):
    """Implementation of the fixed, previously validated retrieval ensemble.

    V3 = .5 * Bernoulli-NLL + .5 * scaled cosine distance
    CLEAN = .6 * binary-query L1 expectation + .4 * scaled cosine distance
    output = .65 * zscore(V3) + .35 * zscore(CLEAN)
    """
    Q = np.asarray(queries, dtype=np.float32)
    G = len(v_logits)
    out = np.empty((len(Q), G), dtype=np.float32)
    # float32 shapes: NLL base [G], NLL delta [G,40]
    # Bernoulli NLL from raw logits avoids clipping for extreme predictions.
    v_base = np.logaddexp(0.0, v_logits).mean(axis=1)
    v_delta = v_logits / 40.0
    h_base = h_probs.mean(axis=1)
    h_delta = (1.0 - 2.0 * h_probs) / 40.0
    for s in range(0, len(Q), 16):
        q = Q[s:s + 16]
        v_attr = v_base[None, :] - q @ v_delta.T
        v_emb = np.float32(0.5) * (1.0 - v_qz[s:s+len(q)] @ v_z.T)
        V = np.float32(0.5)*v_attr + np.float32(0.5)*v_emb
        h_attr = h_base[None, :] + q @ h_delta.T
        h_emb = np.float32(0.5) * (1.0 - h_qz[s:s+len(q)] @ h_z.T)
        H = np.float32(0.6)*h_attr + np.float32(0.4)*h_emb
        out[s:s+len(q)] = np.float32(0.65)*_zrows(V) + np.float32(0.35)*_zrows(H)
    if not np.isfinite(out).all():
        raise ValueError('Non-finite retrieval scores')
    return out


@torch.inference_mode()
def rank_gallery(sample: dict[str, Any]) -> dict[str, np.ndarray]:
    if _MODELS is None:
        load_model()
    trained = list(_CONFIG['attribute_names'])
    incoming = list(sample['attribute_names'])
    if len(incoming) != 40 or len(set(incoming)) != 40 or set(incoming) != set(trained):
        raise ValueError('Unexpected attribute names (UPAR 40-attribute vocabulary required)')
    lookup = {n:i for i,n in enumerate(incoming)}
    q = np.asarray(sample['queries'], dtype=np.float32)[:, [lookup[n] for n in trained]]
    if q.ndim != 2 or q.shape[1] != 40 or not np.isin(q, [0.0,1.0]).all():
        raise ValueError('Expected binary query array of shape Qx40')
    gallery = list(sample['gallery'])
    if not gallery or not len(q):
        return {'distances': np.empty((len(q), len(gallery)), dtype=np.float32)}
    v_logits, v_z, h_probs, h_z = _gallery_vectors(gallery)
    vq, hq = _MODELS[1], _MODELS[3]
    q_t = torch.from_numpy(q).to(_DEVICE)
    vq_z = vq(q_t).detach().cpu().numpy().astype(np.float32)
    hq_z = hq(q_t).detach().cpu().numpy().astype(np.float32)
    distances = _score_from_vectors(q,v_logits,v_z,h_probs,h_z,vq_z,hq_z)
    return {'distances': distances}
