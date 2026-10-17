"""UPAR Track 2: ConvNeXt-Tiny attribute-based pedestrian retrieval."""
import os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torchvision.models import convnext_tiny
from torchvision import transforms
from PIL import Image

ROOT = Path(__file__).resolve().parent
MODEL = None
CHECKPOINT = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = int(os.environ.get("UPAR_BATCH_SIZE", "64"))
# Verify that resize/normalization match your original validation code.
HEIGHT = int(os.environ.get("UPAR_HEIGHT", "256"))
WIDTH = int(os.environ.get("UPAR_WIDTH", "128"))
PREPROCESS = transforms.Compose([
    transforms.Resize((HEIGHT, WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))
])

class PedestrianConvNeXt(nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()
        self.backbone = convnext_tiny(weights=None)
        self.backbone.classifier[2] = nn.Identity()
        self.attr_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(768, 40))
    def forward(self, x):
        return self.attr_head(self.backbone(x))

def load_model():
    global MODEL, CHECKPOINT
    if MODEL is not None:
        return
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    path = ROOT / "best_dev.pt"
    CHECKPOINT = torch.load(path, map_location="cpu", weights_only=False)
    state = CHECKPOINT["model_state"]
    dropout = float(CHECKPOINT.get("config", {}).get("dropout", 0.2))
    model = PedestrianConvNeXt(dropout=dropout)
    model.load_state_dict(state, strict=True)
    MODEL = model.to(DEVICE).eval()
    print(f"Loaded model epoch {CHECKPOINT.get('epoch')} on {DEVICE}", flush=True)

def _predict_gallery(gallery):
    output = np.empty((len(gallery), 40), dtype=np.float32)
    for start in range(0, len(gallery), BATCH_SIZE):
        chunk = gallery[start:start+BATCH_SIZE]
        batch = []
        for item in chunk:
            with Image.open(item["image_path"]) as im:
                batch.append(PREPROCESS(im.convert("RGB")))
        with torch.inference_mode():
            logits = MODEL(torch.stack(batch).to(DEVICE))
            output[start:start+len(chunk)] = torch.sigmoid(logits).cpu().numpy()
        if start == 0 or (start // BATCH_SIZE + 1) % 20 == 0:
            print(f"Gallery: {start+len(chunk)}/{len(gallery)}", flush=True)
    return output

def rank_gallery(sample: dict) -> dict:
    if MODEL is None:
        load_model()
    names = list(sample["attribute_names"])
    trained_names = list(CHECKPOINT["attribute_names"])
    if len(names) != 40 or len(set(names)) != 40:
        raise ValueError("Expected 40 unique attribute names")
    if set(names) != set(trained_names):
        raise ValueError("Attribute names differ from checkpoint")
    queries = np.asarray(sample["queries"], dtype=np.float32)
    gallery = sample["gallery"]
    if queries.ndim != 2 or queries.shape[1] != 40:
        raise ValueError(f"Expected Qx40 queries; got {queries.shape}")
    if not np.isin(queries, [0, 1]).all():
        raise ValueError("Queries must be binary")
    probs = _predict_gallery(gallery)
    indices = [trained_names.index(name) for name in names]
    probs = np.clip(probs[:, indices].astype(np.float64), 1e-6, 1-1e-6)
    minus_log_p = -np.log(probs).T
    minus_log_not_p = -np.log1p(-probs).T
    distances = queries @ (minus_log_p-minus_log_not_p) + minus_log_not_p.sum(axis=0)[None,:]
    if distances.shape != (len(queries),len(gallery)) or not np.isfinite(distances).all():
        raise ValueError("Invalid distances")
    print(f"Ranked {len(queries)} queries against {len(gallery)} images", flush=True)
    return {"distances": distances.astype(np.float32)}
