# UPAR Challenge 2027 — Track 2

ConvNeXt-Tiny baseline for attribute-based pedestrian retrieval.

## Checkpoints and epochs

Training configuration: **8 epochs total**. The checkpoint `best_dev.pt` contains `epoch: 6`, the best model selected on development retrieval performance, not a separately trained 6-epoch run. `last.pt` represents the end-of-training checkpoint at epoch 8. Local full-validation metrics for `best_dev.pt` with uniform NLL were mADM **0.6313169495**, mAP **0.2047432391**, and Rank-1 **0.2383015598** (3,462 queries). These are **not** official hidden-test scores.

## Codabench submission

The archive needs `run.py`, `metadata.yaml`, and `best_dev.pt` at ZIP root. The model weights are about **111 MB** and must be added via **Git LFS**; the GitHub connector cannot transfer the binary checkpoint. Download your `best_dev.pt` from the original Google Drive folder or use the prepared ChatGPT ZIP. Run `git lfs install && git lfs track "*.pt"` before committing the weight.

**Important:** The current `run.py` uses 256x128 resizing and ImageNet normalization, which have NOT been checked against the original training transforms. Verify before relying on leaderboard accuracy. A locally passing smoke test does not establish the official hidden-set score.
