# UPAR Challenge 2027 — Track 2

ConvNeXt-Tiny baseline for attribute-based pedestrian retrieval.

## Checkpoints and epochs

Training configuration: **8 epochs total**. The checkpoint `best_dev.pt` contains `epoch: 6`, the best model selected on development retrieval performance, not a separately trained 6-epoch run. `last.pt` represents the end-of-training checkpoint at epoch 8. Local full-validation metrics for `best_dev.pt` with uniform NLL were mADM **0.6313169495**, mAP **0.2047432391**, and Rank-1 **0.2383015598** (3,462 queries). These are **not** official hidden-test scores.

## Codabench submission

The archive needs `run.py`, `metadata.yaml`, and `best_dev.pt` at ZIP root. The model weights are about **111 MB** and must be added via **Git LFS**; the GitHub connector cannot transfer the binary checkpoint. Download your `best_dev.pt` from the original Google Drive folder or use the prepared ChatGPT ZIP. Run `git lfs install && git lfs track "*.pt"` before committing the weight.

**Important:** The current `run.py` uses 256x128 resizing and ImageNet normalization, which have NOT been checked against the original training transforms. Verify before relying on leaderboard accuracy. A locally passing smoke test does not establish the official hidden-set score.

## Final ensemble archives (Git LFS)

The finalized 65% V3 + 35% CLEAN Hybrid experiment has two separate archives:

- `UPLOAD_TO_CODABENCH_65_35.zip` — Codabench code-submission archive (~212 MB).
- `UPAR_FINAL_ALL_IN_ONE_2027.zip` — full reproducibility and coursework archive (~425 MB).

**These archives are not yet committed to this repository.** GitHub's ordinary file upload limit is 100 MB, so upload them with Git LFS from a computer where both ZIPs have been downloaded:

```powershell
git clone https://github.com/hamza26051/cv-assignment.git
cd cv-assignment
git lfs install
git lfs track "*.zip"
# Copy both downloaded ZIPs into this repository folder before continuing.
git add .gitattributes UPLOAD_TO_CODABENCH_65_35.zip UPAR_FINAL_ALL_IN_ONE_2027.zip
git commit -m "Add final UPAR ensemble submission and research archives"
git push origin main
```

After pushing, verify that both GitHub file pages show an LFS pointer and downloadable binary, and check Git LFS storage/bandwidth availability. **Submit the original Codabench ZIP**, not a ZIP generated from GitHub's repository source download (that source download may only contain LFS pointer text).
