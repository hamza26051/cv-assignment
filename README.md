# UPAR Challenge 2027 — Track 2: Attribute-Based Pedestrian Retrieval

## Project Report

[**Project Report (PDF)**](./Project_Report.pdf)

## Project Demo

<!-- Add your demo video link here. -->

## Project Overview

This repository documents our UPAR Challenge 2027 Track 2 project: retrieving pedestrian gallery images from a **40-dimensional binary attribute query**. The query represents visible attributes (age group, upper/lower-body clothing colors, accessories, etc.), not an input photograph. The model assigns distances and ranks all gallery images. Ground-truth retrieval relevance is based on complete attribute patterns.

The project compares three independently trained vision-backbone runs (ResNet-50, ConvNeXt-Tiny baseline, and ConvNeXt-Tiny Hybrid), an additional V3 VisualQuery retrieval head trained on baseline features, and a fixed-weight score ensemble.

**Team:** Emman Abrar (23K-0051), Areeba Hasnain (23K-0059), Hamza Sheikh (23K-0060).

## Dataset

The [UPAR Challenge 2027 organizers' dataset](https://github.com/speckean/UPAR-Challenge-2027) harmonizes **Market1501, PA-100K, and PETA** using 40 binary pedestrian attributes.

| Split | Images | Distinct attribute queries |
|---|---:|---:|
| Train | 97,669 | 7,204 |
| Public validation | 33,407 | 3,462 |

Our local full-validation evaluation ranks the 33,407-image gallery for all 3,462 distinct queries.

## Models and Training

| Model | Role | Key training setup |
|---|---|---|
| **ResNet-50 baseline** | 40-attribute visual classifier | 6 epochs, batch 64, LR 1e-4, weight decay 5e-4 |
| **ConvNeXt-Tiny baseline** | Stronger 40-attribute classifier | 8 epochs, batch 64, domain-balanced training |
| **ConvNeXt-Tiny Hybrid (CLEAN)** | Attribute and metric-learning model | 8 epochs, 256-dimensional embedding, contrastive loss weight 0.3 |
| **V3 VisualQuery** | Image/query alignment head using ConvNeXt baseline features | Best recorded retrieval head at epoch 5; frozen ConvNeXt visual backbone |
| **Final 65/35 Ensemble** | Score-level combination, no joint training | 65% V3 VisualQuery + 35% CLEAN Hybrid |

Three distinct visual-backbone training runs were performed. V3 trains retrieval heads on features of the ConvNeXt baseline, not an additional independent image backbone. Four TorchScript networks in the submission are **two image models and two query heads**, not four image backbones.

### Retrieval scoring

V3 computes 50% Bernoulli negative log-likelihood distance plus 50% scaled cosine embedding distance. CLEAN Hybrid computes 60% expected binary-attribute L1 distance plus 40% scaled cosine embedding distance. Each query's gallery scores are independently standardized by mean and standard deviation, then combined:

```text
final_distance = 0.65 * zscore(V3_distance)
               + 0.35 * zscore(CLEAN_Hybrid_distance)
```

Lower final distances produce higher ranks. This is the actual implementation in `Models/Final_65_35_Ensemble/run.py`.

## Experimental Results

**Local public validation mADM — 3,462 attribute queries, 33,407 gallery images:**

| System | mADM |
|---|---:|
| ResNet-50 baseline | 58.53% |
| ConvNeXt-Tiny baseline | 63.13% |
| V3 VisualQuery | 66.56% |
| CLEAN Hybrid | 67.26% |
| **Final 65/35 ensemble** | **69.63%** |

The same 65/35 ensemble recorded **71.22% mADM on a separate 600-query development cohort** and **67.89% on a distinct 500-query cohort**. Those cohorts are not directly comparable to full-validation results.

**Official Codabench evidence:** submission **1001454** was marked `Finished` with a displayed score of **0.5361** (earlier submission **1000643** showed **0.4826**). These official scores are **not** the local 69.63% mADM measurement; evaluation split/metric differences have not been conclusively resolved. See [Project Report](./Project_Report.pdf) for the screenshot and discussion.

### Failure analysis and limitations

An audit of public validation attribute predictions reported **2.3766 wrong attributes per image** on average at a 0.5 threshold; only **20.18%** of images had all 40 predicted attributes correct. Particularly difficult attributes included bags, normal glasses, backpacks, and lower-body clothing colors. Limitations include domain shift, class imbalance, partial/occluded attributes, public-validation model-selection bias, and reproducibility of the full Codabench runtime. No hidden-test score of 69.63% is claimed.

## Repository Structure

```text
Project_Report.pdf
README.md
Source_Code/
  V3_VisualQuery_Submission/
  Original_ConvNeXt_Hybrid_Submission/
  Ensemble_Evaluation/
  Original_Baseline_Code/
Models/
  ResNet50_Baseline/
  ConvNeXt_Tiny_Baseline/
  ConvNeXt_Tiny_Hybrid/
  V3_VisualQuery/
  Final_65_35_Ensemble/
    run.py
    metadata.yaml
    assets/config.json
Training_Notebooks/
```

The text code, notebooks, and configurations are kept at their Dropbox directory paths. **Large model binaries may be absent from this GitHub checkout until transferred with Git LFS**; consult the verified [complete Dropbox collection](https://www.dropbox.com/scl/fo/e4j1yv6lo7di76v0xg3la/AH1xDgCNOQ0gju2Yfoymbhk?rlkey=0k5eugr8harkmhejy4sntq53j&dl=0) for the original `.pt` checkpoints and four exported `.ts` inference networks.

### Required competition runtime files

```text
Models/Final_65_35_Ensemble/
  run.py
  metadata.yaml
  assets/
    config.json
    v3_image.ts
    v3_query.ts
    hybrid_image.ts
    hybrid_query.ts
```

`run.load_model()` loads the TorchScript files. `run.rank_gallery(sample)` consumes ordered `attribute_names`, a Qx40 binary `queries` array, and a `gallery` with `image_path` entries, returning `{"distances": float32[Q, G]}`. Keep this exact structure, model versions and image preprocessing for reproduction.

**Codabench submission:** the separate `UPLOAD_TO_CODABENCH_65_35.zip` in Dropbox is the self-contained competition submission. Do not submit this full research repository as the competition ZIP.

## Tools

Python, PyTorch, TorchScript, NumPy, Pillow, torchvision, Google Colab, Codabench.

## References

- [UPAR Challenge 2027 Track 2](https://www.codabench.org/competitions/18196/)
- [UPAR Challenge 2027 organizer repository and data guide](https://github.com/speckean/UPAR-Challenge-2027)
- Specker, A., Cormier, M., and Beyerer, J. (2023), *UPAR: Unified Pedestrian Attribute Recognition and Person Retrieval*, WACV.

