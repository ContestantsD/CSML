# Cortical Transformer (CSML)

Official code release for "Cortical Surface Morphology Learning Model: An Anatomy-informed Large-scale Pretrained Model for Decoding Cortical Manifolds".

CSML is a self-supervised framework for learning representations of subject-specific 3D white-surface cortical morphology. It combines a **Canonical Surface Generator (CSG)**, which converts each white-matter surface into canonically organized morphology patches, with a **Cortical Transformer (CT)**, which conditions local attention on subject-specific mesh-geodesic distances while retaining global self-attention.

The subdivision connectivity operations build on [SubdivNet](https://github.com/lzhengning/SubdivNet), and the Cortical Transformer extends the patch-based masked-autoencoding backbone of [MeshMAE](https://github.com/liang3588/MeshMAE).

## Repository structure

| Directory | Content |
|---|---|
| `CSG/` | Simplification, subdivision, and two-stage LDDMM deformation pipeline (`CSG/README.md`) |
| `CT/` | Offline feature extraction, pretraining, and downstream task entry points (`CT/README.md`) |
| `three_stage_qc/` | Three-stage surface quality-control audit scripts (`three_stage_qc/README.md`) |
| `subject_lists/` | Pretraining-cohort subject identifier lists (ABCD / AGING / CHCP) documenting the pretraining composition |

Parameter-level documentation for every entry point is provided in the README of the corresponding subdirectory.

## Quick start

Typical usage (per-hemisphere; see `CT/README.md` for full parameter tables):

```bash
# 1. Extract offline patch features from mapped surfaces
python CT/offline_extract_features.py --surface-dir <mapped> --out-dir <feat_lh> \
    --surface-type mapped --hemi LH --vtk-dir <vtk_lh> --ref-obj <template.obj> --ref-vtk <template.vtk>

# 2. Build subject-specific mesh-graph metric distances
python CT/offline_geodesic.py <feat_lh>

# 3. Self-supervised pretraining
python CT/train_pretrain_offline.py --feat-dirs <feat_lh> --save-dir <ckpt_lh>

# 4. Downstream phenotype prediction
python CT/train_phenotype.py --lh-ckpt <ckpt_lh> --rh-ckpt <ckpt_rh> \
    --lh-dir <feat_lh> --rh-dir <feat_rh> --cohort-dir <cohort> --seed 1 \
    --phenotype <name> --results-root <root> --run-name <run>
```

## Requirements

See `CSG/requirements.txt` and `CT/requirements.txt`.

## Pretrained weights

The pretrained cortical surface encoders are released **by application only**; see the "Pretrained weights" section in `CT/README.md`.

## Data availability

The underlying datasets (ABCD, HCP / HCP-Aging, CHCP, ABIDE-I) are governed by their respective data use agreements. Neither the datasets nor their derivatives are redistributed in this repository. Downstream splits are reproducible from the released split-generation code (`CT/build_cohort_splits.py`) with fixed seeds. The group-average template surfaces referenced by the pipeline (the canonical-slot reference `--ref-obj` / `--ref-vtk` of `CT/offline_extract_features.py` and the P_able anchor template `--pable-vtk` of `CSG/datagen_batch.py`) are likewise derivatives of the pretraining cohort and are distributed **by application only**; see the "Templates" section in `CSG/README.md`.

## Citation

Citation information will be added upon publication.
