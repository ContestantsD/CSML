# CT

## `offline_extract_features.py`

| Parameter | Required | Default | Description |
|---|---|---|---|
| `--surface-dir` | yes | — | surface root containing `{hemi}/` subdirectories (mapped or deformed obj) |
| `--out-dir` | yes | — | output directory |
| `--surface-type` | yes | — | `mapped` or `deformed` |
| `--hemi` | yes | — | `LH` or `RH` |
| `--target` | no | `b1024_d3` | tokenization configuration (n_patches, depth) |
| `--limit` | no | `0` | >0: process only first N subjects (debug) |
| `--sid-list` | no | — | subject ID list file (one base sid per line); only process listed objs |
| `--vtk-dir` | when `--surface-type mapped` | — | vtk directory for the corresponding hemisphere |
| `--name-template` | no | `{sid}.white_MSMAll.32k_{hemi}` | [mapped] filename template (without extension) |
| `--alpha-center` | no | `0.25` | center-cost weight in the canonical slot assignment |
| `--ref-obj` | when `--surface-type mapped` | — | [mapped] template mapped .obj whose patches define the canonical slots (available by application, see `../CSG/README.md` Templates) |
| `--ref-vtk` | when `--surface-type mapped` | — | [mapped] template native .vtk carrying the fs_LR vertex identities (available by application, see `../CSG/README.md` Templates) |
| `--skip-self-check` | no | off | skip the post-extraction self check |
| `--sid-regex` | no | — | [mapped] regex to extract base sid from obj filename (must contain group `sid`) |
| `--mapped-feature-dir` | when `--surface-type deformed` | — | [deformed] mapped feature output dir (`canonical_mapping.npy` / `hop.npy` / `subject_ids.npy`) |
| `--frames` | no | even frames 0–18 | [deformed] comma-separated frame numbers |

## `offline_geodesic.py`

| Parameter | Required | Default | Description |
|---|---|---|---|
| `feat_dir` (positional) | yes | — | offline feature directory (writes `geodesic.npy` next to it) |
| `limit` (positional) | no | `0` | >0: process only first N subjects |
| `--workers` | no | `8` | parallel worker count |
| `--subject-list` | no | — | subject ID list file; restrict to listed subjects |

## `prep_fc.py`

| Parameter | Required | Default | Description |
|---|---|---|---|
| `--subject-ids` | yes | — | offline `subject_ids.npy` defining the subject set |
| `--fc-dir` | yes | — | parcel size and matrix directory, e.g. `100=/path/to/fc_matrices`; repeatable |
| `--out` | yes | — | label output root (writes `OUT/FC<SIZE>/`) |
| `--pattern` | no | `{sid}_REST1_LR_Full.mat` | .mat filename template (`{sid}` substituted) |

## `prep_hcp_phenotype.py`

| Parameter | Required | Default | Description |
|---|---|---|---|
| `csv` (positional) | yes | — | phenotype CSV (one row per subject, `Subject` column) |
| `subject_ids` (positional) | yes | — | offline `subject_ids.npy` defining the subject set |
| `out_dir` (positional) | yes | — | output directory |

## `build_cohort_splits.py`

| Parameter | Required | Default | Description |
|---|---|---|---|
| `--lh-dir` | yes | — | left-hemisphere offline feature directory (`subject_ids.npy`) |
| `--rh-dir` | yes | — | right-hemisphere offline feature directory (`subject_ids.npy`) |
| `--labels-dir` | yes | — | labels directory (`phenotypes.npy` / `phenotype_names.json` / `brain_size_labels.csv` / `subject_ids.npy`) |
| `--out-dir` | yes | — | output directory (`cohort.csv` / `splits.csv` / `audit.json`) |

## `train_pretrain_offline.py`

| Parameter | Required | Default | Description |
|---|---|---|---|
| `--feat-dirs` | yes | — | comma-separated offline npy directories (with `hop.npy`) |
| `--save-dir` | yes | — | checkpoint output directory |
| `--sid-list` | no | — | comma-separated subject-id list files; only listed subjects are used |
| `--epochs` | no | `50` | training epochs |
| `--batch` | no | `16` | batch size |
| `--n-worker` | no | `8` | dataloader workers |
| `--seed` | no | `0` | random seed |
| `--embed-dim` | no | `384` | transformer embedding dimension |
| `--encoder-depth` | no | `6` | encoder block count |
| `--heads` | no | `6` | attention head count |
| `--decoder-depth` | no | `6` | decoder block count |
| `--decoder-dim` | no | `512` | decoder embedding dimension |
| `--decoder-num-heads` | no | `8` | decoder attention head count |
| `--patch-size` | no | `64` | vertices per patch |
| `--num-patches` | no | `0` | number of patch tokens; 0 = auto-infer from first feat dir `data.npy` shape[2] |
| `--channels` | no | `10` | input feature channels |
| `--mask-ratio` | no | `0.5` | masked-token ratio |
| `--feat-weight` | no | `0.2` | feature reconstruction loss weight |
| `--lr` | no | `5e-4` | learning rate |
| `--weight-decay` | no | `0.05` | weight decay |
| `--warmup-ratio` | no | `0.05` | warmup fraction of total steps |
| `--amp` | no | off | mixed precision |
| `--cosine-tmax-epochs` | no | `0` | cosine T_max in epochs (0 = use `--epochs`) |
| `--nan` | no | off | per-dataset NAN normalization (each dir computes its own mu_j/sigma_j) |
| `--dist` | no | `geodesic` | attention distance matrix: `hop` (BFS hops) \| `geodesic` (surface mm, reads `geodesic.npy`) |
| `--path-mode` | no | `dual` | `dual` = dual-path fusion; `local` = region_attn only; `global` = standard_attn only |
| `--init-sigma` | no | `0.0` | bias_generator sigma init (>0 to activate); geodesic default 50 mm, else hop default 2.0 |
| `--frames-per-subj` | no | `0` | deformed frame subsampling: k equally-spaced frames per subject (0 = all, 3 = subsample 3) |
| `--no-ram-cache` | no | off | disable ram_cache (default on: load selected samples into RAM) |
| `--grad-accum` | no | `1` | gradient accumulation steps (>1: micro-batch = `--batch` accumulated N steps) |
| `--save-epochs` | no | — | comma-separated epoch checkpoints (e.g. `50,100,150`) |
| `--resume` | no | — | resume from checkpoint (e.g. `A_last.pkl`); optimizer/sched restart from scratch |

## `train_phenotype.py`

| Parameter | Required | Default | Description |
|---|---|---|---|
| `--lh-ckpt` | yes | — | path to the pretrained left-hemisphere encoder checkpoint |
| `--rh-ckpt` | yes | — | path to the pretrained right-hemisphere encoder checkpoint |
| `--lh-dir` | yes | — | left-hemisphere offline feature directory |
| `--rh-dir` | yes | — | right-hemisphere offline feature directory |
| `--cohort-dir` | yes | — | cohort directory (`cohort.csv` / `splits.csv`) |
| `--seed` | yes | — | split seed (selects the train/val/test partition) |
| `--phenotype` | yes | — | target phenotype (one of the 28 HCP columns) |
| `--results-root` | yes | — | result root directory |
| `--run-name` | yes | — | run name used in the output directory name |
| `--dist` | no | `geodesic` | attention distance of the pretrained encoder (`hop` \| `geodesic`) |
| `--path-mode` | no | `dual` | encoder pathway configuration (must match ckpt) |
| `--nan` | no | off | apply NAN normalization (stats from the split's training subjects); use with NAN pretrained ckpts |
| `--batch` | no | `10` | batch size |
| `--n-worker` | no | `4` | dataloader workers |
| `--lr` | no | `5e-5` | finetune-phase learning rate |
| `--weight-decay` | no | `0.05` | weight decay |
| `--probe-only` | no | off | head-only probe only, no finetune phase |
| `--from-scratch` | no | off | do not load a pretrained ckpt (randomly initialized encoder) |
| `--probe-epochs` | no | `50` | probe phase epochs |
| `--probe-lr` | no | `1e-3` | probe phase learning rate |
| `--probe-patience` | no | `10` | probe phase early-stop patience |
| `--ft-epochs` | no | `45` | finetune phase epochs |
| `--ft-patience` | no | `15` | finetune phase early-stop patience |
| `--min-delta` | no | `1e-6` | minimum val-loss improvement counted as best |
| `--ft-warmup-epochs` | no | `3` | finetune lr warmup epochs |
| `--ft-warmup-start-lr` | no | `1e-6` | finetune warmup starting lr |
| `--smoke` | no | off | smoke test: 3 ep, no ckpt save |
| `--embed-dim` | no | `384` | transformer embedding dimension |
| `--encoder-depth` | no | `6` | encoder block count |
| `--heads` | no | `6` | attention head count |
| `--patch-size` | no | `64` | vertices per patch |
| `--channels` | no | `10` | input feature channels |
| `--drop-path` | no | `0.1` | drop-path rate |

## `train_fc.py`

| Parameter | Required | Default | Description |
|---|---|---|---|
| `--lh-dir` | yes | — | left-hemisphere offline feature directory |
| `--rh-dir` | yes | — | right-hemisphere offline feature directory |
| `--fc-dir` | yes | — | FC label directory (with `fc.npy`, `subject_ids.npy`) |
| `--lh-ckpt` | yes | — | path to the pretrained left-hemisphere encoder checkpoint |
| `--rh-ckpt` | yes | — | path to the pretrained right-hemisphere encoder checkpoint |
| `--splits` | yes | — | frozen cohort splits csv; `--seed` selects the train/val/test partition |
| `--save` | yes | — | output run directory |
| `--batch-size` | no | `16` | batch size |
| `--epochs` | no | `30` | head-only training epochs |
| `--lr` | no | `1e-3` | learning rate |
| `--wd` | no | `1e-4` | weight decay |
| `--seed` | no | `42` | random / partition seed |
| `--cv-folds` | no | `1` | >1 runs N-fold CV (head-only) |
| `--random-encoder` | no | off | skip pretrained weights (random encoder ablation) |
| `--n-queries` | no | `8` | attention pool query count |
| `--path-mode` | no | `dual` | path mode for the encoder Block (must match ckpt) |
| `--dist` | no | `geodesic` | attention distance: `hop` \| `geodesic` (geodesic requires geodesic pretrained ckpt) |
| `--nan` | no | off | apply NAN normalization to feats; use with NAN pretrained ckpt |
| `--eval-only` | no | off | skip training; load existing ckpts and re-run evaluation |

## `train_tfmri_bilateral.py`

| Parameter | Required | Default | Description |
|---|---|---|---|
| `--lh-dir` | yes | — | left-hemisphere offline feature directory |
| `--rh-dir` | yes | — | right-hemisphere offline feature directory |
| `--l-target-prefix` | yes | — | left target file prefix (`<prefix>_targets_z.npy` etc.) |
| `--r-target-prefix` | yes | — | right target file prefix (`<prefix>_targets_z.npy` etc.) |
| `--lh-ckpt` | yes | — | path to the pretrained left-hemisphere encoder checkpoint |
| `--rh-ckpt` | yes | — | path to the pretrained right-hemisphere encoder checkpoint |
| `--splits` | yes | — | frozen cohort splits csv; `--seed` selects the train/val/test partition |
| `--save` | yes | — | output run directory |
| `--cc-l` | no | — | left vertex indices to mask out (e.g. corpus callosum) |
| `--cc-r` | no | — | right vertex indices to mask out (e.g. corpus callosum) |
| `--batch-size` | no | `16` | batch size |
| `--epochs` | no | `20` | epochs |
| `--lr` | no | `1e-3` | learning rate |
| `--wd` | no | `1e-4` | weight decay |
| `--seed` | no | `42` | random / partition seed |
| `--cv-folds` | no | `1` | `1` = paper protocol: frozen split selected by `--splits` + `--seed`; `>1` runs N-fold subject CV |
| `--random-encoder` | no | off | skip pretrained weights (random encoder ablation) |
| `--path-mode` | no | `dual` | ablation: passed to the encoder Block (must match ckpt) |
| `--dist` | no | `geodesic` | ablation: attention distance `hop` \| `geodesic` (geodesic requires geodesic pretrained ckpt) |
| `--nan` | no | off | apply NAN normalization to feats (stats from the split's training subjects) |
| `--exclude-subtasks` | no | — | comma-separated subtask indices to exclude (0-based) |
| `--eval-only` | no | off | skip training; load existing ckpts and re-run evaluation |

## `train_ados.py`

| Parameter | Required | Default | Description |
|---|---|---|---|
| `--ckpt-lh` | yes | — | path to the pretrained left-hemisphere encoder checkpoint |
| `--ckpt-rh` | yes | — | path to the pretrained right-hemisphere encoder checkpoint |
| `--lh-dir` | yes | — | ADOS left-hemisphere offline feature directory |
| `--rh-dir` | yes | — | ADOS right-hemisphere offline feature directory |
| `--label-dir` | yes | — | label directory (`labels.npy` / `subject_ids.npy`) |
| `--out-dir` | yes | — | output directory |
| `--nan` | no | off | apply NAN normalization to encoder inputs (stats from each fold's training subjects) |

## `compute_ig.py`

| Parameter | Required | Default | Description |
|---|---|---|---|
| `--lh-dir` | yes | — | left-hemisphere offline feature directory |
| `--rh-dir` | yes | — | right-hemisphere offline feature directory |
| `--cohort-dir` | yes | — | cohort directory (`cohort.csv` / `splits.csv`) |
| `--out-dir` | yes | — | output directory |
| `--phenotype` | yes | — | phenotype column the checkpoints were trained on |
| `--ckpt-template` | yes | — | checkpoint path template containing `{seed}` |

## `map_ig_to_32k_vtk.py`

| Parameter | Required | Default | Description |
|---|---|---|---|
| `--res-dir` | yes | — | directory containing `ig_mean_LH.npy` / `ig_mean_RH.npy` |
| `--out-dir` | yes | — | output directory |
| `--subject` | yes | — | subject id of the source 32k VTK pair |
| `--src-dir` | yes | — | directory containing the source 32k VTKs |
| `--lh-dir` | yes | — | left-hemisphere offline feature directory (`faces.npy` / `coordinates.npy`) |
| `--rh-dir` | yes | — | right-hemisphere offline feature directory (`faces.npy` / `coordinates.npy`) |
| `--src-pattern` | no | `{subject}_tfMRI_EMOTION_{tag}.vtk` | source VTK filename template; `{subject}` and `{tag}` (L or R) are substituted |

## `compare_csml_meshmae_bootstrap.py`

| Parameter | Required | Default | Description |
|---|---|---|---|
| `--csml-root` | yes | — | root directory of the CSML phenotype run outputs |
| `--csml-run` | yes | — | CSML run name (prefix before `__seed<seed>__`) |
| `--meshmae-root` | yes | — | root directory of the MeshMAE phenotype run outputs |
| `--meshmae-run` | yes | — | MeshMAE run name |
| `--out` | yes | — | output directory for the bootstrap tables |

## Pretrained weights

The pretrained cortical surface encoders (all hemispheres and granularities described in the paper) are released **by application only**. The datasets used for pretraining are governed by data use agreements, so we distribute checkpoints after review rather than through a public link.

To request the weights, send an email to **kigga1998@gmail.com** with the subject line

```
[CSML weights request] <Your name> - <Your institution>
```

and the following information in the body:

```
Name:
Position:
Institution / affiliation:
Country:
Institutional email:
Requested checkpoints (hemisphere / granularity, e.g. LH b1024, RH b2048):
Intended use (describe the research project):
Planned dataset(s):
I confirm that the weights will be used for non-commercial research
purposes only and will not be redistributed or shared with third parties: yes/no
```

Please use an institutional email address. Requests are reviewed by our team; approved applicants receive a download link together with the applicable terms of use. We aim to respond within two weeks.

The BrainUGDL and Yang et al. reimplementations used in our experiments are likewise available upon reasonable request.
