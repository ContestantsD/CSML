# Three-Stage Surface QC

## `stage1_vtk_readability.py`

| Parameter | Required | Default | Description |
|---|---|---|---|
| `--dataset` | yes | — | original VTK directory, `NAME=DIR`; repeatable |
| `--out` | yes | — | report output directory |
| `--limit` | no | `0` | per-dataset cap (0 = all) |

## `stage2_mapped_qc.py`

| Parameter | Required | Default | Description |
|---|---|---|---|
| `--dataset` | yes | — | original VTK directory, `NAME=ORI_DIR`; repeatable |
| `--mapped-root` | yes | — | root containing one mapped subdirectory per dataset |
| `--out` | yes | — | report output directory |
| `--subdir` | no | — | mapped subdirectory for a dataset, `NAME=SUBDIR` (default: dataset name); repeatable |
| `--limit` | no | `0` | cap objs checked per dataset (0 = all; completeness still uses full lists) |
| `--workers` | no | `1` | parallel worker count |

## `stage3_deformed_fidelity.py`

| Parameter | Required | Default | Description |
|---|---|---|---|
| `--cases` | yes | — | `aggregate_case_rows.csv` from the deformation run; repeatable |
| `--out` | yes | — | report output directory |
| `--hd95-mm` | no | `3.0` | HD95 flag threshold (mm) |
| `--asd-mm` | no | `0.8` | ASD flag threshold (mm) |
| `--cc-low` | no | `0.3` | cross-correlation flag threshold |

## `geometry_audit.py`

| Parameter | Required | Default | Description |
|---|---|---|---|
| `--dataset` | yes | — | original VTK directory, `NAME=DIR`; repeatable |
| `--out` | yes | — | report output directory |
| `--limit` | no | `0` | per-dataset cap (0 = all) |
| `--workers` | no | `1` | parallel worker count |

## `geometry_audit_mapped.py`

| Parameter | Required | Default | Description |
|---|---|---|---|
| `--dataset` | yes | — | mapped dataset directory containing `LH/` and `RH/`, `NAME=DIR`; repeatable |
| `--out` | yes | — | report output directory |
| `--limit` | no | `0` | per-dataset cap (0 = all) |
| `--workers` | no | `1` | parallel worker count |

## `summarize_three_stage.py`

| Parameter | Required | Default | Description |
|---|---|---|---|
| `--stage1-dir` | yes | — | stage 1 output directory (`readability_summary.csv` / `all_readability.csv`) |
| `--stage2-dir` | yes | — | stage 2 output directory (`mapped_qc_summary.csv` / `all_mapped_checks.csv` / `all_completeness_issues.csv`) |
| `--stage3-dir` | yes | — | stage 3 output directory (`deformed_summary.csv` / `deformed_case_flags.csv`) |
| `--out` | yes | — | report output directory |
