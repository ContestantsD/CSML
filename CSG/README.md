# CSG

## `datagen_batch.py`

| Parameter | Required | Default | Description |
|---|---|---|---|
| `--src-root` | yes | — | source root containing the original VTK meshes |
| `--dst-root` | yes | — | output root for the mapped OBJ meshes |
| `--pable-vtk` | yes | — | P_able template VTK providing the anchor points (distributed by application; see [Templates](#templates)) |
| `--reject-log-path` | yes | — | log file listing VTKs rejected by the inf/nan ASCII precheck |
| `--failure-log-path` | yes | — | TSV log of failed meshes (source, output, elapsed, reason) |
| `--log-path` | yes | — | log file listing the output paths of failed meshes |
| `--base-size` | no | `1024` | target base mesh size before subdivision |
| `--max-base-size` | no | `1024` | reject meshes whose actual base mesh exceeds this size |
| `--depth` | no | `3` | subdivision depth (patch size = `4**depth`) |
| `--src-pattern` | no | `*.vtk` | source file glob under `--src-root` |
| `--src-recursive` | no | off | search `--src-root` recursively |
| `--preserve-source-tree` | no | auto | mirror the source directory tree in `--dst-root` (auto: on iff `--src-recursive`) |
| `--n-variation` | no | `1` | number of mapping variations per source mesh |
| `--n-worker` | no | `8` | worker process count (<=0 runs in-process) |
| `--worker-maxtasksperchild` | no | `1` | Pool maxtasksperchild (<=0 disables restarts) |
| `--timeout` | no | `10000` | per-mesh MAPS timeout in seconds |
| `--hard-timeout` | no | `14400` | hard per-job timeout in seconds |
| `--verbose` | no | off | print worker tracebacks |
| `--reject-nonfinite-vtk` | no | on | precheck VTKs for inf/nan ASCII tokens and skip offenders (`--no-reject-nonfinite-vtk` to disable) |
| `--append-failure-logs` | no | on | append to (vs recreate) the failure/reject logs |
| `--pable-scalar` | no | `P_able_1200` | point-data array name in `--pable-vtk` marking the anchor points |
| `--pable-ids-path` | no | — | optional text file of anchor vertex ids (overrides `--pable-vtk`) |
| `--pable-ids-out` | no | — | optional output file for the resolved anchor ids |
| `--anatomy-weight` | no | `1.0` | anatomy term weight in the MAPS objective |
| `--protect-pable` | no | off | protect anchor points during simplification |
| `--soft-last-pable` | no | on | soft anchor protection at the last simplification level |

## `two_stage_deform.py`

| Parameter | Required | Default | Description |
|---|---|---|---|
| `--source` | yes | — | source mesh (.obj/.vtk) — the mapped surface |
| `--target` | yes | — | target mesh (.vtk) — the original surface |
| `--out` | yes | — | output dir |
| `--driver` | yes | — | path to `run_deformetrica.py` |
| `--deform-py` | no | `python` | python interpreter used to run the deformetrica driver |

## `vtk_obj_convert.py`

| Parameter | Required | Default | Description |
|---|---|---|---|
| `input` (positional) | one of the two modes | — | single `.vtk` or `.obj` file |
| `output` (positional) | with `input` | — | single-file output path |
| `--batch` | with `--out-dir` | — | deformation output directory (single case, or root of per-subject case directories) |
| `--subject` | no | — | subject id used in batch output file names (single-case `--batch` only) |
| `--out-dir` | with `--batch` | — | batch output directory |
| `--final-frame` | no | `-1` | frame number assigned to `final.vtk` (default: one past the last `tp_XX` index) |

## Templates

The group-average template surfaces used by this pipeline are not included in this repository. They are group averages of the cortical surfaces of the 12,000 pretraining subjects, and because the underlying datasets are governed by data use agreements, they are distributed **by application only**:

- the `--pable-vtk` anchor templates of `datagen_batch.py` (`group_average_{LH,RH}.P_able_sulc_curv.vtk`, carrying the `P_able_1200` scalar);
- the canonical-slot reference templates (`--ref-obj` / `--ref-vtk`) of `CT/offline_extract_features.py`.

To request the templates, send an email to **kigga1998@gmail.com** with the subject line

```
[CSML template request] <Your name> - <Your institution>
```

and the following information in the body:

```
Name:
Position:
Institution / affiliation:
Country:
Institutional email:
Requested templates (LH / RH / both; anchor and/or canonical-slot):
Intended use (describe the research project):
Planned dataset(s):
I confirm that the templates will be used for non-commercial research
purposes only and will not be redistributed or shared with third parties: yes/no
```

Please use an institutional email address. Requests are reviewed by our team; approved applicants receive a download link together with the applicable terms of use. We aim to respond within two weeks.
