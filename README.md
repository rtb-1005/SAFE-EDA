# SAFE-EDA

PyTorch implementation of SAFE-EDA for artifact-supervised wrist-EDA affect
recognition, reproducing all 20 numerical tables and four manuscript figures.

**Paper:** *Artifact Annotations Substitute for Per-User Calibration: SAFE-EDA
and a Calibration-Controlled Evaluation of Wrist-EDA Affect Recognition*.
Haochen Chai, Xinbi Luo, Zining Liu, and Fangfang Jiang, 2026. Manuscript.

## Layout

```text
configs/model.json   Model and training settings
data/metrics.zip     Result-level inputs (9 CSV files)
data/literature/     Study coding and supporting source quotations
data/results/        Reference manuscript tables (20 CSV files)
src/safe_eda/        Model, training, inference, analysis, and plotting
tests/              Model and reproduction tests
```

## Install

Run from the repository root. The numerical reproduction environment uses
Python 3.14.5; `requirements.txt` pins the CPU-tested macOS dependencies.
This is the table-reconstruction and CPU test environment, not a complete
record of the GPU training environment.

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m safe_eda check
```

`check` verifies 359,739 target parameters, 296,822 transferred parameters,
and source/target output shapes. The console command `safe-eda` provides
the same interface as `python -m safe_eda`.

## Reproduce Tables and Figures

```bash
python -m safe_eda reproduce
python -m unittest discover -s tests -v
```

`reproduce` reads `data/metrics.zip` and `data/literature/study_coding.csv`,
then writes 20 CSV tables to
`outputs/tables/` and Figures 2, 4, 5, and 6 to `outputs/figures/`.
The tests compare all regenerated tables with `data/results/` byte for byte.
Figure 2 shows the architecture; Figures 4-6 use the numerical tables.
Figures 1 and 3 require waveform examples selected from provider datasets and
therefore require separately obtained source data.

The input archive holds aggregate and participant-level metrics, reference
counts, epoch summaries, and window-level correctness/transition-distance
records. It contains no source waveforms or full class-probability arrays.
Source code is organized as follows:

| Module | Purpose |
|---|---|
| `model.py`, `rf.py` | Target classifier and multiscale temporal blocks |
| `decomposition.py`, `ops.py`, `pooling.py` | Fixed signal decomposition and pooling |
| `transfer.py` | Source artifact head and temporal-weight transfer |
| `training.py`, `cli.py` | Training, checkpoint I/O, prediction, and commands |
| `analysis.py` | Deterministic table reconstruction |
| `plotting.py`, `schematics.py` | Numerical and architecture figures |

## Train and Predict

The training interface starts from **prepared, normalized windows** in a local
NPZ file. It does not download datasets, construct participant folds, resample
recordings, or fit normalization statistics. Prepare each training fold according
to the paper and keep held-out subjects separate. For global-train normalization,
fit statistics on training subjects only; per-subject-z uses the complete
unlabeled subject recording. These are distinct evaluation conditions.

| Input | Arrays | Sampling rate |
|---|---|---|
| Source training | `x`: float32 `[N, 1, 256]`; `target`: binary `[N, 256]` | 32 Hz, after preprocessing |
| Target training | `x`: float32 `[N, 1, 240]`; `y`: integer `[N]` in `{0, 1, 2}` | 4 Hz |
| Prediction | `x` with the shape required by the checkpoint | As above |

Target training requires all three classes. Preserve the dataset's label
mapping consistently between training and evaluation. Only the temporal stem
and residual blocks transfer; target pooling and classification layers are
initialized afresh.

```bash
python -m safe_eda train-source \
  --data datasets/source_train.npz --output runs/source.pt --seed 1005
python -m safe_eda train-target \
  --data datasets/target_train.npz --source runs/source.pt \
  --output runs/target.pt --seed 1005
python -m safe_eda predict \
  --data datasets/target_test.npz --checkpoint runs/target.pt \
  --output runs/predictions.npz
```

Omit `--source` to train the scratch arm. Use the same seed for source and
transfer checkpoints. `--config` selects a local JSON configuration, and
`--device` defaults to `cpu`. Prediction writes `artifact_probability` with
shape `[N, 256]` for source checkpoints or `predicted_class` with shape `[N]`
for target checkpoints. Load only trusted checkpoints created by this code.

## Reproduction Scope

**Directly reproducible.** The supplied result-level inputs reproduce 18
experimental tables, Supplementary Tables S1-S2, and Figures 2 and 4-6. The
literature data include all 50 study codes. `source_quotes.csv` has one row for
each reported survey field, with its coded value, supporting source passage,
and section or page location. Some source passages support several fields and
therefore appear in more than one row.

**Requires provider data.** Model training and Figures 1 and 3 require the
original EDABE, WESAD, or PhysioNet recordings. The repository accepts prepared
windows but does not redistribute or download the recordings.

**Excluded data products.** Trained weights, complete class-probability arrays,
and per-window prediction arrays are outside this repository. The compact
participant-level inputs needed for the published statistics are included.

## Source Data

Obtain the following releases directly from their providers. Original
recordings and annotations, trained checkpoints, and full per-window prediction
arrays are not included. Retraining requires separately prepared data;
recomputing the supplied tables and Figures 2, 4-6 does not.
Questions about existing checkpoints and prediction arrays can be sent to
Haochen Chai at chaihc@mails.neu.edu.cn; these files are not inputs to the
table-reconstruction command.

| Dataset | Release | Identifier and access | Terms |
|---|---|---|---|
| Electrodermal Activity artifact correction BEnchmark (EDABE) | v2 | [10.17632/w8fxrg4pv5.2](https://doi.org/10.17632/w8fxrg4pv5.2), [Mendeley Data](https://data.mendeley.com/datasets/w8fxrg4pv5/2) | CC BY 4.0 |
| Wearable Stress and Affect Detection (WESAD) | ICMI 2018 | [Paper DOI](https://doi.org/10.1145/3242969.3242985), [dataset access](https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html) | Scientific, non-commercial use with attribution under provider terms |
| Wearable Device Dataset from Induced Stress and Structured Exercise Sessions | PhysioNet v1.0.0 | [10.13026/zzf8-xv61](https://doi.org/10.13026/zzf8-xv61), [dataset access](https://physionet.org/content/wearable-device-dataset/1.0.0/) | Open Data Commons Attribution License v1.0 |

## Paper Mapping

Files below are in `data/results/`. T-identifiers denote data tables;
Roman numerals denote tables printed in the manuscript.

| Data table | Manuscript location |
|---|---|
| `T01_protocol_ladder.csv` | Table I: evaluation splits |
| `T04_edabe_source_task.csv` | Results: artifact detection |
| `T05_normalization_ladder.csv` | Table V; Figure 4c |
| `T06_hop_ladder.csv` | Table VI; Figure 4d |
| `T07_cross_configuration_stability.csv` | Results: consistency across configurations |
| `T08_label_permutation.csv` | Table IV; Figure 6 |
| `T10_class_level_net_change.csv` | Results: class-level changes |
| `T11_calibration_subject_level.csv` | Results: probability calibration |
| `T12_headline_result.csv` | Abstract; WESAD headline metrics |
| `T13_epoch_curves.csv` | Data supplement: fixed-budget epoch summaries |
| `T14_method_ladder.csv` | Table IV; Figure 6; control comparisons |
| `T15_equivalence.csv` | Data supplement: paired differences |
| `T16_transition_window_axis.csv` | Data supplement: transition-window diagnostics |
| `T17_normalization_hop_cross_grid.csv` | Table II; Figure 4a-b |
| `T18_interaction_test.csv` | Table III |
| `T19_stability_by_normalization.csv` | Results: across-hop ranges and SD ratios |
| `T20_normalization_value_by_hop.csv` | Results: normalization effects by hop |
| `T21_subject_level_grid.csv` | Figure 5 |
| `S01_literature_study_codes.csv` | Supplementary Table S1: study-level Methods codes |
| `S02_literature_reporting_counts.csv` | Supplementary Table S2: reporting counts and intervals |

The two source files for the literature analysis are not printed tables:
`data/literature/study_coding.csv` contains DOI identifiers, field values, and
the flags used to construct S1 and S2; `data/literature/source_quotes.csv`
links every reported field to a verbatim passage and source location.

## Citation and License

Citation metadata is provided in [CITATION.cff](CITATION.cff).
The software release is archived at
[Zenodo DOI 10.5281/zenodo.22920792](https://doi.org/10.5281/zenodo.22920792),
and the source repository is [GitHub](https://github.com/rtb-1005/SAFE-EDA).
The article is an unpublished manuscript; its article DOI will be added after
publication.
Code and documentation: [MIT](LICENSE).
Derived result data and literature coding: [CC BY 4.0](LICENSE-DATA).
Source datasets retain their separate provider terms.
