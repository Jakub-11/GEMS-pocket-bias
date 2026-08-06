# GEMS pocket-bias

Helper repository for a publication on novel-target generalisation of
protein–ligand scoring functions.

**This repo only holds what is specific to that study**: the out-of-distribution
split definitions, the scripts that train/score/compile them, and the resulting
numbers. Everything needed to actually run GEMS — installation, data
preparation, graph construction, the model itself — lives in the original
repository and is not duplicated here:

**<https://github.com/camlab-ethz/GEMS>**

## The question

Does a scoring function's benchmark performance survive when an entire
binding-pocket family is removed from training?

We train the GEMS18d affinity model nine ways — two references on the standard
PDBbind cross-validation splits, and seven with one PLINDER
`pocket_lddt__50__community` held out — then score all of them on the same
complexes. Every reported number is an **ensemble of the 5 fold models**.

## Results

`results/comparison.md` (also `.csv`, and `full_metrics.json` for everything).

On the held-out PDBbind complexes, the mean scoring gap between an OOD ensemble
and the best reference is **+0.273 Pearson R** across the seven communities,
while whole-benchmark CASF-2016 scoring R barely moves (0.781–0.805 vs
0.791/0.809 for the references). Docking and screening show a smaller but
consistent penalty on the full benchmark.

## Splits

`splits/` — the CV splits (identical to GEMS's `PDBbind_data/`), the seven OOD
splits, and the PLINDER community membership lists. [SPLITS.md](SPLITS.md)
documents how the OOD splits were built and verifies the recipe against the
shipped files. This is the part of the repo worth reusing.

## Scripts

Four steps. They assume the GEMS datasets are already available (see the GEMS
repo) and are pointed at via environment variables.

```bash
export GEMS_REPO=/path/to/GEMS            # needed to unpickle the GEMS datasets
export CASF_ROOT=/path/to/CASF-2016       # the power tests read CoreSet.dat etc.

python train.py --model ood_1nvq --fold 2      # one model, one fold (9 models x 5 folds)
python predict.py --model ood_1nvq             # ensemble the 5 folds
python evaluate.py --model ood_1nvq            # CASF power tests + PDBbind metrics
python compile_results.py                      # -> results/comparison.{md,csv}
```

Other paths (`TRAIN_CLEANSPLIT_PT`, `TRAIN_PDBBIND_PT`, `CASF_SCORING_PT`,
`CASF_DOCKING_PT`, `CASF_SCREENING_DIR`, `RUNS`, `RESULTS`) default to the
locations used for the study; see `gems/paths.py` and override by exporting them.

The nine models:

| model | trains on |
|---|---|
| `cleansplit` | PDBbind CleanSplit CV |
| `pdbbind` | PDBbind original-split CV |
| `ood_<c>` | CleanSplit minus community `<c>`, which becomes the test set |

`<c>` ∈ `1nvq 1sqa 2p15 2vw5 3dd0 3f3e 3o9i`. All nine use identical
architecture and hyperparameters, so the only difference is the training set.

## Layout

```
train.py predict.py evaluate.py compile_results.py    the pipeline
make_splits.py                                        regenerate OOD splits (provenance)
gems/model.py                                         GEMS18d as a Lightning module
gems/data.py                                          which dataset + split each model uses
gems/paths.py                                         paths from environment variables
gems/casf/                                            the 4 CASF-2016 power scripts + runner
splits/                                               CV splits, OOD splits, community lists
results/                                              the final tables
```

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the vendored GEMS and
CASF-2016 code.
