# Final comparison — 5-model ensembles

**Models** (all use identical architecture and hyperparameters; only the
training set differs):

| model | trained on the community? |
|---|---|
| `ood` | **no** — the community was removed from training |
| `ref_cleansplit` | yes — PDBbind CleanSplit CV reference |
| `ref_pdbbind` | yes — PDBbind original-split CV reference |

**Evaluation sets**:

| set | what it is | size |
|---|---|---|
| `heldout` | the PDBbind complexes of that community — exactly what the `ood` model was denied | 171–2492 complexes |
| `casf_ood` | the CASF-2016 entries belonging to that community | 4–50 scoring/docking, 1–10 screening targets |
| `casf_all` | the whole CASF-2016 benchmark | 282 scoring, 285 docking, 57 screening |

Within a community every model is scored on the *identical* complexes.

## 1. Scoring — held-out PDBbind complexes (primary signal)

Pearson R. `gap` = best reference − ood; positive means the exclusion cost accuracy.

| community | n | ood | ref_cleansplit | ref_pdbbind | gap |
|---|---|---|---|---|---|
| 1nvq | 2492 | 0.630 | 0.794 | 0.780 | +0.164 |
| 1sqa | 643 | 0.714 | 0.850 | 0.841 | +0.136 |
| 2p15 | 426 | 0.379 | 0.771 | 0.746 | +0.392 |
| 2vw5 | 171 | 0.605 | 0.726 | 0.712 | +0.121 |
| 3dd0 | 410 | 0.334 | 0.762 | 0.730 | +0.428 |
| 3f3e | 363 | 0.420 | 0.759 | 0.736 | +0.339 |
| 3o9i | 310 | 0.386 | 0.720 | 0.697 | +0.334 |

**Mean gap: +0.273 Pearson R** across the 7 communities.

## 2. Scoring — CASF-2016

Left: the community's own CASF entries (`casf_ood`, small n).
Right: the whole benchmark (`casf_all`, 282 complexes) — one value per model.

| community | casf_ood n | ood | ref_cleansplit | ref_pdbbind | ood casf_all |
|---|---|---|---|---|---|
| 1nvq | 50 | 0.667 | 0.791 | 0.810 | 0.782 |
| 1sqa | 24 | 0.902 | 0.896 | 0.895 | 0.798 |
| 2p15 | 15 | 0.605 | 0.694 | 0.734 | 0.781 |
| 2vw5 | 10 | 0.845 | 0.653 | 0.628 | 0.797 |
| 3dd0 | 4 | 0.991 | 0.624 | 0.955 | 0.805 |
| 3f3e | 5 | -0.120 | -0.301 | 0.680 | 0.786 |
| 3o9i | 5 | 0.676 | 0.793 | 0.863 | 0.798 |

`casf_all` references: ref_cleansplit R = 0.791, ref_pdbbind R = 0.809

## 3. Docking power

Top1 success rate (%) on the community's CASF targets, then Top3.

| community | n targets | ood Top1 | ref_clean Top1 | ref_pdbb Top1 | ood Top3 | ref_clean Top3 | ref_pdbb Top3 |
|---|---|---|---|---|---|---|---|
| 1nvq | 50 | 10.0 | 16.0 | 18.0 | 44.0 | 44.0 | 50.0 |
| 1sqa | 25 | 20.0 | 12.0 | 24.0 | 40.0 | 52.0 | 44.0 |
| 2p15 | 15 | 13.3 | 26.7 | 26.7 | 40.0 | 46.7 | 40.0 |
| 2vw5 | 10 | 10.0 | 10.0 | 30.0 | 60.0 | 20.0 | 30.0 |
| 3dd0 | 4 | 25.0 | 50.0 | 50.0 | 100.0 | 100.0 | 100.0 |
| 3f3e | 5 | 80.0 | 100.0 | 60.0 | 80.0 | 100.0 | 100.0 |
| 3o9i | 5 | 0.0 | 20.0 | 20.0 | 60.0 | 40.0 | 40.0 |

Whole benchmark (285 targets):

| model | Top1 % | Top2 % | Top3 % |
|---|---|---|---|
| cleansplit | 23.2 | 37.5 | 47.0 |
| pdbbind | 23.5 | 36.5 | 42.1 |
| ood_1nvq | 18.9 | 37.2 | 45.6 |
| ood_1sqa | 21.4 | 36.1 | 43.2 |
| ood_2p15 | 21.4 | 35.8 | 46.0 |
| ood_2vw5 | 21.4 | 38.9 | 46.3 |
| ood_3dd0 | 20.7 | 35.1 | 44.6 |
| ood_3f3e | 20.0 | 35.8 | 45.6 |
| ood_3o9i | 22.1 | 37.2 | 45.3 |

## 4. Forward screening power

EF1 on the community's CASF targets, then Top1 success rate (%).

| community | n targets | ood EF1 | ref_clean EF1 | ref_pdbb EF1 | ood Top1 | ref_clean Top1 | ref_pdbb Top1 |
|---|---|---|---|---|---|---|---|
| 1nvq | 10 | 1.25 | 2.78 | 2.78 | 10.0 | 10.0 | 10.0 |
| 1sqa | 5 | 4.00 | 0.00 | 3.33 | 0.0 | 0.0 | 20.0 |
| 2p15 | 3 | 3.70 | 3.70 | 3.70 | 33.3 | 33.3 | 33.3 |
| 2vw5 | 2 | 7.14 | 7.14 | 0.00 | 50.0 | 0.0 | 0.0 |
| 3dd0 | 1 | 16.67 | 16.67 | 16.67 | 100.0 | 100.0 | 100.0 |
| 3f3e | 1 | 20.00 | 20.00 | 20.00 | 0.0 | 0.0 | 0.0 |
| 3o9i | 1 | 14.29 | 14.29 | 14.29 | 0.0 | 100.0 | 100.0 |

Whole benchmark (57 targets):

| model | EF1 | EF5 | EF10 | Top1 % | Top10 % |
|---|---|---|---|---|---|
| cleansplit | 2.08 | 1.27 | 1.28 | 7.0 | 29.8 |
| pdbbind | 1.98 | 1.11 | 1.29 | 8.8 | 28.1 |
| ood_1nvq | 0.78 | 1.46 | 1.10 | 5.3 | 26.3 |
| ood_1sqa | 1.30 | 1.14 | 1.08 | 3.5 | 29.8 |
| ood_2p15 | 1.69 | 1.16 | 1.21 | 7.0 | 31.6 |
| ood_2vw5 | 1.54 | 1.34 | 1.29 | 7.0 | 31.6 |
| ood_3dd0 | 1.07 | 1.33 | 1.31 | 7.0 | 29.8 |
| ood_3f3e | 1.51 | 1.24 | 1.27 | 7.0 | 29.8 |
| ood_3o9i | 1.35 | 1.23 | 1.17 | 5.3 | 31.6 |

## 5. Per-community detail

All columns, as in `comparison.csv`. `*_n` are sample sizes — the
`casf_ood` and screening subsets are small; read them before the rates.

| community | role | model | heldout_n | heldout_R | heldout_rmse | casf_ood_n | casf_ood_R | dock_ood_n | dock_ood_top1 | screen_ood_n | screen_ood_ef1 | casf_all_R | dock_all_top1 | screen_all_ef1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1nvq | ood | ood_1nvq | 2492 | 0.6298 | 1.176 | 50 | 0.667 | 50 | 10 | 10 | 1.25 | 0.782 | 18.9 | 0.78 |
| 1nvq | ref_cleansplit | cleansplit | 2492 | 0.7941 | 0.8892 | 50 | 0.791 | 50 | 16 | 10 | 2.78 | 0.791 | 23.2 | 2.08 |
| 1nvq | ref_pdbbind | pdbbind | 2492 | 0.7805 | 0.9177 | 50 | 0.81 | 50 | 18 | 10 | 2.78 | 0.809 | 23.5 | 1.98 |
| 1sqa | ood | ood_1sqa | 643 | 0.7141 | 1.416 | 24 | 0.902 | 25 | 20 | 5 | 4 | 0.798 | 21.4 | 1.3 |
| 1sqa | ref_cleansplit | cleansplit | 643 | 0.85 | 1.062 | 24 | 0.896 | 25 | 12 | 5 | 0 | 0.791 | 23.2 | 2.08 |
| 1sqa | ref_pdbbind | pdbbind | 643 | 0.8414 | 1.095 | 24 | 0.895 | 25 | 24 | 5 | 3.33 | 0.809 | 23.5 | 1.98 |
| 2p15 | ood | ood_2p15 | 426 | 0.3793 | 1.504 | 15 | 0.605 | 15 | 13.3 | 3 | 3.7 | 0.781 | 21.4 | 1.69 |
| 2p15 | ref_cleansplit | cleansplit | 426 | 0.7706 | 0.961 | 15 | 0.694 | 15 | 26.7 | 3 | 3.7 | 0.791 | 23.2 | 2.08 |
| 2p15 | ref_pdbbind | pdbbind | 426 | 0.7458 | 0.9737 | 15 | 0.734 | 15 | 26.7 | 3 | 3.7 | 0.809 | 23.5 | 1.98 |
| 2vw5 | ood | ood_2vw5 | 171 | 0.6051 | 1.197 | 10 | 0.845 | 10 | 10 | 2 | 7.14 | 0.797 | 21.4 | 1.54 |
| 2vw5 | ref_cleansplit | cleansplit | 171 | 0.7259 | 1.012 | 10 | 0.653 | 10 | 10 | 2 | 7.14 | 0.791 | 23.2 | 2.08 |
| 2vw5 | ref_pdbbind | pdbbind | 171 | 0.7116 | 1.034 | 10 | 0.628 | 10 | 30 | 2 | 0 | 0.809 | 23.5 | 1.98 |
| 3dd0 | ood | ood_3dd0 | 410 | 0.3336 | 1.852 | 4 | 0.991 | 4 | 25 | 1 | 16.67 | 0.805 | 20.7 | 1.07 |
| 3dd0 | ref_cleansplit | cleansplit | 410 | 0.7618 | 1.135 | 4 | 0.624 | 4 | 50 | 1 | 16.67 | 0.791 | 23.2 | 2.08 |
| 3dd0 | ref_pdbbind | pdbbind | 410 | 0.7303 | 1.185 | 4 | 0.955 | 4 | 50 | 1 | 16.67 | 0.809 | 23.5 | 1.98 |
| 3f3e | ood | ood_3f3e | 363 | 0.4202 | 1.576 | 5 | -0.12 | 5 | 80 | 1 | 20 | 0.786 | 20 | 1.51 |
| 3f3e | ref_cleansplit | cleansplit | 363 | 0.7588 | 1.134 | 5 | -0.301 | 5 | 100 | 1 | 20 | 0.791 | 23.2 | 2.08 |
| 3f3e | ref_pdbbind | pdbbind | 363 | 0.7356 | 1.197 | 5 | 0.68 | 5 | 60 | 1 | 20 | 0.809 | 23.5 | 1.98 |
| 3o9i | ood | ood_3o9i | 310 | 0.3858 | 1.915 | 5 | 0.676 | 5 | 0 | 1 | 14.29 | 0.798 | 22.1 | 1.35 |
| 3o9i | ref_cleansplit | cleansplit | 310 | 0.7202 | 1.227 | 5 | 0.793 | 5 | 20 | 1 | 14.29 | 0.791 | 23.2 | 2.08 |
| 3o9i | ref_pdbbind | pdbbind | 310 | 0.6971 | 1.248 | 5 | 0.863 | 5 | 20 | 1 | 14.29 | 0.809 | 23.5 | 1.98 |
