# Splits

Three families, all shipped in `splits/`.

| Family | Files | Test key |
|---|---|---|
| CleanSplit CV | `splits/pdbbind/PDBbind_cleansplit_train_val_split_f{0..4}.json` | no |
| Original CV | `splits/pdbbind/PDBbind_original_train_val_split_f{0..4}.json` | no |
| OOD | `splits/ood/<c>/train_val_test_split_f{0..4}.json` | **yes** |

The two CV families come from GEMS unchanged (`splits/pdbbind/` is byte-identical
to GEMS's `PDBbind_data/`). CleanSplit is the variant with CASF-similar entries
removed, so CASF numbers are not inflated by leakage.

> Naming: the model called **`pdbbind`** uses the **`original`** split files.

## The OOD splits

For seven CASF-2016 targets we take the PLINDER `pocket_lddt__50__community`
cluster — every PDB entry whose pocket falls in the same 50 %-lDDT community —
and hold the entire community out of training.

| Cluster | Community | Held out of training | CASF core-set ∩ | Screening targets ∩ |
|---|---:|---:|---:|---:|
| `1nvq` | 2813 | 2492 | 50 | 10 |
| `1sqa` | 793 | 643 | 25 | 5 |
| `2p15` | 493 | 426 | 15 | 3 |
| `2vw5` | 223 | 171 | 10 | 2 |
| `3dd0` | 485 | 410 | 4 | 1 |
| `3f3e` | 419 | 363 | 5 | 1 |
| `3o9i` | 476 | 310 | 5 | 1 |

Membership lists: `splits/ood_clusters/<c>_pocket_lddt__50__community_pdb_ids.txt`.

The last two columns are why the PDBbind evaluation matters: the CASF subsets are
tiny, while the held-out PDBbind set is hundreds to thousands of complexes.

## How they were built

```
universe   16,491 complexes — everything in the CleanSplit training dataset
           (splits/ood/_universe_cleansplit_16491.txt)
test       universe ∩ community        (identical across all 5 folds)
remaining  universe − community
folds      StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
           over `remaining`, stratified on round(pK)
```

Verified against the shipped JSONs: the universe is the same 16,491 ids for all
seven clusters; `test == universe ∩ community` exactly; train and validation
never intersect the community; `train ∪ validation ∪ test == universe`.

`make_splits.py` implements this recipe for new clusters:

```bash
python make_splits.py --cluster 4mzm --out_dir splits/ood/4mzm
```

It reproduces the `test` list identically and the same fold sizes, but fold
*membership* is not bit-identical to the shipped files — `StratifiedKFold`
shuffling depends on input ordering and the scikit-learn version. **Use the
shipped JSONs to reproduce published numbers**; use the tool for new clusters.

To add a genuinely new community, first get its membership list from PLINDER:

```python
from plinder.core.scores import query_index
q = query_index(columns=["entry_pdb_id", "pocket_lddt__50__community"],
                filters=[("entry_pdb_id", "==", "4mzm")], splits=["*"])
cid = q["pocket_lddt__50__community"].dropna().unique()[0]
members = query_index(columns=["entry_pdb_id"],
                      filters=[("pocket_lddt__50__community", "==", cid)],
                      splits=["*"])["entry_pdb_id"].drop_duplicates()
members.to_csv("splits/ood_clusters/4mzm_pocket_lddt__50__community_pdb_ids.txt",
               index=False, header=False)
```

The same membership lists drive the evaluation (`evaluate.py` slices CASF and
PDBbind predictions by them), so the training exclusion and the evaluation
subset can never drift apart.
