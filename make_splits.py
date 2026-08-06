"""Regenerate the OOD (novel-target) train/validation/test splits.

Recipe — exactly what produced ``splits/ood/<cluster>/train_val_test_split_f{0..4}.json``:

  universe   all complexes present in the training dataset used for the study
             (the PDBbind "CleanSplit" training set as materialised by GEMS:
             16,491 complexes; ``splits/ood/_universe_cleansplit_16491.txt``)
  test       universe INTERSECT <cluster>  — the held-out PLINDER pocket community
  remaining  universe MINUS <cluster>
  folds      StratifiedKFold(n_splits=5, shuffle=True, random_state=42) over
             `remaining`, stratified on round(pK); the same `test` list is
             attached to all five folds

`<cluster>` is a PLINDER ``pocket_lddt__50__community`` membership list — every
PDB entry whose pocket falls in the same 50%-lDDT community as the named CASF-2016
target. Lists live in ``splits/ood_clusters/`` and are produced by
``tools/get_plinder_cluster.py``.

Verified against the shipped splits: the `test` list this script derives is
IDENTICAL to the shipped one for all 7 clusters, and train/validation never
intersect the cluster. Fold *membership* is not bit-reproducible here, because
StratifiedKFold's shuffle depends on the input ordering and on the scikit-learn
version used at the time. **For reproduction, use the shipped JSONs** — this
script exists to document the recipe and to build splits for new clusters.

Examples
--------
    # new cluster, using the shipped universe list + the shipped label dictionary
    python make_splits.py --cluster 4mzm \
        --cluster_file splits/ood_clusters/4mzm_pocket_lddt__50__community_pdb_ids.txt \
        --out_dir splits/ood/4mzm

    # take the universe (and its order) straight from a GEMS .pt dataset
    python make_splits.py --cluster 4mzm \
        --universe_pt $TRAIN_CLEANSPLIT_PT --out_dir splits/ood/4mzm
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gems import paths

DEFAULT_UNIVERSE = str(paths.SPLITS / "ood" / "_universe_cleansplit_16491.txt")
DEFAULT_DATA_DICT = str(paths.SPLITS / "pdbbind" / "PDBbind_data_dict.json")


def load_universe(universe_txt=None, universe_pt=None):
    """Return an ORDERED list of complex ids. Order matters for fold assignment."""
    if universe_pt:
        import torch
        paths.add_gems_repo_to_syspath()   # GEMS pickles reference `Dataset`
        ds = torch.load(universe_pt, weights_only=False)
        graphs = list(ds.input_data.values()) if hasattr(ds, "input_data") else list(ds)
        return [g.id for g in graphs]
    with open(universe_txt or DEFAULT_UNIVERSE) as f:
        return [ln.strip() for ln in f if ln.strip()]


def load_cluster(path):
    with open(path) as f:
        return frozenset(ln.strip() for ln in f if ln.strip())


def load_labels(data_dict_path, ids):
    """pK (log_kd_ki) per complex id, from PDBbind_data_dict.json."""
    with open(data_dict_path) as f:
        dd = json.load(f)
    missing = [i for i in ids if i not in dd]
    if missing:
        raise KeyError(f"{len(missing)} ids missing from {data_dict_path}, e.g. {missing[:5]}")
    return [float(dd[i]["log_kd_ki"]) for i in ids]


def make_splits(universe, cluster, labels, n_folds=5, seed=42):
    from sklearn.model_selection import StratifiedKFold
    import numpy as np

    test = [i for i in universe if i in cluster]
    remaining = [i for i in universe if i not in cluster]
    lab = dict(zip(universe, labels))
    strat = np.array([round(lab[i]) for i in remaining])

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    rem_arr = np.array(remaining)
    folds = []
    for train_idx, val_idx in skf.split(np.zeros(len(remaining)), strat):
        folds.append({
            "train": rem_arr[train_idx].tolist(),
            "validation": rem_arr[val_idx].tolist(),
            "test": list(test),
        })
    return folds


def main():
    ap = argparse.ArgumentParser(description="Regenerate OOD deleaking splits for one cluster.")
    ap.add_argument("--cluster", required=True, help="Cluster name, e.g. 1nvq (the held-out CASF target)")
    ap.add_argument("--cluster_file", default=None,
                    help="PLINDER community id list (default: "
                         "$OOD_CLUSTERS_DIR/<cluster>_pocket_lddt__50__community_pdb_ids.txt)")
    ap.add_argument("--universe_txt", default=None, help=f"Ordered id list (default {DEFAULT_UNIVERSE})")
    ap.add_argument("--universe_pt", default=None, help="Read the universe + its order from a GEMS .pt dataset")
    ap.add_argument("--data_dict", default=DEFAULT_DATA_DICT, help="PDBbind_data_dict.json (affinity labels)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cluster_file = args.cluster_file or os.path.join(
        str(paths.OOD_CLUSTERS_DIR), f"{args.cluster}_pocket_lddt__50__community_pdb_ids.txt")

    universe = load_universe(args.universe_txt, args.universe_pt)
    cluster = load_cluster(cluster_file)
    labels = load_labels(args.data_dict, universe)
    folds = make_splits(universe, cluster, labels, args.n_folds, args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    for f, split in enumerate(folds):
        out = os.path.join(args.out_dir, f"train_val_test_split_f{f}.json")
        with open(out, "w") as fh:
            json.dump(split, fh, indent=2)
        print(f"[make_ood_splits] fold {f}: train={len(split['train'])} "
              f"val={len(split['validation'])} test={len(split['test'])} -> {out}")
    print(f"[make_ood_splits] universe={len(universe)} cluster={len(cluster)} "
          f"held out={len(folds[0]['test'])}")


if __name__ == "__main__":
    main()
