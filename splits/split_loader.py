"""Unified split loader for the OG CV splits and the OOD deleaking splits.

Normalizes both schemas to {train, validation, test, has_test}:
  * OG  (cleansplit / original): PDBbind_{family}_train_val_split_f{fold}.json
      keys {train, validation}                       -> has_test = False
  * OOD (deleaking):             ood/{cluster}/train_val_test_split_f{fold}.json
      keys {train, validation, test}                 -> has_test = True

CLI contract used everywhere: --split_family {cleansplit,original,ood}
                              --ood_cluster {1nvq,1sqa,2p15,2vw5,3dd0,3f3e,3o9i}
                              --fold {0..4}   (ood_cluster required iff family == ood)
"""
import json
import os

OOD_CLUSTERS = ["1nvq", "1sqa", "2p15", "2vw5", "3dd0", "3f3e", "3o9i"]
OG_FAMILIES = ["cleansplit", "original"]

_DEFAULT_SPLITS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)))


def resolve_split_file(split_family, fold, ood_cluster=None, splits_root=None):
    root = splits_root or _DEFAULT_SPLITS_ROOT
    if split_family in OG_FAMILIES:
        return os.path.join(root, "pdbbind", f"PDBbind_{split_family}_train_val_split_f{fold}.json")
    if split_family == "ood":
        if ood_cluster not in OOD_CLUSTERS:
            raise ValueError(f"ood split requires --ood_cluster in {OOD_CLUSTERS}, got {ood_cluster!r}")
        return os.path.join(root, "ood", ood_cluster, f"train_val_test_split_f{fold}.json")
    raise ValueError(f"Unknown split_family {split_family!r}. Choices: {OG_FAMILIES + ['ood']}")


def load_split(split_family, fold, ood_cluster=None, splits_root=None):
    """Return {train, validation, test, has_test, path} with a normalized schema."""
    path = resolve_split_file(split_family, fold, ood_cluster, splits_root)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Split file not found: {path}")
    with open(path) as f:
        d = json.load(f)
    train = d["train"]
    val = d["validation"]
    test = d.get("test")
    return {
        "train": train,
        "validation": val,
        "test": test,
        "has_test": test is not None,
        "path": path,
    }


def enumerate_runs(include_cv=True, include_ood=True):
    """All (split_family, ood_cluster, fold) runs: 2x5 CV + 7x5 OOD = 45 by default."""
    runs = []
    if include_cv:
        for fam in OG_FAMILIES:
            for fold in range(5):
                runs.append((fam, None, fold))
    if include_ood:
        for c in OOD_CLUSTERS:
            for fold in range(5):
                runs.append(("ood", c, fold))
    return runs
