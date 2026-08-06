"""Datasets and splits.

One function decides everything about a model: which GEMS dataset it trains on
and which split JSON defines its folds.

    cleansplit   PDBbind CleanSplit CV        B6AEPL_train_cleansplit.pt
    pdbbind      PDBbind original-split CV    B6AEPL_train_pdbbind.pt
    ood_<c>      CleanSplit minus community   B6AEPL_train_cleansplit.pt
                 <c>, which becomes the test set

Graphs carry `y` scaled to [0,1] (pK / 16) and are matched to split entries by
their `.id` attribute.
"""
import json

import torch
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from gems import paths


def model_spec(model, fold):
    """(dataset .pt, split JSON, held-out cluster or None) for one model + fold."""
    if model == "cleansplit":
        return paths.TRAIN_CLEANSPLIT_PT, paths.SPLITS / "pdbbind" / f"PDBbind_cleansplit_train_val_split_f{fold}.json", None
    if model == "pdbbind":
        return paths.TRAIN_PDBBIND_PT, paths.SPLITS / "pdbbind" / f"PDBbind_original_train_val_split_f{fold}.json", None
    if model.startswith("ood_"):
        cluster = model[len("ood_"):]
        if cluster not in paths.OOD_CLUSTERS:
            raise ValueError(f"unknown cluster {cluster!r}; choices {paths.OOD_CLUSTERS}")
        return paths.TRAIN_CLEANSPLIT_PT, paths.SPLITS / "ood" / cluster / f"train_val_test_split_f{fold}.json", cluster
    raise ValueError(f"unknown model {model!r}; choices {paths.ALL_MODELS}")


def load_graphs(pt_path):
    """Load a GEMS .pt dataset as a plain list of PyG Data objects.

    Deliberately uncached: the screening decoys are ~2 GB per target and there
    are 57 of them, so holding them would exhaust memory on a 1-GPU eval node.
    """
    paths.add_gems_repo_to_syspath()          # pickles reference module `Dataset`
    ds = torch.load(str(pt_path), weights_only=False)
    if hasattr(ds, "input_data"):             # GEMS PDBbind_Dataset
        return list(ds.input_data.values())
    return list(ds)


def load_split(split_file):
    with open(split_file) as f:
        d = json.load(f)
    return {"train": d["train"], "validation": d["validation"], "test": d.get("test")}


class SplitData(pl.LightningDataModule):
    def __init__(self, model, fold, batch_size=32, num_workers=4):
        super().__init__()
        self.model, self.fold = model, fold
        self.batch_size, self.num_workers = batch_size, num_workers
        self.train_ds = self.val_ds = None

    def setup(self, stage=None):
        pt, split_file, _ = model_spec(self.model, self.fold)
        graphs = load_graphs(pt)
        split = load_split(split_file)
        train_ids, val_ids = set(split["train"]), set(split["validation"])
        self.train_ds = [g for g in graphs if g.id in train_ids]
        self.val_ds = [g for g in graphs if g.id in val_ids]
        n_test = len(split["test"]) if split["test"] else 0
        print(f"[data] {self.model} f{self.fold}: train={len(self.train_ds)} "
              f"val={len(self.val_ds)} test={n_test} (of {len(graphs)} graphs)", flush=True)
        missing = len(train_ids) - len(self.train_ds)
        if missing:
            print(f"[data] WARNING {missing} train ids not found in the dataset", flush=True)

    def feature_dims(self):
        g = self.train_ds[0]
        return g.x.shape[1], g.edge_attr.shape[1]

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, persistent_workers=self.num_workers > 0,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=max(512, self.batch_size), shuffle=False,
                          num_workers=self.num_workers, persistent_workers=self.num_workers > 0,
                          pin_memory=True)
