"""Ensemble inference: average the 5 fold models of one model type.

    python predict.py --model ood_1nvq
    python predict.py --model cleansplit --only casf_scoring

Loads <RUNS>/<model>_f{0..4}/checkpoints/best.ckpt (5 models), averages their
pK predictions, and writes everything the evaluation needs:

    results/<model>/scoring.dat                 CASF-2016 core set, 285 rows
    results/<model>/docking/{PDB}_score.dat     285 files
    results/<model>/screening/{T}_score.dat      57 files
    results/<model>/pdbbind.csv                 id,y_true,y_pred over the PDBbind set

Every number reported by this project is an ensemble of 5 — there is no
single-fold path.
"""
import argparse
import csv
import glob
import os
import re
import sys

import torch
from torch_geometric.loader import DataLoader

from gems import paths
from gems.casf import dat
from gems.data import load_graphs, model_spec
from gems.model import GATE18d


def find_checkpoint(model, fold):
    """The lowest-val_rmse checkpoint of one fold.

    Handles both layouts: `checkpoints/best.ckpt` written by this repo's
    train.py, and the older `best_epoch_epoch=NNNN.ckpt` (+ best_ckpt.txt).
    """
    run = paths.run_dir(model, fold)
    direct = run / "checkpoints" / "best.ckpt"
    if direct.exists():
        return str(direct)
    pointer = run / "best_ckpt.txt"
    if pointer.exists():
        p = pointer.read_text().strip()
        if p and os.path.exists(p):
            return p
    cands = glob.glob(str(run / "checkpoints" / "best_epoch_*.ckpt"))
    if cands:   # several "best" snapshots: the last one is the best-so-far
        return max(cands, key=lambda p: int(re.search(r"epoch[=_](\d+)", os.path.basename(p)).group(1)))
    raise FileNotFoundError(f"no checkpoint for {model} fold {fold} under {run}")


def load_ensemble(model, device):
    models = []
    for fold in range(paths.N_FOLDS):
        ck = find_checkpoint(model, fold)
        m = GATE18d.load_from_checkpoint(ck, map_location=device).eval().to(device)
        models.append(m)
        print(f"[ensemble] f{fold}: {os.path.basename(ck)}", flush=True)
    return models


@torch.no_grad()
def predict(models, graphs, device, batch_size=512):
    """Mean pK prediction over the ensemble. Returns {graph_id: pK}."""
    out = {}
    for batch in DataLoader(graphs, batch_size=batch_size, shuffle=False):
        batch = batch.to(device)
        preds = torch.stack([m.predict_pk(batch) for m in models]).mean(0).cpu().numpy()
        ids = list(batch.id)
        for i, gid in enumerate(ids):
            out[str(gid)] = float(preds[i])
    return out


def main():
    ap = argparse.ArgumentParser(description="Ensemble inference for one model type.")
    ap.add_argument("--model", required=True, choices=paths.ALL_MODELS)
    ap.add_argument("--only", nargs="*", default=None,
                    choices=["pdbbind", "casf_scoring", "docking", "screening"],
                    help="restrict to some prediction sets (default: all)")
    ap.add_argument("--batch_size", type=int, default=512)
    args = ap.parse_args()
    want = set(args.only) if args.only else {"pdbbind", "casf_scoring", "docking", "screening"}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = paths.RESULTS / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    models = load_ensemble(args.model, device)

    # --- PDBbind: every complex the model's dataset contains -----------------
    if "pdbbind" in want:
        pt, _, _ = model_spec(args.model, 0)
        graphs = load_graphs(pt)
        scores = predict(models, graphs, device, args.batch_size)
        truth = {str(g.id): float(g.y) * models[0].scale_max for g in graphs}
        with open(out_dir / "pdbbind.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "y_true", "y_pred"])
            for gid in sorted(scores):
                w.writerow([gid, f"{truth[gid]:.4f}", f"{scores[gid]:.4f}"])
        print(f"[predict] pdbbind.csv: {len(scores)} complexes", flush=True)

    # --- CASF-2016 core set (scoring + ranking power) ------------------------
    if "casf_scoring" in want:
        scores = predict(models, load_graphs(paths.CASF_SCORING_PT), device, args.batch_size)
        n = dat.write_scoring(scores, out_dir / "scoring.dat")
        print(f"[predict] scoring.dat: {n} complexes", flush=True)

    # --- CASF docking decoys -------------------------------------------------
    if "docking" in want:
        scores = predict(models, load_graphs(paths.CASF_DOCKING_PT), device, args.batch_size)
        targets = dat.write_docking(scores, out_dir / "docking")
        print(f"[predict] docking: {len(targets)} targets", flush=True)

    # --- CASF screening decoys (one .pt per target) --------------------------
    if "screening" in want:
        pts = sorted(glob.glob(os.path.join(paths.CASF_SCREENING_DIR, "*.pt")))
        for i, pt in enumerate(pts, 1):
            target = re.sub(r"_dataset_.*$", "", os.path.splitext(os.path.basename(pt))[0])
            if (out_dir / "screening" / f"{target}_score.dat").exists():
                continue
            scores = predict(models, load_graphs(pt), device, args.batch_size)
            dat.write_screening(target, scores, out_dir / "screening")
            print(f"[predict] screening {i}/{len(pts)}: {target}", flush=True)

    print(f"[predict] {args.model} -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
