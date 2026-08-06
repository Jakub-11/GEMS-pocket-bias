"""Evaluate one model's ensemble predictions on every set.

    python evaluate.py --model ood_1nvq
    python evaluate.py --model cleansplit          # scored against all 7 cluster masks

For each pocket community <c> the same three questions are asked on three data
sources, always comparing the identical complexes across models:

    scoring    PDBbind   Pearson R / RMSE on the community's held-out complexes
                         (the OOD split's test list: denied to ood_<c>, trained
                          on by the references — identical set for all three)
               CASF      scoring + ranking power on the full 285 core set
               CASF-OOD  the same, restricted to core-set complexes in <c>
    docking    CASF      Top1/2/3 over 285 targets
               CASF-OOD  the same, restricted to targets in <c>
    screening  CASF      Top1/5/10 + EF1/5/10 over 57 targets
               CASF-OOD  the same, restricted to targets in <c>

An `ood_<c>` model is only evaluated against its own community; the two
reference models are evaluated against all seven, so every OOD number has a
matched in-distribution control.

Writes results/<model>/metrics.json.
"""
import argparse
import csv
import json
import math

from gems import paths
from gems.casf import power


def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def held_out_ids(cluster):
    """The complexes the ood_<cluster> model was denied.

    This is the `test` list of the OOD split (identical across the 5 folds) — the
    community intersected with the CleanSplit training universe. Using it, rather
    than the raw community list, keeps all three models on the *same* complexes:
    the `pdbbind` reference trains on a larger universe and would otherwise be
    scored on extra community members the other two never had.
    """
    f = paths.SPLITS / "ood" / cluster / "train_val_test_split_f0.json"
    with open(f) as fh:
        return frozenset(json.load(fh)["test"])


def pdbbind_metrics(csv_path, keep):
    """Pearson R / RMSE / MAE over the complexes in `keep`."""
    ys, ps = [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["id"].split("_")[0] in keep:
                ys.append(float(row["y_true"]))
                ps.append(float(row["y_pred"]))
    if len(ys) < 2:
        return {"n": len(ys)}
    n = len(ys)
    return {"pearsonr": _pearson(ys, ps),
            "rmse": math.sqrt(sum((a - b) ** 2 for a, b in zip(ys, ps)) / n),
            "mae": sum(abs(a - b) for a, b in zip(ys, ps)) / n,
            "n": n}


def evaluate_model(model):
    d = paths.RESULTS / model
    scoring_dat = d / "scoring.dat"
    out = {"model": model, "n_folds": paths.N_FOLDS, "full": {}, "clusters": {}}

    # ---- full CASF-2016, identical for every model --------------------------
    if scoring_dat.exists():
        out["full"]["scoring"] = power.scoring(str(scoring_dat))
        out["full"]["ranking"] = power.ranking(str(scoring_dat))
    if (d / "docking").is_dir():
        out["full"]["docking"] = power.docking(str(d / "docking"))
    if (d / "screening").is_dir():
        out["full"]["screening"] = power.screening(str(d / "screening"))

    # ---- per pocket community ----------------------------------------------
    clusters = [model[len("ood_"):]] if model.startswith("ood_") else paths.OOD_CLUSTERS
    for c in clusters:
        ids = paths.cluster_ids(c)
        entry = {"held_out": model == f"ood_{c}"}
        if (d / "pdbbind.csv").exists():
            entry["pdbbind"] = pdbbind_metrics(d / "pdbbind.csv", held_out_ids(c))
        if scoring_dat.exists():
            entry["scoring"] = power.scoring(str(scoring_dat), keep=ids)
            entry["ranking"] = power.ranking(str(scoring_dat), keep=ids)
        if (d / "docking").is_dir():
            entry["docking"] = power.docking(str(d / "docking"), keep=ids)
        if (d / "screening").is_dir():
            entry["screening"] = power.screening(str(d / "screening"), keep=ids)
        out["clusters"][c] = entry
        print(f"[evaluate] {model} / {c}: done", flush=True)

    return out


def main():
    ap = argparse.ArgumentParser(description="Run the CASF power tests + PDBbind metrics for one model.")
    ap.add_argument("--model", required=True, choices=paths.ALL_MODELS)
    args = ap.parse_args()

    res = evaluate_model(args.model)
    out = paths.RESULTS / args.model / "metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[evaluate] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
