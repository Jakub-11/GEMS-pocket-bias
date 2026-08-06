"""Build the final comparison tables from every model's metrics.json.

    python compile_results.py

Writes into results/:
    comparison.md     headline tables per power + the full per-community breakdown
    comparison.csv    the same numbers, one row per (community, model)
    full_metrics.json every metric of every model in one file

Naming used throughout (models and evaluation sets are deliberately distinct):

  MODELS      ood             trained with the community removed
              ref_cleansplit  reference, PDBbind CleanSplit CV — trained on it
              ref_pdbbind     reference, PDBbind original-split CV — trained on it

  SETS        heldout   the PDBbind complexes of that community (the OOD model's
                        test split) — the largest sample, 171-2492 complexes
              casf_ood  CASF-2016 entries belonging to that community
                        (4-50 scoring/docking targets, 1-10 screening targets)
              casf_all  the whole CASF-2016 benchmark: 282 scoring complexes,
                        285 docking targets, 57 screening targets

Every value is an ensemble of the 5 fold models.
"""
import argparse
import csv
import json

from gems import paths

# role -> the model that plays it for a given community
ROLES = ["ood", "ref_cleansplit", "ref_pdbbind"]


def model_for(role, cluster):
    return {"ood": f"ood_{cluster}", "ref_cleansplit": "cleansplit", "ref_pdbbind": "pdbbind"}[role]


# per-community columns: (csv column, path into the cluster entry)
COLUMNS = [
    ("heldout_n",        ("pdbbind", "n")),
    ("heldout_R",        ("pdbbind", "pearsonr")),
    ("heldout_rmse",     ("pdbbind", "rmse")),
    ("casf_ood_n",       ("scoring", "n")),
    ("casf_ood_R",       ("scoring", "pearsonr")),
    ("casf_ood_sd",      ("scoring", "sd")),
    ("dock_ood_n",       ("docking", "n_targets")),
    ("dock_ood_top1",    ("docking", "top1")),
    ("dock_ood_top3",    ("docking", "top3")),
    ("screen_ood_n",     ("screening", "n_targets")),
    ("screen_ood_top1",  ("screening", "top1")),
    ("screen_ood_ef1",   ("screening", "ef1")),
]
# whole-benchmark columns: identical for every row of a given model
FULL_COLUMNS = [
    ("casf_all_R",        ("scoring", "pearsonr")),
    ("casf_all_sd",       ("scoring", "sd")),
    ("casf_all_spearman", ("ranking", "spearman")),
    ("dock_all_top1",     ("docking", "top1")),
    ("dock_all_top3",     ("docking", "top3")),
    ("screen_all_top1",   ("screening", "top1")),
    ("screen_all_ef1",    ("screening", "ef1")),
]


def dig(d, path):
    for k in path:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def fmt(v, nd=None):
    if v is None:
        return ""
    if isinstance(v, int):
        return str(v)
    return f"{v:.{nd}f}" if nd is not None else f"{v:.4g}"


def _table(head, rows):
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + \
           ["| " + " | ".join(rows_) + " |" for rows_ in rows]


def build_md(rows, metrics):
    by = {(r["community"], r["role"]): r for r in rows}

    def trio(cluster, col, nd):
        return [fmt_or(by.get((cluster, role), {}).get(col), nd) for role in ROLES]

    def fmt_or(v, nd):
        try:
            return f"{float(v):.{nd}f}"
        except (TypeError, ValueError):
            return "-"

    md = ["# Final comparison — 5-model ensembles", "",
          "**Models** (all use identical architecture and hyperparameters; only the",
          "training set differs):", "",
          "| model | trained on the community? |",
          "|---|---|",
          "| `ood` | **no** — the community was removed from training |",
          "| `ref_cleansplit` | yes — PDBbind CleanSplit CV reference |",
          "| `ref_pdbbind` | yes — PDBbind original-split CV reference |", "",
          "**Evaluation sets**:", "",
          "| set | what it is | size |",
          "|---|---|---|",
          "| `heldout` | the PDBbind complexes of that community — exactly what the `ood` model was denied | 171–2492 complexes |",
          "| `casf_ood` | the CASF-2016 entries belonging to that community | 4–50 scoring/docking, 1–10 screening targets |",
          "| `casf_all` | the whole CASF-2016 benchmark | 282 scoring, 285 docking, 57 screening |",
          "",
          "Within a community every model is scored on the *identical* complexes.",
          ""]

    # ---- 1. scoring on the held-out PDBbind complexes -----------------------
    md += ["## 1. Scoring — held-out PDBbind complexes (primary signal)", "",
           "Pearson R. `gap` = best reference − ood; positive means the exclusion cost accuracy.", ""]
    trows, gaps = [], []
    for c in paths.OOD_CLUSTERS:
        o, cs, pb = (by.get((c, r)) for r in ROLES)
        if not (o and cs and pb):
            continue
        vals = trio(c, "heldout_R", 3)
        try:
            gap = max(float(vals[1]), float(vals[2])) - float(vals[0])
            gaps.append(gap); gapstr = f"{gap:+.3f}"
        except ValueError:
            gapstr = "-"
        trows.append([c, str(o.get("heldout_n", "")), *vals, gapstr])
    md += _table(["community", "n", "ood", "ref_cleansplit", "ref_pdbbind", "gap"], trows)
    if gaps:
        md += ["", f"**Mean gap: {sum(gaps)/len(gaps):+.3f} Pearson R** across the 7 communities."]
    md += [""]

    # ---- 2. scoring on CASF ------------------------------------------------
    md += ["## 2. Scoring — CASF-2016", "",
           "Left: the community's own CASF entries (`casf_ood`, small n).",
           "Right: the whole benchmark (`casf_all`, 282 complexes) — one value per model.", ""]
    trows = []
    for c in paths.OOD_CLUSTERS:
        o = by.get((c, "ood"))
        if not o:
            continue
        trows.append([c, str(o.get("casf_ood_n", "")), *trio(c, "casf_ood_R", 3),
                      fmt_or(o.get("casf_all_R"), 3)])
    md += _table(["community", "casf_ood n", "ood", "ref_cleansplit", "ref_pdbbind",
                  "ood casf_all"], trows)
    cs_all = by.get((paths.OOD_CLUSTERS[0], "ref_cleansplit"), {})
    pb_all = by.get((paths.OOD_CLUSTERS[0], "ref_pdbbind"), {})
    md += ["", f"`casf_all` references: ref_cleansplit R = {fmt_or(cs_all.get('casf_all_R'),3)}, "
               f"ref_pdbbind R = {fmt_or(pb_all.get('casf_all_R'),3)}", ""]

    # ---- 3. docking ---------------------------------------------------------
    md += ["## 3. Docking power", "",
           "Top1 success rate (%) on the community's CASF targets, then Top3.", ""]
    trows = []
    for c in paths.OOD_CLUSTERS:
        o = by.get((c, "ood"))
        if not o:
            continue
        trows.append([c, str(o.get("dock_ood_n", "")), *trio(c, "dock_ood_top1", 1),
                      *trio(c, "dock_ood_top3", 1)])
    md += _table(["community", "n targets", "ood Top1", "ref_clean Top1", "ref_pdbb Top1",
                  "ood Top3", "ref_clean Top3", "ref_pdbb Top3"], trows)
    md += ["", "Whole benchmark (285 targets):", ""]
    trows = [[m, fmt_or(dig(metrics[m], ("full", "docking", "top1")), 1),
              fmt_or(dig(metrics[m], ("full", "docking", "top2")), 1),
              fmt_or(dig(metrics[m], ("full", "docking", "top3")), 1)]
             for m in paths.ALL_MODELS if m in metrics]
    md += _table(["model", "Top1 %", "Top2 %", "Top3 %"], trows) + [""]

    # ---- 4. screening -------------------------------------------------------
    md += ["## 4. Forward screening power", "",
           "EF1 on the community's CASF targets, then Top1 success rate (%).", ""]
    trows = []
    for c in paths.OOD_CLUSTERS:
        o = by.get((c, "ood"))
        if not o:
            continue
        trows.append([c, str(o.get("screen_ood_n", "")), *trio(c, "screen_ood_ef1", 2),
                      *trio(c, "screen_ood_top1", 1)])
    md += _table(["community", "n targets", "ood EF1", "ref_clean EF1", "ref_pdbb EF1",
                  "ood Top1", "ref_clean Top1", "ref_pdbb Top1"], trows)
    md += ["", "Whole benchmark (57 targets):", ""]
    trows = [[m, fmt_or(dig(metrics[m], ("full", "screening", "ef1")), 2),
              fmt_or(dig(metrics[m], ("full", "screening", "ef5")), 2),
              fmt_or(dig(metrics[m], ("full", "screening", "ef10")), 2),
              fmt_or(dig(metrics[m], ("full", "screening", "top1")), 1),
              fmt_or(dig(metrics[m], ("full", "screening", "top10")), 1)]
             for m in paths.ALL_MODELS if m in metrics]
    md += _table(["model", "EF1", "EF5", "EF10", "Top1 %", "Top10 %"], trows) + [""]

    # ---- 5. everything ------------------------------------------------------
    md += ["## 5. Per-community detail", "",
           "All columns, as in `comparison.csv`. `*_n` are sample sizes — the",
           "`casf_ood` and screening subsets are small; read them before the rates.", ""]
    show = ["community", "role", "model", "heldout_n", "heldout_R", "heldout_rmse",
            "casf_ood_n", "casf_ood_R", "dock_ood_n", "dock_ood_top1",
            "screen_ood_n", "screen_ood_ef1", "casf_all_R", "dock_all_top1", "screen_all_ef1"]
    md += _table(show, [[str(r.get(k, "")) for k in show] for r in rows])
    return md


def main():
    ap = argparse.ArgumentParser(description="Build the final comparison tables.")
    ap.add_argument("--results", default=None, help="results dir (default: $RESULTS)")
    args = ap.parse_args()
    if args.results:
        import pathlib
        paths.RESULTS = pathlib.Path(args.results)

    metrics = {}
    for m in paths.ALL_MODELS:
        p = paths.RESULTS / m / "metrics.json"
        if p.exists():
            metrics[m] = json.load(open(p))
        else:
            print(f"[compile] missing {p} — skipping {m}")
    if not metrics:
        raise SystemExit("no metrics.json found; run predict.py + evaluate.py first")

    with open(paths.RESULTS / "full_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    header = ["community", "role", "model", "trained_on_community"] \
        + [c for c, _ in COLUMNS] + [c for c, _ in FULL_COLUMNS]
    rows = []
    for c in paths.OOD_CLUSTERS:
        for role in ROLES:
            model = model_for(role, c)
            m = metrics.get(model)
            if not m:
                continue
            entry = m["clusters"].get(c, {})
            row = {"community": c, "role": role, "model": model,
                   "trained_on_community": "no" if entry.get("held_out") else "yes"}
            for col, path in COLUMNS:
                row[col] = fmt(dig(entry, path))
            for col, path in FULL_COLUMNS:
                row[col] = fmt(dig(m.get("full", {}), path))
            rows.append(row)

    csv_path = paths.RESULTS / "comparison.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    (paths.RESULTS / "comparison.md").write_text("\n".join(build_md(rows, metrics)) + "\n")

    print(f"[compile] {len(rows)} rows -> {csv_path}")
    print(f"[compile] {paths.RESULTS / 'comparison.md'}")
    print(f"[compile] {paths.RESULTS / 'full_metrics.json'}")


if __name__ == "__main__":
    main()
