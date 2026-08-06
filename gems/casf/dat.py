"""Read/write the score-file formats the CASF-2016 power scripts expect.

Formats (verified against CASF-2016/power_*/examples/):

    scoring.dat            #code score              1a30 6.2300
    {PDB}_score.dat        #code score              1a30_128 6.0300          (docking)
    {TARGET}_score.dat     #code_ligand_num score   1a30_ligand_101 4.8800   (screening)

Scores are unscaled pK (higher = better binder), so the power scripts are always
run with `-p positive`.
"""
import glob
import os


def pdb_code(graph_id):
    """'1a30' / '1a30_L00001' / '1a30_ligand_3' -> '1a30'."""
    return str(graph_id).split("_")[0]


def write_scoring(scores, path):
    """scores: {id: pK}. One row per PDB code (mean if a code appears twice)."""
    by_code = {}
    for gid, v in scores.items():
        by_code.setdefault(pdb_code(gid), []).append(float(v))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write("#code score\n")
        for code in sorted(by_code):
            vals = by_code[code]
            f.write(f"{code} {sum(vals) / len(vals):.4f}\n")
    return len(by_code)


def write_docking(scores, out_dir):
    """scores: {'{PDB}_{pose}': pK}. Writes one {PDB}_score.dat per target."""
    os.makedirs(out_dir, exist_ok=True)
    by_target = {}
    for pose, v in scores.items():
        by_target.setdefault(pdb_code(pose), {})[pose] = float(v)
    for target, poses in by_target.items():
        with open(os.path.join(out_dir, f"{target}_score.dat"), "w") as f:
            f.write("#code score\n")
            for p in sorted(poses):
                f.write(f"{p} {poses[p]:.4f}\n")
    return sorted(by_target)


def write_screening(target, scores, out_dir):
    """scores: {'{ligand}_ligand_{n}': pK}. Writes one {TARGET}_score.dat."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{target}_score.dat")
    with open(path, "w") as f:
        f.write("#code_ligand_num score\n")
        for p in sorted(scores):
            f.write(f"{p} {float(scores[p]):.4f}\n")
    return path


# ------------------------------------------------------------------ subsetting
def targets_in(score_dir):
    """4-char target codes that have a *_score.dat in this directory."""
    return {os.path.basename(f)[:4] for f in glob.glob(os.path.join(score_dir, "*_score.dat"))}


def subset_scoring(src, dst, keep):
    """Copy scoring.dat keeping only rows whose PDB code is in `keep`."""
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    n = 0
    with open(src) as fi, open(dst, "w") as fo:
        for line in fi:
            s = line.strip()
            if not s or s.startswith("#"):
                fo.write(line)
            elif s.split()[0] in keep:
                fo.write(line)
                n += 1
    return n


def subset_dir(src_dir, dst_dir, keep):
    """Link the *_score.dat files whose target is in `keep` into dst_dir."""
    os.makedirs(dst_dir, exist_ok=True)
    n = 0
    for f in glob.glob(os.path.join(src_dir, "*_score.dat")):
        if os.path.basename(f)[:4] in keep:
            dst = os.path.join(dst_dir, os.path.basename(f))
            if not os.path.exists(dst):
                os.symlink(os.path.abspath(f), dst)
            n += 1
    return n


def subset_table(src, dst, keep):
    """Copy CoreSet.dat / TargetInfo.dat keeping only rows whose first column is in `keep`.

    The power scripts take the number of targets from these tables, so a subset
    scored against the full table reports success rates deflated by the
    subset-size ratio. Always subset both together with the score directory.
    """
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    with open(src) as fi, open(dst, "w") as fo:
        for line in fi:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#") or s.split()[0] in keep:
                fo.write(line)
    return dst
