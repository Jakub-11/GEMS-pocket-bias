"""Run the four CASF-2016 power tests and parse their numbers.

The scripts in this package are faithful Python-3 ports of the official CASF-2016
tools — identical metric computation and identical printed strings. We run them
as subprocesses and read the labelled lines out of stdout.

Every function takes an optional `keep` set of PDB codes; when given, the score
files *and* the CoreSet.dat / TargetInfo.dat denominators are restricted to it,
which is what makes an OOD-subset success rate comparable with the full one.
"""
import os
import re
import subprocess
import sys
import tempfile

from gems import paths
from gems.casf import dat

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = {
    "scoring": os.path.join(HERE, "scoring_power.py"),
    "ranking": os.path.join(HERE, "ranking_power.py"),
    "docking": os.path.join(HERE, "docking_power.py"),
    "screening": os.path.join(HERE, "forward_screening_power.py"),
}


def coreset_dat():
    for sub in ("power_scoring", "power_ranking", "power_docking"):
        p = os.path.join(paths.CASF_ROOT, sub, "CoreSet.dat")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"CoreSet.dat not found under CASF_ROOT={paths.CASF_ROOT}")


def target_info_dat():
    return os.path.join(paths.CASF_ROOT, "power_screening", "TargetInfo.dat")


def decoys_docking_dir():
    return os.path.join(paths.CASF_ROOT, "decoys_docking")


def _abs(p):
    """Power scripts run with cwd=<tmp>, so every path handed to them must be absolute."""
    return os.path.abspath(p)


def _run(kind, args, cwd):
    cmd = [sys.executable, SCRIPTS[kind]] + args
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=cwd)
        return True, r.stdout
    except subprocess.CalledProcessError as e:
        return False, (e.stdout or "") + (e.stderr or "")


def _num(pattern, text, cast=float):
    m = re.search(pattern, text)
    return cast(m.group(1)) if m else None


def scoring(scoring_dat, keep=None, workdir=None):
    """CASF scoring power: Pearson R and SD in fitting."""
    scoring_dat = _abs(scoring_dat)
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        core = coreset_dat()
        sdat = scoring_dat
        if keep is not None:
            sdat = os.path.join(tmp, "scoring.dat")
            n = dat.subset_scoring(scoring_dat, sdat, keep)
            if n < 2:
                return {"n": n}
            core = dat.subset_table(core, os.path.join(tmp, "CoreSet.dat"), keep)
        ok, out = _run("scoring", ["-c", core, "-s", sdat, "-p", "positive",
                                   "-o", os.path.join(tmp, "out")], cwd=tmp)
        if not ok:
            return {"error": out[-300:]}
        return {"pearsonr": _num(r"Pearson correlation coefficient \(R\) = ([-\d.]+)", out),
                "sd": _num(r"Standard deviation in fitting \(SD\) = ([-\d.]+)", out),
                "n": _num(r"Number of favorable sample \(N\) = (\d+)", out, int)}


def ranking(scoring_dat, keep=None, workdir=None):
    """CASF ranking power: Spearman, Kendall tau, predictive index."""
    scoring_dat = _abs(scoring_dat)
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        core = coreset_dat()
        sdat = scoring_dat
        if keep is not None:
            sdat = os.path.join(tmp, "scoring.dat")
            if dat.subset_scoring(scoring_dat, sdat, keep) < 2:
                return {}
            core = dat.subset_table(core, os.path.join(tmp, "CoreSet.dat"), keep)
        ok, out = _run("ranking", ["-c", core, "-s", sdat, "-p", "positive",
                                   "-o", os.path.join(tmp, "out")], cwd=tmp)
        if not ok:
            return {"error": out[-300:]}
        return {"spearman": _num(r"Spearman correlation coefficient \(SP\) = ([-\d.]+)", out),
                "kendall": _num(r"Kendall correlation coefficient \(tau\) = ([-\d.]+)", out),
                "pi": _num(r"Predictive index \(PI\) = ([-\d.]+)", out)}


def docking(score_dir, keep=None, workdir=None):
    """CASF docking power: Top1/2/3 success rates (RMSD <= 2 A)."""
    score_dir = _abs(score_dir)
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        core = coreset_dat()
        sdir = score_dir
        present = dat.targets_in(score_dir)
        if keep is not None:
            present = present & set(keep)
            if not present:
                return {"n_targets": 0}
            sdir = os.path.join(tmp, "scores")
            dat.subset_dir(score_dir, sdir, present)
            core = dat.subset_table(core, os.path.join(tmp, "CoreSet.dat"), present)
        ok, out = _run("docking", ["-c", core, "-s", sdir, "-r", decoys_docking_dir(),
                                   "-p", "positive", "-l", "2",
                                   "-o", os.path.join(tmp, "out")], cwd=tmp)
        if not ok:
            return {"error": out[-300:], "n_targets": len(present)}
        res = {"top1": _num(r"Top1 Success Rate = ([-\d.]+)%", out),
               "top2": _num(r"Top2 Success Rate = ([-\d.]+)%", out),
               "top3": _num(r"Top3 Success Rate = ([-\d.]+)%", out),
               "n_targets": len(present)}
        for s in (2, 5, 10):
            v = _num(rf"Avg Spearman when RMSD <= {s} [ÅA]: ([-\d.]+)", out)
            if v is not None:
                res[f"spearman_rmsd{s}"] = v
        return res


def screening(score_dir, keep=None, workdir=None):
    """CASF forward screening power: Top1/5/10 success rates and EF1/5/10."""
    score_dir = _abs(score_dir)
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        core, tinfo = coreset_dat(), target_info_dat()
        sdir = score_dir
        present = dat.targets_in(score_dir)
        if keep is not None:
            present = present & set(keep)
            if not present:
                return {"n_targets": 0}
            sdir = os.path.join(tmp, "scores")
            dat.subset_dir(score_dir, sdir, present)
            core = dat.subset_table(core, os.path.join(tmp, "CoreSet.dat"), present)
            tinfo = dat.subset_table(tinfo, os.path.join(tmp, "TargetInfo.dat"), present)
        ok, out = _run("screening", ["-c", core, "-s", sdir, "-t", tinfo,
                                     "-p", "positive", "-o", os.path.join(tmp, "out")], cwd=tmp)
        if not ok:
            return {"error": out[-300:], "n_targets": len(present)}
        return {"top1": _num(r"Top1 success rate: ([-\d.]+)%", out),
                "top5": _num(r"Top5 success rate: ([-\d.]+)%", out),
                "top10": _num(r"Top10 success rate: ([-\d.]+)%", out),
                "ef1": _num(r"EF1: ([-\d.]+)", out),
                "ef5": _num(r"EF5: ([-\d.]+)", out),
                "ef10": _num(r"EF10: ([-\d.]+)", out),
                "n_targets": len(present)}
