"""All paths in one place. Every value is an environment variable with a default.

Every value is an environment variable with a default; export it to override.
The defaults point at the locations used for the study.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _p(key, default):
    return os.environ.get(key, str(default))


SCRATCH = _p("SCRATCH", "/scratch/project_465003034")   # only used to build the defaults below

# --- inputs: the GEMS-released datasets ------------------------------------
GEMS_REPO = _p("GEMS_REPO", f"{SCRATCH}/GEMS")
TRAIN_CLEANSPLIT_PT = _p("TRAIN_CLEANSPLIT_PT", f"{GEMS_REPO}/datasets/GEMS_pytorch_datasets/B6AEPL_train_cleansplit.pt")
TRAIN_PDBBIND_PT = _p("TRAIN_PDBBIND_PT", f"{GEMS_REPO}/datasets/GEMS_pytorch_datasets/B6AEPL_train_pdbbind.pt")
CASF_SCORING_PT = _p("CASF_SCORING_PT", f"{GEMS_REPO}/datasets/GEMS_pytorch_datasets/B6AEPL_casf2016.pt")
CASF_DOCKING_PT = _p("CASF_DOCKING_PT", f"{SCRATCH}/GEMS_data_old_processing/CASF2016_docking_decoys_dataset.pt")
CASF_SCREENING_DIR = _p("CASF_SCREENING_DIR", f"{SCRATCH}/GEMS_data_old_processing/CASF2016_screening_decoys")
CASF_ROOT = _p("CASF_ROOT", f"{SCRATCH}/CASF-2016")

# --- repo data --------------------------------------------------------------
SPLITS = Path(_p("SPLITS", REPO / "splits"))
OOD_CLUSTERS_DIR = Path(_p("OOD_CLUSTERS_DIR", SPLITS / "ood_clusters"))

# --- outputs ----------------------------------------------------------------
RUNS = Path(_p("RUNS", REPO / "runs"))            # training checkpoints
RESULTS = Path(_p("RESULTS", REPO / "results"))   # predictions + metrics
LOGS = Path(_p("LOGS", REPO / "logs"))

OOD_CLUSTERS = ["1nvq", "1sqa", "2p15", "2vw5", "3dd0", "3f3e", "3o9i"]
REFERENCE_MODELS = ["cleansplit", "pdbbind"]
ALL_MODELS = REFERENCE_MODELS + [f"ood_{c}" for c in OOD_CLUSTERS]
N_FOLDS = 5


def add_gems_repo_to_syspath():
    """Put the GEMS checkout on sys.path.

    The GEMS datasets were pickled with `Dataset.py` at the root of the GEMS
    repository, so unpickling them needs that module importable. We take it from
    the GEMS checkout rather than keeping a copy here.
    """
    import sys
    if not (Path(GEMS_REPO) / "Dataset.py").exists():
        raise FileNotFoundError(
            f"GEMS_REPO does not look like a GEMS checkout: {GEMS_REPO}\n"
            "Clone https://github.com/camlab-ethz/GEMS and set GEMS_REPO to it.")
    if GEMS_REPO not in sys.path:
        sys.path.insert(0, GEMS_REPO)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))


def cluster_ids(cluster):
    """PDB codes of one PLINDER pocket-lDDT community."""
    p = OOD_CLUSTERS_DIR / f"{cluster}_pocket_lddt__50__community_pdb_ids.txt"
    return frozenset(l.strip() for l in open(p) if l.strip())


def run_dir(model, fold):
    return RUNS / f"{model}_f{fold}"
