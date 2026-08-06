# Third-party code and data

This is a helper repository for one study. GEMS itself — installation, data
preparation, graph construction — is at <https://github.com/camlab-ethz/GEMS>
and is not duplicated here.

| Path | Origin | Notes |
|---|---|---|
| `gems/model.py` | **GEMS** (GEMS18d architecture) | Reimplemented as a single Lightning module; the layer structure and parameter names are unchanged, so GEMS-lineage checkpoints load with a strict `state_dict` match. |
| `gems/casf/{scoring,ranking,docking,forward_screening}_power.py` | **CASF-2016** official power scripts | Ported from Python 2 to Python 3. Faithful ports - identical metric computation and identical printed strings; changes limited to `print()`, `Decimal(float(x))`, `float(np.ravel(...))` and equivalents. Validated: X-Score scoring R = 0.631, matching the CASF-2016 paper. |
| `splits/pdbbind/*` | **GEMS** | Byte-identical to GEMS's `PDBbind_data/`. |
| `splits/ood_clusters/*` | **PLINDER** | `pocket_lddt__50__community` membership lists (PDB accession codes) derived from the PLINDER index, <https://www.plinder.sh/>. |

`PDBbind_Dataset` is **not** vendored: the GEMS datasets were pickled with
`Dataset.py` at the root of the GEMS repository, so it is imported from the GEMS
checkout pointed to by `$GEMS_REPO`.

## Datasets (not redistributed here)

| Dataset | Terms |
|---|---|
| PDBbind v2020 | registered download, <http://www.pdbbind.org.cn/> |
| CASF-2016 | registered download, <http://www.pdbbind.org.cn/casf.php> |
| GEMS-released preprocessed datasets | published with GEMS |

Affinity labels in `splits/pdbbind/PDBbind_data_dict.json` derive from the
PDBbind index files.

## License

No `LICENSE` file yet - pick one before publishing, bearing in mind the GEMS and
CASF-2016 code above.
