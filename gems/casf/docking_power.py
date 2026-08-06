#!/usr/bin/env python3
import numpy as np
import sys, os
import pandas as pd
import getopt
from decimal import *

def usage():
    print("-c or --coreset: specify the location of 'CoreSet.dat'")
    print("-s or --score: specify the directory containing score files")
    print("-p or --prefer: input 'negative' or 'positive' scoring preference")
    print("-r or --rmsd: specify the directory containing RMSD files")
    print("-o or --output: prefix of output result files. Default is My_Docking_Power")
    print("-l or --limit: RMSD cutoff (in angstrom) to define near-native pose")
    print("-h or --help: print help message")
    print("\nExample: python docking_power.py -c CoreSet.dat -s ./X-Score -r ../decoys_docking/ -p 'positive' -l 2 -o 'X-Score' > MyDockingPower.out")

# Handle args
try:
    options, args = getopt.getopt(sys.argv[1:], "hc:s:r:p:l:o:", ["help", "coreset=", "score=", "rmsd=", "prefer=", "limit=", "output="])
except getopt.GetoptError:
    usage()
    sys.exit(2)

# Defaults
out = 'My_Docking_Power'
fav = 'positive'
cut = 2.0

for name, value in options:
    if name in ("-h", "--help"):
        usage()
        sys.exit()
    elif name in ("-c", "--coreset"):
        with open(value, 'r') as f, open('cstemp', 'w') as f1:
            for line in f:
                if not line.startswith('#') or line.startswith('#code'):
                    f1.write(line)
        aa = pd.read_csv('cstemp', sep=r"\s+")
        aa = aa.drop_duplicates(subset=['#code'], keep='first')
        aa.rename(columns={'#code': 'code'}, inplace=True)
    elif name in ("-s", "--score"):
        scorefile = value
    elif name in ("-r", "--rmsd"):
        rmsdfile = value
    elif name in ("-p", "--prefer"):
        fav = value.lower()
    elif name in ("-l", "--limit"):
        cut = float(value)
    elif name in ("-o", "--output"):
        out = value

def dec(x, y):
    return Decimal(x).quantize(Decimal(f'0.{"0"*(y-1)}1'), rounding=ROUND_HALF_UP)

# Structures
pdb = aa['code']
Top1 = pd.DataFrame(index=pdb, columns=['success'])
Top2 = pd.DataFrame(index=pdb, columns=['success'])
Top3 = pd.DataFrame(index=pdb, columns=['success'])
SPs = {s: pd.DataFrame(index=pdb, columns=['spearman']) for s in range(2, 11)}
dockresults = pd.DataFrame(index=range(1, len(pdb) + 1), columns=['code', 'Rank1', 'RMSD1', 'Rank2', 'RMSD2', 'Rank3', 'RMSD3'])

ascending = [False] if fav == 'positive' else [True]
tmp = 1

for i in pdb:
    rmsd_path = os.path.join(rmsdfile, f"{i}_rmsd.dat")
    score_path = os.path.join(scorefile, f"{i}_score.dat")

    try:
        rmsddf = pd.read_csv(rmsd_path, sep=r"\s+")
        scoredf = pd.read_csv(score_path, sep=r"\s+")
    except Exception as e:
        print(f"Error reading {i} files: {e}")
        continue

    # Standardize column name
    if '#code' in rmsddf.columns:
        rmsddf.rename(columns={'#code': 'ligand_id'}, inplace=True)
    if '#code' in scoredf.columns:
        scoredf.rename(columns={'#code': 'ligand_id'}, inplace=True)

    # Merge on ligand ID (e.g., 4llx_276)
    testdf = pd.merge(rmsddf, scoredf, on='ligand_id')
    if testdf.empty:
        print(f"No matching entries for {i}")
        continue

    dfsorted = testdf.sort_values(by=['score'], ascending=ascending)

    dockresults.at[tmp, 'code'] = i
    for rank in range(3):
        if rank >= len(dfsorted):
            break
        dockresults.at[tmp, f'Rank{rank + 1}'] = dfsorted.iloc[rank]['ligand_id']
        dockresults.at[tmp, f'RMSD{rank + 1}'] = float(dfsorted.iloc[rank]['rmsd'])

    tmp += 1

    for j in range(1, 4):
        minrmsd = dfsorted.iloc[:j]['rmsd'].min()
        top = locals()[f"Top{j}"]
        top.at[i, 'success'] = 1 if minrmsd <= cut else 0

    for s in range(2, 11):
        sptemp = testdf[testdf.rmsd <= s]
        if sptemp.shape[0] >= 5:
            sp = SPs[s]
            corr = sptemp[['rmsd', 'score']].corr(method='spearman')
            if 'score' in corr.columns and 'rmsd' in corr.index:
                val = corr.loc['rmsd', 'score']
                if fav == 'positive':
                    val = -val
                sp.at[i, 'spearman'] = val

# Drop NaNs
for s in SPs:
    SPs[s] = SPs[s].dropna(subset=['spearman'])

# Report
top1success = dec(Top1['success'].sum() / Top1.shape[0], 3) * 100
top2success = dec(Top2['success'].sum() / Top2.shape[0], 3) * 100
top3success = dec(Top3['success'].sum() / Top3.shape[0], 3) * 100

print(f"Top1 Success Rate = {top1success}%")
print(f"Top2 Success Rate = {top2success}%")
print(f"Top3 Success Rate = {top3success}%")

for s in range(2, 11):
    if not SPs[s].empty:
        mean_sp = dec(SPs[s]['spearman'].mean(), 3)
        print(f"Avg Spearman when RMSD <= {s} Å: {mean_sp}")
