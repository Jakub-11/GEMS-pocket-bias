#!/usr/bin/env python3
import numpy as np
import os
import sys
import pandas as pd
import getopt
from decimal import Decimal, ROUND_HALF_UP

def usage():
    print("Usage:")
    print("-c or --coreset: path to CoreSet.dat")
    print("-s or --score: directory with XXXX_score.dat files")
    print("-t or --target: path to TargetInfo.dat")
    print("-p or --prefer: 'positive' or 'negative'")
    print("-o or --output: output prefix (default: My_Forward_Screening_Power)")
    print("-h or --help: display this help message")
    print("\nExample:")
    print("python forward_screening_power.py -c CoreSet.dat -s ./scores -t TargetInfo.dat -p positive -o X-Score")
    sys.exit()

# Parse arguments
try:
    opts, args = getopt.getopt(sys.argv[1:], "hc:s:t:p:o:", ["help", "coreset=", "score=", "target=", "prefer=", "output="])
except getopt.GetoptError:
    usage()

coreset_file = None
score_dir = None
target_file = None
prefer = None
out = 'My_Forward_Screening_Power'

for opt, arg in opts:
    if opt in ("-h", "--help"):
        usage()
    elif opt in ("-c", "--coreset"):
        coreset_file = arg
    elif opt in ("-s", "--score"):
        score_dir = arg
    elif opt in ("-t", "--target"):
        target_file = arg
    elif opt in ("-p", "--prefer"):
        prefer = arg
    elif opt in ("-o", "--output"):
        out = arg

if not all([coreset_file, score_dir, target_file, prefer]):
    print("Missing required arguments.\n")
    usage()

def dec(x, precision):
    if precision == 2:
        return Decimal(x).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    elif precision == 3:
        return Decimal(x).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)
    else:
        return Decimal(x).quantize(Decimal('0'), rounding=ROUND_HALF_UP)

# Read CoreSet (whitespace-separated; \s+ avoids the python-engine multi-char-sep quirk)
aa = pd.read_csv(
    coreset_file,
    sep=r'\s+',
    engine='python',
    comment=None,           # do NOT treat '#' as comment (keep the #code header)
)
aa = aa.drop_duplicates(subset=['#code'])

# Read TargetInfo. The file is RAGGED (some targets have >5 binders) and the '#T'
# header line is a comment, so read headerless with explicit names #T,L1..L10 and
# let pandas NaN-pad short rows. Robust across pandas versions.
_TI_NAMES = ['#T'] + [f'L{k}' for k in range(1, 11)]
bb = pd.read_csv(target_file, comment='#', sep=r'\s+', engine='python',
                 header=None, names=_TI_NAMES)
bb = bb.drop_duplicates(subset=['#T'])
cc = bb.set_index('#T')

# Get representative complex for each cluster
def top(df, n=1, column='logKa'):
    return df.sort_values(by=column, ascending=False).head(n)

pdbs = aa['#code']
toptardf = aa.groupby('target').apply(top)
targetlst = [i for i in toptardf['#code'] if i in cc.index]

# Build decoy list and cutoff thresholds
decoylst = list(filter(None, pd.unique(cc.loc[:, cc.columns.str.startswith('L')].values.ravel())))
t1 = int(dec(len(decoylst) * 0.01, 0))
t5 = int(dec(len(decoylst) * 0.05, 0))
t10 = int(dec(len(decoylst) * 0.10, 0))
t1 = max(t1, 1)
t5 = max(t5, 1)
t10 = max(t10, 1)

# Prepare output DataFrames
Top1 = pd.DataFrame(index=targetlst, columns=['success'])
Top5 = pd.DataFrame(index=targetlst, columns=['success'])
Top10 = pd.DataFrame(index=targetlst, columns=['success'])
EF1 = pd.DataFrame(index=targetlst, columns=['enrichment'])
EF5 = pd.DataFrame(index=targetlst, columns=['enrichment'])
EF10 = pd.DataFrame(index=targetlst, columns=['enrichment'])

forwardf = pd.DataFrame(columns=['Target'] + list(range(1, t10 + 1)))

for i, target in enumerate(targetlst, 1):
    try:
        score_path = os.path.join(score_dir, f"{target}_score.dat")
        scoredf = pd.read_csv(score_path, sep=r'[ ,_\t]+', engine='python')
        scoredf = scoredf[scoredf['#code'].isin(decoylst)]
        grouped = scoredf.groupby('#code')

        if prefer == 'positive':
            testdf = grouped['score'].max().to_frame()
            sorted_df = testdf.sort_values('score', ascending=False)
        elif prefer == 'negative':
            testdf = grouped['score'].min().to_frame()
            sorted_df = testdf.sort_values('score', ascending=True)
        else:
            print("Please input 'positive' or 'negative' for --prefer")
            sys.exit()

        row = {'Target': target}
        for m in range(1, t10 + 1):
            row[m] = sorted_df.iloc[m-1:m].index[0]
        forwardf.loc[i] = row

        Topligand = cc.loc[target]['L1']
        Allactivelig = list(cc.loc[target].dropna())
        NTBtotal = len(Allactivelig)

        for name, j in zip(['1', '5', '10'], [t1, t5, t10]):
            top_set = sorted_df.index[:j]
            top_df = locals()[f'Top{name}']
            ef_df = locals()[f'EF{name}']

            top_df.loc[target, 'success'] = int(Topligand in top_set)
            ntb = sum(lig in top_set for lig in Allactivelig)
            ef_value = float(ntb) / (NTBtotal * int(name) * 0.01)
            ef_df.loc[target, 'enrichment'] = ef_value

    except Exception as e:
        print(f"Error processing {target}: {e}")
        continue

# Calculate success rates and enrichment factors
top1success = float(dec(Top1['success'].sum() / Top1.shape[0], 3)) * 100
top5success = float(dec(Top5['success'].sum() / Top5.shape[0], 3)) * 100
top10success = float(dec(Top10['success'].sum() / Top10.shape[0], 3)) * 100
ef1factor = float(dec(EF1['enrichment'].mean(), 2))
ef5factor = float(dec(EF5['enrichment'].mean(), 2))
ef10factor = float(dec(EF10['enrichment'].mean(), 2))

# Print summary
print(f"\nTop1 success rate: {top1success:.1f}%")
print(f"Top5 success rate: {top5success:.1f}%")
print(f"Top10 success rate: {top10success:.1f}%")
print(f"EF1: {ef1factor:.2f}")
print(f"EF5: {ef5factor:.2f}")
print(f"EF10: {ef10factor:.2f}")

# Optionally save results
forwardf.to_csv(f"{out}_rankings.csv", index=False)
Top1.to_csv(f"{out}_top1.csv")
Top5.to_csv(f"{out}_top5.csv")
Top10.to_csv(f"{out}_top10.csv")
EF1.to_csv(f"{out}_ef1.csv")
EF5.to_csv(f"{out}_ef5.csv")
EF10.to_csv(f"{out}_ef10.csv")
