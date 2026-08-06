#!/usr/bin/env python3
"""CASF-2016 ranking power (Python-3 port of CASF-2016/power_ranking/ranking_power.py).

Faithful port: identical Spearman/Kendall/PI computation and output strings, with
Py2 fixes (print(), .loc instead of .ix, Decimal(float(x)), numeric-only corr).
Usage:
  python ranking_power.py -c CoreSet.dat -s score.dat -p positive -o OUT
"""
import numpy as np
import sys
import os
import pandas as pd
import getopt
from decimal import Decimal, ROUND_HALF_UP

if len(sys.argv) < 2:
    print("Please input parameter files or use -h for help")
    sys.exit()

try:
    options, args = getopt.getopt(sys.argv[1:], "hc:s:p:o:", ["help", "coreset=", "score=", "prefer=", "output="])
except getopt.GetoptError:
    sys.exit()


def usage():
    print("-c or --coreset: location of 'CoreSet.dat' (or a subset data file)")
    print("-s or --score: scoring file. 1st column #code, 2nd column score")
    print("-p or --prefer: 'negative' or 'positive'")
    print("-o or --output: prefix of output result files. Default My_Ranking_Power")
    print("-h or --help: print help message")


def cal_PI(df):
    dfsorted = df.sort_values(['logKa'], ascending=True)
    W, WC = [], []
    lst = list(dfsorted.index)
    for i in np.arange(0, 5):
        xi = lst[i]
        score = float(dfsorted.loc[xi, 'score'])
        bindaff = float(dfsorted.loc[xi, 'logKa'])
        for j in np.arange(i + 1, 5):
            xj = lst[j]
            scoretemp = float(dfsorted.loc[xj, 'score'])
            bindafftemp = float(dfsorted.loc[xj, 'logKa'])
            w_ij = abs(bindaff - bindafftemp)
            W.append(w_ij)
            if score < scoretemp:
                WC.append(w_ij)
            elif score > scoretemp:
                WC.append(-w_ij)
            else:
                WC.append(0)
    return float(sum(WC)) / float(sum(W))


def dec(x, y):
    q = {2: '0.01', 3: '0.001', 4: '0.0001'}[y]
    return Decimal(float(x)).quantize(Decimal(q), rounding=ROUND_HALF_UP)


out = 'My_Ranking_Power'
for name, value in options:
    if name in ("-h", "--help"):
        usage()
        sys.exit()
    if name in ("-c", "--coreset"):
        with open(value, 'r') as f, open('cstemp', 'w+') as f1:
            for i in f.readlines():
                if i.startswith('#'):
                    if i.startswith('#code'):
                        f1.writelines(i)
                    else:
                        continue
                else:
                    f1.writelines(i)
        aa = pd.read_csv('cstemp', sep='[,,\t, ]+', engine='python')
        aa = aa.drop_duplicates(subset=['#code'], keep='first')
    if name in ("-s", "--score"):
        filename = value
        bb = pd.read_csv(value, sep='[,,\t, ]+', engine='python')
    if name in ("-p", "--prefer"):
        fav = value
    if name in ("-o", "--output"):
        out = value

# Process the data
testdf1 = pd.merge(aa, bb, on='#code')
if str(fav) == 'negative':
    testdf1['score'] = testdf1['score'].apply(np.negative)
    group = testdf1.groupby('target')
elif str(fav) == 'positive':
    group = testdf1.groupby('target')
else:
    print('please input negative or positive')
    sys.exit()


def top(df, n=1, column='logKa'):
    return df.sort_values(by=column)[-n:]


toptardf = testdf1.groupby('target').apply(top)
targetlst = toptardf['#code'].tolist()

spearman = pd.DataFrame(index=targetlst, columns=['spearman'])
kendall = pd.DataFrame(index=targetlst, columns=['kendall'])
PI = pd.DataFrame(index=targetlst, columns=['PI'])
rankresults = pd.DataFrame(index=range(1, len(targetlst) + 1),
                           columns=['Target', 'Rank1', 'Rank2', 'Rank3', 'Rank4', 'Rank5'])
tmp = 1
for i, j in group.__iter__():
    testdf2 = group.get_group(i)[['#code', 'logKa', 'score']]
    testdf2 = testdf2.sort_values('score', ascending=False)
    tartemp = top(testdf2)['#code'].tolist()
    tar = ''.join(tartemp)
    if len(testdf2) == 5:
        num = testdf2[['logKa', 'score']]
        spearman.loc[tar, 'spearman'] = num.corr(method='spearman').loc['logKa', 'score']
        kendall.loc[tar, 'kendall'] = num.corr(method='kendall').loc['logKa', 'score']
        PI.loc[tar, 'PI'] = cal_PI(df=testdf2)
        rankresults.loc[tmp, 'Rank1'] = ''.join(testdf2[0:1]['#code'].tolist())
        rankresults.loc[tmp, 'Rank2'] = ''.join(testdf2[1:2]['#code'].tolist())
        rankresults.loc[tmp, 'Rank3'] = ''.join(testdf2[2:3]['#code'].tolist())
        rankresults.loc[tmp, 'Rank4'] = ''.join(testdf2[3:4]['#code'].tolist())
        rankresults.loc[tmp, 'Rank5'] = ''.join(testdf2[4:5]['#code'].tolist())
        rankresults.loc[tmp, 'Target'] = tar
        tmp += 1
    else:
        spearman.drop(tar, inplace=True)
        kendall.drop(tar, inplace=True)
        PI.drop(tar, inplace=True)

spearmanmean = dec(float(spearman['spearman'].sum()) / float(spearman.shape[0]), 3)
kendallmean = dec(float(kendall['kendall'].sum()) / float(kendall.shape[0]), 3)
PImean = dec(float(PI['PI'].sum()) / float(PI.shape[0]), 3)
spearman.to_csv(out + '_Spearman.results', sep='\t', index_label='#Target')
kendall.to_csv(out + '_Kendall.results', sep='\t', index_label='#Target')
PI.to_csv(out + '_PI.results', sep='\t', index_label='#Target')
if os.path.exists('cstemp'):
    os.remove('cstemp')

rankresults.dropna(axis=0, inplace=True)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print(rankresults)
print("\nSummary of the ranking power: ===========================================")
print("The Spearman correlation coefficient (SP) = %0.3f" % (dec(spearmanmean, 3)))
print("The Kendall correlation coefficient (tau) = %0.3f" % (dec(kendallmean, 3)))
print("The Predictive index (PI) = %0.3f" % (dec(PImean, 3)))
print("=========================================================================\n")
