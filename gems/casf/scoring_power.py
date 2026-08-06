#!/usr/bin/env python3
"""CASF-2016 scoring power (Python-3 port of CASF-2016/power_scoring/scoring_power.py).

Faithful port: identical metric computation and output strings (so results match
the official tool), with Py2 fixes (print(), Decimal(float(x)), float(np.ravel(...))).
Usage:
  python scoring_power.py -c CoreSet.dat -s score.dat -p positive -o OUT
"""
import numpy as np
import sys
import os
import pandas as pd
import scipy
import scipy.stats
import getopt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from decimal import Decimal, ROUND_HALF_UP

if len(sys.argv) < 2:
    print("Please input parameter files or use -h for help")
    sys.exit()


def usage():
    print("-c or --coreset: location of 'CoreSet.dat' (or a subset data file)")
    print("-s or --score: scoring file. 1st column #code, 2nd column score. Separators: , \\t space")
    print("-p or --prefer: 'negative' or 'positive' depending on scoring preference")
    print("-o or --output: prefix of the output processed scoring files. Default My_Scoring_Power")
    print("-h or --help: print help message")


try:
    options, args = getopt.getopt(sys.argv[1:], "hc:s:p:o:", ["help", "coreset=", "score=", "prefer=", "output="])
except getopt.GetoptError:
    sys.exit()


def dec(x, y):
    q = {2: '0.01', 3: '0.001', 4: '0.0001'}[y]
    return Decimal(float(x)).quantize(Decimal(q), rounding=ROUND_HALF_UP)


out = 'My_Scoring_Power'
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

# Process the data and remove the outliers
testdf1 = pd.merge(aa, bb, on='#code')
if str(fav) == 'positive':
    testdf2 = testdf1[testdf1.score > 0]
    testdf2.to_csv(out + '_processed_score', columns=['#code', 'logKa', 'score'], sep='\t', index=False)
elif str(fav) == 'negative':
    testdf1['score'] = testdf1['score'].apply(np.negative)
    testdf2 = testdf1[testdf1.score > 0]
    testdf2.to_csv(out + '_processed_score', columns=['#code', 'logKa', 'score'], sep='\t', index=False)
else:
    print('please input negative or positive')
    sys.exit()

# Pearson correlation coefficient + regression
regr = linear_model.LinearRegression()
regr.fit(testdf2[['score']], testdf2[['logKa']])
testpredy = regr.predict(testdf2[['score']])
testr = scipy.stats.pearsonr(testdf2['logKa'].values, testdf2['score'].values)[0]
testmse = mean_squared_error(testdf2[['logKa']], testpredy)
num = testdf2.shape[0]
testsd = np.sqrt((testmse * num) / (num - 1))
if os.path.exists('cstemp'):
    os.remove('cstemp')

# Print the output of scoring power evaluation
testdf1.rename(columns={'#code': 'code'}, inplace=True)
testdf1.index = testdf1.index.map(lambda x: x + 1)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print(testdf1[['code', 'logKa', 'score']])
print("\nSummary of the scoring power: ===================================")
print("The regression equation: logKa = %.2f + %.2f * Score" % (
    dec(float(np.ravel(regr.coef_)[0]), 2), dec(float(np.ravel(regr.intercept_)[0]), 2)))
print("Number of favorable sample (N) = %d" % (num))
print("Pearson correlation coefficient (R) = %0.3f" % (dec(testr, 3)))
print("Standard deviation in fitting (SD) = %0.2f" % (dec(testsd, 2)))
print("=================================================================")
