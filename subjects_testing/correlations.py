import sys
import operator
from scipy.stats.stats import pearsonr

# read command line and csv data
try:
    objs_fn = sys.argv[1]
    measures_fn = sys.argv[2]
    pred_fn = sys.argv[3]
    cond = sys.argv[4]
except:
    sys.exit("$python correlations.py " +
             "[object_results_file] [measurements_file] [predicates_csv] [cond] [sort by]")

# read in pred object scores
f = open(objs_fn, 'r')
d = {}
uniform = []
for line in f.readlines():
    pred, data = line.strip().split(':')
    d[pred] = [0 for _ in range(0, 32)]
    pairs = data.split(';')
    last = None
    differentiated = False
    for pair in pairs:
        oidx, dec = pair.split(',')
        d[pred][int(oidx)-1] = float(dec)
        if not differentiated and last is not None and last != float(dec):
            differentiated = True
        else:
            last = float(dec)
    if not differentiated:
        uniform.append(pred)
f.close()

# read in measurements
f = open(measures_fn, 'r')
headers = f.readline().strip().split(',')
m = {}
for line in f.readlines():
    data = line.strip().split(',')
    oidx = int(data[headers.index('new object id')])-1
    for h in ["height", "width", "weight"]:
        if h not in m:
            m[h] = [0 for _ in range(0, 32)]
        m[h][oidx] = float(data[headers.index(h)])
f.close()

# read in n
f = open(pred_fn, 'r')
headers = f.readline().strip().split(',')
n = {}
for line in f.readlines():
    data = line.strip().split(',')
    data_cond = data[headers.index('cond')]
    if data_cond == cond:
        n[data[headers.index('pred')]] = float(data[headers.index('n')])
f.close()

# calculate correlation between each predicate's decisions and the object's properties
correlations = {}  # indexed first by header, then by predicate
significance = {}
n_limit = 10
p_limit = 0.05
r_limit = 0.5
for h in ["height", "width", "weight"]:
    if h not in correlations:
        correlations[h] = {}
        significance[h] = {}
    for pred in d:
        if pred in uniform or n[pred] < n_limit:
            continue
        r, p = pearsonr(d[pred], m[h])
        correlations[h][pred] = r
        significance[h][pred] = p

# sort and print
for h in ["height", "width", "weight"]:
    print h
    for pred, r in sorted(correlations[h].items(), key=operator.itemgetter(1), reverse=True):
        if significance[h][pred] < p_limit and (r > r_limit or r < -r_limit):
            print '\t' + ', '.join([pred, str(r), str(significance[h][pred])])
