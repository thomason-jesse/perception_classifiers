import sys
import os
import operator

# read command line and csv data
try:
    dirs = sys.argv[1].split(',')
except:
    sys.exit("$python unify_cross_validation.py " +
             "[directories_to_draw_from]")

# trace dirs to read in pred object scores and predicate metric csvs
d = {"con": {}, "exp": {}}
headers = ["xval_fold"]
m = []
for dr in dirs:

    # read in pred object scores
    for cond in ["con", "exp"]:
        f = open(os.path.join(dr, cond+"_preds_results.txt"), 'r')
        uniform = []
        for line in f.readlines():
            pred, data = line.strip().split(':')
            if pred not in d[cond]:
                d[cond][pred] = [0 for _ in range(0, 32)]
            pairs = data.split(';')
            for pair in pairs:
                oidx, dec = pair.split(',')
                if int(oidx)-1 not in d[cond][pred]:
                    d[cond][pred][int(oidx)-1] = float(dec)
                else:
                    print "collision! pred '"+pred+"' has multiple entries for object id "+oidx
        f.close()

    # read in pred metrics
    f = open(os.path.join(dr, "predicate_metrics.csv"), 'r')
    if len(headers) == 1:
        headers.extend(f.readline().strip().split(','))
    for line in f.readlines():
        data = line.strip().split(',')
        r = []
        for h in headers:
            if h == "xval_fold":
                r.append(dr)
            else:
                r.append(data[headers.index(h)])
        m.append(r)
    f.close()

# write out aggregated, re-sorted object scores
for cond in ["con", "exp"]:
    f = open(cond+"_preds_results.txt", 'w')
    for pred in d[cond]:
        f.write(pred+':')
        f.write(';'.join([str(idx+1)+','+str(d[cond][pred][idx]) for idx in range(0, len(d[cond][pred]))]) + '\n')
