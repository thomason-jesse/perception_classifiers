import sys
import math
import scipy.stats
import operator  # TEMP

# read command line and csv data
try:
    p_paired_header = None if sys.argv[4] == "None" else sys.argv[4]
    p_value = float(sys.argv[3])
    batch_header = sys.argv[2]
    in_csv = open(sys.argv[1], 'r')
    contents = in_csv.read().split('\n')
    in_csv.close()
except:
    sys.exit("$python analyze_results.py " +
             "[csv_in_filename] [batch_header] [p_value_thresh] [p-paired header OR None] [(opt)constraints]")

# read restrictions
restrictions = {}  # indexed by header, valued at required entry
if len(sys.argv) == 6:
    constraints = sys.argv[5].split(',')
    for constraint in constraints:
        for op in ['=', '>', '<']:
            if op in constraint:
                header, required_value = constraint.split(op)
                if op == '>' or op == '<':
                    rv = float(required_value)
                else:
                    rv = required_value
                restrictions[header] = [rv, op]

# set up structures
headers = [h.strip() for h in contents[0].split(',')]
header_batch_data = {}  # indexed first by header, then by batch_header value
batch_header_values = []
for header in headers:
    header_batch_data[header] = {}

# read in data
for line in contents[1:]:
    if len(line) > 1:
        line_parts = line.split(',')
        restriction_violated = False
        for restriction in restrictions:
            if ((restrictions[restriction][1] == '=' and
                    line_parts[headers.index(restriction)] != restrictions[restriction][0]) or
                    (restrictions[restriction][1] == '<' and
                        float(line_parts[headers.index(restriction)]) >= restrictions[restriction][0]) or
                    (restrictions[restriction][1] == '>' and
                        float(line_parts[headers.index(restriction)]) <= restrictions[restriction][0])):
                restriction_violated = True
        if restriction_violated:
            continue
        batch = line_parts[headers.index(batch_header)]
        if batch not in batch_header_values:
            batch_header_values.append(batch)
        for h in range(0, len(headers)):
            try:
                data = float(line_parts[h])
            except:
                data = line_parts[h]
            if p_paired_header is None:
                if batch in header_batch_data[headers[h]]:
                    header_batch_data[headers[h]][batch].append(data)
                else:
                    header_batch_data[headers[h]][batch] = [data]
            else:
                if batch in header_batch_data[headers[h]]:
                    header_batch_data[headers[h]][batch][line_parts[headers.index(p_paired_header)]] = data
                else:
                    header_batch_data[headers[h]][batch] = {line_parts[headers.index(p_paired_header)]: data}

# if paired data, transform from dictionaries to matching lists
if p_paired_header is not None:
    h_order = [i for i in range(0, len(headers)) if headers[i] != p_paired_header]
    h_order.append(headers.index(p_paired_header))
    for h in h_order:
        batch_lists = {}
        for batch in batch_header_values:
            batch_lists[batch] = []
            for paired_idx in header_batch_data[p_paired_header][batch]:
                pair_present = True
                for _batch in batch_header_values:
                    if _batch == batch:
                        continue
                    if paired_idx not in header_batch_data[p_paired_header][_batch].keys():
                        pair_present = False
                        print "WARNING: "+paired_idx+" pair key from batch "+batch+" missing from batch "+_batch
                if pair_present:
                    batch_lists[batch].append(header_batch_data[headers[h]][batch][paired_idx])

        for batch in batch_header_values:
            header_batch_data[headers[h]][batch] = batch_lists[batch]

# print size of satisfying set
print "\nsatisfying total\t" + str(sum([len(header_batch_data[batch_header][batch]) for batch in batch_header_values]))
batch_sizes = {}
for batch in batch_header_values:
    l = len(header_batch_data[batch_header][batch])
    print "satisfying batch " + batch + "\t" + str(l)
    batch_sizes[batch] = l

# print avg and stddev
print "\nraw stats"
print "stat\t\tbatch\tavg\tstddev\tmin\tmax"
batch_stats = {}
for header in header_batch_data:
    batch_stats[header] = {}
    for batch in batch_header_values:
        batch_stats[header][batch] = []

        # get stats
        if len(header_batch_data[header][batch]):
            try:
                avg = sum(header_batch_data[header][batch]) / len(header_batch_data[header][batch])
                stddev = math.sqrt(sum([math.pow(x - avg, 2.0) for x in header_batch_data[header][batch]]) / len(
                    header_batch_data[header][batch]))
            except:
                avg = stddev = None
        else:
            avg = stddev = None
        batch_stats[header][batch].extend([avg, stddev])

        if batch_sizes[batch] > 0:
            print "\t".join([header, str(batch), str(avg), str(stddev), str(min(header_batch_data[header][batch])),
                             str(max(header_batch_data[header][batch]))])
        else:
            print "\t".join([header, str(batch), "EMPTY"])

# calculate the two-side t-test p value
print "\nstatistical tests"
stats_header = "\t\t"
results_under_p_value = []
for b1 in range(0, len(batch_header_values)):
    for b2 in range(b1 + 1, len(batch_header_values)):
        stats_header += ",".join([batch_header_values[b1], batch_header_values[b2]]) + "\t\t"
print stats_header
for h in header_batch_data:
    if h == batch_header:
        continue
    results_line = h + "\t"
    for b1 in range(0, len(batch_header_values)):
        for b2 in range(b1 + 1, len(batch_header_values)):
            if batch_sizes[batch_header_values[b1]] == 0 or batch_sizes[batch_header_values[b2]] == 0:
                p = None
            elif batch_stats[h][batch_header_values[b1]][0] != batch_stats[h][batch_header_values[b2]][0]:
                if p_paired_header is not None:
                    p = scipy.stats.ttest_rel(header_batch_data[h][batch_header_values[b1]],
                                              header_batch_data[h][batch_header_values[b2]])[1]
                else:
                    p = scipy.stats.ttest_ind(header_batch_data[h][batch_header_values[b1]],
                                              header_batch_data[h][batch_header_values[b2]],
                                              equal_var=False)[1]
            else:
                p = None
            results_line += str(p) + "\t"
            if p < p_value and p is not None:
                results_under_p_value.append(
                    [str(p), h, str((batch_header_values[b1], batch_stats[h][batch_header_values[b1]][0])),
                     str((batch_header_values[b2], batch_stats[h][batch_header_values[b2]][0]))])
    print results_line

# report results below p value
if len(results_under_p_value) > 0:
    print "\nnull rejected tests"
    print "\t" + "\t".join(["p", "header", "(" + batch_header + "1,avg1)", "(" + batch_header + "2,avg2)"])
    for r in results_under_p_value:
        print "\t" + "\t".join(r)
else:
    print "\nfailed to reject null in any test"

# TEMP
# calculate differences in paired data and order them
# header = "kappa"
# n = 10
# diffs = {}
# print header, n
# for pred in header_batch_data[p_paired_header]["con"]:
#     if (header_batch_data["n"]["exp"][header_batch_data[p_paired_header]["exp"].index(pred)] < n or
#        header_batch_data["n"]["con"][header_batch_data[p_paired_header]["con"].index(pred)] < n):
#        continue
#    diff = header_batch_data[header]["exp"][header_batch_data[p_paired_header]["exp"].index(pred)] -\
#        header_batch_data[header]["con"][header_batch_data[p_paired_header]["con"].index(pred)]
#    diffs[pred] = diff
# for pred, diff in sorted(diffs.items(), key=operator.itemgetter(1), reverse=True):
#     print pred, diff, \
#         header_batch_data["n"]["exp"][header_batch_data[p_paired_header]["exp"].index(pred)], \
#         header_batch_data["n"]["con"][header_batch_data[p_paired_header]["con"].index(pred)]
