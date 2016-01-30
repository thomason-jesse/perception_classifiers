#!/usr/bin/env python
__author__ = 'jesse'

import sys
import pickle
import os
import math
import operator


# python analyze_predicates.py [out_dir] [agent pickle] [classifier config] [classifier conf]
def main():

    # read command-line args
    out_dir = sys.argv[1]
    agent_fn = sys.argv[2]
    config_fn = sys.argv[3]
    conf_fn = sys.argv[4]

    # read in pickled agent
    f = open(agent_fn, 'rb')
    a = pickle.load(f)
    f.close()

    print a.predicate_to_classifier_map  # DEBUG
    print a.classifier_to_predicate_map  # DEBUG

    # read in behaviors, modalities, and context sizes
    f = open(config_fn, 'r')
    lines = f.readlines()
    f.close()
    modalities = lines[0].strip().split(',')[1:]
    context_feature_sizes = {}  # indexed by behavior then modality
    behaviors = []
    for l_idx in range(1, len(lines)):
        p = lines[l_idx].strip().split(',')
        behaviors.append(p[0])
        context_feature_sizes[p[0]] = {}
        for p_idx in range(1, len(p)):
            context_feature_sizes[behaviors[-1]][modalities[p_idx-1]] = int(p[p_idx])

    # read in context confidences
    f = open(conf_fn, 'r')
    lines = f.readlines()
    f.close()
    context_confidences = {}  # indexed first by classifier id, then behavior, then modality
    pred_confidences = {}  # indexed by classifier id, valued at sum of all kappa
    for line in lines:
        p = line.split(',')
        context_confidences[int(p[0])] = {}
        pred_confidences[int(p[0])] = 0
        p_idx = 1
        for b in behaviors:
            context_confidences[int(p[0])][b] = {}
            for m in modalities:
                context_confidences[int(p[0])][b][m] = float(p[p_idx])
                pred_confidences[int(p[0])] += float(p[p_idx])
                p_idx += 1
        pred_confidences[int(p[0])] = len(a.predicate_examples[a.classifier_to_predicate_map[int(p[0])]])

    # save predicates and their context matrices to file
    f = open(os.path.join(out_dir, 'pred_conf_matrices.txt'), 'w')
    for c, v in sorted(pred_confidences.items(), key=operator.itemgetter(1), reverse=True):
        f.write(str(c) + ":" + a.classifier_to_predicate_map[c] + '\t' + str(v) + '\n')
        f.write(matrix_str(context_confidences[c], context_feature_sizes) + '\n')
    f.close()

    # calculate cosine distance between predicates in the kappa classifier space
    norm = {}  # indexed by classifier id
    cos_sim = {}  # indexed by (classifier id, classifier id)
    for aidx in context_confidences:
        if aidx not in norm:
            norm[aidx] = calc_norm([context_confidences[aidx][b][m] for b in behaviors for m in modalities])
        if norm[aidx] == 0:
            continue
        for bidx in context_confidences:
            if aidx == bidx or (bidx, aidx) in cos_sim:
                continue
            if bidx not in norm:
                norm[bidx] = calc_norm([context_confidences[bidx][b][m] for b in behaviors for m in modalities])
            if norm[bidx] == 0:
                continue
            cos_sim[(aidx, bidx)] = sum([context_confidences[aidx][b][m]*context_confidences[bidx][b][m]
                                        for b in behaviors for m in modalities]) / (norm[aidx]*norm[bidx])

    # save distances to file
    f = open(os.path.join(out_dir, 'pred_cos_distances.txt'), 'w')
    for k, v in sorted(cos_sim.items(), key=operator.itemgetter(1), reverse=True):
        f.write(','.join([a.classifier_to_predicate_map[k[0]], a.classifier_to_predicate_map[k[1]], str(v)])+'\n')
    f.close()


# take in a list of numbers and return the 2 norm
def calc_norm(l):
    return math.sqrt(sum([math.pow(l[idx], 2) for idx in range(0, len(l))]))

# take in a data map and produce a string representation
def matrix_str(d, fs):
    s = '\t' + '\t'.join(d[d.keys()[0]].keys()) + '\n'
    for ridx in d:
        s += ridx+'\t'
        s += '\t'.join(["%.3f" % d[ridx][k] if fs[ridx][k] > 0 else '-----' for k in d[ridx].keys()]) + '\n'
    return s

if __name__ == "__main__":
        main()