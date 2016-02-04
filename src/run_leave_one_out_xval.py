#!/usr/bin/env python
__author__ = 'jesse'

import pickle
import operator
import copy
from agent_io import *
from perception_classifiers.srv import *


# python run_leave_one_out_xval.py
#   [full_data_agent_pickle]
#   [cond]
#   [metrics_out_csv] [objects_out_file]
def main():

    agent_fn = sys.argv[1]
    cond = sys.argv[2]
    out_fn = sys.argv[3]
    obj_fn = sys.argv[4]

    print "calling ROSpy init"
    rospy.init_node('ispy_retrain')

    print "loading training agent"
    f = open(agent_fn, 'rb')
    a = pickle.load(f)
    f.close()

    print "setting all agent predicates to be retrained"
    for pred in a.predicates:
        a.classifier_data_modified[a.predicate_to_classifier_map[pred]] = True

    print "performing leave-one-out xval..."
    r = {}
    for oidx in range(1, 33):
        print "... object "+str(oidx)

        print "...... updating classifier training data"
        b = copy.deepcopy(a)
        for pred in b.predicates:
            if oidx in b.predicate_examples[pred]:
                print "......... removing '"+pred+"' examples "+str(b.predicate_examples[pred][oidx])
                del b.predicate_examples[pred][oidx]
                b.classifier_data_modified[b.predicate_to_classifier_map[pred]] = True

        print "...... training updated classifiers"
        b.retrain_predicate_classifiers()
        for pred in a.predicates:
            a.classifier_data_modified[a.predicate_to_classifier_map[pred]] = False

        print "...... getting classifier results for all predicates on object"
        r_oidx = b.get_classifier_results(b.predicates, [oidx])
        r[oidx] = r_oidx[oidx]

    # write decisions and confidences out to file in sorted order
    f = open(obj_fn, 'w')
    pred_decs = {pred:
                 {oidx: r[oidx][pred][0]*r[oidx][pred][1]
                  for oidx in range(1, 33)} for pred in a.predicates}
    for pred in a.predicates:
        f.write(pred+":")
        s = sorted(pred_decs[pred].items(), key=operator.itemgetter(1), reverse=True)
        f.write(";".join([str(oidx)+","+str(dec) for oidx, dec in s]))
        f.write("\n")
    f.close()

    # get confusion matrix for each predicate
    print "calculating confusion matrix of training agent decisions against testing agent labels"
    p_cm = {}
    for pred in a.predicates:
        cm = [[0, 0], [0, 0]]
        for i in range(1, 33):
            if i in b.predicate_examples[pred]:
                d = 0 if r[i][pred][0] == -1 else 1  # 0 confidence is assigned a False label
                for label in b.predicate_examples[pred][i]:
                    ld = 1 if label else 0
                    cm[ld][d] += 1
        p_cm[pred] = cm

    # calculate precision, recall, f1, and kappa of predicates
    print "calculating metrics of interest"
    d = []
    for pred in a.predicates:
        cm = p_cm[pred]
        if cm[0][0] + cm[0][1] == 0 or cm[1][0] + cm[1][1] == 0:  # only one class label
            print "... pred '" + pred + "' has only one class label " + str(cm)
            continue
        r = {'cond': cond, 'pred': pred}
        r["precision"] = 0 if cm[0][1] + cm[1][1] == 0 else float(cm[1][1]) / (cm[0][1] + cm[1][1])
        r["recall"] = 0 if cm[1][0] + cm[1][1] == 0 else float(cm[1][1]) / (cm[1][0] + cm[1][1])
        r["f1"] = 0 if cm[1][1] == 0 else 2 * (r["precision"] * r["recall"]) / (r["precision"] + r["recall"])
        c = float(sum([cm[i][j] for i in range(0, 2) for j in range(0, 2)]))
        r["n"] = c
        p_o = (cm[1][1] + cm[0][0]) / c
        gy = (cm[1][0] + cm[1][1]) / c
        cy = (cm[0][1] + cm[1][1]) / c
        p_e = (gy * cy) + ((1 - gy) * (1 - cy))
        r["kappa"] = (p_o - p_e) / (1 - p_e)
        d.append(r)

    # write out to csv
    if len(d) > 0:
        f = open(out_fn, 'a')
        labels = d[0].keys()
        f.write(','.join(labels) + '\n')
        for r in d:
            f.write(','.join([str(r[l]) for l in labels]) + '\n')
        f.close()
        print "records written"
    else:
        print "no records written"

if __name__ == "__main__":
        main()
