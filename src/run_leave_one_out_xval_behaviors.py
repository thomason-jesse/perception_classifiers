#!/usr/bin/env python
__author__ = 'jesse'

import pickle
import copy
import IspyAgent
from agent_io import *
from perception_classifiers.srv import *


# python run_leave_one_out_xval.py
#   [full_data_agent_pickle]
#   [config_fn]
#   [confidences_fn]
#   [confusion_matrix_out_fn]
def main():

    agent_fn = sys.argv[1]
    config_fn = sys.argv[2]
    conf_fn = sys.argv[3]
    out_fn = sys.argv[4]
    
    obj_interval = range(1, 33)

    print "calling ROSpy init"
    rospy.init_node('ispy_retrain')

    # read in behaviors/modalities
    print "reading behaviors and modalities"
    f = open(config_fn, 'r')
    behaviors = []
    modalities = f.readline().strip().split(',')[1:]
    for line in f.readlines():
        parts = line.strip().split(',')
        behaviors.append(parts[0])
    f.close()

    print "loading training agent"
    f = open(agent_fn, 'rb')
    fa = pickle.load(f)
    f.close()

    print "unifying loaded agent with newly created"
    a = IspyAgent.IspyAgent(None, None, None)
    a.unify_with_agent(fa)
    a.io = fa.io
    a.object_IDs = fa.object_IDs
    a.stopwords = fa.stopwords

    print "setting all agent predicates to be retrained"
    for pred in a.predicates:
        a.classifier_data_modified[a.predicate_to_classifier_map[pred]] = True

    print "performing leave-one-out xval..."
    s = {}
    c = {}
    for oidx in obj_interval:
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

        print "...... getting sub classifier results for all predicates on object"
        s_oidx = b.get_sub_classifier_results(b.predicates, [oidx])
        s[oidx] = {}
        for pred in a.predicates:
            s[oidx][pred] = [[0 for m_idx in range(0, len(modalities))] for b_idx in range(0, len(behaviors))]
            c_idx = 0
            for b_idx in range(0, len(behaviors)):
                for m_idx in range(0, len(modalities)):
                    s[oidx][pred][b_idx][m_idx] = s_oidx[oidx][pred][c_idx]
                    c_idx += 1

        # read in predicate context confidence values
        print "...... reading in predicate confidence values"
        f = open(conf_fn, 'r')
        conf = {}  # indexed by predicate
        for line in f.readlines():
            parts = line.strip().split(',')
            pred = a.classifier_to_predicate_map[int(parts[0])]
            conf[pred] = \
                [[0 for m_idx in range(0, len(modalities))] for b_idx in range(0, len(behaviors))]
            i = 1
            for b_idx in range(0, len(behaviors)):
                for m_idx in range(0, len(modalities)):
                    conf[pred][b_idx][m_idx] = float(parts[i])
                    i += 1
        f.close()
        c[oidx] = conf

    # calculate decisions for each (pred, behavior)
    print "calculating decisions across each predicate, behavior tuple"
    r = {}
    for oidx in obj_interval:
        r_oidx = {}
        for pred in a.predicates:
            for b_idx in range(0, len(behaviors)):
                key = (pred, b_idx)
                dec = sum([s[oidx][pred][b_idx][m_idx]*c[oidx][pred][b_idx][m_idx]
                           for m_idx in range(0, len(modalities))])
                r_oidx[key] = dec
        r[oidx] = r_oidx

    # get confusion matrix for each predicate
    print "calculating confusion matrix of training agent decisions against testing agent labels"
    pb_cm = {}
    for pred in a.predicates:
        for b_idx in range(0, len(behaviors)):
            key = (pred, b_idx)
            cm = [[0, 0], [0, 0]]
            for i in obj_interval:
                if i in b.predicate_examples[pred]:
                    d = 0 if r[i][key] <= 0 else 1  # 0 confidence is assigned a False label
                    for label in b.predicate_examples[pred][i]:
                        ld = 1 if label else 0
                        cm[ld][d] += 1
            pb_cm[key] = cm

    # write out confusion matrices to csv
    print "writing confusion matrices out to file"
    f = open(out_fn, 'a')
    f.write("pred, behavior, true_positive, false_positive, false_negative, true_negative\n")
    for pred in a.predicates:
        for b_idx in range(0, len(behaviors)):
            key = (pred, b_idx)
            f.write(pred+','+behaviors[b_idx]+',')
            dec = []
            for prediction_idx in range(1, -1, -1):
                for label_idx in range(1, -1, -1):
                    dec.append(str(pb_cm[key][label_idx][prediction_idx]))
            f.write(','.join(dec) + '\n')
    f.close()

if __name__ == "__main__":
        main()
