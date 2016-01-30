#!/usr/bin/env python
__author__ = 'jesse'

import rospkg
import ast
import pickle
import IspyAgent
from agent_io import *
from perception_classifiers.srv import *


# python test_artificial_agent.py
#   [pickle_dir] [cond]
#   [folds_to_train] [fold_to_test]
#   [out_pickle] [log_dir] [out_dir]
def main():

    # read command-line args
    pickle_dir = sys.argv[1]
    cond = sys.argv[2]
    folds_to_train = [int(f) for f in sys.argv[3].split(',')]
    fold_to_test = int(sys.argv[4])
    out_pickle_fn = sys.argv[5]
    log_dir = sys.argv[6]
    out_dir = sys.argv[7]

    # calculations from command-line
    fold_to_user_ids = [range(0, 10), range(10, 20), range(20, 30), range(30, 42)]
    logs_to_train = []
    for fold in folds_to_train:
        logs_to_train.extend(fold_to_user_ids[fold])
    logs_to_test = fold_to_user_ids[fold_to_test]

    # paths
    path_to_perception_classifiers = rospkg.RosPack().get_path('perception_classifiers')
    stopwords_fn = os.path.join(path_to_perception_classifiers, 'src', 'stopwords_en.txt')

    print "calling ROSpy init"
    rospy.init_node('ispy_retrain')

    try:
        f = open(os.path.join(out_dir, out_pickle_fn), 'rb')
        a = pickle.load(f)
        f.close()
        print "loaded requested IspyAgent from file; ensure classifiers are intact!"
        _ = raw_input()
    except IOError:

        print "instantiating blank ispyAgent"
        a = IspyAgent.IspyAgent(None, None, stopwords_fn)

        # pass over each requested fold's directory, loading agents, subtracting their base, and unifying
        for fold in folds_to_train:
            fold_dirname = get_fold_dirname(cond, fold)
            fold_dir = os.path.join(pickle_dir, fold_dirname)

            # load in base agent
            b = None
            bn = cond+".local.agent"
            if fold > 0:
                print "...loading base agent '"+bn+"'"
                pfn = os.path.join(fold_dir, bn)
                f = open(pfn, 'rb')
                b = pickle.load(f)
                f.close()

            # iterate over fold's directory and unify agents
            for root, dirs, files in os.walk(fold_dir):
                for fn in files:
                    if fn != bn:
                        print "...loading and unifying agent '"+fn+"'"
                        f = open(os.path.join(root, fn), 'rb')
                        n = pickle.load(f)
                        f.close()
                        if b is not None:
                            n.subtract_predicate_examples(b.predicate_examples)
                        a.unify_with_agent(n)

        # load in test agent pickle (formed after fold) and pickle from before it to
        # subtract the second from the first's predicates and establish what predicates
        # were learned in the test fold
        test_agent_fn = os.path.join(pickle_dir, get_fold_dirname(cond, fold_to_test+1), cond+".local.agent")
        f = open(test_agent_fn, 'rb')
        test_agent = pickle.load(f)
        f.close()
        if fold_to_test > 0:
            prev_test_agent_fn = os.path.join(pickle_dir, get_fold_dirname(cond, fold_to_test),
                                              cond+".local.agent")
            f = open(prev_test_agent_fn, 'rb')
            prev_test_agent = pickle.load(f)
            f.close()
            test_fold_preds = [pred for pred in test_agent.predicates if pred not in prev_test_agent.predicates]
        else:
            test_fold_preds = test_agent.predicates[:]
        print "discovered predicates learned in test fold: "+str(test_fold_preds)

        # iterate over training log files in order and remove information gleaned about
        # predicates first learned in test fold until they are seen in descriptions in training logs
        # (e.g. remove info from `red' until someone uses `red'; strip clarifying dialog labels)
        print "walking logs to see train data in-order and strip test fold predicate info as appropriate"
        for user_id in logs_to_train:
            for root, dirs, files in os.walk(log_dir):
                for fn in files:
                    valid, object_IDs = get_log_fn_properties(fn, cond, [user_id])
                    if valid:
                        print "... processing log " + fn
                        f = open(os.path.join(root, fn), 'r')
                        lines = f.readlines()
                        f.close()
                        curr_ob = None
                        for line_idx in range(0, len(lines)):
                            if len(lines[line_idx].strip()) == 0:
                                continue
                            p = lines[line_idx].split(':')
                            # recognize pointing to a new object
                            if p[0] == "point":
                                curr_ob = object_IDs[int(p[1])]
                            # catch removal of banned preds when they are introduced by training speakers
                            if p[0] == "cnf_clauses":
                                cnf_clauses = ast.literal_eval(p[1])
                                for cnf in cnf_clauses:
                                    for pred in cnf:
                                        if pred in test_fold_preds:
                                            print "...... discovered previously banned word '"+pred+"'"
                                            del test_fold_preds[test_fold_preds.index(pred)]
                            # remove training label gathered from pred not introduced in training
                            if p[0] == "say" and p[1][:len("Would you use the word")] == "Would you use the word":
                                for pred in test_fold_preds:
                                    if "'"+pred+"'" in p[1]:  # label to remove
                                        r = lines[line_idx+1].split(':')[1]
                                        if '?' not in r.split():
                                            if a.is_no(r):
                                                print "...... removing negative label for '"+pred+"', "+str(curr_ob)
                                                del a.predicate_examples[pred][curr_ob][
                                                    a.predicate_examples[pred][curr_ob].index(False)]
                                            elif a.is_yes(r):
                                                print "...... removing positive label for '"+pred+"', "+str(curr_ob)
                                                del a.predicate_examples[pred][curr_ob][
                                                    a.predicate_examples[pred][curr_ob].index(True)]

        # call classifier training services
        print "retraining classifiers from specified data"
        a.retrain_predicate_classifiers()

        print "saving perceptual classifiers to file"
        a.save_classifiers()

        print "pickling artificial ispyAgent"
        f = open(os.path.join(out_dir, out_pickle_fn), 'wb')
        pickle.dump(a, f)
        f.close()

    # iterate through logs to test
    print "writing out artificial match scores"
    for root, dirs, files in os.walk(log_dir):
        for fn in files:
                valid, object_IDs = get_log_fn_properties(fn, cond, logs_to_test)
                if valid:
                    print "... processing log " + fn
                    f_out = open(os.path.join(out_dir, fn), 'w')
                    f_in = open(os.path.join(root, fn), 'r')
                    lines = f_in.readlines()
                    f_in.close()
                    for line_idx in range(0, len(lines)):
                        line = lines[line_idx]
                        if len(line) == 0:
                            f_out.write(line)
                            continue
                        p = line.strip().split(':')
                        if p[0] == "match_scores":
                            cnf_clauses = ast.literal_eval(lines[line_idx-1].split(':')[1])
                            a.object_IDs = object_IDs
                            match_scores = a.get_match_scores(cnf_clauses)
                            f_out.write("match_scores:" + str(match_scores) + '\n')
                        else:
                            f_out.write(line)
                    f_out.close()


def get_fold_dirname(cond, fold):
    return cond+"_fold"+str(fold)+"_previous" if fold > 0 else "fold0_previous"

def get_log_fn_properties(fn, cond, ids_valid):
    valid = False
    object_IDs = None
    if len(fn.split('_')) == 3:
        valid = fn[:3] == cond and int(fn.split('_')[1]) in ids_valid
        object_IDs = [int(o) for o in fn.split('_')[2].split('.')[0].split('-')]
    elif len(fn.split('_')) == 2:
        valid = int(fn.split('_')[0]) in ids_valid
        object_IDs = [int(o) for o in fn.split('_')[1].split('.')[0].split('-')]
    return valid, object_IDs

if __name__ == "__main__":
        main()