#!/usr/bin/env python
__author__ = 'jesse'

import pickle
import sys


# python extract_labels.py
#   [agent_pickle_1]
#   [agent_pickle_2]
def main():

    # load agents and get preds
    preds = [{}, {}]
    for idx in range(0, 2):
        f = open(sys.argv[1+idx], 'rb')
        a = pickle.load(f)
        f.close()
        for pred in a.predicate_examples:
            preds[idx][pred] = sum([len(a.predicate_examples[pred][oidx]) for oidx in a.predicate_examples[pred]])

    # print pred label differences
    num_new = 0
    label_difference = 0
    for pred in preds[1]:
        if pred not in preds[0]:
            num_new += 1
            l_diff = preds[1][pred]
        else:
            l_diff = preds[1][pred]-preds[0][pred]
        print pred+": "+str(l_diff)
        label_difference += l_diff
    print "num new: "+str(num_new)
    print "label difference: "+str(label_difference)


if __name__ == "__main__":
        main()
