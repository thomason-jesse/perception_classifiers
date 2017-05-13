#!/usr/bin/env python
__author__ = 'jesse'

import pickle
import sys


# python extract_labels.py
#   [agent_pickle]
#   [output_fn]
def main():

    # get arguments
    agent_fn = sys.argv[1]
    out_fn = sys.argv[2]

    # load agent
    f = open(agent_fn, 'rb')
    a = pickle.load(f)
    f.close()

    # write out labels for each object
    f = open(out_fn, 'w')
    f.write("predicate,object_id,num_true_labels,num_false_labels\n")
    for pred in a.predicate_examples:
        for oidx in a.predicate_examples[pred]:
            tr = sum([1 if b else 0 for b in a.predicate_examples[pred][oidx]])
            fa = len(a.predicate_examples[pred][oidx])-tr
            f.write(','.join([pred, str(oidx), str(tr), str(fa)])+'\n')
    f.close()

if __name__ == "__main__":
        main()
