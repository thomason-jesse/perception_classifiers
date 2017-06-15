#!/usr/bin/env python
__author__ = 'jesse'

import random
from argparse import ArgumentParser


def main(args):

    folds = [[10, 3, 27, 7, 18, 2, 20, 17],
             [5, 14, 8, 15, 1, 30, 29, 31],
             [21, 24, 19, 23, 16, 0, 4, 9],
             [22, 28, 12, 25, 11, 6, 26, 13]]

    test_obs = folds[args.test_fold][:]
    random.shuffle(test_obs)
    train_obs = folds[args.train_fold][:]
    random.shuffle(train_obs)

    for tidx in range(2):
        t1o = train_obs[:4]
        t3o = train_obs[4:]
        if tidx % 2 == 0:
            t2o = test_obs[:4]
        else:
            t2o = test_obs[4:]

        cmd = ("rosrun perception_classifiers run_inquisitive_subject.py" +
               " --stopwords_fn " + args.stopwords_fn +
               " --exp_dir " + args.exp_dir +
               " --table_1_oidxs " + ','.join([str(o) for o in t1o]) +
               " --table_2_oidxs " + ','.join([str(o) for o in t2o]) +
               " --table_3_oidxs " + ','.join([str(o) for o in t3o]) +
               " --uid " + str(args.uid) +
               " --fold " + str(args.train_fold) +
               " --cond " + str(args.cond))
        print cmd


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--stopwords_fn', type=str, required=True,
                        help="File with stop words")
    parser.add_argument('--exp_dir', type=str, required=True,
                        help=("Directory where experiment data for this fold is held. " +
                              "Expects a source/ sub-directory with initial predicates, labels, " +
                              "and, for fold > 0, classifiers pickles."))
    parser.add_argument('--uid', type=int, required=True,
                        help="unique user id number")
    parser.add_argument('--train_fold', type=int, required=True,
                        help="the active train fold")
    parser.add_argument('--test_fold', type=int, required=True,
                        help="the current test fold")
    parser.add_argument('--cond', type=int, required=True,
                        help="the condition to run")
    cmd_args = parser.parse_args()
    
    main(cmd_args)
