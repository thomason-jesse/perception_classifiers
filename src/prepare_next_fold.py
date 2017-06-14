#!/usr/bin/env python
__author__ = 'jesse'

import os
import pickle
from argparse import ArgumentParser


def main(args):

    # Directory structure:
    # exp/
    #   objects/  # these are shared across all conditions
    #       oidxs.pickle, features.pickle
    #   fold0/, fold1/, ...  # each subsequent fold generated from previous; fold0 created by hand
    #       cond1/, cond2/
    #           source/  # contains pre-trained classifiers and preds from all PREVIOUS folds
    #               labels.pickle, predicates.pickle, classifiers.pickle
    #           uid0/, uid1/, ...
    #               toidx0_toidx1_toidx2_toidx3, ...  # directories named for test object ids (4 each, 2 dirs)
    #                   log.txt
    #                   source/  # contains re-trained classifiers and new preds from dialog after the game
    #                       labels.pickle, objects.pickle, predicates.pickle, classifiers.pickle

    fold_dir = os.path.join(args.exp_dir, "fold" + str(args.train_fold))
    new_fold_dir = os.path.join(args.exp_dir, "fold" + str(args.train_fold + 1))

    # For each condition, collect source info for fold, subtract it from all user data, and write new fold
    # with source + user data
    for cond in [1, 2]:
        print "retraining condition " + str(cond) + "..."

        # Read in source information so it can be subtracted from individual user sessions, then added in only once.
        print "... reading in information from fold " + str(args.train_fold) + "..."
        cond_dir = os.path.join(fold_dir, str(cond))
        fold_source_dir = os.path.join(cond_dir, "source")
        with open(os.path.join(fold_source_dir, "predicates.pickle"), 'rb') as f:
            source_predicates = pickle.load(f)
        with open(os.path.join(fold_source_dir, "labels.pickle"), 'rb') as f:
            source_labels = pickle.load(f)
        print "...... done"

        # Change (pidx, oidx, l) label triples to (predicate, oidx, l) to prevent bad indexing.
        source_labels = [(source_predicates[pidx], oidx, l)
                         for pidx, oidx, l in source_labels]

        # Read in user data and subtract source information as we go.
        print "... reading in information from each user and subtracting original data..."
        user_predicates = {}  # indexed by uid, value is list of new user preds
        user_labels = {}  # indexed by uid, value is list of new user label triples
        for root, dirs, fs in os.walk(cond_dir):
            if root != cond_dir and root != "source":
                user_id = int(root)
                print "...... processing user id " + str(user_id)
                user_predicates[user_id] = []
                user_labels[user_id] = []
                for test_dir in dirs:
                    user_source_dir = os.path.join(root, test_dir, "source")
                    with open(os.path.join(user_source_dir, "predicates.pickle"), 'rb') as f:
                        all_user_predicates = pickle.load(f)
                    with open(os.path.join(user_source_dir, "labels.pickle"), 'rb') as f:
                        all_user_labels = pickle.load(f)

                    # Change (pidx, oidx, l) label triples to (predicate, oidx, l) to prevent bad indexing.
                    all_user_labels = [(all_user_predicates[pidx], oidx, l)
                                       for pidx, oidx, l in all_user_labels]

                    # Remove sources and record.
                    for pred in source_predicates:
                        all_user_predicates.remove(pred)
                    for label in source_labels:
                        all_user_labels.remove(label)
                    for pred in all_user_predicates:
                        if pred not in user_predicates[user_id]:
                            user_predicates[user_id].append(pred)
                    user_labels[user_id].extend(all_user_labels)

                print "user " + str(user_id) + ":"
                print "\tnew preds: " + str(user_predicates[user_id])
                print "\tnew labels: " + str(user_labels[user_id])
        print "...... done"

        # Unify predicates and labels.
        print "... unifying information from old fold with nove user information..."
        new_predicates = source_predicates[:]
        new_labels = source_labels[:]
        for uid in user_predicates.keys():
            for pred in user_predicates[uid]:
                if pred not in new_predicates:
                    new_predicates.append(pred)
            new_labels += user_labels[uid]
        print "all preds: " + str(new_predicates)
        print "all labels: " + str(new_labels)
        print "...... done"

        # Change labels from (predicate, oidx, l) to (pidx, oidx, l) with new predicate order established.
        new_labels = [(new_predicates.index(predicate), oidx, l)
                      for predicate, oidx, l in new_labels]

        # Write outfiles.
        print "... writing outfiles for fold " + str(args.train_fold + 1) + "..."
        cond_out_dir = os.path.join(new_fold_dir, str(cond), "source")
        if not os.path.isfile(cond_out_dir):
            cmd = "mkdir -p " + cond_out_dir
            print "> " + cmd
            os.system(cmd)
        with open(os.path.join(cond_out_dir, "predicates.pickle"), 'wb') as f:
            pickle.dump(new_predicates, f)
        with open(os.path.join(cond_out_dir, "labels.pickle"), 'wb') as f:
            pickle.dump(new_labels, f)
        print "...... done"

        print "... done"


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--exp_dir', type=str, required=True,
                        help=("Directory where experiment data for this fold is held. " +
                              "Expects a source/ sub-directory with initial predicates, labels, " +
                              "and, for fold > 0, classifiers pickles."))
    parser.add_argument('--train_fold', type=int, required=True,
                        help="ID of training fold. Next fold directory will be generated automatically.")
    cmd_args = parser.parse_args()
    
    main(cmd_args)
