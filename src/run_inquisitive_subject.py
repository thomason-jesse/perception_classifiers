#!/usr/bin/env python
__author__ = 'jesse'

import pickle
import subprocess
from argparse import ArgumentParser
from Policy import Policy
from InquisitiveIspyAgent import InquisitiveIspyAgent
from agent_io import *


def main(args):

    # Directory structure:
    # exp/
    #   objects/  # these are shared across all conditions
    #       oidxs.pickle, features.pickle
    #   fold0/, fold1/, ...  # each subsequent fold generated from previous; fold0 created by hand
    #       policy0/, policy1/, policy2/
    #           source/  # contains pre-trained classifiers and preds from all PREVIOUS folds
    #               labels.pickle, predicates.pickle, classifiers.pickle
    #           uid0/, uid1/, ...
    #               toidx0_toidx1_toidx2_toidx3, ...  # directories named for test object ids (4 each, 2 dirs)
    #                   log.txt
    #                   source/  # contains re-trained classifiers and new preds from dialog after the game
    #                       labels.pickle, objects.pickle, predicates.pickle, classifiers.pickle

    # Hard-coded parameters
    policy_max_questions = 5
    policy_min_confidence_threshold = 0.1
    policy_min_num_unknown_predicates = 3

    # Create needed data from args and prepare directory structures.
    table_oidxs = [[int(oidx) for oidx in tl.split(',')]
                   for tl in [args.table_1_oidxs, args.table_2_oidxs, args.table_3_oidxs]]
    feature_dir = os.path.join(args.exp_dir, "objects")
    policy_dir = os.path.join(args.exp_dir, "fold" + str(args.fold), args.policy_type)
    user_dir = os.path.join(policy_dir, str(args.uid), '_'.join([str(oidx) for oidx in table_oidxs[1]]))
    if not os.path.isdir(user_dir):
        cmd = "mkdir -p " + user_dir
        print "> " + cmd
        os.system(cmd)
    logfn = os.path.join(user_dir, "log.txt")
    source_dir = os.path.join(user_dir, 'source')
    if not os.path.isdir(source_dir):
        cmd = "mkdir " + source_dir
        print "> " + cmd
        os.system(cmd)
    cmd = "cp " + os.path.join(policy_dir, 'source', '*') + " " + source_dir
    print "> " + cmd
    os.system(cmd)  # fold + policy preds and init classifiers

    # Launch python_classifier_services node with appropriate operating directory.
    cmd = ("rosrun perception_classifiers python_classifier_services.py " +
           " --source_dir " + source_dir +
           " --feature_dir " + feature_dir)
    print "> " + cmd
    pc = subprocess.Popen(cmd.split())

    print "Initializing run_inquisitive_agent node"
    node_name = 'run_inquisitive_agent'
    rospy.init_node(node_name)
    
    print 'Creating policy'
    policy = Policy(args.policy_type, policy_max_questions, policy_min_confidence_threshold,
                    policy_min_num_unknown_predicates)
    
    print 'Loading initial predicates'
    with open(os.path.join(source_dir, 'predicates.pickle'), 'rb') as f:
        initial_predicates = pickle.load(f)

    print 'Instantiating an InquistiveIspyAgent'
    agent = InquisitiveIspyAgent(IORobot(None, logfn, table_oidxs[1]),
                                 table_oidxs, args.stopwords_fn, policy,
                                 logfn, initial_predicates)

    # Run the dialog.
    print "Running experiment..."
    agent.run_dialog()
    print "Concluding..."
    agent.io.say("Thanks for playing.")
    agent.io.point(-1)
    agent.commit_classifier_changes()
    pc.kill()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--stopwords_fn', type=str, required=True,
                        help="File with stop words")
    parser.add_argument('--exp_dir', type=str, required=True,
                        help=("Directory where experiment data for this fold is held. " +
                              "Expects a source/ sub-directory with initial predicates, labels, " +
                              "and, for fold > 0, classifiers pickles."))
    parser.add_argument('--table_1_oidxs', type=str, required=True,
                        help="Comma-separated ids of objects on robot's left")
    parser.add_argument('--table_2_oidxs', type=str, required=True,
                        help="Comma-separated ids of objects on robot's front")
    parser.add_argument('--table_3_oidxs', type=str, required=True,
                        help="Comma-separated ids of objects on robot's right")
    parser.add_argument('--uid', type=int, required=True,
                        help="unique user id number")
    parser.add_argument('--fold', type=int, required=True,
                        help="the active train fold")
    parser.add_argument('--policy_type', type=str, required=True,
                        help="One of 'guess', 'yes_no', 'example'")
    cmd_args = parser.parse_args()
    
    main(cmd_args)
