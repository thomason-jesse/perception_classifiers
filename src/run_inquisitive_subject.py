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
    #       cond1/, cond2/
    #           source/  # contains pre-trained classifiers and preds from all PREVIOUS folds
    #               labels.pickle, predicates.pickle, classifiers.pickle
    #           uid0/, uid1/, ...
    #               toidx0_toidx1_toidx2_toidx3, ...  # directories named for test object ids (4 each, 2 dirs)
    #                   log.txt
    #                   source/  # contains re-trained classifiers and new preds from dialog after the game
    #                       labels.pickle, objects.pickle, predicates.pickle, classifiers.pickle

    # Policy parameters
    assert args.cond == 1 or args.cond == 2
    if args.cond == 1:  # This condition should only try to get an answer for the current dialog preds
        policy_max_questions = 3
        only_dialog_relevant_questions = True
    else:  # This condition should be willing to ask about any preds
        policy_max_questions = 5
        only_dialog_relevant_questions = False
    policy_ask_yes_no_prob = 0.2
    policy_min_confidence_threshold = 0.8
    policy_min_num_unknown_predicates = 0

    # Create needed data from args and prepare directory structures.
    table_oidxs = [[int(oidx) for oidx in tl.split(',')]
                   for tl in [args.table_1_oidxs, args.table_2_oidxs, args.table_3_oidxs]]
    feature_dir = os.path.join(args.exp_dir, "objects")
    cond_dir = os.path.join(args.exp_dir, "fold" + str(args.fold), str(args.cond))
    user_dir = os.path.join(cond_dir, str(args.uid), '_'.join([str(oidx) for oidx in table_oidxs[1]]))
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
    if os.path.isfile(logfn):
        cmd = "rm " + logfn  # clear logfile if it exists, since we're restarting
        print "> " + cmd
        os.system(cmd)
    cmd = "rm " + os.path.join(source_dir, '*')  # clear any existing data in this user/fold/test_objs combo
    print "> " + cmd
    os.system(cmd)
    cmd = "cp " + os.path.join(cond_dir, 'source', '*') + " " + source_dir
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
    policy = Policy(policy_ask_yes_no_prob, policy_max_questions, policy_min_confidence_threshold,
                    policy_min_num_unknown_predicates)
    
    print 'Loading initial predicates'
    with open(os.path.join(source_dir, 'predicates.pickle'), 'rb') as f:
        initial_predicates = pickle.load(f)

    print 'Instantiating an InquistiveIspyAgent'
    agent = InquisitiveIspyAgent(IORobot(None, logfn, table_oidxs[1]),
                                 table_oidxs, args.stopwords_fn, policy,
                                 only_dialog_relevant_questions,
                                 logfn, initial_predicates)

    # Run the dialog.
    print "Running experiment..."
    agent.run_dialog()
    agent.io.say("That was fun. Let's do it again.")
    agent.retract_arm()
    blacklist_from_before = agent.blacklisted_predicates_for_example
    agent.run_dialog(init_blacklist=blacklist_from_before)
    agent.io.say("Thanks for playing.")
    agent.retract_arm()
    print "Committing classifiers to file..."
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
    parser.add_argument('--cond', type=int, required=True,
                        help="experimental condition; one of '1' or '2'")
    cmd_args = parser.parse_args()
    
    main(cmd_args)
