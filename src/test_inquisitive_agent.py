#!/usr/bin/env python
__author__ = 'aishwarya'

import rospy
import pickle
from argparse import ArgumentParser
from Policy import Policy
from InquisitiveIspyAgent import InquisitiveIspyAgent
from agent_io import *

def main(args):
    table_oidxs = [[int(oidx) for oidx in tl.split(',')]
                   for tl in [args.table_1_oidxs, args.table_2_oidxs, args.table_3_oidxs]]
    assert args.io_type == 'std' or args.io_type == 'robot'
    
    print "Calling ROSpy init"
    node_name = 'test_inquisitive_agent'
    rospy.init_node(node_name)
    
    print 'Creating policy'
    policy = Policy(args.policy_type, args.max_questions, args.min_confidence_threshold,
                    args.min_num_unknown_predicates)
    
    print 'Loading initial predicates'
    initial_predicates = pickle.load(open(args.initial_predicates_fn))
    
    print 'Creating agent'
    agent = InquisitiveIspyAgent(None, table_oidxs, args.stopwords_fn, policy, \
                                 args.log_fn, initial_predicates)
    
    io = None
    if io_type == "std":
        print "... with input from keyboard and output to screen"
        io = IOStd(logfn)
    elif io_type == "robot":
        print "... with input and output through embodied robot"
        io = IORobot(None, logfn, table_oidxs[1])  # start facing center table.
    agent.io = io
    
    while True:
        agent.run_dialog()
        
        agent.io.say('Do you want to try another dialog? (yes/no)')
        response = agent.io.get()
        if response.lower() == 'no':
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--table_1_oidxs', type=str, required=True,
                        help="Comma-separated ids of objects on robot's left")
    parser.add_argument('--table_2_oidxs', type=str, required=True,
                        help="Comma-separated ids of objects on robot's front")
    parser.add_argument('--table_3_oidxs', type=str, required=True,
                        help="Comma-separated ids of objects on robot's right")
    parser.add_argument('--io_type', type=str, required=True,
                        help="One of 'std' or 'robot'")
    parser.add_argument('--logfn', type=str, required=True,
                        help="Log filename")
    parser.add_argument('--stopwords_fn', type=str, required=True,
                        help="File with stop words")
    parser.add_argument('--initial_predicates_fn', type=str, required=True,
                        help="Pickle of initial predicates")
    parser.add_argument('--policy_type', type=str, required=True,
                        help="One of 'guess', 'yes_no', 'example'")
    parser.add_argument('--policy_type', type=str, required=True,
                        help="One of 'guess', 'yes_no', 'example'")
    parser.add_argument('--policy_max_questions', type=int, default=5,
                        help="Max # questions in dialog")
    parser.add_argument('--policy_min_confidence_threshold', type=float, default=0.0001,
                        help="Min avg confidence threshold across candidate yes-no questions to allow asking")
    parser.add_argument('--policy_min_num_unknown_predicates', type=int, default=3,
                        help="Min # unknown predicates required before asking for example")                                                                
    args = parser.parse_args()
    
    main(args)
