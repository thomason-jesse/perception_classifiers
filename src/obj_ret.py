#!/usr/bin/env python
__author__ = 'jesse'

import argparse
import ORAgent
from agent_io import *
from perception_classifiers.srv import *


def main():

    table_oidxs = [[int(oidx) for oidx in tl.split(',')]
                   for tl in [FLAGS_table_0_oidxs, FLAGS_table_1_oidxs, FLAGS_table_2_oidxs]]
    io_type = FLAGS_io_type
    logfn = FLAGS_logfn
    assert io_type == 'std' or io_type == 'robot'

    print "calling ROSpy init"
    node_name = 'obj_ret'
    rospy.init_node(node_name)

    print "instantiating ORAgent"
    a = ORAgent.ORAgent(None, table_oidxs)

    io = None
    if io_type == "std":
        print "... with input from keyboard and output to screen"
        io = IOStd(logfn)
    elif io_type == "robot":
        print "... with input and output through embodied robot"
        io = IORobot(None, logfn, table_oidxs[1])  # start facing center table.

    a.io = io

    # Identify objects on main table, then side tables.
    a.io.face_table(0)
    a.io.face_table(2)
    a.io.face_table(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--table_0_oidxs', type=str, required=True,
                        help="comma-separated ids")
    parser.add_argument('--table_1_oidxs', type=str, required=True,
                        help="comma-separated ids")
    parser.add_argument('--table_20_oidxs', type=str, required=True,
                        help="comma-separated ids")
    parser.add_argument('--io_type', type=str, required=True,
                        help="one of 'std' or 'robot'")
    parser.add_argument('--logfn', type=str, required=True,
                        help="log filename")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
