#!/usr/bin/env python
__author__ = 'jesse'

import argparse
import UnitTestAgent
from agent_io import *


def main():

    table_oidxs = [[int(oidx) for oidx in tl.split(',')]
                   for tl in [FLAGS_table_1_oidxs, FLAGS_table_2_oidxs, FLAGS_table_3_oidxs]]
    io_type = FLAGS_io_type
    logfn = FLAGS_logfn
    assert io_type == 'std' or io_type == 'robot'

    print "calling ROSpy init"
    node_name = 'obj_id_unit_test'
    rospy.init_node(node_name)

    print "instantiating UnitTestAgent"
    a = UnitTestAgent.UnitTestAgent(None, 2, table_oidxs)

    io = None
    if io_type == "std":
        print "... with input from keyboard and output to screen"
        io = IOStd(logfn)
    elif io_type == "robot":
        print "... with input and output through embodied robot"
        io = IORobot(None, logfn, table_oidxs[1])  # start facing center table.
    a.io = io

    # Accept commands until told to stop.
    while True:
        print "enter command: "
        c = a.io.get()
        if "face table" in c:  # face table [tid]
            try:
                tid = int(c.split()[-1])
                a.face_table(tid, report=True)
            except (IndexError, ValueError):
                continue
        elif "run classifier" in c and c.split()[3] == "on":  # run classifier [cidx] on [oidx]
            try:
                cp = c.split()
                dec, conf = a.run_classifier_on_object(int(cp[-3]), int(cp[-1]))
                a.io.say("Decision " + str(dec) + " with  confidence " + str(round(conf, 2)) + ".")
            except (IndexError, ValueError):
                continue
        elif "run classifier" in c and c.split()[3] == "at":  # run classifier [cidx] at [pos]
            try:
                cp = c.split()
                dec, conf = a.run_classifier_on_object_at_position(int(cp[-3]), int(cp[-1]))
                a.io.say("Decision " + str(dec) + " with  confidence " + str(round(conf, 2)) + ".")
            except (IndexError, ValueError):
                continue
        elif "point to " in c:  # point to [pos]
            try:
                pos = int(c.split()[-1])
                a.io.say("Pointing to object at position " + str(pos) + ".")
                a.io.point(pos)
            except (IndexError, ValueError):
                continue
        elif "detect touch" == c:  # detect touch and return pos
            a.io.say("Looking for a detected touch...")
            pos, oidx = a.io.get_guess()
            a.io.say("Saw touch at position " + str(pos) + " on object " + str(oidx) + ".")
        elif "stop" == c or "exit" == c:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--table_1_oidxs', type=str, required=True,
                        help="comma-separated ids")
    parser.add_argument('--table_2_oidxs', type=str, required=True,
                        help="comma-separated ids")
    parser.add_argument('--table_3_oidxs', type=str, required=True,
                        help="comma-separated ids")
    parser.add_argument('--io_type', type=str, required=True,
                        help="one of 'std' or 'robot'")
    parser.add_argument('--logfn', type=str, required=True,
                        help="log filename")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
