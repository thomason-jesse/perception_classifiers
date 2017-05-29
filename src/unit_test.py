#!/usr/bin/env python
__author__ = 'jesse'

import argparse
import UnitTestAgent
from agent_io import *
from perception_classifiers.srv import *


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

    # Accept commands indefinitely.
    while True:
        print "enter command: "
        c = a.io.get()
        if "face table" in c:  # face table [tid]
            tid = int(c.split()[-1])
            a.face_table(tid)
        elif "run classifier" in c:  # run classifier [cidx] on [oidx]
            cp = c.split()
            req = PythonRunClassifierRequest()
            req.pidx = int(cp[-3])
            req.oidx = int(cp[-1])
            try:
                rc = rospy.ServiceProxy('python_run_classifier', PythonRunClassifier)
                res = rc(req)
                print "... done; dec=" + str(res.dec) + ", conf=" + str(res.conf)
            except rospy.ServiceException, e:
                print "Service call failed: %s" % e
        elif "point to " in c:  # point to [pidx]
            pos = int(c.split()[-1])
            a.io.point(pos)
        elif "detect touch" == c:  # detect touch and return pidx
            pos = a.io.get_guess()
            print "saw touch at position " + str(pos)
        elif "stop" == c or "exit" == c:
			break

    # TODO: replace robot implementation of 'get' with speech listening


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
