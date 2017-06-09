#!/usr/bin/env python
__author__ = 'jesse'

import rospy
import sys
import rospy
from perception_classifiers.srv import *


class UnitTestAgent:

    # Initialize a new agent.
    # io - an instance of a class in agent_io.py
    # tid - the table id the robot is already facing (1, 2, or 3).
    # table_oidxs - a list of three lists, each containing the four object indices of those on the corresponding table.
    def __init__(self, io, tid, table_oidxs):
        assert 1 <= tid <= 3
        assert len(table_oidxs) == 3

        self.io = io
        self.table_oidxs = table_oidxs
        self.tid = tid
        self.io.point(-1)
        self.arm_pos = -1

    # The robot turns and faces table id tid, updating its io to reflect the new objects on the faced table.
    # tid - the destination table id
    # report - if True, the robot will speak aloud to explain its actions and delays during turning.
    def face_table(self, tid, report=False):
        assert 1 <= tid <= 3

        self.retract_arm()
        if self.io.face_table(tid, self.table_oidxs[tid - 1], report=report):
            self.tid = tid
        else:
            sys.exit("ERROR: failed to face table " + str(tid))

    # The robot retracts its arm to the resting position, enabling all objects to be seen.
    def retract_arm(self):
        if self.arm_pos != -1:
            self.io.point(-1)

    # The robot points to an object at the specified position on the table.
    # pos - position index in {0, 1, 2, 3} for four objects.
    def point_to_position(self, pos):
        assert 0 <= pos <= 3
        self.retract_arm()
        if self.arm_pos != pos:
            self.io.point(pos)

    # The robot points to an object by its id if on the table, or does nothing if the id specified is not on the table.
    # oidx - the object index to be pointed to.
    # return - True if the object was found on the table, False otherwise.
    def point_to_object(self, oidx):
        obj_found = True
        if oidx not in self.table_oidxs[self.tid - 1]:
            obj_found = False
            for tidx in range(0, len(self.table_oidxs)):
                if oidx in self.table_oidxs[tidx]:
                    self.face_table(tidx + 1)
                    obj_found = True
        if obj_found:
            pos = self.table_oidxs[self.tid - 1].index(oidx)
            self.retract_arm()
            if self.arm_pos != pos:
                self.io.point(pos)
        return obj_found

    # The robot will watch for a touch on top of some object, returning the position and object id when detected.
    # returns - a tuple (pos, oidx) of the position and object id of the detected touch.
    def detect_touch(self):
        pos = self.io.get_touch()
        oidx = self.table_oidxs[self.tid - 1][pos]
        return pos, oidx

    # Run a classifier on an object at a specified position.
    def run_classifier_on_object_at_position(self, cidx, pos):
        oidx = self.table_oidxs[self.tid - 1][pos]
        return self.run_classifier_on_object(cidx, oidx)

    # Run a classifier on the specified object.
    # cidx - the index of the classifier to run
    # oidx - the object index to run on
    # returns - a tuple (dec, conf) of the decision in {True, False} and ensemble confidence in [0, 1]
    def run_classifier_on_object(self, cidx, oidx):
        req = PythonRunClassifierRequest()
        req.pidx = cidx
        req.oidx = oidx
        try:
            rc = rospy.ServiceProxy('python_run_classifier', PythonRunClassifier)
            res = rc(req)
            return res.dec, res.conf
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e
            return None, None

    # new_preds - List of predicates for which new classifiers have to
    #       be created. This allows the agent and the classifier service
    #       to have same order of predicates
    # pidxs, oidxs, labels are parallel arrays which give triples of
    # (predicate_idx, obj_idx, {-1, 1}) to be updated
    def update_classifiers(self, new_preds, pidxs, oidxs, labels):
        req = PythonUpdateClassifiersRequest()
        req.new_preds = new_preds
        req.pidxs = pidxs
        req.oidxs = oidxs
        req.labels = labels
        try:
            uc = rospy.ServiceProxy('python_update_classifiers', PythonUpdateClassifiers)
            res = uc(req)
            return res.success
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e
            return False
