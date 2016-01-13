#!/usr/bin/env python
__author__ = 'jesse'

import os
import sys
import time
import math
import rospy
from segbot_arm_perception.srv import *
from segbot_arm_manipulation.srv import *
import roslib
roslib.load_manifest('sound_play')
from sound_play.libsoundplay import SoundClient


def append_to_file(s, fn):
    f = open(fn, 'a')
    f.write(s)
    f.close()


class IOStd:
    def __init__(self, trans_fn):
        self.trans_fn = trans_fn

    def get(self):
        uin = raw_input().lower()
        append_to_file("get:"+str(uin)+"\n", self.trans_fn)
        return uin

    def get_guess(self):
        uin = int(raw_input())
        append_to_file("guess:"+str(uin)+"\n", self.trans_fn)
        return uin

    def say(self, s):
        append_to_file("say:"+str(s)+"\n", self.trans_fn)
        print "SYSTEM: "+s

    def point(self, idx):
        append_to_file("point:"+str(idx)+"\n", self.trans_fn)
        print "SYSTEM POINTS TO SLOT "+str(idx)


class IOFile:
    def __init__(self, get_fn, guess_fn, say_fn, point_fn, trans_fn):
        self.get_fn = get_fn
        self.guess_fn = guess_fn
        self.say_fn = say_fn
        self.point_fn = point_fn
        self.trans_fn = trans_fn

    def get(self):

        # spin until input get exists, then read
        print "waiting for "+self.get_fn
        t = 60*60
        while not os.path.isfile(self.get_fn):
            time.sleep(1)
            t -= 1
            if t == 0:
                print "... FATAL: timed out waiting for "+self.get_fn
                sys.exit()
        f = open(self.get_fn, 'r')
        c = f.read()
        f.close()
        os.system("rm -f "+self.get_fn)
        print "...returning contents of "+self.get_fn+" : '"+str(c)+"'"

        # log gotten get
        append_to_file("get:"+str(c)+"\n", self.trans_fn)

        return c

    def get_guess(self):

        # spin until guess exists, then read
        print "waiting for "+self.guess_fn
        t = 60*60
        while not os.path.isfile(self.guess_fn):
            time.sleep(1)
            t -= 1
            if t == 0:
                print "... FATAL: timed out waiting for "+self.guess_fn
                sys.exit()
        f = open(self.guess_fn, 'r')
        idx = f.read()
        f.close()
        os.system("rm -f "+self.guess_fn)
        print "...returning contents of "+self.guess_fn+" : '"+str(idx)+"'"

        # log gotten guess
        append_to_file("guess:"+str(idx)+"\n", self.trans_fn)

        return int(idx)

    def say(self, s):
        f = open(self.say_fn, 'a')
        f.write(s+"\n")
        f.close()
        os.system("chmod 777 "+self.say_fn)
        append_to_file("say:"+s+"\n", self.trans_fn)

    def point(self, idx):
        f = open(self.point_fn, 'w')
        f.write(str(idx))
        f.close()
        os.system("chmod 777 "+self.point_fn)
        append_to_file("point:"+str(idx)+"\n", self.trans_fn)


class IORobot:

    def __init__(self, get_fn, trans_fn, object_IDs):
        self.get_fn = get_fn
        self.trans_fn = trans_fn
        self.object_IDs = object_IDs

        # get the point cloud objects on the table for pointing / recognizing touches
        tries = 10
        while tries > 0:
            self.pointCloud2_plane, self.cloud_plane_coef, self.pointCloud2_objects = self.get_pointCloud2_objects()
            if len(self.pointCloud2_objects) == len(self.object_IDs):
                break
            tries -= 1
        if tries == 0:
            sys.exit("ERROR: "+str(len(self.pointCloud2_objects))+" PointCloud2 objects detected " +
                     "while "+str(len(self.object_IDs))+" objects were expected")

        # initialize a sound client instance for TTS
        self.sound_client = SoundClient()
        rospy.sleep(1)
        self.sound_client.stopAll()

        # have operator interaction to confirm ordering of objects is correct, terminate if it isn't
        print "touching objects from left-most to right-most... please watch and confirm detection and order"
        for i in range(0, len(object_IDs)):
            print "... touching object in position "+str(i)
            self.point(i, log=False)
            rospy.sleep(2)
            self.point(-1, log=False)
            rospy.sleep(2)
        op_resp = None
        while op_resp != "Y" and op_resp != "N":
            print "confirm detection and ordering[Y/N]:"
            op_resp = raw_input()
            if op_resp == "N":
                sys.exit("Try to fix my detection and try again.")
        self.point(-1, log=False)
        
        # have open-ended operator interaction to confirm detection of touches is working
        op_resp = None
        while op_resp != "Y" and op_resp != "N":
            print "detect a new touch?[Y/N]:"
            op_resp = raw_input()
            if op_resp == "Y":
                print "...waiting to see what you point to"
                t_idx = self.get_guess(log=False, block_until_prompted=False)
                if t_idx == -1:
                    print "...no touch detected"
                else:
                    print "...touching at detected position "+str(t_idx)
                    self.point(t_idx, log=False)
            self.point(-1, log=False)

    # for now, default to IOFile behavior, but might eventually do ASR instead
    def get(self, log=True):

        # spin until input get exists, then read
        print "waiting for "+self.get_fn
        t = 60*60
        while not os.path.isfile(self.get_fn):
            time.sleep(1)
            t -= 1
            if t == 0:
                print "... FATAL: timed out waiting for "+self.get_fn
                sys.exit()
        f = open(self.get_fn, 'r')
        c = f.read()
        f.close()
        os.system("rm -f "+self.get_fn)

        # log gotten get
        append_to_file("get:"+str(c)+"\n", self.trans_fn)

        return c

    # get guesses by detecting human touches on top of objects
    def get_guess(self, log=True, block_until_prompted=True):
        if block_until_prompted:
            operator_okay = self.get(log=False)
        idx = self.detect_touch_client()
        append_to_file("guess:"+str(idx)+"\n", self.trans_fn)
        return int(idx)

    # use built-in ROS sound client to do TTS
    def say(self, s, log=True):
        append_to_file("say:"+str(s)+"\n", self.trans_fn)
        self.sound_client.voiceSound(str(s)).play()
        rospy.sleep(int(math.sqrt(len(str(s).split()))))

    # point using the robot arm
    def point(self, idx, log=True):
        append_to_file("point:"+str(idx)+"\n", self.trans_fn)
        self.touch_client(idx)

    # get PointCloud2 objects from service
    def get_pointCloud2_objects(self):

        # query to get the blobs on the table
        req = TabletopPerceptionRequest()
        rospy.wait_for_service('tabletop_object_detection_service')
        try:
            tabletop_object_detection_service = rospy.ServiceProxy(
                'tabletop_object_detection_service', TabletopPerception)
            res = tabletop_object_detection_service(req)

            if len(res.cloud_clusters) == 0:
                sys.exit("ERROR: no objects detected")

            # re-index clusters so order matches left-to-right indexing expected
            ordered_cloud_clusters = self.reorder_client("x", True)

            return res.cloud_plane, res.cloud_plane_coef, ordered_cloud_clusters
        except rospy.ServiceException, e:
            sys.exit("Service call failed: %s " % e)

    # reorder PointCloud2 objects returned in arbitrary order from table detection
    def reorder_client(self, coord, forward):
        req = TabletopReorderRequest()
        req.coord = coord
        req.forward = forward
        rospy.wait_for_service('tabletop_object_reorder_service')
        try:
            reorder = rospy.ServiceProxy('tabletop_object_reorder_service', TabletopReorder)
            res = reorder(req)
            return res.ordered_cloud_clusters
        except rospy.ServiceException, e:
            sys.exit("Service call failed: %s " % e)

    # use the arm to touch an object
    def touch_client(self, idx):
        req = iSpyTouchRequest()
        req.cloud_plane = self.pointCloud2_plane
        req.cloud_plane_coef = self.cloud_plane_coef
        req.objects = self.pointCloud2_objects
        req.touch_index = idx
        rospy.wait_for_service('ispy/touch_object_service')
        try:
            touch = rospy.ServiceProxy('ispy/touch_object_service', iSpyTouch)
            res = touch(req)
            return res.success
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e

    # detect a touch above an object
    def detect_touch_client(self):
        req = iSpyDetectTouchRequest()
        req.cloud_plane = self.pointCloud2_plane
        req.cloud_plane_coef = self.cloud_plane_coef
        req.objects = self.pointCloud2_objects
        rospy.wait_for_service('ispy/human_detect_touch_object_service')
        try:
            detect_touch = rospy.ServiceProxy('ispy/human_detect_touch_object_service', iSpyDetectTouch)
            res = detect_touch(req)
            return res.detected_touch_index
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e
