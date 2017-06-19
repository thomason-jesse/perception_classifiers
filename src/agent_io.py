#!/usr/bin/env python
__author__ = 'jesse'

import os
import string
import time
import rospy
from bwi_speech_services.srv import *
from segbot_arm_perception.srv import *
from segbot_arm_manipulation.srv import *
from std_srvs.srv import *
import roslib
roslib.load_manifest('sound_play')
from sound_play.libsoundplay import SoundClient


def append_to_file(s, fn):
    f = open(fn, 'a')
    f.write(s)
    f.close()

vowels = ['a', 'e', 'i', 'o', 'u']
secs_per_vowel = 0.4
speech_sec_buffer = 1


class IOStd:
    def __init__(self, trans_fn):
        self.trans_fn = trans_fn

    def get(self):
        uin = raw_input().lower()
        append_to_file("get:"+str(uin)+"\n", self.trans_fn)
        return uin

    def get_touch(self, block_until_prompted=False):
        if block_until_prompted:
            _ = self.get()
        uin = int(raw_input())
        append_to_file("touch:"+str(uin)+"\n", self.trans_fn)
        return uin

    def say(self, s):
        append_to_file("say:"+str(s)+"\n", self.trans_fn)
        print "SYSTEM: "+s

    def point(self, idx):
        append_to_file("point:"+str(idx)+"\n", self.trans_fn)
        print "SYSTEM POINTS TO SLOT "+str(idx)

    def face_table(self, tid, _, report=False):
        append_to_file("face:" + str(tid) + "\n", self.trans_fn)
        print "SYSTEM TURNS TO TABLE " + str(tid)
        return True


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

    def get_guess(self, block_until_prompted=False):

        if block_until_prompted:
            _ = self.get()

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

    def __init__(self, get_fn, trans_fn, oidxs):
        self.get_fn = get_fn
        self.trans_fn = trans_fn
        self.oidxs = oidxs
        self.last_say = None

        # initialize a sound client instance for TTS
        print "IORobot: initializing SoundClient..."
        self.sound_client = SoundClient()
        rospy.sleep(1)
        self.sound_client.stopAll()
        print "IORobot: ... done"
        
        rospy.wait_for_service('tabletop_object_detection_service')
        self.tabletop_object_detection_service = rospy.ServiceProxy('tabletop_object_detection_service', TabletopPerception, persistent=True)

        print "IORobot: getting initial pointclouds..."
        self.pointCloud2_plane, self.cloud_plane_coef, self.pointCloud2_objects = self.obtain_table_objects()
        print "IORobot: ... done"

    # Listen for speech from user.
    def get(self):
        self.listening_mode_toggle_client()
        uin = self.sound_transcript_client()
        uin = uin.lower()  # lowercase
        uin = uin.translate(None, string.punctuation)  # remove punctuation
        uin = uin.strip()  # strip any leading or trailing whitespace
        print "agent_io: get returning '" + uin + "'"
        self.listening_mode_toggle_client()
        append_to_file("get:"+str(uin)+"\n", self.trans_fn)
        return uin

    # get touches by detecting human touches on top of objects
    def get_touch(self, log=True):
        idx = self.detect_touch_client()
        if log:
            append_to_file("touch:"+str(idx)+"\n", self.trans_fn)
        return int(idx)

    # use built-in ROS sound client to do TTS
    def say(self, s, log=True, voice='voice_cmu_us_slt_arctic_clunits'):

        if self.last_say is None:
            self.last_say = s
        else:
            self.last_say += " " + s

        if log:
            append_to_file("say:"+str(s)+"\n", self.trans_fn)

        self.sound_client.say(str(s), voice=voice)
        print "SYSTEM: "+s
        rospy.sleep(int(secs_per_vowel*len([v for v in s if v in vowels]) + 0.5 + speech_sec_buffer))

    # point using the robot arm
    def point(self, idx, log=True):
        if log:
            append_to_file("point:"+str(idx)+"\n", self.trans_fn)
        self.touch_client(idx)

    # Rotate the chassis and establish new objects in line of sight.
    def face_table(self, tid, new_oidxs, log=True, report=False):
        if log:
            append_to_file("face:" + str(tid) + "\n", self.trans_fn)
        if report:
            self.say("I am turning to face table " + str(tid) + ".")
        s = self.face_table_client(tid)
        if report:
            self.say("I am getting the objects on the table into focus.")
        self.pointCloud2_plane, self.cloud_plane_coef, self.pointCloud2_objects = self.obtain_table_objects()
        self.oidxs = new_oidxs
        if report:
            self.say("Okay, I see them.")
        return s

    # get the point cloud objects on the table for pointing / recognizing touches
    def obtain_table_objects(self):
        pointCloud2_plane = cloud_plane_coef = pointCloud2_objects = None
        focus = False
        while not focus:
            tries = 5
            while tries > 0:
                pointCloud2_plane, cloud_plane_coef, pointCloud2_objects = self.get_pointCloud2_objects()
                if len(pointCloud2_objects) == len(self.oidxs):
                    focus = True
                    break
                tries -= 1
                rospy.sleep(1)
            if tries == 0 and not focus:
                self.say("I am having trouble focusing on the objects. The operator will adjust them.")
                rospy.sleep(10)
        return pointCloud2_plane, cloud_plane_coef, pointCloud2_objects

    # get PointCloud2 objects from service
    def get_pointCloud2_objects(self):

        # query to get the blobs on the table
        req = TabletopPerceptionRequest()
        req.apply_x_box_filter = True  # limit field of view to table in front of robot
        req.x_min = -0.25
        req.x_max = 0.8
        try:
            res = self.tabletop_object_detection_service(req)

            if len(res.cloud_clusters) == 0:
                return [], [], []

            # re-index clusters so order matches left-to-right indexing expected
            ordered_cloud_clusters = self.reorder_client("x", True)

            return res.cloud_plane, res.cloud_plane_coef, ordered_cloud_clusters
        except rospy.ServiceException, e:
            sys.exit("Service call failed: %s " % e)

    # Turn on or off the indicator behavior for listening for speech.
    def listening_mode_toggle_client(self):
        rospy.wait_for_service('ispy/listening_mode_toggle')
        try:
            listen_toggle = rospy.ServiceProxy('ispy/listening_mode_toggle', Empty)
            listen_toggle()
        except rospy.ServiceException, e:
            sys.exit("Service call failed: %s " % e)

    # Listen for speech, transcribe it, and return it.
    def sound_transcript_client(self):
        # print "<enter speech text>"  # DEBUG - until snowball is working
        # return raw_input()  # DEBUG - until snowball is working

        rospy.wait_for_service('sound_transcript_server')
        try:
            transcribe = rospy.ServiceProxy('sound_transcript_server', RequestSoundTranscript)
            resp = transcribe()
            if not resp.isGood:
                return ''
            return resp.utterance
        except rospy.ServiceException, e:
            print "Service call failed: %s " % e
            return ''

    # Turn in place to face a new table.
    def face_table_client(self, tid):
        req = iSpyFaceTableRequest()
        req.table_index = tid
        rospy.wait_for_service('ispy/face_table')
        try:
            face = rospy.ServiceProxy('ispy/face_table', iSpyFaceTable)
            res = face(req)
            return res.success
        except rospy.ServiceException, e:
            sys.exit("Service call failed: %s" % e)

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
