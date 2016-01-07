#!/usr/bin/env python
__author__ = 'jesse'

import os
import subprocess
import time
import rospy
from perception_classifiers.srv import *


class IspyServer:

    def __init__(self):

        self.path_to_ispy = '/u/jesse/public_html/ispy'

        self.logs_folder = "/u/jesse/catkin_ws/src/perception_classifiers/logs"
        if not os.path.isdir(self.logs_folder):
            os.system("mkdir "+self.logs_folder+" ; chmod 777 "+self.logs_folder)
        self.package = "perception_classifiers"
        self.script = "ispy.py"

        rospy.init_node('ispy_server')
        self.timeout = rospy.get_param('~timeout', 600.0)
        print "initialized node 'ispy_server'"

        self.send_say_server = rospy.Service(
            'get_say', getSay, self.get_say)

        self.send_point_server = rospy.Service(
            'get_point', getPoint, self.get_point)

        self.start_dialog_server = rospy.Service(
            'start_dialog', startDialog, self.start_dialog)
        self.processes = []

    def __exit__(self):
        self._close_all_processes()

    def start(self):
        try:
            print "spinning"
            rospy.spin()
        except rospy.ROSInterrupException:
            pass
        rospy.loginfo("Server thread shutting down...")

    def get_say(self, req):

        # spin on say file until it exists, then read
        res = getSayResponse()
        fn = os.path.join(self.path_to_ispy, 'communications', req.id+".say.out")
        print "waiting for "+fn
        t = 120
        while not os.path.isfile(fn):
            time.sleep(1)
            t -= 1
            if t == 0:
                print "...ERROR: timeout waiting for "+fn
                return "ERROR: timeout"
        f = open(fn, 'r')
        res.s = f.read()
        f.close()
        os.system("rm "+fn)
        print "...returning contents of "+fn+" : '"+str(res.s)+"'"

        return res

    def get_point(self, req):

        # check for and read point file, allowing up to 5 seconds for the write
        res = getPointResponse()
        fn = os.path.join(self.path_to_ispy, 'communications', req.id+".point.out")
        print "checking for "+fn
        t = 10
        while not os.path.isfile(fn) and t > 0:
            time.sleep(1)
            t -= 1
        if os.path.isfile(fn):
            f = open(fn, 'r')
            res.oidx = int(f.read())
            f.close()
            os.system("rm "+fn)
            print "...returning contents of "+fn+" : '"+str(res.oidx)+"'"
        else:
            res.oidx = -2  # code for not changing behavior

        return res

    def start_dialog(self, req):
        std_log = open(os.path.join(self.logs_folder, req.id + '.std.log'), 'w')
        err_log = open(os.path.join(self.logs_folder, req.id + '.err.log'), 'w')
        log = [std_log, err_log]
        rospy.loginfo("  Logs at: " + self.logs_folder + '/' + req.id + "*.log")
        args = [req.object_ids, "1",
                "/u/jesse/catkin_ws/src/perception_classifiers/src/stopwords_en.txt",
                req.id, "True"]
        if req.exp_cond == "clusters":
            args.append("clusters.local.agent")
        elif req.exp_cond == "classifiers":
            args.append("classifiers.local.agent")
        else:
            args.append("control.local.agent")
        process = self.start_rosrun_process(self.package, self.script, args=args, log=log)
        self.processes.append([process, log])
        return startDialogResponse()

    def _close_all_processes(self):
        for process, log in self.processes:
            process.terminate()
            for lf in log:
                lf.close()
        self.processes = []

    def start_rosrun_process(self, package, binary, args=None, log=None):
        if args is None:
            args = []
        if package is not None:
            command_args = ['rosrun', package, binary]
        else:
            command_args = ['rosrun', binary]
        command_args.extend(args)
        print "Running command: " + ' '.join(command_args)
        return (subprocess.Popen(command_args, stdout=log[0], stderr=log[1])
                if log is not None else subprocess.Popen(command_args))


if __name__ == "__main__":
    try:
        server = IspyServer()
        server.start()
    except rospy.ROSInterruptException:
        pass
