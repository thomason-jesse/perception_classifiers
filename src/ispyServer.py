#!/usr/bin/env python
__author__ = 'jesse'

import sys
import os
import subprocess
import roslib
import rospy

from perception_classifiers.srv import *

class IspyServer():

	def __init__(self):

		self.logs_folder = "logs"
		if not os.path.isdir(self.logs_folder):
			os.system("mkdir logs ; chmod 777 logs")
		self.package = "perception_classifiers"
		self.script = "ispy.py"

		rospy.init_node('ispy_server')
		self.timeout = rospy.get_param('~timeout', 600.0)
		print "initialized node 'ispy_server'"

		self.start_dialog_server = rospy.Service(
			'start_dialog', startDialog,
			self.start_dialog)
		self.processes = []

	def __exit__(self):
		self.shutdown()

	def start(self):
		try:
			print "spinning"
			rospy.spin()
		except rospy.ROSInterrupException:
			pass
		rospy.loginfo("Server thread shutting down...")
		self.shutdown()

	def shutdown(self):
		self._close_all_processes()

	def start_dialog(self, req):
		std_log = open(self.logs_folder + '/' + req.id + '.std.log', 'w')
		err_log = open(self.logs_folder + '/' + req.id + '.err.log', 'w')
		log = [std_log, err_log]
		rospy.loginfo("  Logs at: " + self.logs_folder + '/' + req.id + "*.log")
		args = [req.object_ids, "1",
				"/u/jesse/catkin_ws/src/perception_classifiers/src/stopwords_en.txt",
				req.id, "True"]
		process = self.start_rosrun_process(self.package, self.script, args=args, log=log)
		self.processes.append(process)

	def _close_all_processes(self):
		for process in self.processes:
			elf.stop_roslaunch_process(process)
		self.processes = []

	def start_rosrun_process(self, package, binary, args=None, log=None):
		if args == None:
			args = []
		if package is not None:
			command_args = ['rosrun', package, binary]
		else:
			command_args = ['rosrun', binary]
		command_args.extend(args)
		print "Running command: " + ' '.join(command_args)
		return (subprocess.Popen(command_args, stdout=log[0], stderr=log[1])
				if log != None else subprocess.Popen(command_args))

	def stop_rosrun_process(process):
		process.terminate()

if __name__ == "__main__":
	try:
		server = IspyServer()
		server.start()
	except rospy.ROSInterruptException:
		pass
