#!/usr/bin/env python
__author__ = 'jesse'

import rospy
from perception_classifiers.srv import *
from std_srvs.srv import *
import operator
import math
import random
import cv2
import numpy


class ORAgent:

    def __init__(self, io, table_oidxs):

        self.io = io
        self.table_oidxs = table_oidxs
