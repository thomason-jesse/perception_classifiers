#!/usr/bin/env python
__author__ = 'jesse'

import os
import sys
import time

class InputFromKeyboard:
    def __init__(self):
        pass

    def get(self):
        return raw_input().lower()

    def get_guess(self):
        return int(raw_input())


class OutputToStdout:
    def __init__(self):
        pass

    def say(self, s):
        print "SYSTEM: "+s

    def point(self, idx):
        print "SYSTEM POINTS TO SLOT "+str(idx)


class InputFromFile:
    def __init__(self, get_fn, guess_fn, trans_fn):
        self.get_fn = get_fn
        self.guess_fn = guess_fn
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
        f = open(self.trans_fn, 'a')
        f.write("get:"+str(c)+"\n")
        f.close()

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
        f = open(self.trans_fn, 'a')
        f.write("guess:"+str(idx)+"\n")
        f.close()

        return int(idx)


class OutputToFile:
    def __init__(self, say_fn, point_fn, trans_fn):
        self.say_fn = say_fn
        self.point_fn = point_fn
        self.trans_fn = trans_fn

    def say(self, s):
        f = open(self.say_fn, 'a')
        f.write(s+"\n")
        f.close()
        os.system("chmod 777 "+self.say_fn)
        f = open(self.trans_fn, 'a')
        f.write("say:"+s+"\n")
        f.close()

    def point(self, idx):
        f = open(self.point_fn, 'w')
        f.write(str(idx))
        f.close()
        os.system("chmod 777 "+self.point_fn)
        f = open(self.trans_fn, 'a')
        f.write("point:"+str(idx)+"\n")
        f.close()
