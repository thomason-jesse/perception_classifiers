#!/usr/bin/env python
__author__ = 'jesse'

import rospy
import sys
import os
import random
import time
import pickle
import IspyAgent
from perception_classifiers.srv import *

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
    def __init__(self, get_fn, guess_fn):
        self.get_fn = get_fn
        self.guess_fn = guess_fn

    def get(self):
        print "waiting for "+self.get_fn
        while not os.path.isfile(self.get_fn):
            time.sleep(1)
        f = open(self.get_fn, 'r')
        c = f.read()
        f.close()
        os.system("rm -f "+self.get_fn)
        print "...returning contents of "+self.get_fn+" : '"+str(c)+"'"
        return c

    def get_guess(self):
        print "waiting for "+self.guess_fn
        while not os.path.isfile(self.guess_fn):
            time.sleep(1)
        f = open(self.guess_fn, 'r')
        idx = f.read()
        f.close()
        os.system("rm -f "+self.guess_fn)
        print "...returning contents of "+self.guess_fn+" : '"+str(idx)+"'"
        return int(idx)


class OutputToFile:
    def __init__(self, say_fn, point_fn, uid):
        self.say_fn = say_fn
        self.point_fn = point_fn
        self.uid = uid

    def say(self, s):
        f = open(self.say_fn, 'a')
        f.write(s+"\n")
        f.close()
        os.system("chmod 777 "+self.say_fn)

    def point(self, idx):
        f = open(self.point_fn, 'w')
        f.write(str(idx))
        f.close()
        os.system("chmod 777 "+self.point_fn)


# rosrun nlu_pipeline ispy.py [object_IDs] [num_rounds] [stopwords_fn] [user_id] [simulation=True/False]
# start a game of ispy with user_id or with the keyboard/screen
# if user_id provided, agents are pickled so that an aggregator can later extract
# all examples across users for retraining classifiers and performing splits/merges
# is user_id not provided, classifiers are retrained and saved after each game with just single-user data
def main():

    path_to_ispy = '/u/jesse/public_html/ispy'
    pp = os.path.join(path_to_ispy, "pickles")
    if not os.path.isdir(pp):
        os.system("mkdir "+pp)
        os.system("chmod 777 "+pp)
    cp = os.path.join(path_to_ispy, "communications")
    if not os.path.isdir(cp):
        os.system("mkdir "+cp)
        os.system("chmod 777 "+cp)

    object_IDs = [int(oid) for oid in sys.argv[1].split(',')]
    num_rounds = int(sys.argv[2])
    stopwords_fn = sys.argv[3]
    user_id = sys.argv[4] if sys.argv[4] != "None" else None
    simulation = True if sys.argv[5] == "True" else False

    print "calling ROSpy init"
    node_name = 'ispy' if user_id is None else 'ispy' + str(user_id)
    rospy.init_node(node_name)

    print "instantiating ispyAgent"
    if os.path.isfile(os.path.join(pp, "local.agent")):
        print "... from file"
        f = open(os.path.join(pp, "local.agent"), 'rb')
        A = pickle.load(f)
        A.object_IDs = object_IDs
        f.close()
        print "... loading perceptual classifiers"
        A.load_classifiers()
    else:
        A = IspyAgent.IspyAgent(None, None, object_IDs, stopwords_fn)
    if user_id is None:
        u_in = InputFromKeyboard()
        u_out = OutputToStdout()
    else:
        u_in = InputFromFile(os.path.join(cp, user_id+".get.in"), os.path.join(cp, user_id+".guess.in"))
        u_out = OutputToFile(os.path.join(cp, user_id+".say.out"), os.path.join(cp, user_id+".point.out"), user_id)
    A.u_in = u_in
    A.u_out = u_out
    A.simulation = simulation

    print "beginning game"
    for rnd in range(0, num_rounds):

        # human turn
        h_utterance, h_cnfs, correct_idx = A.human_take_turn()
        if correct_idx is not None:
            for d in h_cnfs:
                for pred in d:
                    A.update_predicate_data(pred, [[object_IDs[correct_idx], True]])

        # robot turn
        idx_selection = correct_idx
        while idx_selection == correct_idx:
            idx_selection = random.randint(0, len(object_IDs)-1)
        r_utterance, r_predicates, num_guesses = A.robot_take_turn(idx_selection)
        labels = A.elicit_labels_for_predicates_of_object(idx_selection, r_predicates)
        for idx in range(0, len(r_predicates)):
            A.update_predicate_data(r_predicates[idx], [[object_IDs[correct_idx], labels[idx]]])
    A.u_out.say("Thanks for playing!")

    if user_id is None:
        print "retraining classifiers from gathered data"
        A.retrain_predicate_classifiers()

        print "detecting synonymy and polysemy across and within attributes"
        A.refactor_predicates()

        print "saving perceptual classifiers to file"
        A.save_classifiers()

        print "pickling ispyAgent"
        f = open(os.path.join(pp, "local.agent"), 'wb')
        pickle.dump(A, f)
        f.close()

    else:
        f = open(os.path.join(pp, user_id+".agent"), 'wb')
        pickle.dump(A, f)
        f.close()


if __name__ == "__main__":
        main()
