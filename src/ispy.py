#!/usr/bin/env python
__author__ = 'jesse'

import rospy
import sys
import os
import random
import time
import pickle
import IspyAgent

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
    def __init__(self, fn):
        self.fn = fn

    def get(self):
        while not os.path.isfile(self.fn):
            time.sleep(1)
        f = open(self.fn, 'r')
        c = f.read()
        f.close()
        os.system("rm "+self.fn)
        return c


class OutputToFile:
    def __init__(self, fn):
        self.fn = fn

    def say(self, s):
        f = open(self.fn, 'w')
        f.write(s)
        f.close()


# rosrun nlu_pipeline ispy.py [object_IDs] [num_rounds] [stopwords_fn] [user_id] [simulation=True/False]
# start a game of ispy with user_id or with the keyboard/screen
# if user_id provided, agents are pickled so that an aggregator can later extract
# all examples across users for retraining classifiers and performing splits/merges
# is user_id not provided, classifiers are retrained and saved after each game with just single-user data
def main():

    fp = "src/perception_classifiers/src/pickles"
    if not os.path.isdir(fp):
        os.system("mkdir "+fp)
    cp = "src/perception_classifiers/src/communications"
    if not os.path.isdir(cp):
        os.system("mkdir "+cp)

    object_IDs = [int(oid) for oid in sys.argv[1].split(',')]
    num_rounds = int(sys.argv[2])
    stopwords_fn = sys.argv[3]
    user_id = sys.argv[4] if sys.argv[4] != "None" else None
    simulation = True if sys.argv[5] == "True" else False

    print "calling ROSpy init"
    node_name = 'ispy' if user_id is None else 'ispy' + str(user_id)
    rospy.init_node(node_name)

    print "instantiating ispyAgent"
    if os.path.isfile(os.path.join(fp, "local.agent")):
        print "... from file"
        f = open(os.path.join(fp, "local.agent"), 'rb')
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
        u_in = InputFromFile(os.path.join(cp, user_id+".in"))
        u_out = OutputToFile(os.path.join(cp, user_id+".out"))
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
        idx_selection = random.randint(0, len(object_IDs)-1)
        r_utterance, r_predicates, num_guesses = A.robot_take_turn(idx_selection)
        labels = A.elicit_labels_for_predicates_of_object(idx_selection, r_predicates)
        for idx in range(0, len(r_predicates)):
            A.update_predicate_data(r_predicates[idx], [[object_IDs[correct_idx], labels[idx]]])

    if user_id is None:
        print "retraining classifiers from gathered data"
        A.retrain_predicate_classifiers()

        print "detecting synonymy and polysemy across and within attributes"
        A.refactor_predicates()

        print "saving perceptual classifiers to file"
        A.save_classifiers()

        print "pickling ispyAgent"
        f = open(os.path.join(fp, "local.agent"), 'wb')
        pickle.dump(A, f)
        f.close()

    else:
        f = open(os.path.join(fp, user_id+".agent"), 'wb')
        pickle.dump(A, f)
        f.close()


if __name__ == "__main__":
        main()
