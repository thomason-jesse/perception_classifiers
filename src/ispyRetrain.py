#!/usr/bin/env python
__author__ = 'jesse'

import rospy
import os
import sys
import pickle
import IspyAgent


# rosrun nlu_pipeline ispy.py [object_IDs] [num_rounds] [stopwords_fn] [user_id=None]
# start a game of ispy with user_id or with the keyboard/screen
# if user_id provided, agents are pickled so that an aggregator can later extract
# all examples across users for retraining classifiers and performing splits/merges
# is user_id not provided, classifiers are retrained and saved after each game with just single-user data
def main():

    fp = "pickles"

    print "calling ROSpy init"
    rospy.init_node('ispy_retrain')

    print "instantiating blank ispyAgent"
    A = IspyAgent.IspyAgent(None, None, None, None)

    print "tracing pickles folder to gather data from user agents"
    found_agents = False
    for root, dirs, files in os.path.walk(fp):
        for fn in files:
            if fn.split('.')[1] == "agent":
                f = open(fn, 'rb')
                user_agent = pickle.load(f)
                f.close()
                A.unify_with_agent(user_agent)
                os.system("mv "+os.path.join(root, fp)+" "+os.path.join(root, fp+".previous"))
                found_agents = True
    if not found_agents:
        sys.exit("ERROR: found no previous agents against which to train")

    print "retraining classifiers from gathered data"
    A.retrain_predicate_classifiers()

    print "detecting synonymy and polysemy across and within attributes"
    A.refactor_predicates()

    print "saving perceptual classifiers to file"
    A.save_classifiers()

    print "pickling ispyAgent"
    f = open(os.path.join(fp, "local.agent"), 'wb')
    f.dump(A)
    f.close()


if __name__ == "__main__":
        main()
