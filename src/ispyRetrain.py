#!/usr/bin/env python
__author__ = 'jesse'

import rospy
import os
import sys
import pickle
import IspyAgent


# python ispyRetrain.py [experimental_cond=True/False] [out_fn_prefix]
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

    if sys.argv[1] == "True":
        print "detecting synonymy and polysemy across and within attributes"
        A.refactor_predicates()
    else:
        print "skipping synonymy and polysemy detection"

    print "saving perceptual classifiers to file"
    A.save_classifiers()

    print "pickling ispyAgent"
    f = open(os.path.join(fp, sys.argv[2]+".local.agent"), 'wb')
    pickle.dump(A, f)
    f.close()


if __name__ == "__main__":
        main()
