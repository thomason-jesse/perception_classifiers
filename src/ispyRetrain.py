#!/usr/bin/env python
__author__ = 'jesse'

import rospy
import pickle
import IspyAgent
from agent_io import *
from perception_classifiers.srv import *


# python ispyRetrain.py [stopwords_en] [experimental_cond=True/False] [out_fn_prefix] [num_objects] [base_agent]
def main():

    stopwords_fn = sys.argv[1]
    experimental_cond = True if sys.argv[2] == "True" else False
    out_fn_prefix = sys.argv[3]
    num_objects = int(sys.argv[4])
    base_agent = None if sys.argv[5] == "None" else sys.argv[5]

    path_to_ispy = '/u/jesse/public_html/ispy'
    pp = os.path.join(path_to_ispy, "pickles")

    print "calling ROSpy init"
    rospy.init_node('ispy_retrain')

    print "instantiating blank ispyAgent"
    A = IspyAgent.IspyAgent(None, None, None, stopwords_fn)

    print "loading existing perceptual classifiers"
    A.load_classifiers()

    found_agents = False
    prev_dir = str(time.time())+"_previous"

    if base_agent is not None:
        print "loading and unifying base agent"
        pfn = os.path.join(pp, base_agent)
        f = open(pfn, 'rb')
        B = pickle.load(f)
        f.close()
        A.unify_with_agent(B)
        os.system("mv "+pfn+" "+os.path.join(pp, prev_dir, base_agent))
    else:
        B = IspyAgent.IspyAgent(None, None, None, stopwords_fn)

    print "tracing pickles folder to gather data from user agents"
    if not os.path.isdir(os.path.join(pp, prev_dir)):
        os.system("mkdir "+os.path.join(pp, prev_dir))
    for root, dirs, files in os.walk(pp):
        if 'previous' not in root:
            for fn in files:
                if 'local' not in fn.split('.'):
                    pfn = os.path.join(pp, fn)
                    print "...loading and unifying agent '"+pfn+"'"
                    f = open(pfn, 'rb')
                    user_agent = pickle.load(f)
                    f.close()
                    A.unify_with_agent(user_agent)
                    A.subtract_predicate_examples(B.predicate_examples)
                    mv_cmd = "mv "+os.path.join(root, pfn)+" "+os.path.join(root, prev_dir, fn)
                    os.system(mv_cmd)
                    found_agents = True
    if not found_agents:
        sys.exit("ERROR: found no previous agents against which to train")

    print "retraining classifiers from gathered data"
    A.retrain_predicate_classifiers()

    if experimental_cond:
        print "detecting synonymy and polysemy across and within attributes"
        A.refactor_predicates(num_objects)
    else:
        print "skipping synonymy and polysemy detection"

    print "saving perceptual classifiers to file"
    A.save_classifiers()

    print "pickling ispyAgent"
    f = open(os.path.join(pp, out_fn_prefix+".local.agent"), 'wb')
    pickle.dump(A, f)
    f.close()


if __name__ == "__main__":
        main()
