#!/usr/bin/env python
__author__ = 'jesse'

import rospkg
import rospy
import pickle, sys
import IspyAgent
from agent_io import *
from perception_classifiers.srv import *


# python ispyRetrain.py [experimental_cond=control/classifiers/clusters] [out_fn_prefix] [num_objects] [base_agent]
def main():

    experimental_cond = sys.argv[1]
    out_fn_prefix = sys.argv[2]
    num_objects = int(sys.argv[3])
    base_agent = None if sys.argv[4] == "None" else sys.argv[4]

    if experimental_cond != "control" and experimental_cond != "classifiers" and experimental_cond != "clusters":
        sys.exit("Unrecognized experimental condition")

    path_to_perception_classifiers = rospkg.RosPack().get_path('perception_classifiers')
    stopwords_fn = os.path.join(path_to_perception_classifiers, 'src', 'stopwords_en.txt')
    path_to_ispy = os.path.join(path_to_perception_classifiers, 'www/')
    pp = os.path.join(path_to_ispy, "pickles")

    print "calling ROSpy init"
    rospy.init_node('ispy_retrain')

    print "instantiating blank ispyAgent"
    A = IspyAgent.IspyAgent(None, None, stopwords_fn)

    print "loading existing perceptual classifiers"
    A.load_classifiers()

    found_agents = False
    if base_agent is None:
        prev_dir = str(time.time())+"_previous"
    else:
        prev_dir = str(base_agent)+"_"+str(time.time())+"_previous"
    if not os.path.isdir(os.path.join(pp, prev_dir)):
        os.system("mkdir "+os.path.join(pp, prev_dir))

    if base_agent is not None:
        print "loading and unifying base agent"
        pfn = os.path.join(pp, base_agent)
        f = open(pfn, 'rb')
        B = pickle.load(f)
        f.close()
        A.unify_with_agent(B)
        os.system("mv "+pfn+" "+os.path.join(pp, prev_dir, base_agent))
    else:
        B = IspyAgent.IspyAgent(None, None, stopwords_fn)

    print "tracing pickles folder to gather data from user agents"
    for root, dirs, files in os.walk(pp):
        if 'previous' not in root:
            for fn in files:
                if 'local' not in fn.split('.') and (base_agent is None or fn[:len(out_fn_prefix)] == str(out_fn_prefix)):
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

    if experimental_cond != "control":
        print "detecting synonymy and polysemy across and within attributes using "+str(experimental_cond)
        A.refactor_predicates(num_objects, experimental_cond, 1)
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
