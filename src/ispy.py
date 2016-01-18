#!/usr/bin/env python
__author__ = 'jesse'

import rospkg
import random
import pickle
import IspyAgent
from agent_io import *
from perception_classifiers.srv import *


# rosrun perception_classifiers ispy.py
#   [object_IDs] [num_rounds] [user_id] [iotype=std|file|robot] [agent_to_load]
# start a game of ispy with user_id or with the keyboard/screen
# if user_id provided, agents are pickled so that an aggregator can later extract
# all examples across users for retraining classifiers and performing splits/merges
# is user_id not provided, classifiers are retrained and saved after each game with just single-user data
def main():

    path_to_perception_classifiers = rospkg.RosPack().get_path('perception_classifiers')
    stopwords_fn = os.path.join(path_to_perception_classifiers, 'src', 'stopwords_en.txt')
    path_to_ispy = os.path.join(path_to_perception_classifiers, 'www/')
    path_to_logs = os.path.join(path_to_perception_classifiers, 'logs/')
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
    user_id = None if sys.argv[3] == "None" else sys.argv[3]
    io_type = sys.argv[4]
    if io_type != "std" and io_type != "file" and io_type != "robot":
        sys.exit("Unrecognized 'iotype'; options std|file|robot")
    agent_fn = None if sys.argv[5] == "None" else sys.argv[5]

    log_fn = os.path.join(path_to_logs, str(user_id)+".trans.log")
    f = open(log_fn, 'a')
    f.write("object_IDs:"+str(object_IDs)+"\n")
    f.write("num_rounds:"+str(num_rounds)+"\n")
    f.write("agent_fn:"+str(agent_fn)+"\n")
    f.close()

    print "calling ROSpy init"
    node_name = 'ispy' if user_id is None else 'ispy' + str(user_id)
    rospy.init_node(node_name)

    print "instantiating ispyAgent"
    if agent_fn is not None and os.path.isfile(os.path.join(pp, agent_fn)):
        print "... from file"
        f = open(os.path.join(pp, agent_fn), 'rb')
        A = pickle.load(f)
        A.object_IDs = object_IDs
        A.log_fn = log_fn
        f.close()
        print "... loading perceptual classifiers"
        A.load_classifiers()
    else:
        print "... from scratch"
        A = IspyAgent.IspyAgent(None, object_IDs, stopwords_fn, log_fn=log_fn)

    io = None
    if io_type == "std":
        print "... with input from keyboard and output to screen"
        io = IOStd(log_fn)
    elif io_type == "file":
        print "... with input and output through files"
        io = IOFile(os.path.join(cp, str(user_id)+".get.in"),
                    os.path.join(cp, str(user_id)+".guess.in"),
                    os.path.join(cp, str(user_id)+".say.out"),
                    os.path.join(cp, str(user_id)+".point.out"),
                    log_fn)
    elif io_type == "robot":
        print "... preemptively calling active predicates on objects to cache results"
        active_predicates = [p for p in A.predicates if A.predicate_active[p]]
        _ = A.get_classifier_results(active_predicates, A.object_IDs)
        
        print "... with input and output through embodied robot"
        io = IORobot(os.path.join(cp, str(user_id))+".get.in", log_fn, object_IDs)

    A.io = io

    print "beginning game"
    robot_choices = []
    for rnd in range(0, num_rounds):

        # human turn
        h_utterance, h_cnfs, correct_idx = A.human_take_turn()
        if correct_idx is not None:
            for d in h_cnfs:
                for pred in d:
                    A.update_predicate_data(pred, [[object_IDs[correct_idx], True]])

        # robot turn
        idx_selection = random.randint(0, len(object_IDs)-1)
        while idx_selection in robot_choices:  # ensures objects already picked by robot not picked again
            idx_selection = random.randint(0, len(object_IDs)-1)
        robot_choices.append(idx_selection)
        r_utterance, r_predicates, num_guesses = A.robot_take_turn(idx_selection)
        labels = A.elicit_labels_for_predicates_of_object(idx_selection, r_predicates)
        for idx in range(0, len(r_predicates)):
            A.update_predicate_data(r_predicates[idx], [[object_IDs[idx_selection], labels[idx]]])

    A.io.say("Thanks for playing!")
    A.io = None  # don't want to pickle IO structures, which get re-instantiated through this script on agent load

    f = open(os.path.join(pp, str(user_id)+"_"+"-".join([str(oid) for oid in object_IDs])+".agent"), 'wb')
    pickle.dump(A, f)
    f.close()


if __name__ == "__main__":
        main()
