#!/usr/bin/env python
__author__ = 'jesse'

import numpy as np
from argparse import ArgumentParser
from scipy.stats import ttest_ind


def main(args):

    data = {"uid": [],
            "fold": [],
            "condition": [],
            "nb_q_init": [],
            "nb_q_yn": [],
            "nb_q_ex": [],
            "nb_g": [],
            "correct": [],
            "predicates_used": [],
            "understand": [],
            "prod_yn": [],
            "prod_ex": [],
            "long": [],
            "slow": [],
            "qs": [],
            "fun": [],
            "use": []}

    # Populates the data structure with uid. Any user for which we don't have log data will be excluded
    # (e.g. if they only have survey data).
    with open(args.log_summary, 'r') as f:
        lines = f.readlines()
        headers = lines[0].strip().split(',')
        for line in lines[1:]:
            ps = line.strip().split(',')
            uid = int(ps[headers.index('uid')])
            for d in ["fold", "condition", "nb_q_init", "nb_q_yn", "nb_q_ex", "nb_g", "correct"]:
                if uid not in data["uid"]:
                    data[d].append(float(ps[headers.index(d)]))
                else:
                    data[d][data["uid"].index(uid)] = (data[d][data["uid"].index(uid)] +
                                                       float(ps[headers.index(d)])) / 2.
            if uid not in data["uid"]:
                for d in ["understand", "prod_yn", "prod_ex", "long", "slow", "qs", "fun", "use"]:
                    data[d].append(None)  # will be filled in when we read the next file
                data["predicates_used"].append(ps[headers.index("predicates_used")].split('_'))
                data["uid"].append(uid)
            else:
                data["predicates_used"][data["uid"].index(uid)].extend(ps[headers.index("predicates_used")].split('_'))

    unseen_uids = data["uid"][:]
    with open(args.survey_responses, 'r') as f:
        lines = f.readlines()
        headers = lines[0].strip().split(',')
        for line in lines[1:]:
            ps = line.strip().split(',')
            uid = int(ps[headers.index('uid')])
            if uid not in data['uid']:
                print "WARNING: uid " + str(uid) + " present in survey responses but not log summary"
                continue
            else:
                unseen_uids.remove(uid)
            for d in ["understand", "prod_yn", "prod_ex", "long", "slow", "qs", "fun", "use"]:
                data[d][data["uid"].index(uid)] = int(ps[headers.index(d)])
    if len(unseen_uids) > 0:
        print ("WARNING: uid(s) " + ','.join([str(uid) for uid in unseen_uids]) +
               " present in logs but do not have survey responses")

    print data

    # Show predicates introduced per fold + re-used from last folds as per Peter's suggestion.

    # Show average for each value across each fold and condition in tables.
    print "average data values per fold/condition:"
    for d in ["nb_q_init", "nb_q_yn", "nb_q_ex", "nb_g", "correct",
              "understand", "prod_yn", "prod_ex", "long", "slow", "qs", "fun", "use"]:
        print d
        print '\t'.join(['', '1', '2'])
        for fold in range(3):
            avgs = []
            for cond in range(1, 3):
                avgs.append(np.mean([data[d][idx] for idx in range(len(data['uid']))
                                     if data['fold'][idx] == fold and data['condition'][idx] == cond]))
            print '\t'.join([str(fold), str(avgs[0]), str(avgs[1])])
        print '\n'

    # Hypothesis: users won't think the robot asks too many questions in the inquisitive condition.
    # Null: 'qs' is different between cond 1, 2 on average between train folds 0 and 1
    avg_qs = [[data['qs'][idx] for idx in range(len(data['uid']))
               if data['fold'][idx] < 2.0 and data['condition'][idx] == cond]
              for cond in range(1, 3)]
    t, p = ttest_ind(avg_qs[0], avg_qs[1])
    print "users won't think the robot asks too many questions in the inquisitive condition (folds 0,1): "
    print "cond 1 avg qs: " + str(avg_qs[0]) + ", " + str(np.mean(avg_qs[0]))
    print "cond 2 avg qs: " + str(avg_qs[1]) + ", " + str(np.mean(avg_qs[1]))
    print "p: " + str(p) + '\n'

    # Hypothesis: users will find the robot understands them better in inquisitive condition.
    # Null: 'understand' is different between cond 1, 2 in train fold 2 (when policies are same but training diff)
    avg_understand = [[data['understand'][idx] for idx in range(len(data['uid']))
                       if data['fold'][idx] == 2.0 and data['condition'][idx] == cond]
                      for cond in range(1, 3)]
    t, p = ttest_ind(avg_understand[0], avg_understand[1])
    print "users will find the robot understands them better in inquisitive condition (fold 2): "
    print "cond 1 avg understand: " + str(avg_understand[0]) + ", " + str(np.mean(avg_understand[0]))
    print "cond 2 avg understand: " + str(avg_understand[1]) + ", " + str(np.mean(avg_understand[1]))
    print "p: " + str(p) + '\n'

    # Hypothesis: the robot will guess the right object more often in the inquisitive condition.
    # Null: 'correct' is different between cond 1, 2 on average between train folds 0, 1, and 2
    avg_correct = [[data['correct'][idx] for idx in range(len(data['uid']))
                    if data['condition'][idx] == cond] for cond in range(1, 3)]
    t, p = ttest_ind(avg_correct[0], avg_correct[1])
    print "the robot will guess the right object more often in the inquisitive condition (all folds): "
    print "cond 1 avg correct: " + str(avg_correct[0]) + ", " + str(np.mean(avg_correct[0]))
    print "cond 2 avg correct: " + str(avg_correct[1]) + ", " + str(np.mean(avg_correct[1]))
    print "p: " + str(p) + '\n'


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--log_summary', type=str, required=True,
                        help="CSV summarizing information extracted from logs")
    parser.add_argument('--survey_responses', type=str, required=True,
                        help="CSV detailing survey responses by uid")
    cmd_args = parser.parse_args()
    
    main(cmd_args)
