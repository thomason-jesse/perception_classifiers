#!/usr/bin/env python
__author__ = 'jesse'

from argparse import ArgumentParser


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
                    data[d].append(int(ps[headers.index(d)]))
                else:
                    data[d][data["uid"].index(uid)] = (data[d][data["uid"].index(uid)] + int(ps[headers.index(d)])) / 2.
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

    # TODO: suite of statistical tests related to hypotheses when there's enough data to sort it out


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--log_summary', type=str, required=True,
                        help="CSV summarizing information extracted from logs")
    parser.add_argument('--survey_responses', type=str, required=True,
                        help="CSV detailing survey responses by uid")
    cmd_args = parser.parse_args()
    
    main(cmd_args)
