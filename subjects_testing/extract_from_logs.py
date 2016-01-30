#!/usr/bin/env python
__author__ = 'jesse'

import sys
import os
import ast
import operator
import copy


# python extract_from_logs.py input_dir output_fn
def main():

    # get command-line args
    in_dir = sys.argv[1]
    out_fn = sys.argv[2]

    # known info
    fold_user_id_ranges = [range(i*10, (i+1)*10) for i in range(0, 4)]
    fold_user_id_ranges[3].extend([40, 41])  # extra subjects
    identifying_headers = ["user_id", "fold", "cond"]
    data_to_extract = ["avg_rg", "avg_hg", "avg_reg", "avg_rr@1", "avg_hr@1"]

    # gather data from log files
    d_to_write = []
    for cond in ["", "con", "exp"]:
        for fold in range(0, 4):
            for user_id in fold_user_id_ranges[fold]:

                # get log file names
                log_fns = []
                for root, dirs, files in os.walk(in_dir):
                    for f in files:
                        if ((f[:3] == cond and f.split("_")[1] == str(user_id)) or
                                (len(cond) == 0 and f.split("_")[0] == str(user_id))):
                            log_fns.append(os.path.join(root, f))
                if len(log_fns) == 0:
                    continue

                # extract and unify information from each
                d = []
                for fn in log_fns:
                    d.append(extract_data_from_log(fn))
                d_avg = {key: sum([d[i][key] for i in range(0, len(d))])/float(len(d)) for key in d[0]}

                # create and add record
                d_avg['user_id'] = user_id
                d_avg['fold'] = fold
                if len(cond) > 0:
                    d_avg['cond'] = cond
                    d_to_write.append(d_avg)
                else:  # add records for fold 0 twice under each condition
                    for _c in ["con", "exp"]:
                        d_avg['cond'] = _c
                        d_to_write.append(copy.copy(d_avg))

    # write records to out file
    f = open(out_fn, 'w')
    headers = identifying_headers[:]
    headers.extend(data_to_extract)
    f.write(','.join(headers)+'\n')
    for r in d_to_write:
        f.write(','.join([str(r[headers[i]]) for i in range(0, len(headers))])+'\n')
    f.close()


# given a filename, makes a pass to extract data returned as a dictionary
def extract_data_from_log(fn):

    # read in file lines
    f = open(fn, 'r')
    lines = f.readlines()
    f.close()

    # dictionary to return
    d = {}

    # pass through the lines to calculate the average guesses taken by the human and robot
    human_guesses = 0
    human_first_guess = 0
    robot_guesses = 0
    robot_first_guess = 0
    robot_expectation_guesses = 0
    robot_expectation_rewards = {}
    last_guesses = []
    first_guess_worth = -1
    num_rounds = 0
    for l_idx in range(0, len(lines)):
        line = lines[l_idx]

        if len(line) == 0:
            continue
        p = line.strip().split(':')

        # zero everything if the game started over
        if p[0] == "object_IDs":
            human_guesses = 0
            human_first_guess = 0
            robot_guesses = 0
            robot_first_guess = 0
            robot_expectation_guesses = 0
            robot_expectation_rewards = {}
            last_guesses = []
            first_guess_worth = -1
            num_rounds = 0

        # record number of rounds
        elif p[0] == "num_rounds":
            num_rounds = int(p[1])

        # count robot guesses
        elif p[0] == "say" and p[1] == "Is this the object you have in mind?":
            robot_guesses += 1
            if first_guess_worth == -1:
                first_guess_worth = robot_expectation_rewards[int(lines[l_idx-1].strip().split(':')[1])]
            if lines[l_idx+2].strip().split(':')[1] == "-1":  # this is the correct guess, so add expected reward
                correct_idx = int(lines[l_idx-1].strip().split(':')[1])
                robot_expectation_guesses += robot_expectation_rewards[correct_idx]
                # in set of first guess, so get a 1st guess reward based on expected average position
                if robot_expectation_rewards[correct_idx] == first_guess_worth:
                    if robot_expectation_rewards[correct_idx] == 2.5:
                        robot_first_guess += 0.25
                    elif robot_expectation_rewards[correct_idx] == 2:
                        robot_first_guess += 0.333333333333
                    elif robot_expectation_rewards[correct_idx] == 1.5:
                        robot_first_guess += 0.5
                    else:
                        robot_first_guess += 1
                first_guess_worth = -1

        # count human guesses, discarding guessing the same object twice in a row
        elif p[0] == "guess" and p[1] != "None" and int(p[1]) not in last_guesses:
            human_guesses += 1
            last_guesses.append(int(p[1]))
            if lines[l_idx+2].strip().split(':')[1] == p[1]:  # this was correct guess
                if human_guesses == 1:
                    human_first_guess += 1
                last_guesses = []

        # calculate net reward based on match scores to remove noise of guessing randomly
        elif p[0] == "match_scores":
            ms = ast.literal_eval(":".join(p[1:]))
            mss = sorted(ms.items(), key=operator.itemgetter(1), reverse=True)
            t = 1
            while len(mss) > 0:
                k, v = mss.pop(0)
                robot_expectation_rewards[k] = t
                n = 1
                o = []
                t += 1
                while len(mss) > 0 and mss[0][1] == v:
                    kn, vn = mss.pop(0)
                    robot_expectation_rewards[k] += t
                    t += 1
                    n += 1
                    o.append(kn)
                robot_expectation_rewards[k] /= float(n)
                for kn in o:
                    robot_expectation_rewards[kn] = robot_expectation_rewards[k]

    # calculate averages
    d["avg_rg"] = robot_guesses / float(num_rounds)
    d["avg_hg"] = human_guesses / float(num_rounds)
    d["avg_reg"] = robot_expectation_guesses / float(num_rounds)
    d["avg_rr@1"] = robot_first_guess / float(num_rounds)
    d["avg_hr@1"] = human_first_guess / float(num_rounds)

    return d


if __name__ == "__main__":
        main()
