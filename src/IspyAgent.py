#!/usr/bin/env python
__author__ = 'jesse'

import rospy
from perception_classifiers.srv import *
from std_srvs.srv import *
import operator
import math
import random
import cv2
import numpy


def join_lists(a, b, allow_duplicates=True):
    c = a[:]
    for item in b:
        if item not in c or allow_duplicates:
            c.append(item)
    return c


def join_dicts(a, b, allow_duplicates=True, warn_duplicates=False):
    c = {}
    for key in a:
        if type(a[key]) is list:
            c[key] = a[key][:]
        else:
            c[key] = a[key]
    for key in b:
        if key in c:
            if type(c[key]) is list:
                for item in b[key]:
                    if item not in c[key] or allow_duplicates:
                        c[key].append(item)
                    elif item in c[key] and warn_duplicates:
                        sys.exit("ERROR: join_dicts warn_duplicates collision '" + item + "' already in '" +
                                 str(c[key]) + "' and being added from '" + str(b[key]) + "'")
            elif type(c[key]) is dict:
                c[key] = join_dicts(c[key], b[key],
                                allow_duplicates=allow_duplicates, warn_duplicates=warn_duplicates)
            elif c[key] != b[key]:
                sys.exit("ERROR: join_dicts collision '"+str(key) +
                         "' trying to take values '" + str(c[key]) + "', '" + str(b[key]) + "'")
        else:
            if type(b[key]) is list:
                c[key] = b[key][:]
            else:
                c[key] = b[key]
    return c


class SVM:

    def __init__(self, C=1, gamma=0.5):
        self.params = dict(kernel_type=cv2.SVM_LINEAR,
                      svm_type=cv2.SVM_C_SVC,
                      c=C, gamma=gamma)
        self.model = cv2.SVM()

    def train(self, samples, responses):
        self.model.train(samples, responses, params=self.params)

    def predict(self, samples):
        r = []
        for s in samples:
            d = self.model.predict(s, returnDFVal=False)
            dist = self.model.predict(s, returnDFVal=True)
            r.append([d, dist])
        return r


class IspyAgent:

    def __init__(self, u_in, u_out, object_IDs, stopwords_fn, log_fn=None, alpha=0.9, simulation=False):

        self.u_in = u_in
        self.u_out = u_out
        self.object_IDs = object_IDs
        self.log_fn = log_fn
        self.alpha = alpha
        self.simulation = simulation

        # lists of predicates and words currently known
        self.predicates = []
        self.words = []

        # maps because predicates can be dropped during merge and split operations
        self.word_counts = {}
        self.predicate_active = {}
        self.words_to_predicates = {}
        self.predicates_to_words = {}
        self.predicate_examples = {}
        self.predicate_to_classifier_map = {}
        self.classifier_to_predicate_map = {}
        self.classifier_data_modified = {}

        # get stopwords
        fin = open(stopwords_fn, 'r')
        self.stopwords = []
        for line in fin.readlines():
            self.stopwords.append(line.strip())
        fin.close()

    # invite the human to describe an object, parse the description, and start formulating response strategy
    def human_take_turn(self):

        self.u_out.say("Please pick an object that you see and describe it to me in one sentence.")

        understood = False
        guess_idx = None
        utterance = None
        cnf_clauses = None
        while not understood:

            utterance = self.u_in.get().strip()
            cnf_clauses = self.get_predicate_cnf_clauses_for_utterance(utterance)

            if self.log_fn is not None:
                f = open(self.log_fn, 'a')
                f.write("cnf_clauses:"+str(cnf_clauses)+"\n")
                f.close()

            # extract predicates and run these classifiers against each of objects_IDs to find best match
            if len(cnf_clauses) > 0:
                understood = True

                # get matrix of results and confidences for each object against each predicate in cnfs
                all_predicates = []
                for d in cnf_clauses:
                    all_predicates.extend(d)
                classifier_results = self.get_classifier_results(all_predicates, self.object_IDs)

                # TODO
                # maybe have option to explicitly ask about whether objects have predicates when
                # classification confidence is below some threshold; getting confirmation/denial
                # can update the classifier_results structure as well as add training data before
                # final guessing begins (talk with Jivko about this option)

                # calculate simple best-fit ranking from interpolation of result and confidence
                match_scores = {}
                for p_oidx in range(0, len(self.object_IDs)):
                    oidx = self.object_IDs[p_oidx]
                    object_score = 0
                    for d in cnf_clauses:  # take the maximum score of predicates in disjunction
                        cnf_scores = []
                        for pred in d:
                            cnf_scores.append(classifier_results[oidx][pred][0] *
                                              classifier_results[oidx][pred][1])
                        object_score += max(cnf_scores)
                    match_scores[p_oidx] = object_score
                sorted_match_scores = sorted(match_scores.items(), key=operator.itemgetter(1), reverse=True)

                if self.log_fn is not None:
                    f = open(self.log_fn, 'a')
                    f.write("match_scores:"+str(match_scores)+"\n")
                    f.close()

                # iteratively take best guess
                correct = False
                guesses = [gidx for (gidx, score) in sorted_match_scores]
                while not correct:
                    guess_idx = guesses.pop(0)
                    if self.simulation:
                        self.u_out.point(guess_idx)
                    else:
                        # TODO: call pointing service with guess_idx
                        pass
                    got_confirmation = False
                    while not got_confirmation:
                        got_confirmation = True
                        self.u_out.say("Is this the object you have in mind?")
                        confirmation = self.u_in.get()
                        if confirmation == 'yes':
                            correct = True
                        elif confirmation != 'no':
                            got_confirmation = False
                            self.u_out.say("I didn't catch that.")
                        # TODO: think about adding passive negative training when user says guess was wrong
                    if not correct and len(guesses) == 0:
                        self.u_out.say("I tried them all!")
                        guesses = [gidx for (gidx, score) in sorted_match_scores]
                if self.simulation:
                    self.u_out.point(-1)  # stop pointing
                else:
                    # TODO: call pointing service to retract arm
                    pass

            # utterance failed to parse, so get a new one
            else:
                self.u_out.say("Sorry; I didn't catch that. Could you re-word your description?")

        return utterance, cnf_clauses, guess_idx

    # given object idx, form description of object from classifier results and describe to human, adhering
    # to Gricean maxim of quantity (eg. say as many predicates as needed to discriminate, but not more)
    def robot_take_turn(self, ob_pos):
        ob_idx = self.object_IDs[ob_pos]

        # get results for each attribute for every object
        active_predicates = [p for p in self.predicates if self.predicate_active[p]]
        r = self.get_classifier_results(active_predicates, self.object_IDs)

        # rank the classifiers favoring high confidence on ob_idx with low confidence or negative
        # decisions on other objects
        pred_scores = {}
        pred_confidence = {}
        for pred in active_predicates:
            score = (r[ob_idx][pred][0]*r[ob_idx][pred][1])*len(self.object_IDs)
            for oidx in self.object_IDs:
                if ob_idx == oidx:
                    continue
                score -= r[oidx][pred][0]*r[oidx][pred][1]
            pred_scores[pred] = score
            pred_confidence[pred] = r[ob_idx][pred][1]

        if self.log_fn is not None:
            f = open(self.log_fn, 'a')
            f.write("pred_scores:"+str(pred_scores)+"\n")
            f.close()

        # choose predicates to best describe object
        predicates_chosen = []
        for pred, score in sorted(pred_scores.items(), key=operator.itemgetter(1), reverse=True):
            if score > 0:
                # choose just one predicate word arbitrarily if multiple are associated with classifier
                predicates_chosen.append(pred)
                if len(predicates_chosen) == 3:
                    break  # don't want to overload user with questions afterwards
        if len(predicates_chosen) == 0:  # we have no classifier information yet, so choose 3 arbitrarily
            preds_shuffled = active_predicates[:]
            random.shuffle(preds_shuffled)
            predicates_chosen.extend(preds_shuffled[:3 if len(preds_shuffled) >= 3 else len(preds_shuffled)])

        # choose predicates we are most unsure about to grab labels for during clarification dialog
        lcps_candidates = []
        cc = None
        for pred, conf in sorted(pred_confidence.items(), key=operator.itemgetter(1)):
            if pred in predicates_chosen:
                continue
            if cc is None or conf != cc:
                cc = conf
                if len(lcps_candidates) > 2:
                    break  # don't want to overload the user with questions afterwards
            lcps_candidates.append(pred)
        random.shuffle(lcps_candidates)
        lcps = lcps_candidates[:2]

        if self.log_fn is not None:
            f = open(self.log_fn, 'a')
            f.write("predicates_chosen:"+str(predicates_chosen)+"\n")
            f.write("lcps:"+str(lcps)+"\n")
            f.close()

        # describe object to user
        if len(predicates_chosen) > 2:
            desc = "I am thinking of an object I would describe as " + \
                ', '.join([self.choose_word_for_pred(pred) for pred in predicates_chosen[:-1]]) + \
                ", and "+self.choose_word_for_pred(predicates_chosen[-1]) + "."
        elif len(predicates_chosen) == 2:
            desc = "I am thinking of an object I would describe as " + \
                   self.choose_word_for_pred(predicates_chosen[0]) + \
                " and " + self.choose_word_for_pred(predicates_chosen[1]) + "."
        else:
            desc = "I am thinking of an object I would describe as " + \
                   self.choose_word_for_pred(predicates_chosen[0]) + "."
        self.u_out.say(desc)

        # wait for user to find and select correct object
        num_guesses = 0
        predicates_to_ask = predicates_chosen[:]
        predicates_to_ask.extend(lcps)
        random.shuffle(predicates_to_ask)
        while True:
            if self.simulation:
                guess_idx = self.object_IDs[self.u_in.get_guess()]
            else:
                # TODO: call looking for hand over object service
                guess_idx = None
            num_guesses += 1
            if guess_idx == ob_idx:
                self.u_out.say("That's the one!")
                return desc, predicates_to_ask, num_guesses
            else:
                self.u_out.say("That's not the object I am thinking of.")
                # TODO: think about adding passive positive examples when user thinks a different
                # TODO: object is being described

    # pick the word users use most often to describe the predicate
    def choose_word_for_pred(self, p):
        wc = [self.word_counts[w] for w in self.predicates_to_words[p]]
        return self.predicates_to_words[p][wc.index(max(wc))]

    # point to object at pos_idx and ask whether it meets the attributes of aidx chosen to point it out
    def elicit_labels_for_predicates_of_object(self, pos_idx, preds):
        l = []
        if self.simulation:
            self.u_out.point(pos_idx)
        else:
            # TODO: call pointing service with pos_idx
            pass
        for pred in preds:
            self.u_out.say("Would you use the word '" + self.predicates_to_words[pred][0] +
                           "' when describing this object?")
            got_r = False
            while not got_r:
                got_r = True
                r = self.u_in.get()
                if r == "yes":
                    l.append(True)
                elif r == "no":
                    l.append(False)
                else:
                    got_r = False
                    self.u_out.say("I didn't catch that.")
        if self.simulation:
            self.u_out.point(-1)  # stop pointing
        else:
            # TODO: call pointing service to retract arm
            pass
        return l

    # get results for each perceptual classifier over all objects so that for any given perceptual classifier,
    # objects have locations in concept-dimensional space for that classifier
    # detect classifiers that should be split into two because this space has two distinct clusters of objects,
    # as well as classifiers whose object spaces look so similar we should collapse the classifiers
    def refactor_predicates(self, num_objects):

        obj_range = range(0, num_objects)

        change_made = True
        while change_made:
            change_made = False
            active_predicates = [p for p in self.predicates if self.predicate_active[p]]
            r_with_confidence = self.get_classifier_results(active_predicates, obj_range)
            r = {}
            for oidx in r_with_confidence:
                r[oidx] = {}
                for pred in r_with_confidence[oidx]:
                    r[oidx][pred] = r_with_confidence[oidx][pred][0]*r_with_confidence[oidx][pred][1]

            # detect synonymy
            # observes the cosine distance between predicate vectors in |O|-dimensional space
            highest_cos_sim = [None, -1]
            norms = {}
            for p in active_predicates:
                norms[p] = math.sqrt(sum([math.pow(r[oi][p], 2) for oi in obj_range]))
            for pidx in range(0, len(active_predicates)):
                p = active_predicates[pidx]
                if norms[p] == 0:
                    continue
                for qidx in range(pidx+1, len(active_predicates)):
                    q = active_predicates[qidx]
                    if norms[q] == 0:
                        continue
                    cos_sim = sum([r[oi][p]*r[oi][q] for oi in obj_range]) / (norms[p]*norms[q])
                    if cos_sim > self.alpha and cos_sim > highest_cos_sim[1]:
                        highest_cos_sim = [[p, q], cos_sim]
            if highest_cos_sim[0] is not None:

                # collapse the two closest predicates into one new predicate
                p, q = highest_cos_sim[0]
                self.collapse_predicates(p, q)
                change_made = True
                self.retrain_predicate_classifiers()  # should fire only for pq
                continue

            # detect polysemy
            for p in active_predicates:
                if self.attempt_predicate_split(p, num_objects):
                    change_made = True
                    self.retrain_predicate_classifiers()
                    break

    # collapse two predicates into one
    def collapse_predicates(self, p, q):

        pq = "("+p+"+"+q+")"
        print "collapsing '"+p+"' and '"+q+"' to form '"+pq+"'"  # DEBUG
        print "...examples for '"+p+"': "+str(self.predicate_examples[p])  # DEBUG
        print "...examples for '"+q+"': "+str(self.predicate_examples[q])  # DEBUG
        self.predicates.append(pq)
        self.predicate_active[p] = False
        self.predicate_active[q] = False
        self.predicate_active[pq] = True
        self.predicates_to_words[pq] = []
        for w in self.words:
            if p in self.words_to_predicates[w]:
                self.words_to_predicates[w].append(pq)
                self.predicates_to_words[pq].append(w)
            if q in self.words_to_predicates[w]:
                self.words_to_predicates[w].append(pq)
                self.predicates_to_words[pq].append(w)
        self.predicate_examples[pq] = join_dicts(self.predicate_examples[p],
                                                 self.predicate_examples[q],
                                                 allow_duplicates=True)
        print "...examples for '"+pq+"': "+str(self.predicate_examples[pq])  # DEBUG
        cid = self.get_free_classifier_id_client()
        self.predicate_to_classifier_map[pq] = cid
        self.classifier_to_predicate_map[cid] = pq
        self.classifier_data_modified[cid] = True

    # attempt to split a predicate
    def attempt_predicate_split(self, p, num_objects):

        # get positive examples from predicate
        objects_to_split = [oidx for oidx in range(0, num_objects)
                            if oidx in self.predicate_examples[p] and
                            True in self.predicate_examples[p][oidx]]
        if len(objects_to_split) < 4:  # heuristic to prevent unnecessary splitting
            return False

        # naive algorithm: assign random class to positive objects and iteratively
        # adjust classes based on SVM margins

        object_v = numpy.asarray(
            self.get_predicate_classifier_decision_conf_matrices(p, objects_to_split),
            dtype=numpy.float32)
        object_l = numpy.asarray(
            [1 if random.random() > 0.5 else -1 for _ in range(0, len(objects_to_split))],
            dtype=numpy.float32)

        # don't allow initialization to single class
        if 1 not in object_l:
            object_l[0] = 1
        elif -1 not in object_l:
            object_l[0] = -1

        # iterate to converge on object_l labels that divide the space
        converged = False
        i = math.pow(len(objects_to_split), 2)
        while not converged and i > 0:
            i -= 1
            m = SVM()
            m.train(object_v, object_l)

            # find max distance mislabeled object
            mdmo = None
            md = None
            r = m.predict(object_v)
            for idx in range(0, len(objects_to_split)):
                d, dist = r[idx]
                if d != object_l[idx]:
                    if md is None or md < dist:
                        md = dist
                        mdmo = idx

            # converge if least confidence mislabel is unfound (e.g. all labeled correctly)
            if md is None:
                converged = True
                break

            # flip the label of the least confidence mislabeled object and iterate again
            if object_l[mdmo] == 1:
                object_l[mdmo] = -1.0
            else:
                object_l[mdmo] = 1.0

            # if this causes a single class to form, then break with failure
            if 1 not in object_l or -1 not in object_l:
                break

        # if convergence happened, split predicate according to found division
        if converged:
            self.split_predicate(p, objects_to_split, object_l)
            return True

        return False

    # split a predicate into two such
    def split_predicate(self, p, obs, l):

        # if predicate is previously collapsed, attempt to divide along established lines
        if not self.split_collapsed_predicate(p, obs, l):

            # then split into two senses
            qs = [p+"_1", p+"_2"]
            print "splitting '"+p+"' to form '"+qs[0]+"' and '"+qs[1]+"'"  # DEBUG
            print "...examples for '"+p+"': "+str(self.predicate_examples[p])  # DEBUG
            self.predicates.extend(qs)
            self.predicate_active[p] = False
            for idx in range(0, len(qs)):
                q = qs[idx]
                self.predicate_active[q] = True
                self.predicates_to_words[q] = []
                for w in self.words:
                    if p in self.words_to_predicates[w]:
                        self.words_to_predicates[w].append(q)
                        self.predicates_to_words[q].append(w)
                self.predicate_examples[q] = {}
                for oidx in self.predicate_examples[p]:
                    self.predicate_examples[q][oidx] = []
                    for b in self.predicate_examples[p][oidx]:
                        if b:  # keep positive example if it's on our side of the found split
                            loidx = obs.index(oidx)
                            self.predicate_examples[q][oidx].append(
                                False if (idx == 0 and l[loidx] == -1)
                                or (idx == 1 and l[loidx] == 1) else True)
                        else:  # keep all negative examples
                            self.predicate_examples[q][oidx].append(False)

                print "...examples for '"+q+"': "+str(self.predicate_examples[q])  # DEBUG
                cid = self.get_free_classifier_id_client()
                self.predicate_to_classifier_map[q] = cid
                self.classifier_to_predicate_map[cid] = q
                self.classifier_data_modified[cid] = True

    # attempt to split a previously collapsed predicate
    def split_collapsed_predicate(self, p, obs, l):

        if '+' not in p or p[0] != '(' or p[-1] != ')':
            return False
        prn = 0
        c = 0
        pstr = p[1:-1]
        for c in range(0, len(pstr)):
            if pstr[c] == '(':
                prn += 1
            elif pstr[c] == ')':
                prn -= 1
            elif pstr[c] == '+' and prn == 0:
                break
        qs = [pstr[:c], pstr[c+1:]]

        # greedily choose pair of preds whose split gives closest match in old label space
        cmpr = None
        cmc = None
        for d in range(0, 2):
            m = 0.0
            for oidx in range(0, len(l)):
                poidx = obs[oidx]
                if ((d == 0 and l[oidx] == -1 and
                        poidx in self.predicate_examples[qs[0]] and
                        False in self.predicate_examples[qs[0]][poidx])
                   or (d == 1 and l[oidx] == 1 and
                        poidx in self.predicate_examples[qs[1]] and
                        True in self.predicate_examples[qs[1]][poidx])):
                    m += 1
                if ((d == 0 and l[oidx] == 1 and
                        poidx in self.predicate_examples[qs[1]] and
                        True in self.predicate_examples[qs[1]][poidx])
                   or (d == 1 and l[oidx] == -1 and
                        poidx in self.predicate_examples[qs[1]] and
                        False in self.predicate_examples[qs[1]][poidx])):
                    m += 1
            if cmc is None or m > cmc:
                cmc = m
                cmpr = [qs[0], qs[1]] if d == 0 else [qs[1], qs[0]]

        # if match is close enough, split the collapsed predicate pair found
        if (cmc /
            (sum([len(self.predicate_examples[cmpr[0]][idx]) for idx in self.predicate_examples[cmpr[0]]])
             + sum([len(self.predicate_examples[cmpr[1]][idx]) for idx in self.predicate_examples[cmpr[1]]]))
                > self.alpha):

            print "splitting '"+p+"' to form '"+cmpr[0]+"' and '"+cmpr[1]+"'"  # DEBUG
            print "...examples for '"+p+"': "+str(self.predicate_examples[p])  # DEBUG
            self.predicate_active[p] = False
            for idx in range(0, len(cmpr)):
                q = cmpr[idx]
                self.predicate_active[q] = True
                for oidx in range(0, len(l)):
                    poidx = obs[oidx]
                    if poidx not in self.predicate_examples[q]:
                        self.predicate_examples[poidx] = []
                    lb = None
                    if idx == 0:
                        lb = True if l[oidx] == 1 else False
                    elif idx == 1:
                        lb = False if l[oidx] == 1 else True
                    if lb not in self.predicate_examples[q][poidx]:
                        self.predicate_examples[q][poidx].append(lb)
                print "...examples for '"+q+"': "+str(self.predicate_examples[q])  # DEBUG
                cid = self.predicate_to_classifier_map[q]
                self.classifier_data_modified[cid] = True

            return True

        return False

    # given vectors of predicates and object idxs, return a map of results
    def get_classifier_results(self, preds, oidxs):
        m = {}
        for oidx in oidxs:
            om = {}
            for pred in preds:
                cidx = self.predicate_to_classifier_map[pred]
                result, confidence, _ = self.run_classifier_client(cidx, oidx)
                om[pred] = [result, confidence]
            m[oidx] = om
        return m

    # given predicate and object idxs, return a vector of behavior/modality decision*conf vectors
    def get_predicate_classifier_decision_conf_matrices(self, pred, oidxs):
        ov = []
        for oidx in oidxs:
            cidx = self.predicate_to_classifier_map[pred]
            _, _, sub_decisions = self.run_classifier_client(cidx, oidx)
            ov.append(sub_decisions)
        return ov

    # given a string input, strip stopwords and use word to predicate map to build cnf clauses
    # such that each clause represents the predicates associated with each word
    # for unknown words, invent and return new predicates
    def get_predicate_cnf_clauses_for_utterance(self, u):
        u_parts = u.split()
        words = [w for w in u_parts if w not in self.stopwords]

        cnfs = []
        for w in words:
            if w not in self.words:
                self.words.append(w)
                self.word_counts[w] = 0
            if w not in self.words_to_predicates:
                self.predicates.append(w)
                self.predicate_active[w] = True
                self.words_to_predicates[w] = [w]
                self.predicates_to_words[w] = [w]
                cid = self.get_free_classifier_id_client()
                self.predicate_to_classifier_map[w] = cid
                self.classifier_to_predicate_map[cid] = w
                self.predicate_examples[w] = {}
            self.word_counts[w] += 1
            cnfs.append([p for p in self.words_to_predicates[w] if self.predicate_active[p]])

        return cnfs

    # add given attribute examples and re-train relevant classifiers
    def update_predicate_data(self, pred, data):
        for oidx, label in data:
            if oidx not in self.predicate_examples[pred]:
                self.predicate_examples[pred][oidx] = []
            self.predicate_examples[pred][oidx].append(label)
        cidx = self.predicate_to_classifier_map[pred]
        self.classifier_data_modified[cidx] = True

    # retrain classifiers that have modified data since last training
    def retrain_predicate_classifiers(self):
        for cidx in self.classifier_data_modified:
            pred = self.classifier_to_predicate_map[cidx]
            if self.classifier_data_modified[cidx]:
                r_oidxs = []
                r_labels = []
                for oidx in self.predicate_examples[pred]:
                    # include all system - slower, more accurate confidence values
                    for l in self.predicate_examples[pred][oidx]:
                        r_oidxs.append(oidx)
                        r_labels.append(l)
                    # voting system - faster, potentially noiser confidence values
                    # r_oidxs.append(oidx)
                    # t = sum([1 if l else -1 for l in self.predicate_examples[pred][oidx]])
                    # if t != 0:
                    #     r_labels.append(True if t > 0 else False)
                self.train_classifier_client(cidx, r_oidxs, r_labels)
                self.classifier_data_modified[cidx] = False

    # fold in data structures from another dialog agent
    def unify_with_agent(self, other):

        # join lists and dicts of word, predicate, and predite examples
        self.predicates = join_lists(self.predicates, other.predicates, allow_duplicates=False)
        self.words = join_lists(self.words, other.words, allow_duplicates=False)
        for w in self.words:
            if w in self.word_counts and w in other.word_counts:
                self.word_counts[w] += other.word_counts[w]
            elif w in other.word_counts:
                self.word_counts[w] = other.word_counts[w]
        self.words_to_predicates = join_dicts(
            self.words_to_predicates, other.words_to_predicates, allow_duplicates=False)
        self.predicates_to_words = join_dicts(
            self.predicates_to_words, other.predicates_to_words, allow_duplicates=False)
        self.predicate_examples = join_dicts(
            self.predicate_examples, other.predicate_examples, allow_duplicates=True)

        # join predicate<->classifier maps into one-to-many maps
        sptcm_list = {p: [self.predicate_to_classifier_map[p]] for p in self.predicate_to_classifier_map}
        optcm_list = {p: [other.predicate_to_classifier_map[p]] for p in other.predicate_to_classifier_map}
        self.predicate_to_classifier_map = join_dicts(sptcm_list, optcm_list, warn_duplicates=True)

        # reduce one-to-many maps back to one-to-one maps by dropping colliding classifier IDs
        # this happens when two users introduce the same new predicate in parallel, which is assigned
        # two unique IDs
        self.classifier_to_predicate_map = {}
        for p in self.predicate_to_classifier_map:
            c = max(self.predicate_to_classifier_map[p])
            self.predicate_to_classifier_map[p] = c
            self.classifier_to_predicate_map[c] = p
            if p not in self.predicate_active:
                self.predicate_active[p] = other.predicate_active[p]

        # build a new map of what needs to be retrained
        cdm = {}
        for cidx in self.classifier_to_predicate_map:
            if cidx in other.classifier_data_modified and other.classifier_data_modified[cidx]:
                cdm[cidx] = True
            if cidx in self.classifier_data_modified and self.classifier_data_modified[cidx]:
                cdm[cidx] = True
        self.classifier_data_modified = cdm

    # remove predicate examples in structure from self
    def subtract_predicate_examples(self, pe):
        for pred in pe:
            if pred in self.predicate_examples:
                for oidx in pe[pred]:
                    if oidx in self.predicate_examples[pred]:
                        for b in pe[pred][oidx]:
                            if b in self.predicate_examples[pred][oidx]:
                                del self.predicate_examples[pred][oidx][
                                    self.predicate_examples[pred][oidx].index(b)]

    # load classifiers
    def load_classifiers(self):
        r = self.load_classifiers_client()
        if not r:
            print "ERROR when loading perceptual classifiers"

    # save classifiers
    def save_classifiers(self):
        r = self.save_classifiers_client()
        if not r:
            print "ERROR when saving perceptual classifiers"

    # access the perceptual classifiers package load classifier service
    def get_free_classifier_id_client(self):
        req = getFreeClassifierIDRequest()
        rospy.wait_for_service('get_free_classifier_ID')
        try:
            get_free_classifier_id = rospy.ServiceProxy('get_free_classifier_ID', getFreeClassifierID)
            res = get_free_classifier_id(req)
            return res.ID
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e

    # access the perceptual classifiers package load classifier service
    def load_classifiers_client(self):
        req = loadClassifiersRequest()
        rospy.wait_for_service('load_classifiers')
        try:
            load_classifiers = rospy.ServiceProxy('load_classifiers', loadClassifiers)
            res = load_classifiers(req)
            return res.success
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e

    # access the perceptual classifiers package save classifier service
    def save_classifiers_client(self):
        req = EmptyRequest()
        rospy.wait_for_service('save_classifiers')
        try:
            save_classifiers = rospy.ServiceProxy('save_classifiers', Empty)
            res = save_classifiers(req)  # TODO: give saveClassifiers a srv so it can respond with success flag
            return True
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e

    # access the perceptual classifiers package run classifier service to get
    # decision result, confidence, and sub classifier weighted decisions
    def run_classifier_client(self, classifier_ID, object_ID):
        req = runClassifierRequest()
        req.classifier_ID = classifier_ID
        req.object_ID = object_ID
        rospy.wait_for_service('run_classifier')
        try:
            run_classifier = rospy.ServiceProxy('run_classifier', runClassifier)
            res = run_classifier(req)
            return res.result, res.confidence, res.sub_classifier_decisions
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e

    # access the perceptual classifiers package run classifier service to get
    # decision result, confidence, and sub classifier weighted decisions
    def train_classifier_client(self, classifier_ID, object_IDs, positive_example):
        req = trainClassifierRequest()
        req.classifier_ID = classifier_ID
        req.object_IDs = object_IDs
        req.positive_example = positive_example
        rospy.wait_for_service('train_classifier')
        try:
            train_classifier = rospy.ServiceProxy('train_classifier', trainClassifier)
            res = train_classifier(req)
            return res.success
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e

