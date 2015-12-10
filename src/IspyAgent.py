#!/usr/bin/env python
__author__ = 'jesse'

import rospy
from perception_classifiers.srv import *
from std_srvs.srv import *
import operator
import math
import random


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
            elif c[key] != b[key]:
                sys.exit("ERROR: join_dicts collision '"+str(key) +
                         "' trying to take values '" + str(c[key]) + "', '" + str(b[key]) + "'")
        else:
            if type(b[key]) is list:
                c[key] = b[key][:]
            else:
                c[key] = b[key]
    return c


class IspyAgent:

    def __init__(self, u_in, u_out, object_IDs, stopwords_fn, alpha=0.9, simulation=False):

        self.u_in = u_in
        self.u_out = u_out
        self.object_IDs = object_IDs
        self.alpha = alpha
        self.simulation = simulation

        # lists of predicates and words currently known
        self.predicates = []
        self.words = []

        # maps because predicates can be dropped during merge and split operations
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

            utterance = self.u_in.get()
            cnf_clauses = self.get_predicate_cnf_clauses_for_utterance(utterance)

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
        r = self.get_classifier_results(self.predicates, self.object_IDs)

        # rank the classifiers favoring high confidence on ob_idx with low confidence or negative
        # decisions on other objects
        pred_scores = {}
        for pred in self.predicate_to_classifier_map:
            score = r[ob_idx][pred][0]*r[ob_idx][pred][1]
            for oidx in self.object_IDs:
                if ob_idx == oidx:
                    continue
                score -= r[oidx][pred][0]*r[oidx][pred][1]
            pred_scores[pred] = score

        # choose predicates to best describe object
        predicates_chosen = []
        for pred, score in sorted(pred_scores.items(), key=operator.itemgetter(1), reverse=True):
            if score > 0:
                # choose just one predicate word arbitrarily if multiple are associated with classifier
                predicates_chosen.append(pred)
        if len(predicates_chosen) == 0:  # we have no classifier information yet, so choose 3 arbitrarily
            preds_shuffled = self.predicates[:]
            random.shuffle(preds_shuffled)
            predicates_chosen.extend(preds_shuffled[:3 if len(preds_shuffled) >= 3 else len(preds_shuffled)])

        # describe object to user
        if len(predicates_chosen) > 2:
            desc = "I am thinking of an object I would describe as " + \
                ', '.join([self.predicates_to_words[pred][0] for pred in predicates_chosen[:-1]]) + \
                ", and "+self.predicates_to_words[predicates_chosen[-1]][0] + "."
        elif len(predicates_chosen) == 2:
            desc = "I am thinking of an object I would describe as " + self.predicates_to_words[predicates_chosen[0]][0] + \
                " and " + self.predicates_to_words[predicates_chosen[1]][0] + "."
        else:
            desc = "I am thinking of an object I would describe as " + self.predicates_to_words[predicates_chosen[0]][0] + "."
        self.u_out.say(desc)

        # wait for user to find and select correct object
        num_guesses = 0
        while True:
            if self.simulation:
                guess_idx = self.object_IDs[self.u_in.get_guess()]
            else:
                # TODO: call looking for hand over object service
                guess_idx = None
            num_guesses += 1
            if guess_idx == ob_idx:
                self.u_out.say("That's the one!")
                return desc, predicates_chosen, num_guesses
            else:
                self.u_out.say("That's not the object I am thinking of.")

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
            r_with_confidence = self.get_classifier_results(self.predicates, obj_range)
            r = {}
            for oidx in r_with_confidence:
                r[oidx] = {}
                for pred in r_with_confidence[oidx]:
                    r[oidx][pred] = r_with_confidence[oidx][pred][0]*r_with_confidence[oidx][pred][1]

            # detect synonymy
            # observes the cosine distance between predicate vectors in |O|-dimensional space
            highest_cos_sim = [None, -1]
            norms = {}
            for p in self.predicates:
                norms[p] = math.sqrt(sum([math.pow(r[oi][p], 2) for oi in obj_range]))
            for p in self.predicates:
                if norms[p] == 0:
                    continue
                for q in self.predicates:
                    if norms[q] == 0 or p == q:
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
            # TODO: get float decisions from all contexts; comparisons must be at context level within
            # TODO: a single predicate to determine whether a split is warranted

    # collapse two predicates into one
    def collapse_predicates(self, p, q):

        pq = p+"+"+q
        print "collapsing '"+p+"' and '"+q+"' to form '"+pq+"'"  # DEBUG
        del self.predicates[self.predicates.index(p)]
        del self.predicates[self.predicates.index(q)]
        self.predicates_to_words[pq] = []
        self.predicates.append(pq)
        for w in self.words:
            if p in self.words_to_predicates[w]:
                del self.words_to_predicates[w][self.words_to_predicates[w].index(p)]
                self.words_to_predicates[w].append(pq)
                self.predicates_to_words[pq].append(w)
            if q in self.words_to_predicates[w]:
                del self.words_to_predicates[w][self.words_to_predicates[w].index(q)]
                self.words_to_predicates[w].append(pq)
                self.predicates_to_words[pq].append(w)
        self.predicate_examples[pq] = self.predicate_examples[p][:]
        self.predicate_examples[pq].extend(self.predicate_examples[q])
        del self.predicate_examples[p]
        del self.predicate_examples[q]
        p_cid = self.predicate_to_classifier_map[p]
        del self.predicate_to_classifier_map[p]
        q_cid = self.predicate_to_classifier_map[q]
        del self.predicate_to_classifier_map[q]
        cid = self.get_free_classifier_id_client()
        self.predicate_to_classifier_map[pq] = cid
        del self.classifier_to_predicate_map[p_cid]
        del self.classifier_to_predicate_map[q_cid]
        if p_cid in self.classifier_data_modified:
            del self.classifier_data_modified[p_cid]
        if q_cid in self.classifier_data_modified:
            del self.classifier_data_modified[q_cid]
        self.classifier_to_predicate_map[cid] = pq
        self.classifier_data_modified[cid] = True

    # given vectors of attribute idxs and object idxs, return a map of results
    def get_classifier_results(self, preds, oidxs):
        print self.words_to_predicates  # DEBUG
        print preds  # DEBUG
        print self.predicate_to_classifier_map  # DEBUG
        m = {}
        for oidx in oidxs:
            om = {}
            for pred in preds:
                cidx = self.predicate_to_classifier_map[pred]
                result, confidence, _ = self.run_classifier_client(cidx, oidx)
                om[pred] = [result, confidence]
            m[oidx] = om
        print m
        return m

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
            if w not in self.words_to_predicates:
                self.predicates.append(w)
                self.words_to_predicates[w] = [w]
                self.predicates_to_words[w] = [w]
                cid = self.get_free_classifier_id_client()
                self.predicate_to_classifier_map[w] = cid
                self.classifier_to_predicate_map[cid] = w
                self.predicate_examples[w] = []
            cnfs.append(self.words_to_predicates[w])

        return cnfs

    # add given attribute examples and re-train relevant classifiers
    def update_predicate_data(self, pred, data):
        self.predicate_examples[pred].extend(data)
        cidx = self.predicate_to_classifier_map[pred]
        self.classifier_data_modified[cidx] = True

    # retrain classifiers that have modified data since last training
    def retrain_predicate_classifiers(self):
        for cidx in self.classifier_data_modified:
            pred = self.classifier_to_predicate_map[cidx]
            if self.classifier_data_modified[cidx]:
                self.train_classifier_client(cidx,
                                             [d[0] for d in self.predicate_examples[pred]],
                                             [d[1] for d in self.predicate_examples[pred]])
                self.classifier_data_modified[cidx] = False

    # fold in data structures from another dialog agent
    def unify_with_agent(self, other):

        # join lists and dicts of word, predicate, and predite examples
        self.predicates = join_lists(self.predicates, other.predicates, allow_duplicates=False)
        self.words = join_lists(self.words, other.words, allow_duplicates=False)
        self.words_to_predicates = join_dicts(
            self.words_to_predicates, other.words_to_predicates, allow_duplicates=False)
        self.predicates_to_words = join_dicts(
            self.predicates_to_words, other.predicates_to_words, allow_duplicates=False)
        self.predicate_examples = join_dicts(
            self.predicate_examples, other.predicate_examples, allow_duplicates=True)

        # join predicate<->classifier maps into one-to-many maps
        sptcm_list = {p:[self.predicate_to_classifier_map[p]] for p in self.predicate_to_classifier_map}
        optcm_list = {p:[other.predicate_to_classifier_map[p]] for p in other.predicate_to_classifier_map}
        self.predicate_to_classifier_map = join_dicts(sptcm_list, optcm_list, warn_duplicates=True)

        # reduce one-to-many maps back to one-to-one maps by dropping colliding classifier IDs
        # this happens when two users introduce the same new predicate in parallel, which is assigned
        # two unique IDs
        self.classifier_to_predicate_map = {}
        for p in self.predicate_to_classifier_map:
            c = max(self.predicate_to_classifier_map[p])
            self.predicate_to_classifier_map[p] = c
            self.classifier_to_predicate_map[c] = p

        # build a new map of what needs to be retrained
        cdm = {}
        for cidx in self.classifier_to_predicate_map:
            if cidx in other.classifier_data_modified and other.classifier_data_modified[cidx]:
                cdm[cidx] = True
            if cidx in self.classifier_data_modified and self.classifier_data_modified[cidx]:
                cdm[cidx] = True
        self.classifier_data_modified = cdm

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
        print "sending training req: "  # DEBUG
        print req  # DEBUG
        try:
            train_classifier = rospy.ServiceProxy('train_classifier', trainClassifier)
            res = train_classifier(req)
            return res.success
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e
