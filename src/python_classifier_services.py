#!/usr/bin/env python
__author__ = 'jesse'

import argparse
import numpy as np
import os
import pickle
import rospy
from perception_classifiers.srv import *
from sklearn.svm import SVC

# Fixed parameters.
kernel = 'linear'


class ClassifierServices:

    def __init__(self, source_dir, classifier_dir):

        # Service parameters.
        self.source_dir = source_dir  # str
        self.classifier_dir = classifier_dir  # str
        self.predicates = None  # list of strs
        self.oidxs = None  # list of ints
        self.labels = None  # list of (pidx, oidx, label) triples for label in {-1, 1}
        self.features = None  # list of behavior, modality indexed dictionaries into lists of observation vectors
        self.behaviors = None  # list of strs
        self.contexts = None  # list of (behavior, modality) str tuples
        self.classifiers = None  # list of behavior, modality indexed dictionaries into SVC classifiers
        self.kappas = None  # list of behavior, modality indexed dictionaries into [0, 1] floats

        # Read in source information.
        print "reading in source information..."
        with open(os.path.join(self.source_dir, "predicates.pickle"), 'rb') as f:
            self.predicates = pickle.load(f)
        with open(os.path.join(self.source_dir, "oidxs.pickle"), 'rb') as f:
            self.oidxs = pickle.load(f)
        with open(os.path.join(self.source_dir, "labels.pickle"), 'rb') as f:
            self.labels = pickle.load(f)
        with open(os.path.join(self.source_dir, "features.pickle"), 'rb') as f:
            self.features = pickle.load(f)
        self.behaviors = ["drop", "grasp", "hold", "lift", "look", "lower", "press", "push"]
        self.modalities = ["audio_ispy", "color", "fpfh", "haptic_ispy"]
        self.contexts = []
        for b in self.behaviors:
            self.contexts.extend([(b, m) for m in self.modalities
                                  if m in self.features[self.oidxs[0]][b].keys()])
        print "... done"

        # Read in cashed classifiers or train fresh ones.
        classifier_fn = os.path.join(classifier_dir, "classifiers.pickle")
        if os.path.isfile(classifier_fn):
            print "reading cached classifiers from file..."
            with open(classifier_fn, 'rb') as f:
                self.classifiers, self.kappas = pickle.load(f)
        else:
            print "training classifiers from source information..."
            self.train_classifiers(range(len(self.predicates)))
        print "... done"

    # Gets the result of specified predicate on specified object.
    def run_classifier(self, req):
        pidx = req.pidx
        oidx = req.oidx
        ds = []
        ks = []
        for b, m in self.contexts:
            x, y, z = get_classifier_results(self.classifiers[pidx][b][m], b, m,
                                             [(oidx, None)], self.features, None)
            ds.append(np.mean(z))
            ks.append(self.kappas[pidx][b][m])
        dec = [ds[idx] * ks[idx] for idx in range(len(self.contexts))]
        res = PythonRunClassiferResponse()
        res.dec = True if dec > 0 else False
        res.conf = abs(dec)
        return res

    # Updates the in-memory classifiers given new labels in the request.
    def update_classifiers(self, req):
        upreds = req.new_preds
        upidxs = req.pidxs
        uoidxs = req.oidxs
        ulabels = req.labels
        self.predicates.extend(upreds)
        retrain_pidxs = []
        for idx in range(len(upidxs)):
            pidx = upidxs[idx]
            if pidx not in retrain_pidxs:
                retrain_pidxs.append(pidx)
            self.labels.append((pidx, uoidxs[idx], ulabels[idx]))
        self.train_classifiers(retrain_pidxs)
        res = PythonUpdateClassifiersResponse()
        res.success = True
        return res

    # Commits the trained classifiers and current labels to the classifier and source directories, respectively.
    def commit_changes(self, req):
        _ = req
        with open(os.path.join(self.source_dir, "predicates.pickle"), 'wb') as f:
            pickle.dump(self.predicates, f)
        with open(os.path.join(self.source_dir, "labels.pickle"), 'wb') as f:
            pickle.dump(self.labels, f)
        with open(os.path.join(self.classifier_dir, "classifiers.pickle"), 'wb') as f:
            pickle.dump([self.classifiers, self.kappas], f)
        res = PythonCommitChangesResponse()
        res.success = True
        return res

    # Get oidx, l from pidx, oidx, l labels.
    def get_pairs_from_labels(self, pidx):
        pairs = []
        for pjdx, oidx, l in self.labels:
            if pjdx == pidx:
                pairs.append((oidx, l))
        return pairs

    # Train all classifiers given boilerplate info and labels.
    def train_classifiers(self, pidxs):
        self.classifiers = []  # pidx, b, m
        self.kappas = []
        for pidx in pidxs:
            train_pairs = self.get_pairs_from_labels(pidx)
            if -1 in [l for _, l in train_pairs] and 1 in [l for _, l in train_pairs]:
                print "... '" + self.predicates[pidx] + "' fitting"
                pc = {}
                pk = {}
                for b, m in self.contexts:
                    if b not in pc:
                        pc[b] = {}
                        pk[b] = {}
                    pc[b][m] = fit_classifier(b, m, train_pairs, self.features)
                    pk[b][m] = get_margin_kappa(pc[b][m], b, m, train_pairs, self.features, xval=train_pairs)
                s = sum([pk[b][m] for b, m in self.contexts])
                for b, m in self.contexts:
                    pk[b][m] = pk[b][m] / float(s) if s > 0 else 1.0 / len(self.contexts)
                self.classifiers.append(pc)
                self.kappas.append(pk)
            else:
                print "... '" + self.predicates[pidx] + "' lacks a +/- pair to fit"
                self.classifiers.append(None)
                self.kappas.append(0)


# Given an SVM c and its training data, calculate the agreement with gold labels according to kappa
# agreement statistic at the observation level.
def get_margin_kappa(c, behavior, modality, pairs, object_feats, xval=None):
    x, y, z = get_classifier_results(c, behavior, modality, pairs, object_feats, xval)
    cm = [[0, 0], [0, 0]]
    for idx in range(len(x)):
        cm[1 if y[idx] == 1 else 0][1 if z[idx] == 1 else 0] += 1
    return get_kappa(cm)


# Given an SVM and its training data, fit that training data, optionally retraining leaving
# one object out at a time.
def get_classifier_results(c, behavior, modality, pairs, object_feats, xval):
    if c is None:
        x, y = get_data_for_classifier(behavior, modality, pairs, object_feats)
        z = [-1 for _ in range(len(x))]  # No classifier trained, so guess majority class no.
    else:
        if xval is None:
            x, y = get_data_for_classifier(behavior, modality, pairs, object_feats)
            z = c.predict(x)
        else:
            x = []
            y = []
            z = []
            rel_oidxs = list(set([oidx for (oidx, l) in pairs]))
            if len(rel_oidxs) > 1:
                for oidx in rel_oidxs:
                    # Train a new classifier without data from oidx.
                    xval_pairs = [(ojdx, l) for (ojdx, l) in xval if ojdx != oidx]
                    ls = list(set([l for ojdx, l in xval_pairs]))
                    if len(ls) == 2:
                        xval_c = fit_classifier(behavior, modality, xval_pairs, object_feats)
                    else:
                        xval_c = None

                    # Evaluate new classifier on held out oidx data and record results.
                    xval_pairs = [(ojdx, l) for (ojdx, l) in pairs if ojdx == oidx]
                    _x, _y = get_data_for_classifier(behavior, modality, xval_pairs, object_feats)
                    if xval_c is not None:
                        _z = xval_c.predict(_x)
                    else:  # If insufficient data, vote the same label as the training data.
                        _z = [1 if len(ls) > 0 and ls[0] == 1 else -1 for _ in range(len(_x))]
                    x.extend(_x)
                    y.extend(_y)
                    z.extend(_z)
            else:
                x, y = get_data_for_classifier(behavior, modality, pairs, object_feats)
                z = [-1 for _ in range(len(x))]  # Single object, so guess majority class no.
    return x, y, z


# Fits a new SVM classifier given a kernel, context, training pairs, and object feature structure.
def fit_classifier(behavior, modality, pairs, object_feats):
    x, y = get_data_for_classifier(behavior, modality, pairs, object_feats)
    assert len(x) > 0  # there is data
    assert min(y) < max(y)  # there is more than one label
    c = SVC(kernel=kernel, degree=2)
    c.fit(x, y)
    return c


# Given a context, label pairs, and object feature structure, returns SVM-friendly x, y training vectors.
def get_data_for_classifier(behavior, modality, pairs, object_feats):
    x = []
    y = []
    for oidx, label in pairs:
        if behavior in object_feats[oidx] and modality in object_feats[oidx][behavior]:
            for obs in object_feats[oidx][behavior][modality]:
                if len(obs) == 1 and type(obs[0]) is list:  # un-nest single observation vectors
                    obs = obs[0]
                x.append(obs)
                l = 1 if label == 1 else -1
                y.append(l)
    return x, y


# Returns non-negative kappa.
def get_kappa(cm):
    return max(0, get_signed_kappa(cm))


# Returns non-negative kappa.
def get_signed_kappa(cm):

    s = float(cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
    po = (cm[1][1] + cm[0][0]) / s
    ma = (cm[1][1] + cm[1][0]) / s
    mb = (cm[0][0] + cm[0][1]) / s
    pe = (ma + mb) / s
    kappa = (po - pe) / (1 - pe)
    return kappa


def main():

    # Read in command-line arguments.
    source_dir = FLAGS_source_dir
    classifier_dir = FLAGS_classifier_dir

    cs = ClassifierServices(source_dir, classifier_dir)

    # Initialize node and advertise services.
    rospy.init_node('python_classifier_services')
    rospy.Service('python_run_classifier', PythonRunClassifier, cs.run_classifier)
    rospy.Service('python_update_classifiers', PythonUpdateClassifiers, cs.update_classifiers)
    rospy.Service('python_commit_changes', PythonCommitChanges, cs.commit_changes)
    rospy.spin()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', type=str, required=True,
                        help="directory where object ids, predicates, and labels pickles live")
    parser.add_argument('--classifier_dir', type=str, required=True,
                        help="directory in which to stash trained classifiers")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
