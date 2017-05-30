#!/usr/bin/env python
__author__ = 'aishwarya'

import sys
import random
import rospy
import operator
import numpy as np
import traceback
from UnitTestAgent import UnitTestAgent
from perception_classifiers.srv import *


class InquisitiveIspyAgent(UnitTestAgent):
    # initial_predicates - The agent needs to track which classifiers 
    #       it can call. This gives the initial list. This has to match 
    #       the list in python classifier services
    def __init__(self, io, table_oidxs, stopwords_fn, policy, log_fn=None, initial_predicates=None):
        tid = 1
        UnitTestAgent.__init__(self, io, 1, table_oidxs)
        
        self.io = io
        self.log_fn = log_fn
        self.policy = policy
        
        # Read stopwords
        self.stopwords = []
        if stopwords_fn is not None:
            fin = open(stopwords_fn, 'r')
            for line in fin.readlines():
                self.stopwords.append(line.strip())
            fin.close()
        
        # Known predicates - Those for which you can ask the classifiers for a label
        # Unknown predicates - Those for which you can't
        self.known_predicates = list()
        self.unknown_predicates = list()
        if initial_predicates is not None:
            self.known_predicates = initial_predicates
        
        # Caching useful classifier info    
        self.classifiers_changed = list() 
        self.min_confidence_objects = dict()
            # Key: predicate; Value: (obj_idx, confidence)
        self.current_classifier_results = None
        
        self.print_debug_messages = True 
        
        self.objects_for_guessing = self.table_oidxs[1]
        self.objects_for_questions = self.table_oidxs[0] + self.table_oidxs[2]
        
        # Some additional state info
        self.num_dialog_turns = 0
        self.cur_dialog_predicates = None
        self.cur_match_scores = None
        

    def debug_print(self, message):
        if self.print_debug_messages:
            print 'PomdpIspyAgent:', message

    
    def log(self, log_str):
        if self.log_fn is not None:
            f = open(self.log_fn, 'a')
            f.write(log_str)
            f.close()


    def get_dialog_state(self):
        dialog_state = dict()
        dialog_state['num_dialog_turns'] = self.num_dialog_turns
        dialog_state['match_scores'] = self.cur_match_scores
        dialog_state['unknown_predicates'] = self.unknown_predicates
        dialog_state['cur_dialog_predicates'] = self.cur_dialog_predicates
        dialog_state['min_confidence_objects'] = self.min_confidence_objects
        return dialog_state


    def make_guess(self, match_scores):
        # match_scores.items() gives (obj_idx, score) tuples
        # Shuffle to randomize order of equally scored items on sorting
        obj_indices_with_scores = np.random.shuffle(match_scores.items())
        # Sort by match score to identify what the robot should guess
        sorted_match_scores = sorted(obj_indices_with_scores, key=operator.itemgetter(1), reverse=True)
        guess_idx = sorted_match_scores[0][0] 

        # Point to best guess
        self.point_to_object(guess_idx)

        # Ask if the guess was right
        self.io.say("Is this the object you have in mind?")
        got_confirmation = False
        correct = False
        while not got_confirmation:
            got_confirmation = True
            self.io.say("Is this the object you have in mind?")
            confirmation = self.io.get()
            if self.is_yes(confirmation):
                correct = True
            elif not self.is_no(confirmation):
                got_confirmation = False
                self.io.say("I didn't catch that.")
                
        # Stop pointing
        self.retract_arm()

        # Add required classifier updates
        # Give a label of +1 to all predicates in current dialog with 
        # the object guess_idx
        new_preds = [predicate for predicate in self.cur_dialog_predicates if predicate not in self.known_predicates]
        self.known_predicates.extend(new_preds) # Needed here to get correct indices of new predicates
        pidxs = [self.known_predicates.index(predicate) for predicate in self.cur_dialog_predicates]
        oidxs = [guess_idx] * len(pidxs)
        labels = [1] * len(pidxs)
        success = self.update_classifiers(self, new_preds, pidxs, oidxs, labels)
        if success:
            for predicate in new_preds:
                self.unknown_predicates.remove(predicate)
        else:
            # Update didn't happen so undo the extension
            for predicate in new_preds:
                self.known_predicates.remove(predicate)
        self.classifiers_changed = self.classifiers_changed + self.cur_dialog_predicates


    # Identify the object and predicate for which a label should be obtained    
    def get_label_question_details(self):
        predicates = self.unknown_predicates + [predicate for predicate in self.min_confidence_objects.keys()]
        # Sample a predicate with probability proportional to 1 - confidence in lowest confidence object
        prob_numerators = [1.0] * len(self.unknown_predicates) + [(1.0 - self.min_confidence_objects[predicate][1]) for predicate in self.min_confidence_objects.keys()]
        probs = [(v / sum(prob_numerators)) for v in prob_numerators]
        predicate = np.random.choice(predicates, p=probs)
        
        if predicate in self.unknown_predicates:
            candidate_object = np.random.choice(self.objects_for_questions)
        else:
            candidate_object = self.min_confidence_objects[predicate]
        
        return predicate, candidate_object

    
    def ask_predicate_label(self):
        predicate, candidate_object = self.get_label_question_details()
        
        # Point to the object about which the question is
        self.point_to_object(candidate_object)
                    
        question_str = 'Would you use the word ' + predicate + ' to describe this object?'
        self.io.say(question_str)
        
        # Get response
        got_answer = False
        label_value = -1
        while not got_answer:
            got_answer = True
            self.io.say(question_str)
            answer = self.io.get()
            if self.is_yes(answer):
                label_value = 1
            elif not self.is_no(answer):
                got_answer = False
                self.io.say("I didn't catch that.")
                
        # Stop pointing
        self.retract_arm()
        
        # Add required classifier update
        if predicate in self.unknown_predicates:
            new_preds = [predicate]
        else:
            new_preds = []
        self.known_predicates.extend(new_preds) # Needed here to get correct indices of new predicates
        pidxs = [self.known_predicates.index(predicate)]
        oidxs = [candidate_object]
        labels = [label_value]
        success = self.update_classifiers(self, new_preds, pidxs, oidxs, labels)
        if success:
            if predicate in self.unknown_predicates:
                self.unknown_predicates.remove(predicate)
            self.classifiers_changed.append(predicate)
        else:
            # Update didn't happen so undo the extension
            if predicate in self.unknown_predicates:
                self.known_predicates.remove(predicate)
        

    def ask_positive_example(self):
        predicate = np.random.choice(self.unknown_predicates)
        
        question_str = 'Could you point to an object that you would describe as ' + predicate + '?'
        self.io.say(question_str)
        
        # Detect touch
        pos_detected, obj_idx_detected = self.detect_touch()
        
        # Add required classifier update
        new_preds = [predicate]
        self.known_predicates.extend(new_preds) # Needed here to get correct indices of new predicates
        pidxs = [self.known_predicates.index(predicate)]
        oidxs = [obj_idx_detected]
        labels = [1]
        success = self.update_classifiers(self, new_preds, pidxs, oidxs, labels)
        if success:
            self.unknown_predicates.remove(predicate)
            self.classifiers_changed.append(predicate)
        else:
            # Update didn't happen so undo the extension
            self.known_predicates.remove(predicate)

        
    # Start of human_take_turn of IspyAgent. Re-coded here because
    # only part of that function is to be used here
    def get_initial_description(self):
        self.io.say("Please pick an object that you see and describe it to me in one phrase.")

        understood = False
        guess_idx = None
        utterance = None
        predicates = None
        
        while not understood:
            user_response = self.io.get().strip()
            self.log('Get : ' + user_response + '\n')
            predicates = self.get_predicates(user_response)
            self.log("Predicates : " + str(predicates) + "\n")

            if len(predicates) > 0:
                understood = True
            # Failed to parse user response, so get a new one
            else:
                self.io.say("Sorry; I didn't catch that. Could you re-word your description?")

        self.cur_dialog_predicates = predicates

    
    def get_predicates(self, user_response):
        response_parts = user_response.split()
        predicates = [w for w in response_parts if w not in self.stopwords]
        unknown_predicates = [predicate for predicate in predicates if predicate not in self.known_predicates]
        return predicates
    
    
    # Given predicates and object idxs, return a map of results
    def get_classifier_results(self, predicates, obj_indices):
        if self.current_classifier_results is None:
            results = dict()
        else:
            results = self.current_classifier_results
            
        for obj_idx in obj_indices:
            this_obj_results = dict()
            for predicate in predicates:
                if predicate in self.known_predicates:
                    if predicate not in results or predicate in self.classifiers_changed:
                        classifier_idx = self.known_predicates.index(predicate)
                        [result, confidence] = self.run_classifier_on_object(classifier_idx, obj_idx)
                else:
                    [result, confidence] = [0, 0.0] 
                        # This is what a "classifier" without enough data 
                        # points for a hyperplane returns 
                this_obj_results[predicate] = [result, confidence]
            results[obj_idx] = this_obj_results
        self.current_classifier_results = results    
            
        return results
    
    
    # Calculate match scores from an object given a set of cnf clauses of predicates
    def get_match_scores(self, cnf_clauses):
        predicates = set(cnf_clauses)
        classifier_results = self.get_classifier_results(predicates, self.objects_for_guessing)

        match_scores = dict()
        sum_match_scores = 0.0
        for obj_idx in self.objects_for_guessing:
            predicate_scores = list()
            for predicate in predicates:
                # Compute per-predicate score as just result * confidence
                # Since result is 0-1, this is basically equal to the 
                # confidence that this predicate holds for this object
                [result, confidence] = classifier_results[obj_idx][predicate]
                predicate_scores.append(result * confidence)
            # Sum the predicate scores across predicates
            match_scores[obj_idx] = sum(predicate_scores)
            sum_match_scores += match_scores[obj_idx]
        
        # We need match scores that work like probabilities so normalize these
        for obj_idx in self.objects_for_guessing:
            match_scores[obj_idx] /= sum_match_scores

        return match_scores
    
    
    def update_min_confidence_objects(self):
        for predicate in self.known_predicates:
            if predicate not in self.min_confidence_objects or predicate in self.classifiers_changed:
                for obj_idx in self.objects_for_questions:
                    classifier_idx = self.known_predicates.index(predicate)
                    [result, confidence] = self.run_classifier_on_object(classifier_idx, obj_idx)
                    if predicate not in self.min_confidence_objects \
                        or confidence < self.min_confidence_objects[predicate][1]:
                            self.min_confidence_objects[predicate] = (obj_idx, confidence)

    
    def run_dialog(self):
        self.debug_print('In run_dialog')
        
        self.current_classifier_results = None
        self.classifiers_changed = list()
        self.num_dialog_turns = 0
        
        # Get initial user description and update state
        self.log('Action : get_initial_description')
        self.cur_dialog_predicates = None
        self.get_initial_description()  # Sets self.cur_dialog_predicates

        self.cur_match_scores = self.get_match_scores(predicates)
        self.log("Match scores :" + str(self.cur_match_scores) + "\n\n")
        
        dialog_state = self.get_dialog_state()
        dialog_action = self.policy.get_next_action(dialog_state)
        
        object_guessed = False
        try:
            while not object_guessed:
                self.log('\n\nAction : ' + dialog_action)
                self.num_dialog_turns += 1
                
                if dialog_action == 'make_guess':
                    self.make_guess(match_scores)
                    object_guessed = True
                    
                elif dialog_action == 'ask_predicate_label':
                    self.ask_predicate_label()
                    
                elif dialog_action == 'ask_positive_example':
                    self.ask_positive_example()
                
                # Update cached data for questions
                self.update_min_confidence_objects()
                self.classifiers_changed = list()
                    
                if not object_guessed:
                    # Recompute match scores because relevant classifiers may
                    # have changed
                    self.cur_match_scores = self.get_match_scores(self.cur_dialog_predicates)
                    self.log("Match scores : " + str(self.cur_match_scores) + "\n")

                    dialog_state = self.get_dialog_state()
                    dialog_action = self.policy.get_next_action(dialog_state)
                    
        except KeyboardInterrupt, SystemExit:
            raise
        except:
            # Log the exception
            self.log('\n' + traceback.format_exc() + '\n')
