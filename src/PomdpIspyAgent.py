#!/usr/bin/env python
__author__ = 'aishwarya'

import sys
import random
import rospy
import operator
import numpy as np
import traceback
from IspyAgent import IspyAgent
from HISBeliefState import HISBeliefState
from SummaryState import SummaryState
from SystemAction import SystemAction
from Utterance import Utterance

# TODO: Functionality for choosing next best question
class PomdpIspyAgent(UnitTestAgent):
    # initial_predicates - The agent needs to track which classifiers 
    #                      it can call. This gives the initial list 
    def __init__(self, io, stopwords_fn, policy, log_fn=None, initial_predicates=None):
        self.io = io
        self.log_fn = log_fn

        self.policy = policy
        self.knowledge = self.policy.knowledge
        
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
        
        # State and previous system action to track dialog progress
        self.knowledge = Knowledge()    # Provides info for initial state
        self.state = None  
        self.previous_system_action = SystemAction('repeat_goal')
        
        self.print_debug_messages = True 
        
        self.objects_for_guessing = self.table_oidxs[1]
        self.objects_for_questions = self.table_oidxs[0] + self.table_oidxs[2]
        
        # Some dummy vars to use in dialog state
        self.goal = 'point' 
        self.param_name = 'patient'
        

    def debug_print(self, message):
        if self.print_debug_messages:
            print 'PomdpIspyAgent:', message

    
    def log(self, log_str):
        if self.log_fn is not None:
            f = open(self.log_fn, 'a')
            f.write(log_str)
            f.close()


    def make_guess(self, match_scores):
        # match_scores.items() gives (obj_idx, score) tuples
        # Shuffle to randomize order of equally scored items on sorting
        obj_indices_with_scores = np.random.shuffle(match_scores.items())
        # Sort by match score to identify what the robot should guess
        sorted_match_scores = sorted(obj_indices_with_scores, key=operator.itemgetter(1), reverse=True)
        guess_idx = sorted_match_scores[0][0] 

        # TODO: Make the robot point the best guess

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
                
        # TODO: Stop pointing

        # TODO: Add any required classifier updates


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
        
        # TODO: Point to the object about which the question is (may require turning)
                    
        question_str = 'Would you use the word ' + predicate + ' to describe this object?'
        self.io.say(question_str)
        
        # Get response
        got_answer = False
        predicate_holds = False
        while not got_answer:
            got_answer = True
            self.io.say(question_str)
            answer = self.io.get()
            if self.is_yes(answer):
                predicate_holds = True
            elif not self.is_no(answer):
                got_answer = False
                self.io.say("I didn't catch that.")
                
        # TODO: Stop pointing
        
        # TODO: Add required classifier updates
        
        self.unknown_predicates.remove(predicate)
        self.known_predicates.append(predicate)


    def ask_positive_example(self):
        predicate = np.random.choice(self.unknown_predicates)
        
        touch_detected = False
        while not touch_detected:
            touch_detected = True
            
            question_str = 'Could you point to an object that you would describe as ' + predicate + '?'
            self.io.say(question_str)
            
            # TODO: Something extra may be needed here to allow for turning    
            # TODO: Detect touch
            
            if not touch_detected:
                self.io.say("I didn't catch that.")
        
        # TODO: Add required classifier updates
        
        self.unknown_predicates.remove(predicate)
        self.known_predicates.append(predicate)
        
        
    # Start of human_take_turn of IspyAgent. Re-coded here because
    # only part of that function is to be used here
    def get_initial_description(self):
        self.io.say("Please pick an object that you see and describe it to me in one phrase.")

        understood = False
        guess_idx = None
        utterance = None
        cnf_clauses = None
        
        while not understood:
            user_response = self.io.get().strip()
            self.log('Get : ' + user_response + '\n')
            cnf_clauses = self.get_predicate_cnf_clauses(user_response)
            self.log("CNF clauses : " + str(cnf_clauses) + "\n")

            if len(cnf_clauses) > 0:
                understood = True
            # Failed to parse user response, so get a new one
            else:
                self.io.say("Sorry; I didn't catch that. Could you re-word your description?")

        return user_response, cnf_clauses

    
    def get_predicate_cnf_clauses(self, user_response):
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
                        # TODO: Get classifier result
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
        
        # Account for prob that none are correct
        sum_match_scores += self.knowledge.non_n_best_prob
            
        # We need match scores that work like probabilities so normalize these
        for obj_idx in self.objects_for_guessing:
            match_scores[obj_idx] /= sum_match_scores

        return match_scores
    
    
    # Convert predicate match scores to utterances for updating state
    def get_utterances_from_match_scores(self, match_scores):
        utterances = list()
        for (obj_idx, score) in match_scores.items():
            params = {self.param_name : obj_idx}
            utterance = Utterance('inform_param', self.goal, params)   
            utterance.parse_prob = np.log(score)
            utterances.append(utterance)
            
        other_utterance = Utterance('-OTHER-', parse_prob=np.log(self.knowledge.non_n_best_prob))
        utterances.append(other_utterance)
        return utterances

    
    def update_min_confidence_objects(self):
        for predicate in self.known_predicates:
            if predicate not in self.min_confidence_objects or predicate in self.classifiers_changed:
                for obj_idx in self.objects_for_questions:
                    # TODO: Fetch [result, confidence] from classifier
                    if predicate not in self.min_confidence_objects \
                        or confidence < self.min_confidence_objects[predicate][1]:
                            self.min_confidence_objects[predicate] = (obj_idx, confidence)

    
    def run_dialog(self):
        self.debug_print('In run_dialog')
        
        self.current_classifier_results = None
        self.classifiers_changed = list()
        self.state = HISBeliefState(self.knowledge)
        self.state.domain_of_discourse = self.objects_for_guessing
        
        self.debug_print('Initial state')
        self.debug_print('Belief state: ' + str(self.state))
        
        # Get initial user description and update state
        self.log('Action : get_initial_description')
        self.previous_system_action = SystemAction('get_initial_description')
        user_response, cnf_clauses = self.get_initial_description()

        match_scores = self.get_match_scores(cnf_clauses)
        self.log("Match scores :" + str(match_scores) + "\n\n")
        
        utterances = self.get_utterances_from_match_scores(match_scores)
        self.state.update(self.previous_system_action, utterances)
        summary_state = SummaryState(self.state)
        dialog_action = self.policy.get_next_action(summary_state)
        self.previous_system_action = SystemAction(dialog_action)
        
        object_guessed = False
        try:
            while not object_guessed:
                self.log('\n\nAction : ' + dialog_action)
                
                if dialog_action == 'make_guess':
                    self.make_guess(match_scores)
                    object_guessed = True
                    
                elif dialog_action == 'ask_predicate_label':
                    self.ask_predicate_label()
                    
                elif dialog_action == 'ask_positive_example':
                    self.ask_positive_example()
                    
                if not object_guessed:
                    # Recompute match scores because relevant classifiers may
                    # have changed
                    match_scores = self.get_match_scores(cnf_clauses)
                    self.log("Match scores : " + str(match_scores) + "\n")
                        
                    # Update state based on new match scores
                    utterances = self.get_utterances_from_match_scores(match_scores)
                    self.state.update(self.previous_system_action, utterances)
                    summary_state = SummaryState(self.state)
                    dialog_action = self.policy.get_next_action(summary_state)
                    
                    # Update cached data for questions
                    self.update_min_confidence_objects()
                    self.classifiers_changed = list()
                    
        except KeyboardInterrupt, SystemExit:
            raise
        except:
            # Log the exception
            self.log('\n' + traceback.format_exc() + '\n')
