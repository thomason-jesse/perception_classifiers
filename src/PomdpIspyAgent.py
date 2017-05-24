#!/usr/bin/env python
__author__ = 'aishwarya'

import sys
import random
import rospy
import operator
from IspyAgent import IspyAgent
from HISBeliefState import HISBeliefState
from SummaryState import SummaryState
from SystemAction import SystemAction
from Utterance import Utterance

class PomdpIspyAgent(IspyAgent):
    def __init__(self, io, object_IDs, stopwords_fn, policy, learner, log_fn=None, alpha=0.9):
        IspyAgent.__init__(self, io, object_IDs, stopwords_fn, log_fn=None, alpha=0.9)
        
        self.policy = policy
        self.learner = learner 
        
        # TODO: self.dialog_actions and self.dialog_action_functions is 
        # a nice abstraction but may not be useful
        self.dialog_actions = ["make_guess", "ask_predicate_label"]
        self.dialog_action_functions = [self.make_guess, self.ask_predicate_label]
        
        # State and previous system action to track dialog progress
        self.knowledge = Knowledge()    # Provides info for initial state
        self.state = None  
        self.previous_system_action = SystemAction('repeat_goal')
        
        # TODO: 
        #   Have variables to store fold information
        #   Define a function to assign these, which updates 
        #   domain_of_discourse of learner
        
        self.print_debug_messages = True 


    def debug_print(self, message):
        if self.print_debug_messages:
            print 'PomdpIspyAgent:', message

    
    def log(self, log_str):
        if self.log_fn is not None:
            f = open(self.log_fn, 'a')
            f.write(log_str)
            f.close()


    def make_guess(self, match_scores):
        # Introduce small, random perturbations to identical scores to mix up guess order
        min_nonzero_margin = sys.maxint
        for i in match_scores:
            for j in match_scores:
                d = abs(match_scores[i]-match_scores[j])
                if 0 < d < min_nonzero_margin:
                    min_nonzero_margin = d
        for ob_idx in match_scores:
            if match_scores.values().count(match_scores[ob_idx]) > 1:
                match_scores[ob_idx] += (random.random()-0.5)*min_nonzero_margin

        # Then sort by match score to identify what the robot should guess
        sorted_match_scores = sorted(match_scores.items(), key=operator.itemgetter(1), reverse=True)
        guess_idx = sorted_match_scores[0][0] # sorted_match_scores is a list of (obj_idx, score) tuples

        # Make the robot take the best guess
        self.io.point(guess_idx)

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
        self.io.point(-1)  

        # TODO: Add any required classifier updates

    
    # Details of question to be asked will be present in 
    # self.previous_system_action['extra_data']
    def ask_predicate_label(self):
        predicate = system_action.extra_data['label_name']
        candidate_object = system_action.extra_data['candidate_idx']
        
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
                
        # Stop pointing
        self.io.point(-1)  
        
        # TODO: Add required classifier updates

        
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
            cnf_clauses = self.get_predicate_cnf_clauses_for_utterance(user_response)
            self.log("cnf_clauses:"+str(cnf_clauses)+"\n")
            
            # extract predicates and run these classifiers against each of objects_IDs to find best match
            if len(cnf_clauses) > 0:
                understood = True
                
            # Failed to parse user response, so get a new one
            else:
                self.io.say("Sorry; I didn't catch that. Could you re-word your description?")

        return user_response, cnf_clauses

    
    # Convert predicate match scores to utterances for updating state
    def get_utterances_from_match_scores(self, match_scores):
        raise NotImplementedError('Function get_utterances_from_match_scores is not defined')
        

    # TODO: Decide whether logging info needs to be changed. 
    # Eg: Should system actions be logged
    def run_dialog(self):
        self.debug_print('In run_dialog')
        
        if self.domain_of_discourse is None:
            raise RuntimeError('Set domain of discourse before starting dialogue!')
            
        self.state = HISBeliefState(self.knowledge)
        # TODO: Set self.state.domain_of_discourse appropriately
        self.state.learner_update(learner)
        
        self.debug_print('Initial state')
        self.debug_print('Belief state: ' + str(self.state))
        
        # Get initial user description and update state
        user_response, cnf_clauses = self.get_initial_description()

        match_scores = self.get_match_scores(cnf_clauses)
        self.log("match_scores:"+str(match_scores)+"\n")
        
        utterances = self.get_utterances_from_match_scores(match_scores)
        # TODO: Remove requirement for grounder to update state. Then add a state update here
        summary_state = SummaryState(self.state)
        (dialog_action, dialog_action_arg) = self.policy.get_next_action(summary_state)
        
        object_guessed = False
        try:
            while not object_guessed:
                if dialog_action == 'make_guess':
                    self.make_guess(match_scores)
                    object_guessed = True
                    
                elif dialog_action == 'ask_predicate_label':
                    self.previous_system_action = dialog_action_arg
                    self.ask_predicate_label()
                    
                # TODO: Update classifiers

                if not object_guessed:
                    # Recompute match scores because relevant classifiers may
                    # have changed
                    match_scores = self.get_match_scores(cnf_clauses)
                    self.log("match_scores:"+str(match_scores)+"\n")
                        
                    # Update state based on new match scores
                    utterances = self.get_utterances_from_match_scores(match_scores)
                    # TODO: Remove requirement for grounder to update state. Then add a state update here
                    # TODO: Make state fetch a new question from learner
                    summary_state = SummaryState(self.state)
                    (dialog_action, dialog_action_arg) = self.policy.get_next_action(summary_state)
                    
        except:
            # TODO: Save log before raising exception
            raise

        # TODO: Save log
