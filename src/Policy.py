#!/usr/bin/env python
__author__ = 'aishwarya'

import random 


class Policy:
    # ask_yes_no_prob - Probability that a question is yes-no rather 
    #   than example
    # max_questions - Maximum number of questions the robot is allowed
    #   to ask in a dialog
    # min_confidence_threshold - Minimum average confidence threshold
    #   across candidate yes-no questions to allow asking such a question
    # min_num_unknown_predicates - Minimum number of unknown predicates 
    #   required before asking for example
    def __init__(self, ask_yes_no_prob, max_questions, 
                 min_confidence_threshold, min_num_unknown_predicates):
        self.debug_print_level = 2
        self.ask_yes_no_prob = ask_yes_no_prob
        self.max_questions = max_questions
        self.min_confidence_threshold = min_confidence_threshold
        self.min_num_unknown_predicates = min_num_unknown_predicates

    # A util to control how much debug stuff is printed
    def debug_print(self, message, debug_level=2):
        if debug_level <= self.debug_print_level:
            print 'Policy:', message

    def ask_example_allowed(self, state):
        if state['only_dialog_relevant_questions']:
            candidates = set(state['cur_dialog_predicates']).difference(state['blacklisted_predicates'])
        else:
            candidates = set(state['unknown_predicates']).difference(state['blacklisted_predicates'])
        return len(candidates) > self.min_num_unknown_predicates

    def yes_no_allowed(self, state):
        confidences = [confidence for (predicate, (obj_idx, confidence)) in state['min_confidence_objects'].items()
                       if confidence is not None]
        self.debug_print('confidences = ' + str(confidences), 2)
        self.debug_print('state[\'unknown_predicates\'] = ' + str(state['unknown_predicates']), 2)
        num_candidates = len(state['min_confidence_objects'].items()) + len(state['unknown_predicates'])
        avg_confidence = sum(confidences) / float(num_candidates)
        self.debug_print('avg_confidence = ' + str(avg_confidence), 1)
        self.debug_print('self.min_confidence_threshold = ' + str(self.min_confidence_threshold), 1)
        return bool(avg_confidence < self.min_confidence_threshold)

    def get_next_action(self, state):
        if state['num_dialog_turns'] >= self.max_questions:
            return 'make_guess'
            
        else:
            ask_example_allowed = self.ask_example_allowed(state)
            yes_no_allowed = self.yes_no_allowed(state)
            self.debug_print('ask_example_allowed = ' + str(ask_example_allowed))
            self.debug_print('yes_no_allowed = ' + str(yes_no_allowed))
            
            if ask_example_allowed and yes_no_allowed:
                r = random.random()
                if r <= self.ask_yes_no_prob:
                    return 'ask_predicate_label'
                else:
                    return 'ask_positive_example'
            elif ask_example_allowed:
                return 'ask_positive_example'
            elif yes_no_allowed:
                return 'ask_predicate_label'
            else:
                return 'make_guess'
